from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence, Set

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from .bitflip import flip_many_sign_bits_inplace


LAYER_RE = re.compile(r"\bmodel\.layers\.(\d+)\.")


@dataclass(frozen=True)
class LlmAttackConfig:
    first_l_layers: int = 5
    target_groups: Sequence[str] = ("attn", "mlp")
    attn_projections: Sequence[str] = ("q", "k", "v", "o")
    mlp_projections: Sequence[str] = ("gate", "up", "down")
    target_embeddings: str = "none"
    include_norms: bool = False


@dataclass(frozen=True)
class RankedLlmWeight:
    score: float
    param_name: str
    flat_index: int


@dataclass(frozen=True)
class GenerationResult:
    prompt_text: str
    generated_text: str
    generated_tokens: int


class SanitizeLogitsProcessor(LogitsProcessor):
    def __init__(self, clamp: float = 1e4):
        super().__init__()
        self.clamp = float(clamp)

    def __call__(self, input_ids, scores):  # type: ignore[override]
        torch.nan_to_num_(scores, nan=-self.clamp, posinf=self.clamp, neginf=-self.clamp)
        scores.clamp_(min=-self.clamp, max=self.clamp)
        return scores


def resolve_default_device_map() -> str | None:
    if not torch.cuda.is_available():
        return None
    if torch.cuda.device_count() == 1:
        return "cuda:0"
    return "balanced"


def _resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    torch_dtype = getattr(torch, str(dtype), None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return torch_dtype


def _load_tokenizer(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _load_model_class(model_name: str):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    auto_map = getattr(config, "auto_map", None) or {}
    reference = auto_map.get("AutoModelForCausalLM")
    if not reference:
        return AutoModelForCausalLM
    if isinstance(reference, (list, tuple)):
        reference = reference[0]
    return get_class_from_dynamic_module(reference, model_name, trust_remote_code=True)


def load_causal_lm(
    model_name: str,
    *,
    device_map: str | None = None,
    dtype: str | torch.dtype | None = torch.bfloat16,
):
    tokenizer = _load_tokenizer(model_name)
    model_cls = _load_model_class(model_name)

    load_kwargs = {"trust_remote_code": True}
    torch_dtype = _resolve_dtype(dtype)
    if torch_dtype is not None:
        load_kwargs["dtype"] = torch_dtype
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    try:
        model = model_cls.from_pretrained(model_name, **load_kwargs)
    except TypeError as exc:
        if "dtype" not in load_kwargs or "unexpected keyword argument" not in str(exc):
            raise
        torch_dtype = load_kwargs.pop("dtype")
        load_kwargs["torch_dtype"] = torch_dtype
        model = model_cls.from_pretrained(model_name, **load_kwargs)

    model.eval()
    return model, tokenizer


def _build_prompt(
    tokenizer,
    *,
    user_prompt: str,
    system_prompt: str,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = []
        if str(system_prompt).strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt")
        return prompt_text, encoded["input_ids"], encoded["attention_mask"]

    prompt_text = f"{system_prompt}\n\n{user_prompt}\nAssistant:".strip()
    encoded = tokenizer(prompt_text, return_tensors="pt")
    return prompt_text, encoded["input_ids"], encoded["attention_mask"]


def generate_text(
    model,
    tokenizer,
    *,
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 96,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int | None = 20,
) -> GenerationResult:
    prompt_text, input_ids, attention_mask = _build_prompt(
        tokenizer,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )

    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)
    attention_mask = attention_mask.to(first_device)

    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "do_sample": float(temperature) > 0.0,
        "pad_token_id": tokenizer.pad_token_id,
        "logits_processor": LogitsProcessorList([SanitizeLogitsProcessor()]),
    }
    if top_k is not None:
        generation_kwargs["top_k"] = int(top_k)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )[0]

    prompt_tokens = int(input_ids.shape[-1])
    generated_ids = output[prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return GenerationResult(
        prompt_text=prompt_text,
        generated_text=generated_text,
        generated_tokens=int(generated_ids.numel()),
    )


def parse_layer_index(param_name: str) -> int | None:
    match = LAYER_RE.search(param_name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _normalize_set(items: Sequence[str]) -> Set[str]:
    return {str(item).strip().lower() for item in items if str(item).strip()}


def eligible_param_names(
    all_param_names: Iterable[str],
    cfg: LlmAttackConfig,
) -> list[str]:
    target_groups = _normalize_set(cfg.target_groups)
    attn_proj = _normalize_set(cfg.attn_projections)
    mlp_proj = _normalize_set(cfg.mlp_projections)
    emb_mode = (cfg.target_embeddings or "none").strip().lower()

    include_attn = ("attn" in target_groups) or ("all" in target_groups)
    include_mlp = ("mlp" in target_groups) or ("ffn" in target_groups) or ("all" in target_groups)
    include_norms = cfg.include_norms or ("norms" in target_groups)

    out: list[str] = []
    for name in all_param_names:
        is_weight = str(name).endswith(".weight")
        is_moe_expert_param = (".mlp.experts." in str(name)) and (
            str(name).endswith(".gate_up_proj")
            or str(name).endswith(".down_proj")
            or str(name).endswith(".gate_proj")
            or str(name).endswith(".up_proj")
        )
        if not (is_weight or is_moe_expert_param):
            continue

        layer_index = parse_layer_index(str(name))

        if "embed_tokens.weight" in str(name) or str(name).endswith("embed_tokens.weight"):
            if emb_mode in {"input", "both"}:
                out.append(str(name))
            continue
        if str(name).endswith("lm_head.weight") or ".lm_head.weight" in str(name):
            if emb_mode in {"output", "both"}:
                out.append(str(name))
            continue

        if layer_index is None:
            if include_norms and (str(name).endswith("model.norm.weight") or str(name).endswith(".norm.weight")):
                out.append(str(name))
            continue
        if layer_index >= int(cfg.first_l_layers):
            continue

        if include_attn and (".self_attn." in str(name) or ".attention." in str(name)):
            for projection in ("q", "k", "v", "o"):
                if projection not in attn_proj:
                    continue
                if (
                    f".self_attn.{projection}_proj.weight" in str(name)
                    or f".attention.{projection}_proj.weight" in str(name)
                    or str(name).endswith(f".{projection}_proj.weight")
                ):
                    out.append(str(name))
                    break
            else:
                wants_kv = ("k" in attn_proj) or ("v" in attn_proj)
                wants_qkv = bool({"q", "k", "v"} & attn_proj)
                if wants_kv and (".kv_proj.weight" in str(name)):
                    out.append(str(name))
                elif wants_qkv and (".qkv_proj.weight" in str(name)):
                    out.append(str(name))
            continue

        if include_mlp and ".mlp." in str(name):
            mapping = {
                "gate": ("gate_proj", "w1", "gate_up_proj"),
                "up": ("up_proj", "w3", "gate_up_proj"),
                "down": ("down_proj", "w2", "down_proj"),
            }
            for short_name, candidates in mapping.items():
                if short_name not in mlp_proj:
                    continue
                if str(name).endswith(".weight"):
                    if any(f".{candidate}.weight" in str(name) for candidate in candidates):
                        out.append(str(name))
                        break
                else:
                    if any(str(name).endswith(f".{candidate}") for candidate in candidates):
                        out.append(str(name))
                        break
            continue

        if include_norms and (
            str(name).endswith(".input_layernorm.weight")
            or str(name).endswith(".post_attention_layernorm.weight")
        ):
            out.append(str(name))
            continue

    return out


def _topk_candidates_for_param(
    param_name: str,
    score_tensor: torch.Tensor,
    top_k: int,
) -> list[RankedLlmWeight]:
    flat = score_tensor.view(-1)
    local_k = min(int(top_k), int(flat.numel()))
    if local_k <= 0:
        return []

    values, indices = torch.topk(flat, k=local_k, largest=True, sorted=True)
    scores = values.detach().float().cpu().tolist()
    flat_indices = indices.detach().cpu().tolist()
    return [
        RankedLlmWeight(score=float(score), param_name=param_name, flat_index=int(flat_index))
        for score, flat_index in zip(scores, flat_indices)
    ]


@torch.no_grad()
def compute_dnl_ranking(
    model,
    cfg: LlmAttackConfig,
    *,
    top_k: int,
) -> list[RankedLlmWeight]:
    named_tensors = {
        **{name: buffer for name, buffer in model.named_buffers()},
        **{name: param for name, param in model.named_parameters()},
    }
    candidates: list[RankedLlmWeight] = []

    for param_name in eligible_param_names(named_tensors.keys(), cfg):
        tensor = named_tensors[param_name]
        if tensor is None or not tensor.is_floating_point() or int(tensor.numel()) == 0:
            continue
        candidates.extend(
            _topk_candidates_for_param(
                param_name,
                tensor.detach().float().abs(),
                top_k=top_k,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[: int(top_k)]


def _input_device(model) -> torch.device:
    for name, param in model.named_parameters():
        if "embed_tokens.weight" in name:
            return param.device
    return next(model.parameters()).device


def compute_one_pass_grads(
    model,
    tokenizer,
    *,
    seed: int = 0,
    seq_len: int = 32,
) -> None:
    model.zero_grad(set_to_none=True)
    model.eval()

    for param in model.parameters():
        if param is None:
            continue
        try:
            param.requires_grad_(bool(param.is_floating_point()))
        except Exception:
            param.requires_grad_(False)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, int(seq_len)),
        generator=generator,
        dtype=torch.long,
    ).to(_input_device(model))

    with torch.enable_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
        logits.sum().backward()


def compute_1p_dnl_ranking(
    model,
    tokenizer,
    cfg: LlmAttackConfig,
    *,
    top_k: int,
    seed: int = 0,
    seq_len: int = 32,
) -> list[RankedLlmWeight]:
    compute_one_pass_grads(model, tokenizer, seed=seed, seq_len=seq_len)

    named_tensors = {
        **{name: buffer for name, buffer in model.named_buffers()},
        **{name: param for name, param in model.named_parameters()},
    }
    candidates: list[RankedLlmWeight] = []

    for param_name in eligible_param_names(named_tensors.keys(), cfg):
        tensor = named_tensors[param_name]
        if tensor is None or not tensor.is_floating_point() or int(tensor.numel()) == 0:
            continue
        grad = getattr(tensor, "grad", None)
        if grad is None:
            continue

        weight = tensor.detach().float()
        grad = grad.detach().float()
        score = weight.abs() + (weight * grad + 0.5 * weight.square() * grad.square()).abs()
        candidates.extend(_topk_candidates_for_param(param_name, score, top_k=top_k))

    model.zero_grad(set_to_none=True)
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[: int(top_k)]


@torch.no_grad()
def apply_llm_attack(
    model,
    ranking: list[RankedLlmWeight],
    *,
    k: int,
) -> None:
    named_params = {name: param for name, param in model.named_parameters()}
    flip_many_sign_bits_inplace(
        named_params,
        [(candidate.param_name, candidate.flat_index) for candidate in ranking[: int(k)]],
    )
