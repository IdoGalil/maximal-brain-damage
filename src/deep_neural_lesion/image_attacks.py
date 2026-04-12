from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torchvision.models import ResNet50_Weights

from .bitflip import flip_many_sign_bits_inplace


@dataclass(frozen=True)
class RankedImageWeight:
    score: float
    param_name: str
    flat_index: int
    layer_name: str
    filter_index: int


@dataclass(frozen=True)
class ImagePrediction:
    top_index: int
    top_label: str
    top_probability: float
    top5: list[tuple[str, float]]


def ensure_demo_dalmatian_image(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected demo image at {path}")
    return path


def load_timm_imagenet_model(
    model_name: str,
    *,
    device: str | torch.device | None = None,
):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    resolved_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(resolved_device)

    data_config = resolve_data_config({}, model=model)
    transform = create_transform(**data_config)
    class_names = list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
    input_size = tuple(int(v) for v in data_config["input_size"])
    return model, transform, class_names, input_size


def load_timm_resnet50(device: str | torch.device | None = None):
    return load_timm_imagenet_model("resnet50", device=device)


def _iter_attackable_layers(
    model: torch.nn.Module,
    first_l_layers: int,
) -> Iterable[tuple[str, torch.nn.Module]]:
    seen = 0
    for layer_name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            yield layer_name, module
            seen += 1
            if seen >= int(first_l_layers):
                break


def _rowwise_top_candidates(
    param_name: str,
    layer_name: str,
    score_tensor: torch.Tensor,
) -> list[RankedImageWeight]:
    rows = score_tensor.view(int(score_tensor.shape[0]), -1)
    row_scores, row_indices = torch.max(rows, dim=1)
    row_width = int(rows.shape[1])

    scores = row_scores.detach().float().cpu().tolist()
    indices = row_indices.detach().cpu().tolist()
    out: list[RankedImageWeight] = []
    for filter_index, (score, inner_index) in enumerate(zip(scores, indices)):
        flat_index = int(filter_index) * row_width + int(inner_index)
        out.append(
            RankedImageWeight(
                score=float(score),
                param_name=param_name,
                flat_index=flat_index,
                layer_name=layer_name,
                filter_index=int(filter_index),
            )
        )
    return out


@torch.no_grad()
def compute_dnl_ranking(
    model: torch.nn.Module,
    *,
    first_l_layers: int = 10,
    top_k: int = 8,
) -> list[RankedImageWeight]:
    candidates: list[RankedImageWeight] = []
    for layer_name, module in _iter_attackable_layers(model, first_l_layers=first_l_layers):
        param_name = f"{layer_name}.weight" if layer_name else "weight"
        score = module.weight.detach().float().abs()
        candidates.extend(_rowwise_top_candidates(param_name, layer_name, score))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[: int(top_k)]


def compute_1p_dnl_ranking(
    model: torch.nn.Module,
    *,
    input_size: tuple[int, int, int],
    first_l_layers: int = 10,
    top_k: int = 8,
    seed: int = 0,
) -> list[RankedImageWeight]:
    model.zero_grad(set_to_none=True)
    model.eval()

    for param in model.parameters():
        if param is not None and param.is_floating_point():
            param.requires_grad_(True)

    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_input = torch.randn((1, *input_size), generator=generator, dtype=torch.float32).to(device)

    with torch.enable_grad():
        logits = model(random_input)
        logits.sum().backward()

    candidates: list[RankedImageWeight] = []
    for layer_name, module in _iter_attackable_layers(model, first_l_layers=first_l_layers):
        grad = module.weight.grad
        if grad is None:
            continue

        weight = module.weight.detach().float()
        grad = grad.detach().float()
        score = weight.abs() + (weight * grad + 0.5 * weight.square() * grad.square()).abs()
        param_name = f"{layer_name}.weight" if layer_name else "weight"
        candidates.extend(_rowwise_top_candidates(param_name, layer_name, score))

    model.zero_grad(set_to_none=True)
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[: int(top_k)]


@torch.no_grad()
def apply_image_attack(
    model: torch.nn.Module,
    ranking: list[RankedImageWeight],
    *,
    k: int,
) -> None:
    named_params = dict(model.named_parameters())
    flip_many_sign_bits_inplace(
        named_params,
        [(candidate.param_name, candidate.flat_index) for candidate in ranking[: int(k)]],
    )


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    transform,
    class_names: list[str],
    image_path: str | Path,
    *,
    top_k: int = 5,
) -> ImagePrediction:
    image = Image.open(image_path).convert("RGB")
    device = next(model.parameters()).device
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities[0], k=int(top_k))

    probs = top_probs.detach().float().cpu().tolist()
    indices = top_indices.detach().cpu().tolist()
    top5 = [(class_names[int(idx)], float(prob)) for idx, prob in zip(indices, probs)]

    return ImagePrediction(
        top_index=int(indices[0]),
        top_label=class_names[int(indices[0])],
        top_probability=float(probs[0]),
        top5=top5,
    )
