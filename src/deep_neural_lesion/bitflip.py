from __future__ import annotations

import torch


def _sign_mask(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 1 << 31
    if dtype in (torch.float16, torch.bfloat16):
        return 1 << 15
    raise TypeError(f"Unsupported dtype for sign-bit flip: {dtype}")


@torch.no_grad()
def flip_sign_bit_inplace(param: torch.Tensor, flat_index: int) -> None:
    """Flip the sign bit of a single floating-point scalar in-place."""
    if param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"Unsupported dtype for sign-bit flip: {param.dtype}")
    if not param.is_contiguous():
        param.data = param.data.contiguous()

    flat = param.view(-1)
    idx = int(flat_index)
    if idx < 0 or idx >= int(flat.numel()):
        raise IndexError(f"flat_index out of range: {flat_index}")

    if param.dtype == torch.float32:
        flat.view(torch.int32)[idx] ^= _sign_mask(param.dtype)
        return

    flat.view(torch.int16)[idx] ^= _sign_mask(param.dtype)


@torch.no_grad()
def flip_many_sign_bits_inplace(
    named_params: dict[str, torch.Tensor],
    selections: list[tuple[str, int]],
) -> None:
    """Flip a list of `(param_name, flat_index)` entries in-place.

    Reapplying the same selections restores the original values.
    """
    for param_name, flat_index in selections:
        param = named_params.get(str(param_name))
        if param is None:
            raise KeyError(f"Unknown parameter: {param_name}")
        flip_sign_bit_inplace(param, int(flat_index))
