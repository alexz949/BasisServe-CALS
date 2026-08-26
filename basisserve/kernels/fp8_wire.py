"""Static-scaled E4M3 communication-wire utilities."""

from __future__ import annotations

import torch
from torch import Tensor


FP8_E4M3_DTYPE = torch.float8_e4m3fn
FP8_E4M3_MAX = float(torch.finfo(FP8_E4M3_DTYPE).max)


def quantize_e4m3_static(value: Tensor, scale: Tensor | float) -> Tensor:
    """Quantize a floating tensor with one calibrated dequantization scale."""

    if not value.is_floating_point():
        raise TypeError("E4M3 wire quantization requires a floating-point tensor")
    work_scale = torch.as_tensor(scale, device=value.device, dtype=torch.float32)
    if work_scale.numel() != 1:
        raise ValueError("static E4M3 wire quantization requires one scalar scale")
    if work_scale.device.type == "cpu" and (
        not bool(torch.isfinite(work_scale).item())
        or not bool((work_scale > 0).item())
    ):
        raise ValueError("static E4M3 wire scale must be finite and positive")
    normalized = value.float() / work_scale
    return normalized.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_E4M3_DTYPE)


def decode_e4m3_bytes(value: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Interpret raw E4M3 bytes and cast their unscaled codes to FP16/BF16."""

    if value.dtype != torch.uint8:
        raise TypeError("E4M3 communication arena must use torch.uint8 storage")
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("E4M3 wire decoding supports FP16 or BF16 output")
    return value.view(FP8_E4M3_DTYPE).to(dtype=dtype)


__all__ = [
    "FP8_E4M3_DTYPE",
    "FP8_E4M3_MAX",
    "decode_e4m3_bytes",
    "quantize_e4m3_static",
]
