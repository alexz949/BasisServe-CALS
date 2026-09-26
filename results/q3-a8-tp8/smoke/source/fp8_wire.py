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


def quantize_e4m3_tensorwise_col_major(value: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a matrix and return the column-major layout required by scaled MM.

    The returned scale is the dequantization multiplier.  ``torch._scaled_mm``
    requires its right operand to be column-major on CUDA, so the layout change
    is performed once when a serving module is installed rather than in the hot
    path.
    """

    if value.ndim != 2:
        raise ValueError(f"FP8 decoder must be a matrix, got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError("FP8 decoder quantization requires floating-point weights")
    if value.numel() == 0:
        raise ValueError("FP8 decoder must be nonempty")
    scale = (value.detach().float().abs().amax() / FP8_E4M3_MAX).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    row_major = quantize_e4m3_static(value, scale)
    column_major = row_major.transpose(0, 1).contiguous().transpose(0, 1)
    return column_major, scale


def scaled_mm_e4m3_static(
    left: Tensor,
    right: Tensor,
    *,
    left_scale: Tensor,
    right_scale: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Run one tensorwise-scaled E4M3 GEMM with a BF16/FP16 result."""

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("scaled E4M3 inputs must be matrices")
    if int(left.shape[1]) != int(right.shape[0]):
        raise ValueError(
            f"scaled E4M3 K dimensions differ: {left.shape[1]} != {right.shape[0]}"
        )
    if left.dtype != FP8_E4M3_DTYPE or right.dtype != FP8_E4M3_DTYPE:
        raise TypeError("scaled E4M3 GEMM requires two float8_e4m3fn operands")
    if not left.is_cuda or not right.is_cuda or left.device != right.device:
        raise ValueError("scaled E4M3 GEMM requires colocated CUDA operands")
    if left.stride(1) != 1 or left.stride(0) <= left.stride(1):
        raise ValueError(f"left FP8 operand must be row-major, got {left.stride()}")
    if right.stride(0) != 1 or right.stride(1) <= right.stride(0):
        raise ValueError(f"right FP8 operand must be column-major, got {right.stride()}")
    if int(left.shape[1]) % 16:
        raise ValueError("scaled E4M3 GEMM K must be divisible by 16")
    if int(right.shape[1]) % 16:
        raise ValueError("scaled E4M3 GEMM N must be divisible by 16")
    if out_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("scaled E4M3 GEMM output must be FP16 or BF16")
    selected_scales: list[Tensor] = []
    for name, scale in (("left", left_scale), ("right", right_scale)):
        if not isinstance(scale, Tensor):
            raise TypeError(f"{name} scale must be a tensor")
        if scale.dtype != torch.float32 or scale.device != left.device:
            raise ValueError(f"{name} scale must be FP32 on the operand device")
        if scale.numel() != 1:
            raise ValueError(f"{name} scale must contain one value")
        selected_scales.append(scale)
    if torch.is_grad_enabled() and (
        left.requires_grad
        or right.requires_grad
        or selected_scales[0].requires_grad
        or selected_scales[1].requires_grad
    ):
        raise RuntimeError("scaled E4M3 decode is inference-only")
    return torch._scaled_mm(
        left,
        right,
        selected_scales[0],
        selected_scales[1],
        out_dtype=out_dtype,
        use_fast_accum=False,
    )


__all__ = [
    "FP8_E4M3_DTYPE",
    "FP8_E4M3_MAX",
    "decode_e4m3_bytes",
    "quantize_e4m3_static",
    "quantize_e4m3_tensorwise_col_major",
    "scaled_mm_e4m3_static",
]
