"""Fused E4M3 conversion into/out of the feature-major communication arena."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(X, S, Y, M: tl.constexpr, K: tl.constexpr, S0: tl.constexpr,
          S1: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = (m[:, None] < M) & (k[None, :] < K)
    x = tl.load(X + m[:, None] * S0 + k[None, :] * S1, mask, 0).to(tl.float32)
    scale = tl.load(S)
    normalized = tl.div_rn(x, scale)
    normalized = tl.minimum(tl.maximum(normalized, -448.0), 448.0)
    # Avoid the FP32 -> FP16 -> FP8 double rounding in the generic lowering.
    codes = tl.inline_asm_elementwise(
        "{ .reg .b16 v; cvt.rn.satfinite.e4m3x2.f32 v, $1, $1; cvt.u32.u16 $0, v; }",
        constraints="=r,f", args=[normalized], dtype=tl.uint32, is_pure=True, pack=1)
    tl.store(Y + k[None, :] * M + m[:, None], codes.to(tl.uint8), mask)


@triton.jit
def _unpack(X, S, Y, M: tl.constexpr, K: tl.constexpr, DEQUANT: tl.constexpr,
            BM: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = (m[:, None] < M) & (k[None, :] < K)
    codes = tl.load(X + k[None, :] * M + m[:, None], mask, 0)
    if DEQUANT:
        value = codes.to(tl.float8e4nv, bitcast=True).to(tl.float32) * tl.load(S)
    else:
        value = codes
    tl.store(Y + m[:, None] * K + k[None, :], value, mask)


def pack_latent_a8(value, scale, feature_bytes):
    assert value.ndim == 2 and value.dtype == torch.bfloat16 and value.is_cuda
    m, k = value.shape
    assert feature_bytes.shape == (k, m) and feature_bytes.dtype == torch.uint8
    assert feature_bytes.is_contiguous() and feature_bytes.device == value.device == scale.device
    assert scale.dtype == torch.float32 and scale.numel() == 1
    bm, bk = min(16, triton.next_power_of_2(m)), 64
    _pack[(triton.cdiv(m, bm), triton.cdiv(k, bk))](
        value, scale, feature_bytes, m, k, *value.stride(), bm, bk, num_warps=4)
    return feature_bytes


def unpack_latent_a8(feature_bytes, scale, output):
    assert feature_bytes.ndim == 2 and feature_bytes.dtype == torch.uint8
    k, m = feature_bytes.shape
    assert output.shape == (m, k) and output.is_contiguous() and feature_bytes.is_contiguous()
    assert output.dtype in (torch.bfloat16, torch.float8_e4m3fn)
    assert output.device == feature_bytes.device == scale.device
    bm, bk = min(16, triton.next_power_of_2(m)), 64
    target = output if output.dtype == torch.bfloat16 else output.view(torch.uint8)
    _unpack[(triton.cdiv(m, bm), triton.cdiv(k, bk))](
        feature_bytes, scale, target, m, k, output.dtype == torch.bfloat16,
        bm, bk, num_warps=4)
    return output
