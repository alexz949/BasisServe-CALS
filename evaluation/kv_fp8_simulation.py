"""Simulated FP8 (E4M3) KV cache for the routing evaluators: quantize -> dequantize, BF16 storage.
Numerics of a physical E4M3 cache without packed kernels. Keys (post-RoPE, as cached) get one FP32 absmax scale per
token and KV head; the V96 Value latent gets one FP32 absmax scale per page of `page_size` tokens and KV head (the
routing page), decode tokens forming single-token pages. Prefill attention keeps the fresh BF16 K/V (the cache is
written after it); every decode step reads the dequantized cache."""
import torch

from basisserve.core.fp8_value_latent import quantize_paged_e4m3
from basisserve.kernels.fp8_wire import FP8_E4M3_MAX


def fp8_key_roundtrip(key):
    """key: [batch, kv_heads, tokens, head_dim] -> same shape/dtype after an E4M3 round trip with per-token scales."""
    scale = (key.float().abs().amax(-1, keepdim=True) / FP8_E4M3_MAX).clamp_min(1e-30)
    codes = (key.float() / scale).to(torch.float8_e4m3fn)
    return (codes.float() * scale).to(key.dtype)


def fp8_value_roundtrip(value, page_size):
    """value: [batch, kv_heads, tokens, width] V latent -> E4M3 round trip with one scale per page of `page_size` tokens."""
    return quantize_paged_e4m3(value, page_size=page_size).dequantize(value.dtype)
