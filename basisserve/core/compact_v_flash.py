"""Fused attention over selected exact K and compact V support."""

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def compact_v_flash_attention(query, key, value, *, scale):
    assert query.is_cuda and query.dtype in (torch.float16, torch.bfloat16)
    assert key.dtype == value.dtype == query.dtype
    assert key.shape[:3] == value.shape[:3] and value.shape[-1] <= query.shape[-1]
    rank = value.shape[-1]
    padded = F.pad(value, (0, query.shape[-1]-rank))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = F.scaled_dot_product_attention(query, key, padded, scale=scale,
            dropout_p=0.0, is_causal=False, enable_gqa=True)
    return output[..., :rank]
