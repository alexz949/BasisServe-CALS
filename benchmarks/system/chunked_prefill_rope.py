"""Bound inference-only RoPE temporaries using the existing elementwise operator."""

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


PREFILL_ROPE_CHUNK = 2048


def apply_prefill_rope_(query, key, cos, sin):
    """Overwrite independent Q/K projection buffers, preserving their layout."""
    assert not torch.is_grad_enabled()
    assert query.shape[2] == key.shape[2] == cos.shape[-2] == sin.shape[-2]
    for start in range(0, query.shape[2], PREFILL_ROPE_CHUNK):
        stop = start + PREFILL_ROPE_CHUNK
        q, k = apply_rotary_pos_emb(
            query[:, :, start:stop], key[:, :, start:stop],
            cos[..., start:stop, :], sin[..., start:stop, :],
        )
        query[:, :, start:stop].copy_(q)
        key[:, :, start:stop].copy_(k)
    return query, key
