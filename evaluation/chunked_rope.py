"""Token-blocked in-place RoPE for inference-only Q/K projection outputs."""
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


@torch.inference_mode()
def chunked_rotary_inplace(q, k, cos, sin, unsqueeze_dim=1):
    assert unsqueeze_dim == 1 and q.ndim == k.ndim == 4
    assert cos.ndim == sin.ndim == 3
    for start in range(0, q.shape[2], 1024):
        stop = start + 1024
        qs, ks = apply_rotary_pos_emb(q[:, :, start:stop], k[:, :, start:stop],
            cos[:, start:stop], sin[:, start:stop], unsqueeze_dim=unsqueeze_dim)
        q[:, :, start:stop].copy_(qs)
        k[:, :, start:stop].copy_(ks)
    return q, k


@torch.inference_mode()
def chunked_rotary_preserve_keys(q, k, cos, sin, unsqueeze_dim=1):
    # ShadowKV separately fits its SVD to the original pre-RoPE keys.
    return chunked_rotary_inplace(q, k.clone(), cos, sin, unsqueeze_dim)
