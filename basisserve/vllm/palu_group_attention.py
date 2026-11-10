"""vLLM attention for a PaLU G-LRD group latent replicated on every rank of its group.

Each token slot stores this rank's dense Keys (one 128-wide row per local KV head) followed by
one copy of the group's low-rank Value latent. All local query heads read that same latent:
the fused DiffKV kernel is given a Value view whose KV-head stride is zero, so no attention
kernel is modified. Per-layer ranks produce per-layer backends and per-layer page sizes.
"""

from dataclasses import replace
from functools import lru_cache

import torch
import triton
import triton.language as tl
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVBackend,
    TritonAttentionDiffKVImpl,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import unified_attention_diffkv

KEY_DIM = 128


@triton.jit
def _pack_group_slot(key_ptr, latent_ptr, cache_ptr, slot_ptr, key_stride, latent_stride,
                     block_stride, token_stride, BLOCK_SIZE: tl.constexpr,
                     KEY_WIDTH: tl.constexpr, RANK: tl.constexpr, BLOCK_K: tl.constexpr,
                     BLOCK_R: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.load(slot_ptr + token).to(tl.int64)
    if slot < 0:
        return
    base = cache_ptr + (slot // BLOCK_SIZE) * block_stride + (slot % BLOCK_SIZE) * token_stride
    offs_k = tl.arange(0, BLOCK_K)
    keys = tl.load(key_ptr + token * key_stride + offs_k, mask=offs_k < KEY_WIDTH, other=0.0)
    tl.store(base + offs_k, keys, mask=offs_k < KEY_WIDTH)
    offs_r = tl.arange(0, BLOCK_R)
    latent = tl.load(latent_ptr + token * latent_stride + offs_r, mask=offs_r < RANK, other=0.0)
    tl.store(base + KEY_WIDTH + offs_r, latent, mask=offs_r < RANK)


def pack_group_slots(key, latent, cache, slot_mapping):
    """Write per-token Keys [tokens, kv_heads, 128] and latents [tokens, rank] into
    cache [blocks, block_size, kv_heads * 128 + rank]."""
    tokens, kv_heads, _ = key.shape
    rank = latent.shape[-1]
    key = key.reshape(tokens, kv_heads * KEY_DIM)
    _pack_group_slot[(slot_mapping.shape[0],)](
        key, latent, cache, slot_mapping, key.stride(0), latent.stride(0),
        cache.stride(0), cache.stride(1), BLOCK_SIZE=cache.shape[1],
        KEY_WIDTH=kv_heads * KEY_DIM, RANK=rank,
        BLOCK_K=next_power_of_2(kv_heads * KEY_DIM), BLOCK_R=next_power_of_2(rank))


def group_cache_views(kv_cache, kv_heads, rank):
    """Key view [blocks, block_size, kv_heads, 128] and a Value view [blocks, block_size,
    kv_heads, rank] whose head stride is zero, both over the logical [B, 1, N, C] cache."""
    flat = kv_cache[:, 0]
    keys = flat[..., :kv_heads * KEY_DIM].unflatten(-1, (kv_heads, KEY_DIM))
    latent = flat[..., kv_heads * KEY_DIM:kv_heads * KEY_DIM + rank]
    return flat, keys, latent.unsqueeze(2).expand(-1, -1, kv_heads, -1)


class PaluGroupMetadataBuilder(TritonAttentionMetadataBuilder):
    """Split-KV partial outputs are latent-shaped, not Key-shaped."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.softmax_segm_output = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments,
             next_power_of_2(kv_cache_spec.head_size_v)),
            dtype=torch.float32, device=device)


class PaluGroupImpl(TritonAttentionDiffKVImpl):
    def __init__(self, *args, value_rank, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_rank = value_rank

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        # Both local KV heads carry the same latent; store it once.
        flat = kv_cache[:, 0]
        pack_group_slots(key, value[:, 0], flat, slot_mapping)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return output.fill_(0)
        count = attn_metadata.num_actual_tokens
        _, keys, latent = group_cache_views(kv_cache, self.num_kv_heads, self.value_rank)
        unified_attention_diffkv(
            q=query[:count], k=keys, v=latent, out=output[:count],
            cu_seqlens_q=attn_metadata.query_start_loc, seqused_k=attn_metadata.seq_lens,
            softmax_scale=self.scale, causal=True, alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt, window_size=self.sliding_window,
            block_table=attn_metadata.block_table, softcap=self.logits_soft_cap,
            sinks=self.sinks, max_seqlen_q=attn_metadata.max_query_len,
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum)
        return output


@lru_cache(maxsize=None)
def palu_group_backend(rank):
    class PaluGroupBackend(TritonAttentionDiffKVBackend):
        head_size_v = rank

        @classmethod
        def customize_spec(cls, spec):
            # One slot per token: all local Keys plus one shared latent, in BF16.
            return replace(spec, num_head_slots=1,
                           state_content_bytes=(spec.num_kv_heads * KEY_DIM + rank) * 2)

        @staticmethod
        def get_impl_cls():
            return PaluGroupImpl

        @staticmethod
        def get_builder_cls():
            return PaluGroupMetadataBuilder

    PaluGroupBackend.__name__ = f"PaluGroupBackendR{rank}"
    PaluGroupBackend.__qualname__ = PaluGroupBackend.__name__
    return PaluGroupBackend
