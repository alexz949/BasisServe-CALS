"""Tensor-parallel model adapter around the official ShadowKV CPU-offload cache (TP8, TP4, or one GPU).

Rank r of W runs KV heads [r*8/W, (r+1)*8/W) and their query heads; the official cache is instantiated for
those local heads (shadowkv_tp8.make_tp8_cache). Landmarks, outliers, selection, Key reconstruction and Value
fetching are per KV head and run unchanged; the global SVD is gathered on the layer's owner.

A prefill forward fills cache rows starting at cache.prefilled_batch, so a batch may be prefilled at once or
one request at a time (the official cache advances prefilled_batch after the last layer). Decode attention is
FlashAttention-2 (vLLM's build), reading each request's selected-token buffer as one paged-KV block.
"""

from contextlib import nullcontext
from types import MethodType

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from basisserve.core.llama31_8b_tp8_combined import decoder_layer_forward
from basisserve.core.shadowkv_tp8 import (HEAD_DIM, KV_HEADS, QUERY_HEADS, build_global_factors,
                                         make_tp8_cache, world_size)
from evaluation.chunked_prefill_mlp import ChunkedTokenwise


class FlashDecode:
    """FlashAttention-2 decode of one query token per request over each request's first `length` buffer tokens."""

    def __init__(self, batch, device):
        self.blocks = torch.arange(batch, device=device, dtype=torch.int32)[:, None]
        self.queries = torch.arange(batch + 1, device=device, dtype=torch.int32)
        self.lengths = torch.empty(batch, device=device, dtype=torch.int32)

    def __call__(self, query, key_buffer, value_buffer, length, scale):
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

        self.lengths.fill_(length)
        # [B, kv, capacity, 128] -> [blocks=B, block_size=capacity, kv, 128] (strided view, no copy)
        return flash_attn_varlen_func(
            query[:, :, 0], key_buffer.transpose(1, 2), value_buffer.transpose(1, 2), 1, self.queries, length,
            seqused_k=self.lengths, softmax_scale=scale, causal=True, block_table=self.blocks, fa_version=2)


def _local(weight):
    return weight.to_local() if hasattr(weight, "to_local") else weight


class ShadowKVTP8Attention(nn.Module):
    def __init__(self, original, cache, cos_sin, phase, decode):
        super().__init__()
        # Local weight shards used directly (as every other V2Q8 arm), bypassing HF tensor-parallel hooks; the
        # row-parallel output is AllReduced explicitly, exactly what HF RowwiseParallel does.
        self.q_weight = _local(original.q_proj.weight.detach())
        self.k_weight = _local(original.k_proj.weight.detach())
        self.v_weight = _local(original.v_proj.weight.detach())
        self.o_weight = _local(original.o_proj.weight.detach())
        self.layer_idx = original.layer_idx
        self.scaling = original.scaling
        self.kv_heads = KV_HEADS // world_size()
        self.query_heads = QUERY_HEADS // world_size()
        self.cache = cache
        self.cos_sin = cos_sin
        self.phase = phase
        self.decode = decode
        self.length = 0
        self.factor_records = []

    def _output(self, attention):
        projected = F.linear(attention, self.o_weight)
        if world_size() > 1:
            dist.all_reduce(projected)
        return projected

    @torch.inference_mode()
    def forward(self, hidden_states, position_embeddings, **kwargs):
        batch, tokens, _ = hidden_states.shape
        cos, sin = position_embeddings
        cache = self.cache
        layer = self.layer_idx
        kv, heads = self.kv_heads, self.query_heads
        if tokens > 1:
            row = cache.prefilled_batch
            key = torch.empty(batch, kv, tokens, HEAD_DIM, device=hidden_states.device, dtype=hidden_states.dtype)
            value = torch.empty_like(key)
            for left in range(0, tokens, 2048):
                right = min(left + 2048, tokens)
                block = hidden_states[:, left:right]
                key[:, :, left:right] = F.linear(block, self.k_weight).view(batch, right - left, kv, HEAD_DIM).transpose(1, 2)
                value[:, :, left:right] = F.linear(block, self.v_weight).view(batch, right - left, kv, HEAD_DIM).transpose(1, 2)
            # Upstream get_svd runs gc.collect() + empty_cache() before every SVD (once per layer per prefill batch).
            # It is memory housekeeping, not part of the method; with one-request-at-a-time prefill it would run
            # B times per layer and inflate ShadowKV's prefill, so it is not issued here.
            self.factor_records.append(build_global_factors(
                key, layer, cache.U[layer][row:row + batch], cache.SV[layer][row:row + batch], self.phase))
            for left in range(0, tokens, 2048):
                right = min(left + 2048, tokens)
                _, rotated = apply_rotary_pos_emb(key[:, :, left:right], key[:, :, left:right],
                                                  cos[:, left:right], sin[:, left:right])
                key[:, :, left:right].copy_(rotated)
            last_query = F.linear(hidden_states[:, -1:], self.q_weight).view(batch, 1, heads, HEAD_DIM).transpose(1, 2)
            last_query, _ = apply_rotary_pos_emb(last_query, last_query, cos[:, -1:], sin[:, -1:])
            with self.phase("landmarks_outliers_and_value_offload", layer, -1):
                cache.prefill_kv_cache(value, layer, key, last_query)
            for left in range(0, tokens, 2048):
                right = min(left + 2048, tokens)
                query = F.linear(hidden_states[:, left:right], self.q_weight).view(
                    batch, right - left, heads, HEAD_DIM).transpose(1, 2)
                query, _ = apply_rotary_pos_emb(query, query, cos[:, left:right], sin[:, left:right])
                with self.phase("prefill_attention", layer, -1), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attention = F.scaled_dot_product_attention(
                        query, key[:, :, :right], value[:, :, :right],
                        attn_mask=causal_lower_right(right - left, right),
                        scale=self.scaling, enable_gqa=True)
                hidden_states[:, left:right].copy_(self._output(
                    attention.transpose(1, 2).reshape(batch, right - left, heads * HEAD_DIM).contiguous()))
            self.length = tokens
        else:
            query = F.linear(hidden_states, self.q_weight).view(batch, 1, heads, HEAD_DIM).transpose(1, 2)
            key = F.linear(hidden_states, self.k_weight).view(batch, 1, kv, HEAD_DIM).transpose(1, 2)
            value = F.linear(hidden_states, self.v_weight).view(batch, 1, kv, HEAD_DIM).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            with self.phase("append_new_token", layer, -1):
                cache.update_kv_cache(key, value, layer)
            with self.phase("selection_landmark_scores_topk", layer, -1):
                positions = cache.get_retrieval_position_ids(layer, query)
            current = torch.cuda.current_stream()
            # Upstream overlaps the Value fetch (copy stream) with Key reconstruction; breakdown timers serialize them.
            with self.phase("value_miss_fetch_h2d", layer, -1), torch.cuda.stream(cache.copy_stream):
                cache.copy_stream.wait_stream(current)
                selected_value = cache.get_value_cache(layer, positions)
            with self.phase("key_miss_reconstruct", layer, -1):
                selected_key = cache.get_key_cache(layer, positions, None, self.cos_sin)
                current.wait_stream(cache.copy_stream)
            # selected_* are views of the first L tokens of this layer's buffers.
            with self.phase("decode_attention", layer, -1):
                attention = self.decode(query, cache.k_cache_buffer[layer], cache.v_cache_buffer[layer],
                                        selected_key.shape[2], self.scaling)
            del selected_value
            hidden_states.copy_(self._output(attention.reshape(batch, 1, heads * HEAD_DIM).contiguous()))
            self.length += 1
        return hidden_states, None


@torch.inference_mode()
def install_tp8_shadowkv(model, cache_class, *, batch, length, decode_steps, phase=None):
    assert model.config.model_type == "llama" and model.config.num_hidden_layers == 32
    assert model.config.num_attention_heads == QUERY_HEADS and model.config.num_key_value_heads == KV_HEADS
    if phase is None:
        phase = lambda name, layer, request: nullcontext()
    device = model.get_input_embeddings().weight.device
    cache = make_tp8_cache(cache_class, layers=32, batch=batch, length=length,
                           decode_steps=decode_steps, device=device)
    positions = torch.arange(length + decode_steps, device=device)[None]
    cos, sin = model.model.rotary_emb(torch.empty(1, device=device, dtype=torch.bfloat16), positions)
    cos_sin = torch.cat((cos[0, :, :64], sin[0, :, :64]), -1).contiguous()
    decode = FlashDecode(batch, device)
    modules = []
    for layer in model.model.layers:
        attention = ShadowKVTP8Attention(layer.self_attn, cache, cos_sin, phase, decode)
        layer.self_attn = attention
        layer.input_layernorm = ChunkedTokenwise(layer.input_layernorm, chunk_size=2048)
        layer.forward = MethodType(decoder_layer_forward, layer)
        modules.append(attention)
    model.model.norm = ChunkedTokenwise(model.model.norm, chunk_size=2048)
    return modules, cache


def cache_state_bytes(cache):
    groups = {
        "u": ("U",), "sv": ("SV",), "landmarks": ("k_landmark", "k_landmark_idx"),
        "cpu_value": ("v_cache_cpu",),
        "selected_buffers": ("k_cache_buffer", "v_cache_buffer", "temp", "output"),
        "routing_workspace": ("offsets", "cnts", "signals", "position_ids", "gemm_o", "softmax_o", "norm", "sum"),
    }
    return {group: {"bytes": sum(getattr(cache, name).numel() * getattr(cache, name).element_size()
                                 for name in names),
                    "devices": sorted({str(getattr(cache, name).device) for name in names})}
            for group, names in groups.items()}
