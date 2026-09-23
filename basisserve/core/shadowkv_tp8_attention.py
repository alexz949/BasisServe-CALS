"""TP8 model adapter around the official ShadowKV CPU-offload cache."""

from contextlib import nullcontext
import gc
from types import MethodType

import torch
from flash_attn import flash_attn_with_kvcache
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from basisserve.core.llama31_8b_tp8_combined import decoder_layer_forward
from basisserve.core.shadowkv_tp8 import build_global_factors, make_tp8_cache
from evaluation.chunked_prefill_mlp import ChunkedTokenwise


class ShadowKVTP8Attention(nn.Module):
    def __init__(self, original, cache, cos_sin, phase):
        super().__init__()
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.layer_idx = original.layer_idx
        self.scaling = original.scaling
        self.cache = cache
        self.cos_sin = cos_sin
        self.phase = phase
        self.length = 0
        self.factor_records = []

    @torch.inference_mode()
    def forward(self, hidden_states, position_embeddings, **kwargs):
        batch, tokens, _ = hidden_states.shape
        cos, sin = position_embeddings
        cache = self.cache
        layer = self.layer_idx
        if self.length == 0:
            assert tokens > 1
            key = torch.empty(batch, 1, tokens, 128, device=hidden_states.device, dtype=hidden_states.dtype)
            value = torch.empty_like(key)
            for left in range(0, tokens, 2048):
                right = min(left + 2048, tokens)
                block = hidden_states[:, left:right]
                key[:, :, left:right] = self.k_proj(block).view(batch, right - left, 1, 128).transpose(1, 2)
                value[:, :, left:right] = self.v_proj(block).view(batch, right - left, 1, 128).transpose(1, 2)
            with self.phase("other_prepare", layer, -1):
                # Upstream get_svd performs this housekeeping before its SVD.
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            self.factor_records.append(build_global_factors(
                key, layer, cache.U[layer], cache.SV[layer], self.phase))
            for left in range(0, tokens, 2048):
                right = min(left + 2048, tokens)
                _, rotated = apply_rotary_pos_emb(key[:, :, left:right], key[:, :, left:right],
                                                  cos[:, left:right], sin[:, left:right])
                key[:, :, left:right].copy_(rotated)
            last_query = self.q_proj(hidden_states[:, -1:]).view(batch, 1, 4, 128).transpose(1, 2)
            last_query, _ = apply_rotary_pos_emb(last_query, last_query, cos[:, -1:], sin[:, -1:])
            with self.phase("other_prepare", layer, -1):
                cache.prefill_kv_cache(value, layer, key, last_query)
            for left in range(0, tokens, 2048):
                right = min(left + 2048, tokens)
                query = self.q_proj(hidden_states[:, left:right]).view(batch, right - left, 4, 128).transpose(1, 2)
                query, _ = apply_rotary_pos_emb(query, query, cos[:, left:right], sin[:, left:right])
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attention = F.scaled_dot_product_attention(
                        query, key[:, :, :right], value[:, :, :right],
                        attn_mask=causal_lower_right(right - left, right),
                        scale=self.scaling, enable_gqa=True)
                hidden_states[:, left:right].copy_(self.o_proj(
                    attention.transpose(1, 2).reshape(batch, right - left, 512).contiguous()))
        else:
            assert tokens == 1
            query = self.q_proj(hidden_states).view(batch, 1, 4, 128).transpose(1, 2)
            key = self.k_proj(hidden_states).view(batch, 1, 1, 128).transpose(1, 2)
            value = self.v_proj(hidden_states).view(batch, 1, 1, 128).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            cache.update_kv_cache(key, value, layer)
            positions = cache.get_retrieval_position_ids(layer, query)
            current = torch.cuda.current_stream()
            with torch.cuda.stream(cache.copy_stream):
                cache.copy_stream.wait_stream(current)
                selected_value = cache.get_value_cache(layer, positions)
            selected_key = cache.get_key_cache(layer, positions, None, self.cos_sin)
            current.wait_stream(cache.copy_stream)
            attention = flash_attn_with_kvcache(
                q=query.transpose(1, 2), k_cache=selected_key.transpose(1, 2),
                v_cache=selected_value.transpose(1, 2), causal=True, softmax_scale=self.scaling)
            hidden_states.copy_(self.o_proj(attention.reshape(batch, 1, 512).contiguous()))
        self.length += tokens
        return hidden_states, None


@torch.inference_mode()
def install_tp8_shadowkv(model, cache_class, *, batch, length, decode_steps, phase=None):
    assert model.config.model_type == "llama" and model.config.num_hidden_layers == 32
    assert model.config.num_attention_heads == 32 and model.config.num_key_value_heads == 8
    if phase is None:
        phase = lambda name, layer, request: nullcontext()
    device = model.get_input_embeddings().weight.device
    cache = make_tp8_cache(cache_class, layers=32, batch=batch, length=length,
                           decode_steps=decode_steps, device=device)
    positions = torch.arange(length + decode_steps, device=device)[None]
    cos, sin = model.model.rotary_emb(torch.empty(1, device=device, dtype=torch.bfloat16), positions)
    cos_sin = torch.cat((cos[0, :, :64], sin[0, :, :64]), -1).contiguous()
    modules = []
    for layer in model.model.layers:
        attention = ShadowKVTP8Attention(layer.self_attn, cache, cos_sin, phase)
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
