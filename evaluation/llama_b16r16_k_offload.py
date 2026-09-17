"""Llama B16R16 with GPU routing/V96 and pinned-CPU exact K."""
from types import MethodType

import torch

from evaluation import eval_k_routing_ruler as runtime
from evaluation.chunked_prefill_mlp import install_chunked_prefill_norms


class KOffloadRoutingCache(runtime.RoutingCache):
    def __init__(self, config):
        super().__init__(config=config)
        self.key_capacity = int(config.max_position_embeddings)
        self.key_storage = {}
        self.key_staging = {}


def _offload_prefill_key(cache, layer, key):
    capacity = cache.key_capacity
    assert key.dtype == torch.bfloat16 and key.is_cuda and key.shape[2] <= capacity
    storage = torch.empty((*key.shape[:2], capacity, key.shape[-1]),
                          dtype=key.dtype, device='cpu', pin_memory=True)
    storage[:, :, :key.shape[2]].copy_(key)
    cache.key_storage[layer] = storage
    cache.layers[layer].keys = storage[:, :, :key.shape[2]]


def _append_key(cache, layer, key, previous):
    storage = cache.key_storage[layer]
    assert key.shape[2] == 1 and previous < storage.shape[2]
    storage[:, :, previous:previous + 1].copy_(key)
    cache.layers[layer].keys = storage[:, :, :previous + 1]


def _selected_attention(query, host_key, value, ids, valid, scale, cache, layer):
    from basisserve.kernels.split_indexed_attention import split_indexed_attention

    batch, kv_heads, count = ids.shape
    safe_cpu = ids.clamp_min(0).to(device='cpu')
    staging = cache.key_staging.get(layer)
    shape = (batch, kv_heads, count, host_key.shape[-1])
    if staging is None or staging.shape != shape:
        staging = torch.empty(shape, dtype=host_key.dtype, device='cpu', pin_memory=True)
        cache.key_staging[layer] = staging
    torch.gather(host_key, 2, safe_cpu[..., None].expand(shape), out=staging)
    selected_key = staging.to(device=query.device, non_blocking=True)
    safe_gpu = safe_cpu.to(device=query.device, non_blocking=True)
    selected_value = value.gather(
        2, safe_gpu[..., None].expand(batch, kv_heads, count, value.shape[-1]))
    packed_ids = torch.arange(count, device=query.device).view(1, 1, count).expand(
        batch, kv_heads, count).masked_fill(~valid, -1)
    packed_ids = packed_ids.repeat_interleave(query.shape[1] // kv_heads, dim=1)
    return split_indexed_attention(query, selected_key, selected_value, packed_ids, scale=scale)


@torch.inference_mode()
def forward(self, hidden_states, position_embeddings, attention_mask=None,
            past_key_values=None, **kwargs):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from evaluation.llama_sink_recent_routing import page_support

    assert isinstance(past_key_values, KOffloadRoutingCache)
    batch, length, _ = hidden_states.shape
    assert batch == 1 and self._routing_arm == 'ours'
    q = self.q_norm(self.q_proj(hidden_states).view(
        batch, length, self.num_attention_heads, self.head_dim)).transpose(1, 2)
    pre = self.k_norm(self.k_proj(hidden_states).view(
        batch, length, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(
        batch, length, self.num_key_value_heads, self.value_head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    if length == 1:
        q, k = apply_rotary_pos_emb(q, pre, cos, sin)
    else:
        k = torch.empty_like(pre)
        for start in range(0, length, 1024):
            stop = start + 1024
            qr, kr = apply_rotary_pos_emb(q[:, :, start:stop], pre[:, :, start:stop],
                                         cos[:, start:stop], sin[:, start:stop])
            q[:, :, start:stop], k[:, :, start:stop] = qr, kr
        del qr, kr
    del pre
    previous = past_key_values.get_seq_length(self.layer_idx)
    assert previous == 0 or length == 1
    factors = self._routing_factors
    current = runtime.build_conditional_routing_sidecar(
        v, k, base_left=factors['base_left_b16'], base_right=factors['base_right_b16'],
        base_bias=factors['base_bias_b16'],
        residual_encoder=factors['residual_encoder_b16_r16'], cos=cos, sin=sin)
    if previous:
        past_key_values.sidecars[self.layer_idx] = torch.cat(
            (past_key_values.sidecars[self.layer_idx], current), 2)
        _append_key(past_key_values, self.layer_idx, k, previous)
        cached = past_key_values.layers[self.layer_idx]
        cached.values = torch.cat((cached.values, v), 2)
        host_key, v = cached.keys, cached.values
        if attention_mask is not None:
            assert attention_mask.shape[-2] == 1
            assert bool(attention_mask.all()) if attention_mask.dtype == torch.bool else bool((attention_mask == 0).all())
        group_heads = self.num_attention_heads // self.num_key_value_heads
        projector = self._routing_projector
        codes = torch.einsum('bhqd,hdr->bhqr', q.float(), projector.float())
        codes = codes[:, :, 0].reshape(batch, self.num_key_value_heads, group_heads, -1)
        sidecar = past_key_values.sidecars[self.layer_idx]
        scores = (codes @ sidecar.float().transpose(-1, -2)) * self.scaling
        ids, valid = page_support(scores)
        output = _selected_attention(q, host_key, v, ids, valid, self.scaling,
                                     past_key_values, self.layer_idx)
        selected_mean = float(valid.sum(-1).float().mean())
        past_key_values.statistics[self.layer_idx] = dict(
            selected_tokens_mean=selected_mean, sink_tokens=32, recent_tokens=64,
            token_budget=2048, exact_key_storage='pinned_cpu',
            routing_sidecar_storage='cuda', value_storage='cuda',
            exact_key_bytes_fetched_mean=selected_mean*self.head_dim*2)
    else:
        past_key_values.sidecars[self.layer_idx] = current
        k, v = past_key_values.update(k, v, self.layer_idx)
        output = runtime.compressed_v_prefill_attention(q, k, v, scale=self.scaling)
        _offload_prefill_key(past_key_values, self.layer_idx, k)
    del q, k, v
    output = output.transpose(1, 2).contiguous().reshape(batch, length, -1)
    return self.o_proj(output), None


def install(model):
    install_chunked_prefill_norms(model)
    runtime.RoutingCache = KOffloadRoutingCache
    for layer in model.model.layers:
        layer.self_attn.forward = MethodType(forward, layer.self_attn)
