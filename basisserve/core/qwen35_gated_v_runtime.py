"""Compact V-only attention adapter for the audited Transformers 5.3 API."""

import hashlib
import copy
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


def factor_hash(bank):
    digest = hashlib.sha256()
    for index, layer in sorted(bank.items()):
        digest.update(str(index).encode())
        for name in ('E_V', 'R_V'):
            value = layer[name].detach().cpu().contiguous()
            digest.update(str((name, value.shape, value.dtype)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _schema_checked_update(cache, key_states, value_states, layer_idx, cache_kwargs=None):
    expected = cache._basisserve_v_schemas.get(layer_idx)
    if expected is not None:
        assert (cache_kwargs or {}).get('basisserve_v_schema') == expected, 'Dense/latent cache or factor bank mismatch'
    return cache._basisserve_native_update(key_states, value_states, layer_idx, cache_kwargs)


def fork_gated_v_cache(cache):
    fork = copy.copy(cache)
    for name in ('key_cache', 'value_cache', 'conv_states', 'recurrent_states'):
        setattr(fork, name, [None if value is None else value.detach().clone() for value in getattr(cache, name)])
    if hasattr(cache, '_basisserve_v_schemas'):
        fork._basisserve_v_schemas = copy.deepcopy(cache._basisserve_v_schemas)
        fork._basisserve_native_update = MethodType(type(cache).update, fork)
        fork.update = MethodType(_schema_checked_update, fork)
    return fork


def reset_gated_v_cache(cache):
    for name in ('key_cache', 'value_cache', 'conv_states', 'recurrent_states'):
        values = getattr(cache, name)
        for index in range(len(values)):
            values[index] = None
    if hasattr(cache, '_basisserve_v_schemas'):
        cache._basisserve_v_schemas.clear()


class GatedVAttention(nn.Module):
    def __init__(self, native, encoder, decoder, *, query_chunk=128):
        super().__init__()
        self.native = native
        self.o_proj = native.o_proj
        self.config = native.config
        self.layer_idx = native.layer_idx
        self.head_dim = native.head_dim
        self.query_chunk = query_chunk
        kv_heads = native.k_proj.out_features // self.head_dim
        groups, group_width, rank = encoder.shape
        assert group_width % self.head_dim == 0
        self.kv_per_group = group_width // self.head_dim
        assert groups * self.kv_per_group == kv_heads
        assert decoder.shape == (groups, rank, group_width)
        assert native.v_proj.out_features == groups * group_width
        assert query_chunk > 0 and torch.isfinite(encoder).all() and torch.isfinite(decoder).all()
        e = encoder.to(native.v_proj.weight)
        r = decoder.to(native.v_proj.weight)
        dense_weight = native.v_proj.weight.detach().reshape(groups, group_width, -1)
        self.register_buffer('latent_weight', (e.transpose(1, 2) @ dense_weight).reshape(groups * rank, -1))
        bias = native.v_proj.bias
        self.register_buffer('latent_bias', None if bias is None else torch.einsum('gd,gdr->gr', bias.reshape(groups, group_width), e).flatten())
        self.register_buffer('decoder', r)
        self.groups, self.rank, self.kv_heads = groups, rank, kv_heads
        self.schema = ('qwen35_compact_v_v1', groups, rank, self.kv_per_group,
            factor_hash({self.layer_idx: {'E_V': e, 'R_V': r}}))

    @torch.no_grad()
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        n = self.native
        batch, length, _ = hidden_states.shape
        q, logits = n.q_proj(hidden_states).view(batch, length, -1, 2 * self.head_dim).chunk(2, dim=-1)
        heads = q.shape[2]
        q = n.q_norm(q).transpose(1, 2)
        k = n.k_norm(n.k_proj(hidden_states).view(batch, length, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = F.linear(hidden_states, self.latent_weight, self.latent_bias).view(batch, length, self.groups, self.rank).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if past_key_values is not None:
            schemas = getattr(past_key_values, '_basisserve_v_schemas', {})
            existing = past_key_values.value_cache[self.layer_idx]
            assert existing is None or schemas.get(self.layer_idx) == self.schema
            schemas[self.layer_idx] = self.schema
            past_key_values._basisserve_v_schemas = schemas
            if not hasattr(past_key_values, '_basisserve_native_update'):
                past_key_values._basisserve_native_update = past_key_values.update
                past_key_values.update = MethodType(_schema_checked_update, past_key_values)
            k, v = past_key_values.update(k, v, self.layer_idx,
                {'cache_position': cache_position, 'basisserve_v_schema': self.schema})
            assert v.shape[1] == self.groups and v.shape[-1] == self.rank
        mapping = torch.arange(heads, device=q.device) // n.num_key_value_groups
        group_mapping = mapping // self.kv_per_group
        k_heads, v_heads = k[:, mapping], v[:, group_mapping]
        full_length = k.shape[2]
        positions = cache_position if cache_position is not None else torch.arange(full_length - length, full_length, device=q.device)
        outputs = []
        for start in range(0, length, self.query_chunk):
            stop = min(start + self.query_chunk, length)
            if attention_mask is None:
                mask = torch.arange(full_length, device=q.device)[None, :] <= positions[start:stop, None]
            else:
                assert attention_mask.ndim == 4
                mask = attention_mask[..., start:stop, :full_length]
            # Q/K keep native width and scaling; V remains latent throughout.
            out = F.scaled_dot_product_attention(q[:, :, start:stop], k_heads, v_heads,
                attn_mask=mask, dropout_p=0.0, scale=n.scaling)
            outputs.append(out)
        latent = torch.cat(outputs, dim=2).transpose(1, 2)
        head_decoder = self.decoder[group_mapping].reshape(heads, self.rank, self.kv_per_group, self.head_dim)
        local_head = mapping % self.kv_per_group
        head_decoder = head_decoder[torch.arange(heads, device=q.device), :, local_head, :]
        restored = torch.einsum('bqhr,hrd->bqhd', latent, head_decoder)
        post = (restored * logits.sigmoid()).reshape(batch, length, -1)
        return self.o_proj(post), None


class GatedVRuntime:
    def __init__(self, model, bank, *, query_chunk=128):
        self.model, self.bank, self.query_chunk = model, bank, query_chunk
        self.originals = {}
        self.hash = factor_hash(bank)

    def install(self):
        assert not self.originals
        replacements = {}
        for index, factors in self.bank.items():
            native = self.model.model.layers[int(index)].self_attn
            assert not isinstance(native, GatedVAttention)
            replacements[int(index)] = GatedVAttention(native, factors['E_V'], factors['R_V'], query_chunk=self.query_chunk)
        for index, replacement in replacements.items():
            self.originals[index] = self.model.model.layers[index].self_attn
            self.model.model.layers[index].self_attn = replacement
        return self

    def restore(self):
        for index, original in self.originals.items():
            self.model.model.layers[index].self_attn = original
        self.originals.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *args):
        self.restore()
        assert factor_hash(self.bank) == self.hash
