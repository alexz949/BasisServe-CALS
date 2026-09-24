"""Qwen3.5 routing policies with shared full Flash prefill and exact selected attention."""

import torch

from basisserve.core.c1_conditional_page_attention import _selected_pages
from basisserve.core.c1_lrqk import LRQKConfig, LRQKState, gather_heads
from basisserve.core.c1_shadowkv import C1ShadowKVState, gather_group
from basisserve.core.c1_v_k_index import apply_rotary
from basisserve.core.compact_v_flash import compact_v_flash_attention
from basisserve.core.qwen35_gated_v_runtime import GatedVAttention


def page_support(scores, budget=2048, page_size=32):
    """Route historical pages plus exact recent 64; no pinned sink, hard budget."""
    batch, kv_heads, heads_per_group, length = scores.shape
    assert length > 0 and budget >= 64 and budget % page_size == 0 and 64 % page_size == 0
    historical = max(0, length-64)
    recent = torch.arange(historical, length, device=scores.device).view(1, 1, -1).expand(batch, kv_heads, -1)
    if historical == 0 or budget == 64:
        return recent, torch.ones_like(recent, dtype=torch.bool)
    proxy = scores[..., :historical].reshape(batch, kv_heads*heads_per_group, 1, historical)
    selected, _ = _selected_pages(proxy, torch.ones_like(proxy, dtype=torch.bool),
        kv_heads=kv_heads, page_size=page_size, page_budget=(budget-64)//page_size,
        pinned_prefix_pages=0)
    selected = selected.squeeze(-2).sort(-1).values
    ids = (selected[..., None]*page_size+torch.arange(page_size, device=scores.device)).flatten(-2)
    valid = torch.cat((ids < historical, torch.ones_like(recent, dtype=torch.bool)), -1)
    return torch.cat((ids, recent), -1), valid


def build_qwen35_routing_sidecar(value, key, cos, sin, *, base_rank, residual_rank, factors):
    left, right, bias = [factors[f'base_{name}_b{base_rank}'].to(device=value.device, dtype=torch.float32)
        for name in ('left', 'right', 'bias')]
    base = torch.einsum('bhtv,hvr,hrd->bhtd', value.float(), left, right)+bias[None, :, None]
    base = apply_rotary(base, cos, sin)
    residual_encoder = factors[f'residual_encoder_b{base_rank}_r{residual_rank}'].to(device=value.device, dtype=torch.float32)
    code = torch.einsum('bhtd,hdr->bhtr', key.float()-base, residual_encoder)
    return torch.cat((base, code), -1)


class Qwen35RoutingAttention(GatedVAttention):
    def __init__(self, native, encoder, decoder, *, arm, factors=None, loki_basis=None, base_rank=None, residual_rank=None,
                 budget=2048, lrqk_topk=2048, loki_topk=2048, shadowkv_budget=2048):
        super().__init__(native, encoder, decoder)
        # Physical budgets: page routing (recent 64 inside), LRQK/Loki per-query-head top-k, ShadowKV routed tokens.
        assert budget >= 64 and budget % 32 == 0 and lrqk_topk > 0 and loki_topk > 0 and shadowkv_budget % 8 == 0
        self.budget, self.lrqk_topk, self.loki_topk, self.shadowkv_budget = int(budget), int(lrqk_topk), int(loki_topk), int(shadowkv_budget)
        assert self.kv_per_group == 1 and arm in ('full', 'exact_sparse', 'b16r16', 'b32r32', 'ours', 'loki', 'lrqk', 'shadowkv')
        self.arm, self.routing_factors, self.loki_basis = arm, factors, loki_basis
        if arm in ('b16r16', 'b32r32'):
            base_rank = residual_rank = 16 if arm == 'b16r16' else 32
        if arm in ('b16r16', 'b32r32', 'ours'):
            # 'ours' is any Base/residual split; the factor names carry both ranks.
            assert factors is not None and base_rank and residual_rank
            self.base_rank, self.residual_rank = int(base_rank), int(residual_rank)
            assert f'residual_query_b{self.base_rank}_r{self.residual_rank}' in factors
        if arm == 'loki':
            assert loki_basis is not None and loki_basis.shape == (self.kv_heads, self.head_dim, 32)

    def sidecar(self, value, key, cos, sin):
        return build_qwen35_routing_sidecar(value, key, cos, sin,
            base_rank=self.base_rank, residual_rank=self.residual_rank, factors=self.routing_factors)

    def attention(self, q, k, v, *, pre_key, position_embeddings, attention_mask, cache_position, cache):
        assert cache is not None and q.shape[0] == 1 and attention_mask is None
        assert q.shape[2] == k.shape[2] or q.shape[2] == 1
        if self.arm == 'full':
            return super().attention(q, k, v, pre_key=pre_key, position_embeddings=position_embeddings,
                attention_mask=attention_mask, cache_position=cache_position, cache=cache)
        if not hasattr(cache, '_q35_routing'):
            cache._q35_routing = {}
        cos, sin = position_embeddings
        prefill = q.shape[2] == k.shape[2]
        if prefill:
            output = super().attention(q, k, v, pre_key=pre_key, position_embeddings=position_embeddings,
                attention_mask=attention_mask, cache_position=cache_position, cache=cache)
            if self.arm in ('b16r16', 'b32r32', 'ours'):
                state = self.sidecar(v, k, cos, sin)
            elif self.arm == 'loki':
                state = torch.einsum('bhtd,hdr->bhtr', k, self.loki_basis.to(k))
            elif self.arm == 'lrqk':
                state = LRQKState(q, k, LRQKConfig(topk=self.lrqk_topk), layer=self.layer_idx)
            elif self.arm == 'shadowkv':
                # A short prompt fits entirely in the native routed/outlier/local
                # capacity. Keep all prompt tokens and the growing generated tail.
                state = (C1ShadowKVState(pre_key, k, cos, sin, budget=self.shadowkv_budget)
                         if k.shape[2]//8-4 > 48+self.shadowkv_budget//8 else None)
            else:
                state = None
            cache._q35_routing[self.layer_idx] = state
            return output
        state = cache._q35_routing[self.layer_idx]
        if self.arm == 'lrqk':
            return state.decode(q, k, v, self.native.scaling).transpose(1, 2)
        if self.arm == 'shadowkv':
            if state is None:
                return super().attention(q, k, v, pre_key=pre_key, position_embeddings=position_embeddings,
                    attention_mask=attention_mask, cache_position=cache_position, cache=cache)
            return state.decode(q, k[:, :, -1:], v, self.native.scaling).transpose(1, 2)
        heads_per_group = q.shape[1]//self.kv_heads
        if self.arm == 'loki':
            basis = self.loki_basis.to(q)
            state = torch.cat((state, torch.einsum('bhtd,hdr->bhtr', k[:, :, -1:], basis)), 2)
            codes = torch.einsum('bhgd,hdr->bhgr', q[:, :, 0].reshape(1, self.kv_heads, heads_per_group, self.head_dim), basis)
            scores = (codes@state.transpose(-1, -2)).reshape(1, q.shape[1], k.shape[2])
            # Per-query-head support; its physical GQA union is not capped.
            ids = scores.topk(min(self.loki_topk, k.shape[2]), dim=-1).indices.sort(-1).values
            output = compact_v_flash_attention(q, gather_heads(k, ids), gather_heads(v, ids), scale=self.native.scaling)
        else:
            if self.arm == 'exact_sparse':
                query = q[:, :, 0].float().reshape(1, self.kv_heads, heads_per_group, self.head_dim)
                scores = (query@k.float().transpose(-1, -2))*self.native.scaling
            else:
                state = torch.cat((state, self.sidecar(v[:, :, -1:], k[:, :, -1:], cos, sin)), 2)
                projector = self.routing_factors[f'residual_query_b{self.base_rank}_r{self.residual_rank}'].to(device=q.device, dtype=torch.float32)
                residual_query = torch.einsum('bhd,hdr->bhr', q[:, :, 0].float(), projector)
                query = torch.cat((q[:, :, 0].float(), residual_query), -1).reshape(1, self.kv_heads, heads_per_group, -1)
                scores = (query@state.transpose(-1, -2))*self.native.scaling
            ids, valid = page_support(scores, budget=self.budget)
            # At most one partial page exists; remove its invalid token slots.
            safe_ids = ids.clamp_max(k.shape[2]-1)
            selected_k, selected_v = gather_group(k, safe_ids), gather_group(v, safe_ids)
            if k.shape[2] % 32 == 0:
                output = compact_v_flash_attention(q, selected_k, selected_v, scale=self.native.scaling)
            else:
                # Different KV groups may select the partial page; group each head's
                # valid support without padding before the fused attention call.
                pieces = []
                for group in range(self.kv_heads):
                    keep = valid[0, group]
                    pieces.append(compact_v_flash_attention(
                        q[:, group*heads_per_group:(group+1)*heads_per_group],
                        selected_k[:, group:group+1, keep], selected_v[:, group:group+1, keep],
                        scale=self.native.scaling))
                output = torch.cat(pieces, 1)
        cache._q35_routing[self.layer_idx] = state
        self.last_attention_backend = 'selected_flash'
        return output.transpose(1, 2)
