"""Qwen3.5 decode adapter using split indexed attention; frozen routing rules."""
import torch
from basisserve.core.qwen35_k_routing_runtime import Qwen35RoutingAttention, page_support
from basisserve.core.qwen35_gated_v_runtime import GatedVAttention
from basisserve.core.c1_lrqk import LRQKState, LRQKConfig
from basisserve.core.c1_shadowkv import C1ShadowKVState
from basisserve.kernels.split_indexed_attention import split_indexed_attention
from evaluation.eval_qwen35_k_routing_ruler import install as install_reference

class SplitRoutingAttention(Qwen35RoutingAttention):
    def attention(self, q, k, v, *, pre_key, position_embeddings, attention_mask, cache_position, cache):
        assert cache is not None and q.shape[0] == 1 and attention_mask is None
        assert q.shape[2] == k.shape[2] or q.shape[2] == 1
        if self.arm == 'full':
            return GatedVAttention.attention(self,q, k, v, pre_key=pre_key, position_embeddings=position_embeddings,
                attention_mask=attention_mask, cache_position=cache_position, cache=cache)
        if not hasattr(cache, '_q35_routing'):
            cache._q35_routing = {}
        cos, sin = position_embeddings
        prefill = q.shape[2] == k.shape[2]
        if prefill:
            output = GatedVAttention.attention(self,q, k, v, pre_key=pre_key, position_embeddings=position_embeddings,
                attention_mask=attention_mask, cache_position=cache_position, cache=cache)
            if self.arm in ('b16r16', 'b32r32'):
                state = self.sidecar(v, k, cos, sin)
            elif self.arm == 'loki':
                state = torch.einsum('bhtd,hdr->bhtr', k, self.loki_basis.to(k))
            elif self.arm == 'lrqk':
                state = LRQKState(q, k, LRQKConfig(), layer=self.layer_idx)
            elif self.arm == 'shadowkv':
                # A short prompt fits entirely in the native routed/outlier/local
                # capacity. Keep all prompt tokens and the growing generated tail.
                state = C1ShadowKVState(pre_key, k, cos, sin) if k.shape[2]//8-4 > 48+2048//8 else None
            else:
                state = None
            cache._q35_routing[self.layer_idx] = state
            return output
        state = cache._q35_routing[self.layer_idx]
        if self.arm == 'lrqk':
            return state.decode(q, k, v, self.native.scaling).transpose(1, 2)
        if self.arm == 'shadowkv':
            if state is None:
                return GatedVAttention.attention(self,q, k, v, pre_key=pre_key, position_embeddings=position_embeddings,
                    attention_mask=attention_mask, cache_position=cache_position, cache=cache)
            return state.decode(q, k[:, :, -1:], v, self.native.scaling).transpose(1, 2)
        heads_per_group = q.shape[1]//self.kv_heads
        if self.arm == 'loki':
            basis = self.loki_basis.to(q)
            state = torch.cat((state, torch.einsum('bhtd,hdr->bhtr', k[:, :, -1:], basis)), 2)
            codes = torch.einsum('bhgd,hdr->bhgr', q[:, :, 0].reshape(1, self.kv_heads, heads_per_group, self.head_dim), basis)
            scores = (codes@state.transpose(-1, -2)).reshape(1, q.shape[1], k.shape[2])
            # Per-query-head support; its physical GQA union is not capped.
            ids = scores.topk(min(2048, k.shape[2]), dim=-1).indices.sort(-1).values
            output = split_indexed_attention(q, k, v, ids, scale=self.native.scaling)
        else:
            if self.arm == 'exact_sparse':
                query = q[:, :, 0].float().reshape(1, self.kv_heads, heads_per_group, self.head_dim)
                scores = (query@k.float().transpose(-1, -2))*self.native.scaling
            else:
                state = torch.cat((state, self.sidecar(v[:, :, -1:], k[:, :, -1:], cos, sin)), 2)
                rank = self.routing_rank
                projector = self.routing_factors[f'residual_query_b{rank}_r{rank}'].to(device=q.device, dtype=torch.float32)
                residual_query = torch.einsum('bhd,hdr->bhr', q[:, :, 0].float(), projector)
                query = torch.cat((q[:, :, 0].float(), residual_query), -1).reshape(1, self.kv_heads, heads_per_group, -1)
                scores = (query@state.transpose(-1, -2))*self.native.scaling
            ids, valid = page_support(scores)
            head_ids = ids.masked_fill(~valid, -1).repeat_interleave(heads_per_group, dim=1)
            output = split_indexed_attention(q, k, v, head_ids, scale=self.native.scaling)
        cache._q35_routing[self.layer_idx] = state
        self.last_attention_backend = 'split_indexed'
        return output.transpose(1, 2)

def install(model, args, bank):
    sources=install_reference(model,args,bank)
    for layer in bank['schedule']:
        model.model.layers[layer].self_attn.__class__=SplitRoutingAttention
    return sources
