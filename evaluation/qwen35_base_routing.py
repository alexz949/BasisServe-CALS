"""Base16-only routing, with no residual codes or residual scoring."""
from argparse import Namespace
import torch
from basisserve.core.qwen35_gated_v_runtime import GatedVAttention
from basisserve.core.qwen35_k_routing_runtime import Qwen35RoutingAttention, page_support
from basisserve.core.c1_v_k_index import apply_rotary
from basisserve.kernels.split_indexed_attention import split_indexed_attention
from evaluation.eval_qwen35_k_routing_ruler import install as install_reference


class BaseRoutingAttention(Qwen35RoutingAttention):
    def base_key(self, value, cos, sin):
        left, right, bias = [self.routing_factors[f'base_{name}_b16'].float()
            for name in ('left', 'right', 'bias')]
        base = torch.einsum('bhtv,hvr,hrd->bhtd', value.float(), left, right) + bias[None, :, None]
        return apply_rotary(base, cos, sin)

    def attention(self, q, k, v, *, pre_key, position_embeddings, attention_mask, cache_position, cache):
        assert cache is not None and q.shape[0] == 1 and attention_mask is None
        cos, sin = position_embeddings
        if q.shape[2] == k.shape[2]:
            output = GatedVAttention.attention(self, q, k, v, pre_key=pre_key,
                position_embeddings=position_embeddings, attention_mask=attention_mask,
                cache_position=cache_position, cache=cache)
            if not hasattr(cache, '_q35_base'):
                cache._q35_base = {}
            cache._q35_base[self.layer_idx] = self.base_key(v, cos, sin)
            return output
        assert q.shape[2] == 1
        state = torch.cat((cache._q35_base[self.layer_idx], self.base_key(v[:, :, -1:], cos, sin)), 2)
        groups = q.shape[1] // self.kv_heads
        query = q[:, :, 0].float().reshape(1, self.kv_heads, groups, self.head_dim)
        scores = (query @ state.transpose(-1, -2)) * self.native.scaling
        ids, valid = page_support(scores)
        head_ids = ids.masked_fill(~valid, -1).repeat_interleave(groups, dim=1)
        output = split_indexed_attention(q, k, v, head_ids, scale=self.native.scaling)
        cache._q35_base[self.layer_idx] = state
        return output.transpose(1, 2)


def install(model, args, bank):
    assert args.arm == 'base16'
    reference_args = Namespace(**{**vars(args), 'arm': 'b16r16'})
    sources = install_reference(model, reference_args, bank)
    for layer in bank['schedule']:
        attention = model.model.layers[layer].self_attn
        attention.__class__ = BaseRoutingAttention
        attention.arm = 'base16'
        attention.routing_factors = {k:v for k,v in attention.routing_factors.items() if k.startswith('base_')}
    return sources
