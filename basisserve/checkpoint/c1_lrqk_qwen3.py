"""Single-stream, unpadded Qwen3 C1 inference with cache-owned LRQK state."""
from types import MethodType

import torch
from torch.nn import functional as F
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention
from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from basisserve.core.c1_lrqk import LRQKConfig, LRQKState
from basisserve.kernels.compressed_v_decode_attention import compressed_v_prefill_attention


class C1LRQKCache(DynamicCache):
    def __init__(self,config=None):
        super().__init__(config=config)
        self.lrqk_states = {}


@torch.inference_mode()
def _forward(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,
             cache_position=None,**kwargs):
    assert isinstance(past_key_values,C1LRQKCache)
    assert not kwargs.get('output_attentions',False)
    batch,length,_ = hidden_states.shape
    assert batch == 1  # No padding, beam reordering, or continuous batching in this reference.
    cfg = self._lrqk_config
    q = self.q_norm(self.q_proj(hidden_states).view(batch,length,self.num_attention_heads,self.head_dim)).transpose(1,2)
    k = self.k_norm(self.k_proj(hidden_states).view(batch,length,self.num_key_value_heads,self.head_dim)).transpose(1,2)
    v = self.v_proj(hidden_states).view(batch,length,self.num_key_value_heads,self.value_head_dim).transpose(1,2)
    cos,sin = position_embeddings
    q,k = apply_rotary_pos_emb(q,k,cos,sin)
    previous = past_key_values.get_seq_length(self.layer_idx)
    assert previous == 0 or length == 1
    if attention_mask is not None:
        assert attention_mask.ndim == 4
        # Require exactly ordinary causal support, with no padding/custom sparsity.
        expected = torch.arange(previous+length,device=q.device)[None,:] <= (
            previous+torch.arange(length,device=q.device)[:,None])
        valid = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
        assert torch.equal(valid.expand(batch,1,length,previous+length)[0,0],expected)
    k,v = past_key_values.update(k,v,self.layer_idx,{'cos':cos,'sin':sin,'cache_position':cache_position})
    if previous == 0:
        assert self.layer_idx not in past_key_values.lrqk_states
        state = LRQKState(q,k,cfg,self.layer_idx)
        if getattr(self, '_lrqk_official', False):
            state.bind_values(v)
        past_key_values.lrqk_states[self.layer_idx] = state
        if cfg.prefill_backend == 'triton':
            output = compressed_v_prefill_attention(q,k,v,scale=self.scaling)
        else:
            output = F.scaled_dot_product_attention(q,k,v,enable_gqa=True,is_causal=True,scale=self.scaling)
    else:
        state = past_key_values.lrqk_states[self.layer_idx]
        assert state.config == cfg and state.length == previous
        output = state.decode(q,k,v,self.scaling)
    output = output.transpose(1,2).contiguous().reshape(batch,length,-1)
    return self.o_proj(output),None


def install_c1_lrqk(model,config=LRQKConfig()):
    """Install after C1 factor export; preserve Q/K norms, RoPE, C1 writer/decoder."""
    for _, module in c1_attention_layers(model):
        assert isinstance(module,GQATiedVOQwen3Attention) and module.key_projector is None
        assert module.reverse_shadow_config is None and not hasattr(module,'_lrqk_config')
        module._lrqk_config = config
        module.forward = MethodType(_forward,module)
