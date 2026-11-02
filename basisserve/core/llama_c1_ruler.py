"""Llama C1 runtime with exact K and two recent physical pages in sparse decode."""

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from basisserve.core.c1_strict_recent_decode import c1_strict_recent_decode
from basisserve.kernels.compressed_v_decode_attention import (
    compressed_v_prefill_attention, compressed_v_decode_attention_triton,
)


class LlamaC1RulerCache(DynamicCache):
    def __init__(self,capacity):
        super().__init__()
        self.capacity = capacity
        self.lengths = []
        self.sidecars = []

    def get_seq_length(self,layer_idx=0):
        return self.lengths[layer_idx] if layer_idx < len(self.lengths) else 0

    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        while len(self.lengths) <= layer_idx:
            self.lengths.append(0)
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.sidecars.append(None)
        previous = self.lengths[layer_idx]
        stop = previous+key_states.shape[2]
        assert stop <= self.capacity
        if previous == 0:
            batch,groups,_,dim = key_states.shape
            self.key_cache[layer_idx] = torch.empty(batch,groups,self.capacity,dim,
                device=key_states.device,dtype=key_states.dtype)
            self.value_cache[layer_idx] = torch.empty(batch,groups,self.capacity,value_states.shape[-1],
                device=value_states.device,dtype=value_states.dtype)
        self.key_cache[layer_idx][:,:,previous:stop].copy_(key_states)
        self.value_cache[layer_idx][:,:,previous:stop].copy_(value_states)
        self.lengths[layer_idx] = stop
        if layer_idx == 0:
            self._seen_tokens = stop
        return self.key_cache[layer_idx][:,:,:stop],self.value_cache[layer_idx][:,:,:stop]

class LlamaC1RulerAttention(nn.Module):
    def __init__(self,original,encoder,decoder,router):
        super().__init__()
        self.config = original.config
        self.layer_idx = original.layer_idx
        self.head_dim = original.head_dim
        self.scaling = original.scaling
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.rank = encoder.shape[-1]
        groups = self.config.num_key_value_heads
        heads = self.config.num_attention_heads
        hidden = self.config.hidden_size
        assert encoder.shape == (groups,self.head_dim,self.rank)
        assert decoder.shape == (heads,self.rank,hidden)
        assert original.v_proj.bias is None and original.o_proj.bias is None
        dtype,device = original.v_proj.weight.dtype,original.v_proj.weight.device
        weight = torch.bmm(encoder.float().transpose(1,2),
            original.v_proj.weight.float().reshape(groups,self.head_dim,hidden))
        self.v_proj = nn.Linear(hidden,groups*self.rank,bias=False,device=device,dtype=dtype)
        self.o_proj = nn.Linear(heads*self.rank,hidden,bias=False,device=device,dtype=dtype)
        with torch.no_grad():
            self.v_proj.weight.copy_(weight.reshape(groups*self.rank,hidden))
            self.o_proj.weight.copy_(decoder.permute(2,0,1).reshape(hidden,heads*self.rank))
        for name,tensor in router.items():
            self.register_buffer(name,tensor.to(device=device,dtype=dtype))
        self.sparse = True
        self.prefill_calls = 0
        self.decode_calls = 0
        self.routing_calls = 0
        self.selected_max = 0

    @torch.inference_mode()
    def forward(self,hidden_states,position_embeddings,attention_mask=None,
                past_key_value=None,cache_position=None,**kwargs):
        cache = past_key_value
        assert isinstance(cache,LlamaC1RulerCache)
        batch,length,_ = hidden_states.shape
        heads,groups = self.config.num_attention_heads,self.config.num_key_value_heads
        q = self.q_proj(hidden_states).view(batch,length,heads,self.head_dim).transpose(1,2)
        k = self.k_proj(hidden_states).view(batch,length,groups,self.head_dim).transpose(1,2)
        v = self.v_proj(hidden_states).view(batch,length,groups,self.rank).transpose(1,2)
        cos,sin = position_embeddings
        q,k = apply_rotary_pos_emb(q,k,cos,sin)
        previous = cache.get_seq_length(self.layer_idx)
        assert previous == 0 or length == 1
        exact_key,values = cache.update(k,v,self.layer_idx)
        sidecar = None
        if self.sparse:
            for start in range(0,length,2048):
                stop = min(start+2048,length)
                part = build_conditional_routing_sidecar(v[:,:,start:stop],k[:,:,start:stop],
                    base_left=self.base_left_b16,base_right=self.base_right_b16,
                    base_bias=self.base_bias_b16,residual_encoder=self.residual_encoder_b16_r8,
                    cos=cos[:,start:stop],sin=sin[:,start:stop])
                if cache.sidecars[self.layer_idx] is None:
                    cache.sidecars[self.layer_idx] = torch.empty(batch,groups,cache.capacity,part.shape[-1],
                        device=part.device,dtype=part.dtype)
                cache.sidecars[self.layer_idx][:,:,previous+start:previous+stop].copy_(part)
            sidecar = cache.sidecars[self.layer_idx][:,:,:previous+length]
        with torch.cuda.device(q.device):
            if previous == 0:
                output = compressed_v_prefill_attention(q,exact_key,values,scale=self.scaling)
                self.prefill_calls += 1
            elif self.sparse:
                output,ids = c1_strict_recent_decode(q,exact_key,values,sidecar,
                    self.residual_query_b16_r8,scale=self.scaling)
                self.decode_calls += 1
                self.routing_calls += 1
                self.selected_max = max(self.selected_max,int(ids.shape[-1]))
            else:
                output = compressed_v_decode_attention_triton(q,exact_key,values,scale=self.scaling)
                self.decode_calls += 1
        return self.o_proj(output.transpose(1,2).contiguous().reshape(batch,length,-1)),None
