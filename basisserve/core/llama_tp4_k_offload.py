"""Llama K128 TP4 caches, dense prefill and B16R16 mapped-host decode."""
from pathlib import Path
from types import MethodType
import torch
from torch import nn
from safetensors.torch import load_file
from flash_attn import flash_attn_func,flash_attn_with_kvcache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from basisserve.kernels.mapped_host_paged_attention import (
    mapped_host_bf16_empty,append_mapped_host_key,mapped_host_paged_attention,
    conditional_router_page_lse,select_fixed_group_max_pages_cuda)
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from evaluation.chunked_prefill_mlp import ChunkedTokenwise


class LlamaTP4KCacheAttention(nn.Module):
    def __init__(self,original,*,mode,batch,capacity,root,rank):
        super().__init__()
        self.q_proj=original.q_proj;self.k_proj=original.k_proj
        self.v_proj=original.v_proj;self.o_proj=original.o_proj
        self.layer_idx=original.layer_idx;self.scaling=original.scaling
        self.mode=mode;self.capacity=capacity;self.length=0
        self.register_buffer('value_cache',torch.empty(batch,2,capacity,128,device='cuda',dtype=torch.bfloat16),persistent=False)
        if mode=='dense':
            self.register_buffer('key_cache',torch.empty_like(self.value_cache),persistent=False)
        else:
            self.host_key=mapped_host_bf16_empty(batch=batch,kv_heads=2,capacity=capacity)
            self.register_buffer('base_cache',torch.empty(batch,2,capacity,16,device='cuda',dtype=torch.bfloat16),persistent=False)
            self.register_buffer('residual_cache',torch.empty_like(self.base_cache),persistent=False)
            f=load_file(str(Path(root)/'ours_b16r16'/f'layer_{self.layer_idx:03d}.safetensors'))
            self.factors={name:t[rank*(8 if name.startswith('residual_query') else 2):(rank+1)*(8 if name.startswith('residual_query') else 2)].cuda().bfloat16().contiguous() for name,t in f.items()}
            self.register_buffer('workspace',torch.empty(batch*8,128,130,device='cuda'),persistent=False)

    @torch.inference_mode()
    def forward(self,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,**kwargs):
        assert past_key_values is None and attention_mask is None
        x=hidden_states;b,n,_=x.shape;start=self.length;end=start+n
        assert end<=self.capacity and (start==0 or n==1)
        cos,sin=position_embeddings
        if start==0:
            key=torch.empty(b,2,n,128,device=x.device,dtype=x.dtype)
        else:key=torch.empty(b,2,1,128,device=x.device,dtype=x.dtype)
        for left in range(0,n,2048):
            right=min(left+2048,n);block=x[:,left:right]
            pre=self.k_proj(block).view(b,right-left,2,128).transpose(1,2)
            _,post=apply_rotary_pos_emb(pre,pre,cos[:,left:right],sin[:,left:right])
            key[:,:,left:right]=post
            value=self.v_proj(block).view(b,right-left,2,128).transpose(1,2)
            self.value_cache[:,:,start+left:start+right]=value
            if self.mode=='offload':
                f=self.factors
                base=torch.einsum('bhtd,hdr->bhtr',value,f['base_left_b16'])
                side=build_conditional_routing_sidecar(value,post,base_left=f['base_left_b16'],base_right=f['base_right_b16'],
                    base_bias=f['base_bias_b16'],residual_encoder=f['residual_encoder_b16_r16'],cos=cos[:,left:right],sin=sin[:,left:right])
                self.base_cache[:,:,start+left:start+right]=base
                self.residual_cache[:,:,start+left:start+right]=side[...,-16:]
        self.length=end
        if self.mode=='dense':self.key_cache[:,:,start:end]=key
        else:append_mapped_host_key(self.host_key,key,start=start)
        # Full dense prefill consumes temporary exact K before releasing it.
        for left in range(0,n,2048):
            right=min(left+2048,n)
            q=self.q_proj(x[:,left:right]).view(b,right-left,8,128).transpose(1,2)
            q,_=apply_rotary_pos_emb(q,q,cos[:,left:right],sin[:,left:right])
            if start==0:
                attention=flash_attn_func(q.transpose(1,2),key[:,:,:right].transpose(1,2),self.value_cache[:,:,:right].transpose(1,2),
                    softmax_scale=self.scaling,causal=True).transpose(1,2)
            elif self.mode=='dense':
                attention=flash_attn_with_kvcache(q.transpose(1,2),self.key_cache[:,:,:end].transpose(1,2),self.value_cache[:,:,:end].transpose(1,2),
                    softmax_scale=self.scaling,causal=False,num_splits=16).transpose(1,2)
            else:
                if end<=2048:
                    ids=torch.arange(end,device=x.device).expand(b,2,-1).contiguous()
                else:
                    historical=end-64;f=self.factors
                    logs=conditional_router_page_lse(q,self.base_cache[:,:,:historical],self.residual_cache[:,:,:historical],
                        base_right=f['base_right_b16'],base_bias=f['base_bias_b16'],residual_query=f['residual_query_b16_r16'],
                        rope_cos=self.rope_cos[:historical],rope_sin=self.rope_sin[:historical],scale=self.scaling)
                    pages=select_fixed_group_max_pages_cuda(logs,pages_per_kv_head=62,pinned_prefix_pages=1,force_current_page=False)
                    ids=(pages[...,None]*32+torch.arange(32,device=x.device)).flatten(-2)
                    ids=ids.masked_fill(ids>=historical,-1)
                    ids=torch.cat((ids,torch.arange(historical,end,device=x.device).expand(b,2,-1)),-1)
                attention=mapped_host_paged_attention(self.host_key,q,self.value_cache,ids,sequence_length=end,page_size=1,
                    splits=32,workspace=self.workspace,scale=self.scaling)
            projected=self.o_proj(attention.transpose(1,2).contiguous().reshape(b,right-left,1024))
            x[:,left:right]=projected
        return x,None


@torch.inference_mode()
def decoder(self,hidden_states,attention_mask=None,position_ids=None,past_key_values=None,use_cache=False,position_embeddings=None,**kwargs):
    residual=hidden_states
    hidden_states=self.input_layernorm(hidden_states)
    hidden_states,_=self.self_attn(hidden_states=hidden_states,position_embeddings=position_embeddings,
        attention_mask=attention_mask,past_key_values=past_key_values)
    hidden_states.add_(residual)
    for left in range(0,hidden_states.shape[1],1024):
        block=hidden_states[:,left:left+1024]
        block.add_(self.mlp(self.post_attention_layernorm(block)))
    return hidden_states


def install(model,*,mode,batch,capacity,root,rank):
    assert model.config.model_type=='llama' and model.config.num_hidden_layers==32
    model.model.norm=ChunkedTokenwise(model.model.norm)
    modules=[]
    for layer in model.model.layers:
        attn=LlamaTP4KCacheAttention(layer.self_attn,mode=mode,batch=batch,capacity=capacity,root=root,rank=rank)
        layer.self_attn=attn;layer.input_layernorm=ChunkedTokenwise(layer.input_layernorm)
        layer.forward=MethodType(decoder,layer);modules.append(attn)
    return modules
