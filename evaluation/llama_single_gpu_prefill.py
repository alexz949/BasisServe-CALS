"""Query-blocked native FlashAttention prefill; unchanged indexed Triton decode."""
from types import MethodType
import torch
from flash_attn import flash_attn_func
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from evaluation import eval_k_routing_ruler as common


@torch.inference_mode()
def prefill(self,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,**kwargs):
    batch,length,hidden=hidden_states.shape
    if length==1:
        return common.routing_forward(self,hidden_states,position_embeddings,attention_mask,past_key_values,**kwargs)
    assert batch==1 and past_key_values.get_seq_length(self.layer_idx)==0 and attention_mask is None
    hq,hkv,dim=self.num_attention_heads,self.num_key_value_heads,self.head_dim
    assert self.value_head_dim==dim
    cos,sin=position_embeddings
    key=torch.empty(batch,hkv,length,dim,device=hidden_states.device,dtype=hidden_states.dtype)
    value=torch.empty_like(key)
    chunk=2048
    for start in range(0,length,chunk):
        stop=min(start+chunk,length);x=hidden_states[:,start:stop]
        k=self.k_norm(self.k_proj(x).view(batch,stop-start,hkv,dim)).transpose(1,2)
        # Native RoPE arithmetic, bounded to one token block.
        _,k=apply_rotary_pos_emb(k,k,cos[:,start:stop],sin[:,start:stop])
        key[:,:,start:stop]=k
        value[:,:,start:stop]=self.v_proj(x).view(batch,stop-start,hkv,dim).transpose(1,2)
    key,value=past_key_values.update(key,value,self.layer_idx)
    if self._routing_arm=='ours':
        t=self._routing_factors
        pure=self._routing_base_rank==0
        sidecar=torch.empty(batch,hkv,length,16 if pure else dim+16,device=key.device,dtype=key.dtype)
        for start in range(0,length,chunk):
            stop=min(start+chunk,length)
            if pure:
                code=torch.einsum('bhtd,hdr->bhtr',key[:,:,start:stop],t['residual_encoder'].to(key.dtype))
            else:
                code=build_conditional_routing_sidecar(value[:,:,start:stop],key[:,:,start:stop],
                    base_left=t['base_left'],base_right=t['base_right'],base_bias=t['base_bias'],
                    residual_encoder=t['residual_encoder'],cos=cos[:,start:stop],sin=sin[:,start:stop])
            sidecar[:,:,start:stop]=code
        past_key_values.sidecars[self.layer_idx]=sidecar
    output=hidden_states  # K/V and sidecars are complete; consume each Q input block before overwriting it.
    for start in range(0,length,chunk):
        stop=min(start+chunk,length)
        q=self.q_norm(self.q_proj(hidden_states[:,start:stop]).view(batch,stop-start,hq,dim)).transpose(1,2)
        # A small dummy K keeps the native helper's operation ordering unchanged.
        q,_=apply_rotary_pos_emb(q,key[:,:,start:stop],cos[:,start:stop],sin[:,start:stop])
        attention=flash_attn_func(q.transpose(1,2),key[:,:,:stop].transpose(1,2),value[:,:,:stop].transpose(1,2),
            causal=True,softmax_scale=self.scaling)
        output[:,start:stop]=self.o_proj(attention.reshape(batch,stop-start,hidden))
    return output,None


class InplaceChunkedMLP(torch.nn.Module):
    def __init__(self,inner):
        super().__init__();self.inner=inner

    @torch.inference_mode()
    def forward(self,hidden_states):
        if hidden_states.shape[1]<=1024:return self.inner(hidden_states)
        # This is the fresh post-attention RMSNorm output, not the residual stream.
        for start in range(0,hidden_states.shape[1],1024):
            hidden_states[:,start:start+1024].copy_(self.inner(hidden_states[:,start:start+1024]))
        return hidden_states


@torch.inference_mode()
def decoder(self,hidden_states,attention_mask=None,position_ids=None,past_key_values=None,
            use_cache=False,position_embeddings=None,**kwargs):
    residual=hidden_states
    hidden_states=self.input_layernorm(hidden_states)
    hidden_states,_=self.self_attn(hidden_states=hidden_states,attention_mask=attention_mask,
        position_ids=position_ids,past_key_values=past_key_values,use_cache=use_cache,
        position_embeddings=position_embeddings,**kwargs)
    hidden_states.add_(residual)
    for start in range(0,hidden_states.shape[1],1024):
        block=hidden_states[:,start:start+1024]
        update=self.mlp(self.post_attention_layernorm(block))
        block.add_(update)
    return hidden_states


def install(model):
    for _,module in common.c1_attention_layers(model):module.forward=MethodType(prefill,module)
    model.model.norm=InplaceChunkedMLP(model.model.norm.inner)
    for layer in model.model.layers:
        layer.mlp=InplaceChunkedMLP(layer.mlp.inner)
        layer.forward=MethodType(decoder,layer)
