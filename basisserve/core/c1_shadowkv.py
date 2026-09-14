"""ShadowKV accuracy equations with resident C1 values.

Adapted from ByteDance-Seed/ShadowKV models/kv_cache.py, ShadowKVCache,
revision e51904cdeab7d4d34013370f09f2cf5fcd655e15 (Apache-2.0).
Retains shared pre-RoPE SVD, post-RoPE landmarks, outliers and exact local tail.
"""
import math
import torch
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import rotate_half


def gather_group(tensor,ids):
    return tensor.gather(2,ids[...,None].expand(*ids.shape,tensor.shape[-1]))


def chunk_tokens(chunks,size):
    return (chunks[...,None]*size+torch.arange(size,device=chunks.device)).flatten(-2)


class C1ShadowKVState:
    @torch.inference_mode()
    def __init__(self,pre_key,post_key,cos,sin,rank=160,budget=2048,chunk=8,outliers=48):
        b,h,length,d=pre_key.shape
        assert b==1 and pre_key.shape==post_key.shape and budget%chunk==0
        assert pre_key.dtype==post_key.dtype and 0<rank<=min(length,h*d)
        self.rank,self.budget,self.chunk=rank,budget,chunk
        self.outliers=outliers;self.prompt_length=length;self.length=length;self.steps=0
        self.chunks=length//chunk-4
        assert self.chunks>outliers+budget//chunk
        self.context_end=self.chunks*chunk
        flat=pre_key.transpose(1,2).reshape(b,length,h*d)
        u,s,v=torch.svd(flat.float())
        self.u=u[:,:,:rank].to(pre_key.dtype).contiguous()
        self.sv=(torch.diag_embed(s[:,:rank])@v.transpose(1,2)[:,:rank]).to(pre_key.dtype).view(b,rank,h,d).transpose(1,2).contiguous()
        self.cos=cos.clone();self.sin=sin.clone()
        ctx=post_key[:,:,:self.context_end].reshape(b,h,self.chunks,chunk,d)
        landmarks=ctx.mean(-2)
        similarity=F.cosine_similarity(landmarks.unsqueeze(3).expand_as(ctx),ctx,dim=-1)
        outlier_ids=similarity.min(-1).values.topk(outliers,largest=False).indices
        all_ids=torch.arange(self.chunks,device=pre_key.device).view(1,1,-1).expand(b,h,-1)
        mask=torch.ones_like(all_ids,dtype=torch.bool).scatter(-1,outlier_ids,False)
        self.landmark_ids=all_ids.masked_select(mask).view(b,h,-1)
        self.landmarks=landmarks.gather(2,self.landmark_ids[...,None].expand(b,h,self.chunks-outliers,d)).contiguous()
        local=torch.arange(self.context_end,length,device=pre_key.device).view(1,1,-1).expand(b,h,-1)
        self.fixed_ids=torch.cat((local,chunk_tokens(outlier_ids,chunk)),-1)
        self.fixed_key=gather_group(post_key,self.fixed_ids).clone()
        self.generated_key=post_key[:,:,:0].clone()
        self.selected_ids=None

    def select(self,query):
        b,h,_,d=self.landmarks.shape
        assert query.shape[2]==1 and query.shape[1]%h==0
        heads=query.shape[1]//h
        scores=torch.einsum('bhgqd,bhdc->bhgqc',query.reshape(b,h,heads,1,d),self.landmarks.transpose(2,3))/math.sqrt(d)
        mass=torch.softmax(scores,dim=-1,dtype=torch.float32).to(query.dtype).sum(-2)
        group_mass=mass.max(-2).values
        selected=group_mass.topk(self.budget//self.chunk,dim=-1).indices
        return chunk_tokens(self.landmark_ids.gather(-1,selected),self.chunk)

    def reconstruct(self,ids):
        b,h,n=ids.shape
        selected_u=self.u[:,None].expand(b,h,-1,self.rank).gather(2,ids[...,None].expand(b,h,n,self.rank))
        pre=torch.einsum('bhrk,bhkd->bhrd',selected_u,self.sv)
        width=self.cos.shape[-1]
        cos=self.cos[:,None].expand(b,h,-1,-1).gather(2,ids[...,None].expand(b,h,n,width))
        sin=self.sin[:,None].expand(b,h,-1,-1).gather(2,ids[...,None].expand(b,h,n,width))
        rotated=pre[...,:width]*cos+rotate_half(pre[...,:width])*sin
        return torch.cat((rotated,pre[...,width:]),dim=-1)

    @torch.inference_mode()
    def decode(self,query,current_key,resident_value,scale):
        assert current_key.shape[2]==1 and resident_value.shape[2]==self.length+1
        self.generated_key=torch.cat((self.generated_key,current_key),2)
        self.length+=1;self.steps+=1
        routed=self.select(query)
        reconstructed=self.reconstruct(routed)
        b,h=resident_value.shape[:2]
        generated=torch.arange(self.prompt_length,self.length,device=query.device).view(1,1,-1).expand(b,h,-1)
        self.selected_ids=torch.cat((self.fixed_ids,routed,generated),-1)
        key=torch.cat((self.fixed_key,reconstructed,self.generated_key),2)
        value=gather_group(resident_value,self.selected_ids)
        groups=query.shape[1]//h
        assert torch.isfinite(key).all() and key.dtype==value.dtype==query.dtype
        if query.is_cuda and query.dtype in (torch.float16,torch.bfloat16) and value.shape[-1]<=query.shape[-1]:
            from basisserve.core.compact_v_flash import compact_v_flash_attention
            return compact_v_flash_attention(query,key,value,scale=scale)
        return F.scaled_dot_product_attention(query,key.repeat_interleave(groups,1),value.repeat_interleave(groups,1),
            scale=scale,dropout_p=0,is_causal=False)

    def statistics(self):
        if self.steps == 0:
            assert self.selected_ids is None and self.length == self.prompt_length
            return dict(length=self.length,decode_steps=0,svd_rank=self.rank,
                routed_tokens=0,outlier_tokens=0,prompt_local_tokens=0,generated_tokens=0,
                physical_tokens_per_group=0)
        assert self.selected_ids is not None
        return dict(length=self.length,decode_steps=self.steps,svd_rank=self.rank,
            routed_tokens=self.budget,outlier_tokens=self.outliers*self.chunk,
            prompt_local_tokens=self.prompt_length-self.context_end,generated_tokens=self.steps,
            physical_tokens_per_group=self.selected_ids.shape[-1])
