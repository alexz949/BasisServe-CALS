"""Dense-V Basis K offload inside the upstream ShadowKV Llama runtime."""
from pathlib import Path
import torch
from safetensors.torch import load_file
from flash_attn import flash_attn_with_kvcache
from basisserve.kernels.mapped_host_paged_attention import (
    mapped_host_bf16_empty,append_mapped_host_key,mapped_host_paged_attention,
    conditional_router_page_lse,select_fixed_group_max_pages_cuda)


class BasisCache:
    def __init__(self,llm,width):
        self.width=width;self.length=0;self.llm=llm;self.validate=False;self.validated=set()
        self.cos=llm.cos_sin_cache[:,:64].contiguous();self.sin=llm.cos_sin_cache[:,64:].contiguous()
        b=llm.batch_size;t=llm.max_length+128
        self.values=[];self.keys=[];self.bases=[];self.residuals=[];self.factors=[]
        root=Path('/home/zhangal/BasisServe-CALS-runs/llama31_8b_64k_densev/ours_b16r16')
        for layer in range(llm.num_layers):
            self.values.append(torch.empty(b,8,t,128,device='cuda',dtype=torch.bfloat16))
            self.keys.append(mapped_host_bf16_empty(batch=b,kv_heads=8,capacity=t))
            self.bases.append(torch.empty(b,8,t,width,device='cuda',dtype=torch.bfloat16))
            self.residuals.append(torch.empty_like(self.bases[-1]))
            self.factors.append({k:v.cuda().bfloat16().contiguous() for k,v in load_file(str(root/f'layer_{layer:03d}.safetensors')).items()})
        for f in self.factors:
            f['right']=f['base_right_b16'][:,:width].contiguous()
            f['query']=f['residual_query_b16_r16'][...,:width].contiguous()
            f['encoder']=f['residual_encoder_b16_r16'][...,:width].contiguous()
        self.workspace=torch.empty(b*32,32,130,device='cuda',dtype=torch.float32)

    def clear(self):self.length=0
    def get_kv_len(self):return self.length
    def H2D(self):pass
    def print_stats(self):print('Basis Dense V, mapped CPU K, rank',self.width,flush=True)

    def attention(self,q,k,v,layer,positions):
        n=q.shape[2];start=self.length;end=start+n;r=self.width
        assert n==1 or start==0
        self.values[layer][:,:,start:end]=v
        append_mapped_host_key(self.keys[layer],k,start=start)
        f=self.factors[layer];rope=self.llm.cos_sin_cache
        for left in range(0,n,2048):
            right=min(n,left+2048);x=v[:,:,left:right]
            full_base=x@f['base_left_b16']
            predicted=(full_base@f['base_right_b16']+f['base_bias_b16'][None,:,None]).bfloat16()
            c=rope[start+left:start+right,:64].repeat(1,2)[None,None]
            s=rope[start+left:start+right,64:].repeat(1,2)[None,None]
            rotated=torch.cat((-predicted[...,64:],predicted[...,:64]),-1)
            predicted=(predicted*c+rotated*s).bfloat16()
            # Slice the original B16R16 codes, preserving the fitted residual target.
            residual=(k[:,:,left:right]-predicted)@f['encoder']
            self.bases[layer][:,:,start+left:start+right]=full_base[...,:r]
            self.residuals[layer][:,:,start+left:start+right]=residual
        if n>1:
            out=flash_attn_with_kvcache(q=q.transpose(1,2),k_cache=k.transpose(1,2),v_cache=v.transpose(1,2),causal=True)
        else:
            historical=end-64
            logs=conditional_router_page_lse(q,self.bases[layer][:,:,:historical],self.residuals[layer][:,:,:historical],
                base_right=f['right'],base_bias=f['base_bias_b16'],
                residual_query=f['query'],
                rope_cos=self.cos[:historical],rope_sin=self.sin[:historical],scale=128**-.5)
            pages=select_fixed_group_max_pages_cuda(logs,pages_per_kv_head=62,pinned_prefix_pages=1,force_current_page=False)
            ids=(pages[...,None]*32+torch.arange(32,device='cuda')).flatten(-2)
            ids=ids.masked_fill(ids>=historical,-1)
            ids=torch.cat((ids,torch.arange(historical,end,device='cuda').expand(q.shape[0],8,-1)),-1)
            out=mapped_host_paged_attention(self.keys[layer],q,self.values[layer],ids,sequence_length=end,page_size=1,
                splits=32,workspace=self.workspace,scale=128**-.5).transpose(1,2)
        if n==1 and self.validate and layer not in self.validated:
            index=ids.clamp_min(0)[...,None].expand(-1,-1,-1,128)
            selected_k=self.keys[layer].gather(2,index.cpu()).cuda()
            selected_v=self.values[layer].gather(2,index)
            # Per-KV support shared by its four query heads; invalid tail slots masked.
            expanded_k=selected_k.repeat_interleave(4,dim=1).float()
            expanded_v=selected_v.repeat_interleave(4,dim=1).float()
            score=(q.float()@expanded_k.transpose(-1,-2))*128**-.5
            score=score.masked_fill((ids<0).repeat_interleave(4,dim=1)[:,:,None],-torch.inf)
            expected=(score.softmax(-1)@expanded_v).transpose(1,2)
            torch.testing.assert_close(out.float(),expected,rtol=.03,atol=.03)
            self.validated.add(layer)
        if layer==self.llm.num_layers-1:self.length=end
        return out


def basis_llama_class(native,width):
    class BasisLlama(native):
        def init_kv_cache(self,*args,**kwargs):self.kv_cache=BasisCache(self,width)

        def layer_compute(self,buffer,layer_idx,hidden_states,position_ids):
            residual=hidden_states;b,n,_=hidden_states.shape
            q,k,v=self.pre_attention_compute(hidden_states,buffer,self.num_heads,self.num_key_value_heads,self.head_dim)
            q,k=self.apply_rotary_pos_emb(q,k,position_ids)
            out=self.kv_cache.attention(q,k,v,layer_idx,position_ids).reshape(b,n,self.hidden_size)
            if b*n>64*1024:
                output=torch.empty_like(out);chunk=b*n//(b*n//8192)
                for left in range(0,n,chunk):
                    output[:,left:left+chunk]=self.post_attention_compute(out[:,left:left+chunk],residual[:,left:left+chunk],buffer)
                return output
            return self.post_attention_compute(out,residual,buffer)
    return BasisLlama
