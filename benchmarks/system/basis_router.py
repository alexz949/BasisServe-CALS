"""Use the production CUDA B16R16 scanner with an isolated Base prefix cache."""
import math
import torch
from transformers.models.llama.modeling_llama import rotate_half
from basisserve.kernels.mapped_host_paged_attention import (
    conditional_router_page_lse,conditional_router_query_code,select_fixed_group_max_pages_cuda)
from evaluation.llama_sink_recent_routing import page_support


class BasisRouter:
    def __init__(self,q,key,base,tail,factors,cos,sin,budget=2048):
        self.q=q.contiguous();self.base=base.contiguous().clone();self.tail=tail.contiguous().clone()
        self.factors={k:v.bfloat16().contiguous() for k,v in factors.items()}
        self.cos=cos;self.sin=sin;self.budget=budget
        self.batch,self.heads,self.length,_=base.shape
        assert self.base.shape[-1]==16 and self.length>budget
        assert self.base.untyped_storage().data_ptr()!=self.tail.untyped_storage().data_ptr()
        pre=(self.base@self.factors['base_right_b16']+self.factors['base_bias_b16'][None,:,None]).bfloat16()
        post=(pre*cos[:,None]+rotate_half(pre)*sin[:,None]).bfloat16()
        self.residual=((key-post)@self.factors['residual_encoder_b16_r16']).contiguous()
        self.historical=self.length-64
        self.rope_cos=cos[0,:self.historical,:64].contiguous();self.rope_sin=sin[0,:self.historical,:64].contiguous()
        self.query_code=torch.empty(self.batch,self.heads,4,16,device=q.device,dtype=q.dtype)
        self.logs=torch.empty(self.batch,self.heads,4,math.ceil(self.historical/32),device=q.device,dtype=torch.float32)
        self.pages=torch.empty(self.batch,self.heads,(budget-64)//32,device=q.device,dtype=torch.int64)
        self.ids=torch.empty(self.batch,self.heads,budget,device=q.device,dtype=torch.int64)
        self.offsets=torch.arange(32,device=q.device)
        self.recent=torch.arange(self.historical,self.length,device=q.device).expand(self.batch,self.heads,-1)

    def preprocess(self):
        return conditional_router_query_code(self.q,self.factors['residual_query_b16_r16'],self.query_code)

    def scan(self,prepared=True):
        conditional_router_page_lse(self.q,self.base[:,:,:self.historical],self.residual[:,:,:self.historical],
            base_right=self.factors['base_right_b16'],base_bias=self.factors['base_bias_b16'],
            residual_query=self.factors['residual_query_b16_r16'],rope_cos=self.rope_cos,rope_sin=self.rope_sin,
            scale=128**-0.5,query_code=self.query_code,query_code_prepared=prepared,output=self.logs)
        select_fixed_group_max_pages_cuda(self.logs,pages_per_kv_head=self.pages.shape[-1],pinned_prefix_pages=1,
            force_current_page=False,output=self.pages)
        historical=(self.pages[...,None]*32+self.offsets).flatten(-2)
        historical=historical.masked_fill(historical>=self.historical,-1)
        self.ids.copy_(torch.cat((historical,self.recent),-1))
        return self.ids

    def full(self):
        self.preprocess();return self.scan()

    def reference(self):
        pre=(self.base@self.factors['base_right_b16']+self.factors['base_bias_b16'][None,:,None]).bfloat16()
        post=(pre*self.cos[:,None]+rotate_half(pre)*self.sin[:,None]).bfloat16()
        q=self.q.reshape(self.batch,self.heads,4,128)
        scores=(q@post.transpose(-1,-2)+(self.query_code@self.residual.transpose(-1,-2))).bfloat16()
        scores=(scores*(128**-0.5)).bfloat16()
        ids,valid=page_support(scores,budget=self.budget)
        return ids.masked_fill(~valid,-1)

    def validate(self):
        self.preprocess();self.scan();actual=self.ids.clone();logs=self.logs.clone()
        self.scan(prepared=False)
        assert torch.equal(logs,self.logs) and torch.equal(actual,self.ids)
        reference=self.reference()
        expected=reference.sort(-1).values;observed=actual.sort(-1).values
        assert torch.equal(expected,observed),dict(mismatched_ids=int((expected!=observed).sum()))
        saved_tail=self.tail.clone()
        self.tail.fill_(float('nan'));self.scan()
        assert torch.equal(actual,self.ids)
        self.tail.copy_(saved_tail)
        return dict(reference_ids_equal=True,split_query_preprocessing_bitwise_equal=True,
                    nonrouting_payload_poisoned_without_effect=True,physical_tokens_per_kv=(self.ids>=0).sum(-1).tolist(),
                    physical_pages_per_kv=[torch.unique(row[row>=0]//32).numel() for row in self.ids.reshape(-1,self.budget)])
