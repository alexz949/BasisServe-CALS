"""Install the validated two-stage router with per-request cache reset."""
import torch
import benchmarks.system.native_basis_cache as basis
from benchmarks.system.two_stage_router import Metadata,candidates
from basisserve.kernels.mapped_host_paged_attention import mapped_host_device_pointer,select_fixed_group_max_pages_cuda
from basisserve.kernels.slot_indexed_attention import slot_indexed_attention
from benchmarks.system.validate_local_kernels import workspace


def install(full,fine,slots):
    current={};parent=basis.BasisCache
    class EvaluationCache(parent):
        def __init__(self,llm,width):
            super().__init__(llm,width);self.mode='full';self.metadata={};self.slot_states={}
        def clear(self):
            super().clear();self.metadata.clear();self.slot_states.clear();self.validated=set()
        def attention(self,q,k,v,layer,positions):
            current['cache']=self;current['layer']=layer
            return super().attention(q,k,v,layer,positions)
    basis.BasisCache=EvaluationCache
    append=basis.append_mapped_host_key
    def append_key(host,k,*,start):
        cache=current['cache'];layer=current['layer']
        if cache.mode=='two':
            if start==0:cache.metadata[layer]=Metadata(k,host.shape[2])
            else:cache.metadata[layer].advance(k)
        return append(host,k,start=start)
    basis.append_mapped_host_key=append_key
    def choose(logs):
        return select_fixed_group_max_pages_cuda(logs,pages_per_kv_head=62,pinned_prefix_pages=1,force_current_page=False)
    def route(q,b,res,**kw):
        cache=current['cache'];n=b.shape[2]
        code=torch.empty(q.shape[0],b.shape[1],4,16,device=q.device,dtype=q.dtype)
        args=(q,b,res,kw['base_right'],kw['base_bias'],kw['residual_query'],kw['rope_cos'],kw['rope_sin'],code)
        if cache.mode=='two':
            meta=cache.metadata[current['layer']];assert meta.n==n
            ids=candidates(meta.scores(q),n)
            logs=torch.empty(q.shape[0],b.shape[1],4,ids.shape[-1],device=q.device)
            fine.conditional_router_page_lse(*args,logs,kw['scale'],False,ids)
            current['selected']=ids.gather(-1,choose(logs))
        else:
            logs=torch.empty(q.shape[0],b.shape[1],4,(n+31)//32,device=q.device)
            full.conditional_router_page_lse(*args,logs,kw['scale'],False)
            current['selected']=choose(logs)
        return logs
    basis.conditional_router_page_lse=route
    basis.select_fixed_group_max_pages_cuda=lambda logs,**kw:current['selected']
    def attention(host,q,v,ids,**kw):
        cache=current['cache'];layer=current['layer'];b,h,n=ids.shape
        if layer not in cache.slot_states:
            cache.slot_states[layer]=dict(key=torch.empty(b,h,n,128,device=q.device,dtype=q.dtype),
                resident=torch.full_like(ids,-1),lookup=torch.full(v.shape[:3],-1,device=q.device,dtype=torch.int32),
                selected=torch.empty_like(ids),missing=torch.empty_like(ids,dtype=torch.int32),
                counts=torch.empty(b,h,2,device=q.device,dtype=torch.int32),pointer=mapped_host_device_pointer(host),work=workspace(q,v.shape[-1]))
        c=cache.slot_states[layer]
        slots.refresh(c['pointer'],c['key'],ids,c['resident'],c['lookup'],c['selected'],c['missing'],c['counts'],True)
        return slot_indexed_attention(q,c['key'],v,ids,c['selected'],c['work'],scale=kw['scale'])
    basis.mapped_host_paged_attention=attention
