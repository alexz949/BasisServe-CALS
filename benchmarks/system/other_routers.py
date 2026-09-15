"""Repository Loki and upstream ShadowKV routing, independent per-query support."""
from functools import lru_cache
from pathlib import Path
import torch
from basisserve.core.c1_shadowkv import C1ShadowKVState,chunk_tokens
from basisserve.core.c1_k_routing_sidecar import build_routing_sidecar
from basisserve.kernels.indexed_sparse_decode_attention import gqa_proxy_scores_triton


def shadow_provenance():
    import hashlib
    from benchmarks.system.common import command
    root=Path('external/ShadowKV')
    return dict(commit=command(['git','-C',str(root),'rev-parse','HEAD']),
        dirty=command(['git','-C',str(root),'diff']),
        cutlass_commit=command(['git','-C',str(root/'3rdparty/cutlass'),'rev-parse','HEAD']),
        cutlass_dirty=command(['git','-C',str(root/'3rdparty/cutlass'),'status','--porcelain']),
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [root/'kernels/batch_gemm_softmax.cu',root/'kernels/batch_gemm_softmax.h',root/'models/kv_cache.py']},
        cache_reordering='Excluded: native slot-reuse bookkeeping preserves the selected set; common backend deduplicates its own fetch requests')


@lru_cache(None)
def shadow_extension():
    from torch.utils.cpp_extension import load
    root=Path('external/ShadowKV').resolve();cutlass=root/'3rdparty/cutlass'
    assert (cutlass/'include/cutlass/cutlass.h').exists()
    return load('basisserve_shadow_router_v1',sources=[str(Path('benchmarks/system/shadow_binding.cpp').resolve()),str(root/'kernels/batch_gemm_softmax.cu')],
        extra_include_paths=[str(cutlass/'include'),str(cutlass/'examples/common'),str(cutlass/'tools/util/include'),str(root/'kernels')],
        extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17','--expt-relaxed-constexpr'])


def support_stats(ids,length):
    b,h,g,n=ids.shape
    unique=[torch.unique(row[row>=0]) for row in ids.reshape(b*h,-1)]
    return dict(tokens_per_query_head=(ids>=0).sum(-1).tolist(),
        unique_tokens_per_kv=[x.numel() for x in unique],pages32_per_kv=[torch.unique(x//32).numel() for x in unique],
        all_ids_in_range=bool(((ids>=0)&(ids<length)).all()))


class LokiRouter:
    has_preprocessing=True
    def __init__(self,q,k,basis):
        self.q=q.contiguous();self.codes=build_routing_sidecar(k,basis)
        self.basis=basis.repeat_interleave(4,0).contiguous();self.length=k.shape[2]
        self.batch,self.heads=k.shape[:2]
        self.preprocess()
    def preprocess(self):
        self.query_code=torch.einsum('bhqd,hdr->bhqr',self.q,self.basis).contiguous()
    def scan(self):
        self.scores=gqa_proxy_scores_triton(self.query_code[:,:,0].contiguous(),self.codes,scale=128**-.5)
        self.ids=self.scores.topk(2048,dim=-1,sorted=False).indices.reshape(self.batch,self.heads,4,2048)
        return self.ids
    def full(self):self.preprocess();return self.scan()
    def validate(self):
        self.full()
        reference=(self.query_code.float().reshape(self.batch,self.heads,4,32)@self.codes.float().transpose(-1,-2)*(128**-.5)).bfloat16()
        expected=reference.topk(2048,dim=-1,sorted=False).indices
        assert torch.equal(expected.sort(-1).values,self.ids.sort(-1).values)
        return dict(reference_ids_equal=True,**support_stats(self.ids,self.length))
    def state_stats(self):
        return dict(routing_read_dimensions_per_token=32,routing_state_bytes=self.codes.numel()*2+self.basis.numel()*2,
                    rank=32,sink=0,recent=0,nominal_budget=2048,selection='independent query-head token topk, repository Triton kernel')


class ShadowRouter:
    has_preprocessing=False
    def __init__(self,q,pre,k,cos,sin):
        self.q=q.contiguous();self.length=k.shape[2];self.batch,self.heads=k.shape[:2]
        self.state=C1ShadowKVState(pre,k,cos.expand(self.batch,-1,-1),sin.expand(self.batch,-1,-1),landmark_alignment=8)
        self.extension=shadow_extension()
        n=self.state.landmarks.shape[2];shape=(self.batch,self.heads,4,n)
        self.gemm=torch.empty(shape,device=q.device,dtype=q.dtype)
        self.softmax=torch.empty_like(self.gemm)
        self.norm=torch.empty(self.batch*self.heads,4,(n+255)//256,device=q.device,dtype=torch.float32)
        self.sums=torch.empty_like(self.norm)
    def preprocess(self):pass
    def scan(self):
        n=self.state.landmarks.shape[2]
        self.extension.batch_gemm_softmax(self.q,self.state.landmarks,self.gemm,self.norm,self.sums,self.softmax,
            self.batch*self.heads,4,n,128,128**-.5,0.)
        chosen=self.softmax.amax(2).topk(256,dim=-1).indices
        routed=chunk_tokens(self.state.landmark_ids.gather(-1,chosen),8)
        group_ids=torch.cat((self.state.fixed_ids,routed),-1)
        self.ids=group_ids[:,:,None].expand(-1,-1,4,-1)
        return self.ids
    def full(self):return self.scan()
    def validate(self):
        self.scan()
        q=self.q.float().reshape(self.batch,self.heads,4,128)
        score=(q@self.state.landmarks.float().transpose(-1,-2)*(128**-.5)).bfloat16()
        # CUTLASS applies the scaling before its BF16 GEMM output.
        reference=score.float().softmax(-1).bfloat16()
        torch.testing.assert_close(self.softmax,reference,rtol=.02,atol=.0005)
        chosen=reference.amax(2).topk(256,dim=-1).indices
        expected=torch.cat((self.state.fixed_ids,chunk_tokens(self.state.landmark_ids.gather(-1,chosen),8)),-1)
        assert torch.equal(expected.sort(-1).values,self.ids[:,:,0].sort(-1).values)
        return dict(reference_ids_equal=True,**support_stats(self.ids,self.length))
    def state_stats(self):
        s=self.state
        return dict(routing_read_dimensions_per_token=16,
            routing_state_bytes=s.landmarks.numel()*2+s.landmark_ids.numel()*8,
            native_lowrank_state_bytes=s.u.numel()*2+s.sv.numel()*2,
            rank=160,chunk_size=8,landmark_alignment_chunks=8,outlier_tokens=384,local_tokens=self.length-s.context_end,
            nominal_budget=2048,generated_tokens=0,
            selection='upstream CUTLASS GEMM/softmax + native landmark topk; current-stream plumbing only; prefill-state query')
