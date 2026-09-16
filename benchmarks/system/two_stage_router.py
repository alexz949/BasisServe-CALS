"""Exact-K Page32 metadata and candidate-only register MMA routing."""
import hashlib
from pathlib import Path
import torch
import triton as tr
import triton.language as tl
from torch.utils.cpp_extension import load
from benchmarks.system.local_kernel_candidates import write_source
from benchmarks.system.register_router import compile_register


@tr.jit
def _summary(K, MN, MX, N:tl.constexpr, H:tl.constexpr, CAP:tl.constexpr,
             S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr):
    row=tl.program_id(0);p=tl.program_id(1)
    t=p*32+tl.arange(0,32);d=tl.arange(0,128)
    x=tl.load(K+row//H*S0+row%H*S1+t[:,None]*S2+d[None,:],t[:,None]<N,other=0).to(tl.float32)
    lo=tl.min(tl.where(t[:,None]<N,x,float('inf')),0)
    hi=tl.max(tl.where(t[:,None]<N,x,-float('inf')),0)
    tl.store(MN+(row*CAP+p)*128+d,lo);tl.store(MX+(row*CAP+p)*128+d,hi)


@tr.jit
def _advance(K,RING,MN,MX,H:tl.constexpr,CAP:tl.constexpr,POS,SLOT,
             S0:tl.constexpr,S1:tl.constexpr):
    row=tl.program_id(0);d=tl.arange(0,128)
    addr=(row*64+SLOT)*128+d
    old=tl.load(RING+addr).to(tl.float32)
    off=(row*CAP+POS//32)*128+d
    lo=tl.load(MN+off);hi=tl.load(MX+off)
    tl.store(MN+off,tl.where(POS%32==0,old,tl.minimum(lo,old)))
    tl.store(MX+off,tl.where(POS%32==0,old,tl.maximum(hi,old)))
    new=tl.load(K+row//H*S0+row%H*S1+d)
    tl.store(RING+addr,new)


@tr.jit
def _coarse(Q,MN,MX,U,H:tl.constexpr,CAP:tl.constexpr,P:tl.constexpr,N,
            Q0:tl.constexpr,Q1:tl.constexpr):
    head=tl.program_id(0);row=head//4
    p=tl.program_id(1)*16+tl.arange(0,16);d=tl.arange(0,128)
    q=tl.load(Q+row//H*Q0+(row%H*4+head%4)*Q1+d).to(tl.float32)
    valid=(p<P)&(p*32<N)
    lo=tl.load(MN+(row*CAP+p[:,None])*128+d[None,:],valid[:,None],other=0).to(tl.float32)
    hi=tl.load(MX+(row*CAP+p[:,None])*128+d[None,:],valid[:,None],other=0).to(tl.float32)
    x=tl.sum(tl.maximum(q[None,:]*lo,q[None,:]*hi),1)*0.08838834764831845
    x+=tl.log(tl.minimum(32,tl.maximum(1,N-p*32)).to(tl.float32))
    tl.store(U+head*P+p,tl.where(valid,x,-float('inf')),p<P)


@tr.jit
def _group(U,G,P:tl.constexpr,TAIL:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);p=tl.arange(0,B);h=tl.arange(0,4)
    x=tl.load(U+(row*4+h[:,None])*P+p[None,:],p[None,:]<P,other=-float('inf'))
    eligible=(p>0)&(p<TAIL)&(p<P)
    x=tl.where(eligible[None,:],x,-float('inf'))
    m=tl.max(x,1);z=tl.log(tl.sum(tl.exp(x-m[:,None]),1))+m
    score=tl.max(x-z[:,None],0)
    fixed=(p==0)|(p>=TAIL)
    tl.store(G+row*P+p,tl.where(fixed,float('inf'),score),p<P)


class Metadata:
    def __init__(self,k,capacity):
        b,h,n,_=k.shape;self.h=h;self.n=n-64;self.cap=tr.cdiv(capacity,32)
        self.minimum=torch.empty(b,h,self.cap,128,device=k.device,dtype=k.dtype)
        self.maximum=torch.empty_like(self.minimum)
        self.ring=k[:,:,-64:].contiguous().clone();self.slot=0
        _summary[(b*h,tr.cdiv(self.n,32))](k,self.minimum,self.maximum,self.n,h,self.cap,*k.stride()[:3])
    def advance(self,k):
        _advance[(k.shape[0]*self.h,)](k,self.ring,self.minimum,self.maximum,self.h,self.cap,self.n,self.slot,*k.stride()[:2])
        self.n+=1;self.slot=(self.slot+1)%64
    def scores(self,q):
        p=tr.cdiv(self.n+64,32)
        out=torch.empty(q.shape[0],self.h,4,p,device=q.device)
        _coarse[(q.shape[0]*self.h*4,tr.cdiv(p,16))](q,self.minimum,self.maximum,out,self.h,self.cap,p,self.n,*q.stride()[:2])
        return out


def candidates(scores,historical):
    b,h,_,p=scores.shape
    if p<=512:return torch.arange(p,device=scores.device).expand(b,h,p).contiguous()
    group=torch.empty(b,h,p,device=scores.device)
    _group[(b*h,)](scores,group,p,historical//32,tr.next_power_of_2(p),num_warps=8)
    return group.topk(512,dim=-1,sorted=False).indices.sort(dim=-1).values


def compile_fine(root):
    full=compile_register(root/'full',16,8)
    folder=root/'fine';folder.mkdir(parents=True,exist_ok=True)
    src=root/'full/b16_w8'
    text=(src/'conditional_router_page32.cu').read_text()
    left=text.index('__global__ void conditional_router_page_lse_kernel(')
    right=text.index('__global__ void conditional_router_append_decode_kernel(',left)
    body=text[left:right]
    body=body.replace('float scale) {','float scale, const int64_t* ids) {')
    body=body.replace('int64_t start=group*T;',
        'auto token_at = [&](int t) { int64_t idx=group*P+t/32; return idx<pages ? ids[kvrow*pages+idx]*32+t%32 : tokens; };')
    body=body.replace('start+t<tokens','token_at(t)<tokens').replace('(start+t)*base_stride_token','token_at(t)*base_stride_token')
    body=body.replace('token=start+t0+v*8','token=token_at(t0+v*8)')
    body=body.replace('token=start+p*32+lane','token=token_at(p*32+lane)')
    body=body.replace('=maximum+__logf(sum);','=maximum == -CUDART_INF_F ? -CUDART_INF_F : maximum+__logf(sum);')
    text=text[:left]+body+text[right:]
    text=text.replace('double scale, bool query_code_prepared) {','double scale, bool query_code_prepared, const at::Tensor& ids) {')
    text=text.replace('(base_code.size(2) + kPageSize - 1) / kPageSize;','ids.size(2);')
    text=text.replace('static_cast<float>(scale));','static_cast<float>(scale), ids.data_ptr<int64_t>());',1)
    write_source(folder/'conditional_router_page32.cu',text)
    cpp=(src/'mapped_host_paged_attention.cpp').read_text()
    cpp=cpp.replace('double scale, bool query_code_prepared);','double scale, bool query_code_prepared, const at::Tensor& ids);')
    cpp=cpp.replace('pybind11::arg("query_code_prepared"));','pybind11::arg("query_code_prepared"), pybind11::arg("ids"));')
    write_source(folder/'mapped_host_paged_attention.cpp',cpp)
    write_source(folder/'mapped_host_paged_attention.cu',(src/'mapped_host_paged_attention.cu').read_text())
    digest=hashlib.sha256((text+cpp).encode()).hexdigest()[:10]
    fine=load(name=f'two_stage_{digest}',sources=[str(folder/n) for n in ['mapped_host_paged_attention.cpp','mapped_host_paged_attention.cu','conditional_router_page32.cu']],extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17','--use_fast_math','-DBASIS_VALUE_DIM=80','-DBASIS_GQA=4','-DBASIS_PAGE_SIZE=32','-DBASIS_BASE_RANK=16','-DBASIS_RESIDUAL_RANK=16','-DREGISTER_WARPS=8'])
    return full,fine
