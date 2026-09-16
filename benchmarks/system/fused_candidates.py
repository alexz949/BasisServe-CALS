"""GQA normalization and top512 compaction into ascending page IDs."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _select(U,OUT,P:tl.constexpr,TAIL:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);p=tl.arange(0,B);h=tl.arange(0,4)
    x=tl.load(U+(row*4+h[:,None])*P+p[None,:],p[None,:]<P,other=-float('inf'))
    eligible=(p>0)&(p<TAIL)&(p<P)
    x=tl.where(eligible[None,:],x,-float('inf'))
    m=tl.max(x,1);z=tl.log(tl.sum(tl.exp(x-m[:,None]),1))+m
    score=tl.max(x-z[:,None],0)
    score=tl.where((p==0)|(p>=TAIL),3.4028234663852886e38,score)
    score=tl.where(p<P,score,-3.4028234663852886e38)
    bits=score.to(tl.uint32,bitcast=True)
    key=tl.where(score>=0.,bits^0x80000000,~bits)
    threshold=tl.full((),0,tl.uint32)
    # Exact radix selection finds the 512th largest key without sorting all pages.
    for bit in range(31,-1,-1):
        trial=threshold|(tl.full((),1,tl.uint32)<<bit)
        count=tl.sum(((p<P)&(key>=trial)).to(tl.int32),0)
        threshold=tl.where(count>=512,trial,threshold)
    above=(p<P)&(key>threshold)
    equal=(p<P)&(key==threshold)
    # PyTorch topk leaves cutoff ties unspecified. Use ascending original ID.
    equal_rank=tl.cumsum(equal.to(tl.int32),0)
    chosen=above|(equal&(equal_rank<=512-tl.sum(above.to(tl.int32),0)))
    index=tl.cumsum(chosen.to(tl.int32),0)-1
    tl.store(OUT+row*512+index,p.to(tl.int64),chosen)


def candidates(scores,historical):
    b,h,_,p=scores.shape
    if p<=512:return torch.arange(p,device=scores.device).expand(b,h,p).contiguous()
    ids=torch.empty(b,h,512,device=scores.device,dtype=torch.int64)
    _select[(b*h,)](scores,ids,p,historical//32,tr.next_power_of_2(p),num_warps=8)
    return ids
