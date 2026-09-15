"""Compact fetch requests while retaining each query's original support map."""
import triton
import triton.language as tl


@triton.jit
def encode_requests(ids,encoded,LENGTH:tl.constexpr,GROUP_REQUESTS:tl.constexpr,COUNT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    token=tl.load(ids+i,i<COUNT,other=-1)
    # A single sentinel represents every invalid request across all KV heads.
    sentinel=COUNT//GROUP_REQUESTS*LENGTH
    value=tl.where((token>=0)&(token<LENGTH),i//GROUP_REQUESTS*LENGTH+token,sentinel)
    tl.store(encoded+i,value,i<COUNT)


@triton.jit
def pack_requests(sorted_ids,permutation,positions,unique_ids,inverse,count,
                  SENTINEL:tl.constexpr,COUNT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    token=tl.load(sorted_ids+i,i<COUNT,other=SENTINEL)
    previous=tl.load(sorted_ids+i-1,(i>0)&(i<COUNT),other=-1)
    slot=tl.load(positions+i,i<COUNT,other=0)-1
    original=tl.load(permutation+i,i<COUNT,other=0)
    valid=(i<COUNT)&(token<SENTINEL)
    tl.store(unique_ids+slot,token,valid&(token!=previous))
    tl.store(inverse+original,tl.where(valid,slot,-1),i<COUNT)
    if tl.program_id(0)==tl.cdiv(COUNT,BLOCK)-1:
        last=tl.load(sorted_ids+COUNT-1)
        total=tl.load(positions+COUNT-1)
        tl.store(count,total-tl.cast(last==SENTINEL,tl.int64))
