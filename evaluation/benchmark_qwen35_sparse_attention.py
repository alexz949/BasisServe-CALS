"""Compare the current gather/Flash path to resident indexed Triton attention."""
import json
import torch
import triton
from basisserve.kernels.split_indexed_attention import split_indexed_attention
from basisserve.core.qwen35_k_routing_runtime import page_support
from basisserve.core.c1_shadowkv import gather_group
from basisserve.core.compact_v_flash import compact_v_flash_attention
from basisserve.kernels.indexed_sparse_decode_attention import gqa_indexed_sparse_decode_attention_triton

torch.set_num_threads(2)
torch.manual_seed(73)
for length in (32769, 64507):
 for rank in (128,192,224,256):
  q=torch.randn(1,16,1,256,device='cuda',dtype=torch.bfloat16)
  k=torch.randn(1,4,length,256,device='cuda',dtype=torch.bfloat16)
  v=torch.randn(1,4,length,rank,device='cuda',dtype=torch.bfloat16)
  ids,valid=page_support(torch.randn(1,4,4,length,device='cuda'))
  head_ids=ids.masked_fill(~valid,-1).repeat_interleave(4,dim=1)
  def reference():
   sk,sv=gather_group(k,ids.clamp_max(length-1)),gather_group(v,ids.clamp_max(length-1))
   pieces=[]
   for g in range(4):
    keep=valid[0,g]
    pieces.append(compact_v_flash_attention(q[:,g*4:(g+1)*4],sk[:,g:g+1,keep],sv[:,g:g+1,keep],scale=1/16))
   return torch.cat(pieces,1)
  def indexed():
   return gqa_indexed_sparse_decode_attention_triton(q,k,v,head_ids,scale=1/16)
  def split():
   return split_indexed_attention(q,k,v,head_ids,scale=1/16)
  ref,out=reference(),indexed()
  split_out=split()
  split_rel=float((ref.float()-split_out.float()).norm()/ref.float().norm())
  assert split_rel<0.01 and torch.isfinite(split_out).all()
  rel=float((ref.float()-out.float()).norm()/ref.float().norm())
  assert rel<0.01 and torch.isfinite(out).all()
  old=triton.testing.do_bench(reference,warmup=100,rep=300)
  new=triton.testing.do_bench(indexed,warmup=100,rep=300)
  split_ms=triton.testing.do_bench(split,warmup=100,rep=300)
  print(json.dumps(dict(split_ms=split_ms,split_speedup=old/split_ms,split_relative_error=split_rel,length=length,rank=rank,relative_error=rel,gather_flash_ms=old,indexed_ms=new,speedup=old/new)),flush=True)
