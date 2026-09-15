"""A fetch union must not turn into the attention support for every head."""
import json
from pathlib import Path
import pytest
import torch
from basisserve.core.c1_k_offload import PinnedCPUExactKeyPageStore,PreparedQueryKeyFetch
from benchmarks.system.numa_memory import bind_host_allocations,audit_host_tensor,slurm_gpu_numa
from basisserve.kernels.indexed_sparse_decode_attention import gqa_indexed_sparse_decode_attention_triton

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU required')


@torch.inference_mode()
def test_query_support_survives_fetch_union():
    hardware=json.loads(Path('results/system_benchmarks/l40s/hardware.json').read_text())
    node=slurm_gpu_numa(hardware)['numa_node'];library=bind_host_allocations(node)
    b,h,t,d,g,n=2,2,128,128,4,6
    host=torch.empty(b,h,t,d,dtype=torch.bfloat16,pin_memory=True)
    host.copy_(torch.randn(host.shape,dtype=torch.bfloat16))
    fetch=PreparedQueryKeyFetch(PinnedCPUExactKeyPageStore(host),groups=g,tokens_per_query=n,device=torch.device('cuda',0))
    pointers=[x.data_ptr() for x in [fetch.destination,fetch.staging,fetch.host_indices,fetch.inverse_ids]]
    request=torch.tensor([[1,4,8,10,-1,-1],[4,5,9,11,12,13],[1,4,8,10,-1,-1],[2,3,7,6,14,15]])
    q=torch.randn(b,h*g,1,d,device='cuda',dtype=torch.bfloat16)
    value=torch.randn(b,h,t,96,device='cuda',dtype=torch.bfloat16)
    prefix=value[...,:16].contiguous();tail=value[...,16:].contiguous()
    output=torch.empty(b,h*g,1,96,device='cuda',dtype=torch.bfloat16)
    for shift in [0,32,64,0]:
        ids=request.expand(b,h,-1,-1).clone()
        ids[ids>=0]+=shift
        actual=fetch(ids.cuda()).clone()
        inverse=fetch.inverse_ids.cpu();torch.cuda.synchronize()
        assert torch.equal(inverse>=0,ids>=0)
        expected=host[:,:,None].expand(b,h,g,t,d).gather(3,ids.clamp_min(0)[...,None].expand(b,h,g,n,d))
        reconstructed=actual[inverse.clamp_min(0).cuda()].cpu()
        assert torch.equal(reconstructed[ids>=0],expected[ids>=0])
        per_group_unique=torch.unique(ids[0,0][ids[0,0]>=0]).numel()
        assert fetch.actual_count==b*h*per_group_unique
        assert fetch.actual_count<int((ids>=0).sum())
        assert not torch.equal(inverse[:,:,0],inverse[:,:,1])
        assert torch.equal(inverse[:,:,0],inverse[:,:,2])
        assert fetch.traffic()['h2d_dma_payload_bytes']==fetch.actual_count*d*2
        assert pointers==[x.data_ptr() for x in [fetch.destination,fetch.staging,fetch.host_indices,fetch.inverse_ids]]
        gpu_ids=ids.cuda().reshape(b,h*g,n)
        observed=gqa_indexed_sparse_decode_attention_triton(q,fetch.destination,tail,gpu_ids,
            selected_key_rows=fetch.inverse_ids.view(b,h*g,n),value_prefix=prefix,output=output,scale=d**-.5)
        selected_value=value[:,:,None].expand(b,h,g,t,96).gather(3,ids.cuda().clamp_min(0)[...,None].expand(b,h,g,n,96)).float()
        logits=(q.reshape(b,h,g,1,d).float()*expected.cuda().float()).sum(-1)*d**-.5
        reference=(logits.masked_fill(ids.cuda()<0,-torch.inf).softmax(-1)[...,None]*selected_value).sum(-2)
        torch.testing.assert_close(observed.reshape_as(reference).float(),reference,rtol=.015,atol=.004)
        # The resident-key path uses the same kernel with no indirection or split V.
        resident=gqa_indexed_sparse_decode_attention_triton(q,host.cuda(),value,gpu_ids,scale=d**-.5)
        torch.testing.assert_close(observed,resident,rtol=0,atol=0)
    for tensor in [host,fetch.host_count,fetch.host_indices,fetch.staging]:
        audit_host_tensor(tensor,node,library)
    empty=torch.full(fetch.shape,-1,device='cuda',dtype=torch.int64)
    fetch(empty)
    assert fetch.actual_count==0 and bool((fetch.inverse_ids==-1).all())
    observed=gqa_indexed_sparse_decode_attention_triton(q,fetch.destination,tail,empty.view(b,h*g,n),
        selected_key_rows=fetch.inverse_ids.view(b,h*g,n),value_prefix=prefix,output=output,scale=d**-.5)
    assert bool((observed==0).all())
