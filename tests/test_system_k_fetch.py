"""Correctness gates for split V and allocation-free staged page fetch."""
import json
from pathlib import Path
import pytest
import torch
from basisserve.core.c1_k_offload import PinnedCPUExactKeyPageStore,PreparedExactKeyPageFetch
from basisserve.kernels.mapped_host_paged_attention import (
    mapped_host_bf16_empty,mapped_host_paged_attention,gpu_paged_attention)
from benchmarks.system.numa_memory import bind_host_allocations,audit_host_tensor

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU required')


@pytest.mark.parametrize('page',[1,32])
@torch.inference_mode()
def test_split_value_attention(page):
    torch.manual_seed(777)
    b,h,n=2,2,130
    key=torch.randn(b,h,n,128,device='cuda',dtype=torch.bfloat16)
    value=torch.randn(b,h,n,96,device='cuda',dtype=torch.bfloat16)
    query=torch.randn(b,h*4,1,128,device='cuda',dtype=torch.bfloat16)
    prefix=value[...,:16].contiguous();tail=value[...,16:].contiguous()
    ids=torch.tensor([0,2,4,-1],device='cuda').expand(b,h,-1).contiguous()
    positions=(ids[...,None]*page+torch.arange(page,device='cuda')).flatten(-2)
    valid=(positions>=0)&(positions<n)
    safe=positions.clamp(0,n-1)
    selected_k=key.gather(2,safe[...,None].expand(*safe.shape,128)).float()
    selected_v=value.gather(2,safe[...,None].expand(*safe.shape,96)).float()
    logits=(query.float().reshape(b,h,4,128)@selected_k.transpose(-1,-2))*128**-.5
    expected=logits.masked_fill(~valid[:,:,None],-torch.inf).softmax(-1)@selected_v
    host=mapped_host_bf16_empty(batch=b,kv_heads=h,capacity=n);host.copy_(key.cpu())
    for function,source in [(gpu_paged_attention,key),(mapped_host_paged_attention,host)]:
        for splits in [1,3]:
            observed=function(source,query,tail,ids,sequence_length=n,page_size=page,splits=splits,value_prefix=prefix)
            contiguous=function(source,query,value,ids,sequence_length=n,page_size=page,splits=splits)
            torch.testing.assert_close(observed,contiguous,rtol=0,atol=0)
            torch.testing.assert_close(observed.float().reshape(b,h,4,96),expected,rtol=.015,atol=.004)


@torch.inference_mode()
def test_prepared_staged_fetch():
    hardware=json.loads(Path('results/system_benchmarks/l40s/hardware.json').read_text())
    library=bind_host_allocations(hardware['gpu_numa'][0]['numa_node'])
    host=mapped_host_bf16_empty(batch=2,kv_heads=2,capacity=128)
    host.copy_(torch.randn(host.shape,dtype=torch.bfloat16))
    store=PinnedCPUExactKeyPageStore(host)
    fetch=PreparedExactKeyPageFetch(store,pages_per_head=3,page_size=32,device=torch.device('cuda',0))
    audit_host_tensor(host,0,library)
    audit_host_tensor(fetch.staging,0,library)
    pointers=(fetch.destination.data_ptr(),fetch.staging.data_ptr(),fetch.host_indices.data_ptr())
    for values in [[0,3,1],[2,1,3]]:
        ids=torch.tensor(values,device='cuda').expand(2,2,-1).contiguous()
        result=fetch(ids);torch.cuda.synchronize()
        positions=(torch.tensor(values)[:,None]*32+torch.arange(32)).flatten()
        assert torch.equal(result.cpu(),host[:,:,positions])
        assert pointers==(fetch.destination.data_ptr(),fetch.staging.data_ptr(),fetch.host_indices.data_ptr())
