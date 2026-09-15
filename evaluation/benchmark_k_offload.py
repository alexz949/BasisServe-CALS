"""Matched K128 sparse-attention benchmark: resident CUDA/Triton versus mapped host."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import torch
import torch.distributed as dist
from basisserve.kernels.mapped_host_paged_attention import (
    mapped_host_bf16_empty,append_mapped_host_key,mapped_host_device_pointer,
    mapped_host_paged_attention,gpu_paged_attention)
from basisserve.kernels.split_indexed_attention import split_indexed_attention
from flash_attn import flash_attn_with_kvcache
from evaluation.v96kl_common import configure,sha256,write_json


def timing(fn, flush, repeats):
    for _ in range(5):fn()
    torch.cuda.synchronize()
    result={}
    for mode in ('warm','cold'):
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        graph=torch.cuda.CUDAGraph()
        calls=32 if mode=='warm' else 1
        with torch.cuda.graph(graph):
            for _ in range(calls):fn()
        for _ in range(5):graph.replay()
        torch.cuda.synchronize()
        samples=[]
        for _ in range(repeats):
            if dist.is_initialized():dist.barrier()
            # Queue eviction ahead of timing events; its work is excluded.
            # Warm timing amortizes graph-launch overhead across 32 calls.
            if mode=='cold':flush.add_(1)
            start.record();graph.replay();end.record();end.synchronize()
            samples.append(start.elapsed_time(end)*1000/calls)
        result[mode]={'median_us':statistics.median(samples),'p95_us':sorted(samples)[int(.95*(len(samples)-1))],
                      'min_us':min(samples),'samples':len(samples),'samples_us':samples}
    return result


@torch.inference_mode()
def case(length,vdim,budget,repeats,kv_heads,batch):
    query_heads=4*kv_heads
    torch.manual_seed(42+length+vdim+budget)
    k=torch.randn(batch,kv_heads,length,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn(batch,kv_heads,length,vdim,device='cuda',dtype=torch.bfloat16)
    q=torch.randn(batch,query_heads,1,128,device='cuda',dtype=torch.bfloat16)
    pages=torch.stack([torch.randperm(length//32,device='cuda')[:budget//32].sort().values for _ in range(batch*kv_heads)]).reshape(batch,kv_heads,-1)
    ids=(pages[...,None]*32+torch.arange(32,device='cuda')).flatten(-2)
    head_ids=ids.repeat_interleave(4,dim=1)
    host=mapped_host_bf16_empty(batch=batch,kv_heads=kv_heads,capacity=length)
    append_mapped_host_key(host,k,start=0)
    ptr=mapped_host_device_pointer(host)
    sk=k.gather(2,ids[...,None].expand(batch,kv_heads,budget,128)).float()
    sv=v.gather(2,ids[...,None].expand(batch,kv_heads,budget,vdim)).float()
    reference=(((q.float().reshape(batch,kv_heads,4,128)@sk.transpose(-1,-2))*128**-.5).softmax(-1)@sv).reshape(batch,query_heads,1,vdim)
    # More than twice the L40S L2; actual property used when exposed by PyTorch.
    l2=getattr(torch.cuda.get_device_properties(0),'L2_cache_size',96*1024**2)
    flush=torch.zeros(max(2*l2,192*1024**2)//4,device='cuda')
    dense_reference=(((q.float().reshape(batch,kv_heads,4,128)@k.float().transpose(-1,-2))*128**-.5).softmax(-1)@v.float()).reshape(batch,query_heads,1,vdim)
    flash_k=k.transpose(1,2).contiguous()
    flash_v=torch.nn.functional.pad(v,(0,128-vdim)).transpose(1,2).contiguous()
    flash_q=q.transpose(1,2).contiguous()
    records=[]
    for backend in ('resident_cuda','mapped_host','resident_triton','dense_flash'):
        for splits in ((16,32,64) if backend!='resident_triton' else (None,)):
            workspace=torch.empty(batch*query_heads,128,vdim+2,device='cuda',dtype=torch.float32)
            output=torch.empty(batch,query_heads,1,vdim,device='cuda',dtype=torch.bfloat16)
            if backend=='dense_flash':
                def fn():return flash_attn_with_kvcache(flash_q,flash_k,flash_v,softmax_scale=128**-.5,causal=False,num_splits=splits).transpose(1,2)[...,:vdim]
            elif backend=='mapped_host':
                def fn():return mapped_host_paged_attention(host,q,v,pages,sequence_length=length,splits=splits,
                    host_key_device_pointer=ptr,workspace=workspace,output=output)
            elif backend=='resident_cuda':
                def fn():return gpu_paged_attention(k,q,v,pages,sequence_length=length,splits=splits,workspace=workspace,output=output)
            else:
                def fn():return split_indexed_attention(q,k,v,head_ids,scale=128**-.5)
            actual=fn();torch.cuda.synchronize()
            expected=dense_reference if backend=='dense_flash' else reference
            torch.testing.assert_close(actual.float(),expected,rtol=.015,atol=.004)
            rec=dict(backend=backend,splits=splits,max_abs_error=float((actual.float()-expected).abs().max()),
                timing=timing(fn,flush,repeats))
            records.append(rec)
            print('CASE',length,vdim,budget,backend,splits,rec['timing']['cold']['median_us'],flush=True)
    return dict(sequence_length=length,value_dim=vdim,budget=budget,batch=batch,kv_heads=kv_heads,query_heads=query_heads,page_size=32,
        memory=dict(gpu_key_cache_bytes=k.numel()*k.element_size(),mapped_host_key_bytes=host.numel()*host.element_size(),
                    mapped_host_gpu_key_cache_bytes=0,gpu_value_cache_bytes=v.numel()*v.element_size(),
                    selected_key_bytes=batch*kv_heads*budget*128*2,dense_flash_padded_value_bytes=flash_v.numel()*flash_v.element_size(),scope='logical tensor payloads, one layer; excludes allocator, workspace and driver mappings'),
        cache_flush_bytes=flush.numel()*flush.element_size(),measurements=records)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['smoke','run']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kv-heads',type=int,choices=[2,8],required=True)
    p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--sequence-lengths',type=int,nargs='+',choices=[65536,131072],required=True)
    a=p.parse_args();configure();assert a.batch_size>0
    rank=int(os.environ.get('LOCAL_RANK','0'))
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        torch.cuda.set_device(rank);dist.init_process_group('gloo')
        allowed=sorted(os.sched_getaffinity(0));assigned=allowed[rank::dist.get_world_size()]
        if assigned:os.sched_setaffinity(0,assigned)
    else:assert torch.cuda.device_count()==1
    output=a.output/f'rank{rank}' if dist.is_initialized() else a.output
    root=Path(__file__).resolve().parents[1]
    source={str(p.relative_to(root)):sha256(p) for p in [Path(__file__).resolve(),root/'basisserve/kernels/mapped_host_paged_attention.py',
        root/'basisserve/kernels/csrc/mapped_host_paged_attention.cu',root/'basisserve/kernels/split_indexed_attention.py']}
    if a.stage=='run':
        smoke=json.loads((output/'smoke.json').read_text());assert smoke['status']=='complete' and smoke['source_sha256']==source
    grid=[(a.sequence_lengths[0],96,1024)] if a.stage=='smoke' else [(n,v,b) for n in a.sequence_lengths for v in (96,128) for b in (1024,2048)]
    records=[]
    for n,v,b in grid:
        records.append(case(n,v,b,10 if a.stage=='smoke' else 100,a.kv_heads,a.batch_size));gc.collect();torch.cuda.empty_cache()
    metadata=dict(status='complete',source_sha256=source,device=torch.cuda.get_device_name(rank),rank=rank,world_size=dist.get_world_size() if dist.is_initialized() else 1,torch=torch.__version__,
        cuda=torch.version.cuda,python=sys.executable,command=' '.join(sys.argv),
        dense_baseline='FlashAttention KV-cache decode, full GPU K/V; V96 zero-padded to128 before timing, K/V layout conversion excluded',
        timing='CUDA Graph replay with external timing events; warm 32 calls/replay; cold one call/replay; TP ranks use a CPU barrier before each replay, barrier excluded after untimed L2 flush; includes replay overhead; allocations/compile/D2H prefill/routing excluded',
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),cpu_affinity=sorted(os.sched_getaffinity(0)),
        cases=records)
    write_json(output/('smoke.json' if a.stage=='smoke' else 'summary.json'),metadata)
    if dist.is_initialized():dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
