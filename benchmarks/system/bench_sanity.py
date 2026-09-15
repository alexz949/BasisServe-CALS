"""TP4 default NCCL collectives and NUMA-local pinned H2D sanity checks."""
import argparse
import json
import os
from pathlib import Path
import statistics
import torch
import torch.distributed as dist
from benchmarks.system.common import metadata,save
from benchmarks.system.numa_memory import bind_host_allocations,audit_host_tensor
from evaluation.benchmark_tp4_interconnect import _critical_cuda_timings,_summary


def timing(fn,device):
    raw=_critical_cuda_timings(fn,warmup=100,iterations=500,device=device)
    return dict(**_summary(raw),stddev_ms=statistics.pstdev(raw),raw_ms=raw,warmup=100,iterations=500,
                aggregation='per-iteration maximum across TP ranks')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    p.add_argument('--section',choices=['all','h2d'],default='all')
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK'])
    hardware=json.loads((a.output/'hardware.json').read_text());node=hardware['gpu_numa'][rank]['numa_node']
    library=bind_host_allocations(node)
    torch.cuda.set_device(rank);torch.set_num_threads(2)
    dist.init_process_group('nccl');assert dist.get_world_size()==4
    assert 'NCCL_ALGO' not in os.environ and 'NCCL_PROTO' not in os.environ
    device=torch.device('cuda',rank);records=[]
    for size in ([2**power for power in range(8,25)] if a.section=='all' else []):
        source=torch.full((size//2,),rank,dtype=torch.bfloat16,device=device)
        dist.all_reduce(source);assert torch.all(source==6);source.zero_()
        t=timing(lambda:dist.all_reduce(source),device)
        bw=size/(t['p50_ms']*1e6)
        records.append(dict(operation='all_reduce',input_bytes_per_rank=size,output_bytes_per_rank=size,
            algorithm_gbps=bw,bus_gbps=bw*1.5,bandwidth_definition='NCCL-test analytical bus factor 2*(N-1)/N, not measured PCIe counters',timing=t))
        source.fill_(rank);out=torch.empty(size//2*4,dtype=torch.bfloat16,device=device)
        dist.all_gather_into_tensor(out,source)
        assert all(torch.all(out.view(4,-1)[i]==i) for i in range(4))
        t=timing(lambda:dist.all_gather_into_tensor(out,source),device)
        bw=size*4/(t['p50_ms']*1e6)
        records.append(dict(operation='all_gather',input_bytes_per_rank=size,output_bytes_per_rank=size*4,
            algorithm_gbps=bw,bus_gbps=bw*.75,bandwidth_definition='NCCL-test output-size algorithm bandwidth, bus factor (N-1)/N',timing=t))
        del source,out
        if rank==0:print(dict(stage='nccl',input_bytes=size,status='measured'),flush=True)
    if rank==0 and a.section=='all':save(a.output/'nccl_raw.json',dict(metadata=metadata(),dtype='bfloat16',tp=4,records=records))
    copies=[]
    # Measure one physical GPU at a time; other ranks only participate in barriers.
    for active in range(4):
        for mib in [1,4,16,64,256,1024]:
            n=mib*1024*1024
            if rank==active:
                host=torch.empty(n,dtype=torch.uint8,pin_memory=True);host.fill_(23)
                audit=audit_host_tensor(host,node,library)
                target=torch.empty_like(host,device=device);target.copy_(host,non_blocking=True)
                torch.cuda.synchronize();assert torch.all(target==23)
            t=timing(lambda:target.copy_(host,non_blocking=True) if rank==active else None,device)
            if rank==active:
                copies.append(dict(gpu=rank,numa_node=node,bytes=n,concurrency='single active GPU',timing=t,
                    one_direction_h2d_gbps=n/(t['p50_ms']*1e6),host_audit=audit))
                del host,target
                print(dict(stage='h2d',gpu=rank,mib=mib,p50_ms=t['p50_ms']),flush=True)
    save(a.output/f'h2d_rank{rank}.json',dict(metadata=metadata(),records=copies))
    dist.barrier();torch.cuda.empty_cache();dist.destroy_process_group()

if __name__=='__main__':main()
