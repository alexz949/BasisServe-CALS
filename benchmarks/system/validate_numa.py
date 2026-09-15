"""Validate NUMA residency independently of CUDA pinning on all four GPUs."""
import argparse
import json
import os
from pathlib import Path
import torch
from benchmarks.system.common import metadata,save
from benchmarks.system.numa_memory import bind_host_allocations,audit_host_tensor
from basisserve.kernels.mapped_host_paged_attention import mapped_host_bf16_empty


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    hardware=json.loads((a.output/'hardware.json').read_text())
    node=next(g['numa_node'] for g in hardware['gpu_numa'] if g['gpu']==rank)
    assert node>=0
    library=bind_host_allocations(node)
    staged=torch.empty(32*1024*1024,dtype=torch.bfloat16,pin_memory=True);staged.fill_(1)
    first=audit_host_tensor(staged,node,library)
    mapped=mapped_host_bf16_empty(batch=1,kv_heads=2,capacity=131072);mapped.fill_(1)
    second=audit_host_tensor(mapped,node,library)
    save(a.output/f'numa_validation_rank{rank}.json',dict(metadata=metadata(),gpu=rank,node=node,staged=first,mapped=second))
    print(dict(rank=rank,node=node,staged_pinned=first['pinned'],mapped_pinned=second['pinned'],status='passed'),flush=True)

if __name__=='__main__':main()
