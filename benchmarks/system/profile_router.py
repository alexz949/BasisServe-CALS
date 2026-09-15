"""Profile one validated real 64K Basis scan; no full-model NCU replay."""
import argparse
from pathlib import Path
import torch
from benchmarks.system.bench_router import load_basis
from benchmarks.system.common import metadata,save


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'));a=p.parse_args()
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    router=load_basis(a.root,a.output,65536,1)
    audit=router.validate()
    for _ in range(100):router.full()
    torch.cuda.synchronize();router.preprocess();torch.cuda.synchronize()
    with torch.cuda.nvtx.range('basis_router_profile'):router.scan()
    torch.cuda.synchronize()
    save(a.output/'profile_router_input.json',dict(metadata=metadata(),audit=audit,
        length=65536,batch=1,layer=3,budget=2048,base_rank=16,residual_rank=16,warmup=100,
        interpretation='Single-layer warmed cache; NCU uses cache-control none and clock-control none; DRAM and cache traffic are separate metrics'))


if __name__=='__main__':main()
