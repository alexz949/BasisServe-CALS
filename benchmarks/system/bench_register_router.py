"""Reuse matched decode harness with current production versus register router."""
import argparse
from pathlib import Path
import sys
import benchmarks.system.bench_local_kernels as harness
from benchmarks.system.register_router import compile_register


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['trace','baseline','candidate'],default='trace')
    p.add_argument('--rank',type=int,choices=[8,16],default=16)
    p.add_argument('--warps',type=int,choices=[4,8],default=4)
    p.add_argument('--length',type=int,default=65536)
    p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    root=Path('results/system_benchmarks/register_router')
    harness.RESULT_ROOT=root
    harness.TRACE_SLOT_OPERATORS=False
    current=compile_register(root/'source',a.rank,0)
    candidate=compile_register(root/'source',a.rank,a.warps)
    harness.compile_router=lambda root,rank,reference=False: current if reference or a.mode=='baseline' else candidate
    # Both paths use the already-deployed vector fetch; only the router changes.
    original_slots=harness.compile_slots
    harness.compile_slots=lambda build_root,vector:original_slots(root/'source',True)
    mode='trace' if a.mode=='trace' else 'optimized'
    # Separate matched baseline/candidate directories without changing any serving parameter.
    if a.mode!='trace':
        harness.RESULT_ROOT=root/a.mode
    sys.argv=['bench_register_router','--mode',mode,'--rank',str(a.rank),'--warps',str(a.warps),
        '--length',str(a.length),'--repeat',str(a.repeat)]
    if a.smoke:sys.argv.append('--smoke')
    harness.main()

if __name__=='__main__':main()
