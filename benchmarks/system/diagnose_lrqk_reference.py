"""Inspect a native BF16 versus independent FP32 top-k disagreement."""
import argparse
from pathlib import Path
import torch
from benchmarks.system.bench_router import load_other
from benchmarks.system.common import metadata,save
from benchmarks.system.numa_memory import bind_host_allocations,slurm_gpu_numa
import json
import os


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    args=parser.parse_args()
    hardware=json.loads((args.output/'hardware.json').read_text())
    rank=int(os.environ.get('LOCAL_RANK',0))
    bind_host_allocations(slurm_gpu_numa(hardware,rank)['numa_node'])
    torch.cuda.set_device(rank);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    router=load_other('lrqk',args.output,32768,8)
    router.full()
    assert torch.equal(router.ids.sort(-1).values,router.native_ids.sort(-1).values)
    native=(router.codes@router.query_code.transpose(-1,-2))[...,0]
    fp32=(router.codes.float()@router.query_code.float().transpose(-1,-2))[...,0]
    rounded=fp32.to(native.dtype)
    nidx=native.topk(2048,dim=-1).indices
    ridx=rounded.topk(2048,dim=-1).indices
    disagree=(nidx.sort(-1).values!=ridx.sort(-1).values).any(-1)
    rows=[]
    for batch,head in disagree.nonzero().cpu().tolist():
        ni=nidx[batch,head];ri=ridx[batch,head]
        native_only=ni[~torch.isin(ni,ri)];reference_only=ri[~torch.isin(ri,ni)]
        ids=torch.cat((native_only,reference_only))
        boundary=rounded[batch,head].topk(2048).values[-1]
        rows.append(dict(batch=batch,query_head=head,native_only=native_only.cpu().tolist(),
            reference_only=reference_only.cpu().tolist(),native_score=native[batch,head,ids].float().cpu().tolist(),
            fp32_score=fp32[batch,head,ids].cpu().tolist(),rounded_score=rounded[batch,head,ids].float().cpu().tolist(),
            reference_cutoff=float(boundary),native_cutoff=float(native[batch,head].topk(2048).values[-1]),
            all_difference_ids_at_reference_cutoff=bool((rounded[batch,head,ids]==boundary).all())))
    delta=(native.float()-fp32).abs()
    data=dict(metadata=metadata(),length=32768,batch=8,rank=32,budget=2048,
        original_native_ids_equal=True,score_dtype=str(native.dtype),fp32_reference_dtype=str(fp32.dtype),
        native_vs_rounded_different_scores=int((native!=rounded).sum()),
        max_native_vs_fp32_abs=float(delta.max()),disagreeing_query_heads=len(rows),details=rows,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        interpretation='Diagnostic only; does not relax the failed benchmark correctness gate')
    save(args.output/f'lrqk_reference_check_t32768_b8_rank{rank}.json',data)
    print({key:value for key,value in data.items() if key not in ['metadata','details']},flush=True)


if __name__=='__main__':main()
