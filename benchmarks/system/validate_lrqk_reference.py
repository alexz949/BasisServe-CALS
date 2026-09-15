"""Validate native selection and FP32 score bounds on smoke and rounding case."""
import json
import os
from pathlib import Path
import torch
from benchmarks.system.bench_router import load_other
from benchmarks.system.common import metadata,save
from benchmarks.system.numa_memory import bind_host_allocations,slurm_gpu_numa


@torch.inference_mode()
def main():
    root=Path('results/system_benchmarks/l40s')
    rank=int(os.environ['LOCAL_RANK']);assert rank in (0,1)
    length,batch=(4096,1) if rank==0 else (32768,8)
    hardware=json.loads((root/'hardware.json').read_text())
    bind_host_allocations(slurm_gpu_numa(hardware,rank)['numa_node'])
    torch.cuda.set_device(rank);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    router=load_other('lrqk',root,length,batch,smoke=rank==0)
    audit=router.validate()
    save(root/f'lrqk_validation_t{length}_b{batch}.json',dict(metadata=metadata(),length=length,batch=batch,audit=audit))
    print(dict(length=length,batch=batch,native_ids_equal=audit['reference_ids_equal'],
        fp32_score_bound=audit['independent_fp32_score_error_bound_passed'],
        fp32_rounded_ids_equal=audit['fp32_rounded_topk_ids_equal']),flush=True)


if __name__=='__main__':main()
