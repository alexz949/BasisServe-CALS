"""Repeat setup OOM cases after releasing redundant input assembly buffers."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from benchmarks.system.common import save


def main():
    root=Path('results/system_benchmarks/l40s')
    names=['benchmarks/system/bench_router.py','benchmarks/system/lrqk_router.py',
        'evaluation/official_lrqk_state.py','external/LRQK/lrqk_attention.py']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in names}
    def worker(case):
        rank,length,batch=case
        cmd=[sys.executable,'-m','benchmarks.system.bench_router','--method','lrqk',
            '--root','/home/zhangal/BasisServe-CALS-runs/llama31_8b_64k','--length',str(length),'--batch',str(batch)]
        log=root/f'router_lrqk_capacity_t{length}_b{batch}.log';assert not log.exists()
        print('COMMAND',' '.join(cmd),flush=True)
        with log.open('w') as stream:
            result=subprocess.run(cmd,env=dict(os.environ,LOCAL_RANK=str(rank)),stdout=stream,stderr=subprocess.STDOUT)
        text=log.read_text(errors='replace')
        outcome='complete' if result.returncode==0 else 'gpu_oom' if 'CUDA out of memory' in text or 'torch.OutOfMemoryError' in text else 'error'
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        record=dict(length=length,batch=batch,local_rank=rank,status=outcome,returncode=result.returncode,
            command=cmd,log=str(log),source_sha256=hashes,
            original_status_file=str(root/f'router_lrqk_status_t{length}_b{batch}.json'))
        print(record,flush=True)
        return record
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(worker,[(0,131072,8),(1,65536,8),(2,131072,4)]))
    save(root/'lrqk_capacity_review.json',dict(records=records,
        change='Release redundant per-sample input tensors after concatenation; timed native algorithm unchanged'))


if __name__=='__main__':main()
