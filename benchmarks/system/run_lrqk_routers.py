"""Independent processes prevent one LRQK setup OOM from killing other cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from benchmarks.system.common import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'));a=p.parse_args()
    paths=['benchmarks/system/lrqk_router.py','benchmarks/system/bench_router.py',
           'evaluation/official_lrqk_state.py','external/LRQK/lrqk_attention.py']
    sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths}
    smoke=json.loads((a.output/'router_lrqk_smoke_t4096_b1.json').read_text())
    assert smoke['records'][0]['audit']['native_exact_key_cache_verified']
    def worker(rank):
        length=[16384,32768,65536,131072][rank];records=[]
        env=dict(os.environ,LOCAL_RANK=str(rank))
        for batch in [1,4,8]:
            status=a.output/f'router_lrqk_status_t{length}_b{batch}.json'
            if status.exists():
                record=json.loads(status.read_text());assert record['source_sha256']==sources
                records.append(record);continue
            log=a.output/f'router_lrqk_t{length}_b{batch}.log'
            cmd=[sys.executable,'-m','benchmarks.system.bench_router','--method','lrqk','--root',str(a.root),
                 '--output',str(a.output),'--length',str(length),'--batch',str(batch)]
            print('COMMAND',' '.join(cmd),flush=True)
            with log.open('w') as stream:result=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
            assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in sources.items())
            report=a.output/f'router_lrqk_t{length}_b{batch}.json'
            if result.returncode==0:
                assert json.loads(report.read_text())['records'][0]['audit']['reference_ids_equal']
                outcome='complete'
            else:
                text=log.read_text(errors='replace')
                outcome='gpu_oom' if 'CUDA out of memory' in text or 'torch.OutOfMemoryError' in text else 'error'
            record=dict(length=length,batch=batch,status=outcome,returncode=result.returncode,
                command=cmd,source_sha256=sources,log=str(log),report=str(report) if result.returncode==0 else None)
            save(status,record);records.append(record)
            print(dict(length=length,batch=batch,status=outcome),flush=True)
        return records
    with ThreadPoolExecutor(max_workers=4) as executor:groups=list(executor.map(worker,range(4)))
    records=[r for group in groups for r in group]
    save(a.output/'lrqk_router_grid.json',dict(records=records,
        status='complete' if all(r['status']=='complete' for r in records) else 'completed_with_failed_cases'))


if __name__=='__main__':main()
