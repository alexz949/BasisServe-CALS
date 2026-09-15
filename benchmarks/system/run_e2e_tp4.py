"""Run the prescribed full-model grid in fresh TP4 processes, recording OOMs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from benchmarks.system.common import save


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    p.add_argument('--lengths',type=int,nargs='+',default=[16384,32768,65536]);a=p.parse_args()
    assert len(a.lengths)==len(set(a.lengths)) and set(a.lengths)<={16384,32768,65536,131072}
    gate=json.loads((a.output/'e2e_smoke_validation.json').read_text())
    assert gate['status']=='complete' and {r['batch'] for r in gate['records']}=={1,4}
    names=['basisserve/core/llama_tp4_c1_system.py','basisserve/core/llama_tp4_k_offload.py',
           'benchmarks/system/bench_e2e_tp4.py','benchmarks/system/collective_audit.py','basisserve/kernels/mapped_host_paged_attention.py',
           'basisserve/kernels/csrc/mapped_host_paged_attention.cu','basisserve/kernels/csrc/conditional_router_page32.cu']
    sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
    records=[]
    for length in a.lengths:
        for batch in [1,4,8,16]:
            for mode in ['dense','c1','sparse_local','offload']:
                folder=a.output/'e2e'/f'{mode}_t{length}_b{batch}';folder.mkdir(parents=True,exist_ok=True)
                result_path=folder/'trial.json'
                if result_path.exists():
                    record=json.loads(result_path.read_text());assert record['source_sha256']==sources
                    assert record['status'] in ['complete','gpu_oom']
                    records.append(record);continue
                cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
                    '-m','benchmarks.system.bench_e2e_tp4','--root',str(a.root),'--output',str(a.output),
                    '--mode',mode,'--length',str(length),'--batch',str(batch)]
                print('COMMAND',' '.join(cmd),flush=True)
                with (folder/'run.log').open('w') as stream:result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
                assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in sources.items()),'Measured source changed during trial'
                record=dict(mode=mode,length=length,batch=batch,source_sha256=sources,command=cmd,returncode=result.returncode)
                if result.returncode==0:
                    reports=[json.loads((folder/f'rank{rank}.json').read_text()) for rank in range(4)]
                    assert all(r['status']=='complete' and r['generated_tokens']==256 for r in reports)
                    record.update(status='complete',ranks=reports)
                else:
                    text=(folder/'run.log').read_text(errors='replace')
                    oom='CUDA out of memory' in text or 'torch.OutOfMemoryError' in text
                    record.update(status='gpu_oom' if oom else 'error',
                        phases=[json.loads(path.read_text()) for path in sorted(folder.glob('phase*.json'))],log=str(folder/'run.log'))
                if record['status']!='error':save(result_path,record)
                records.append(record)
                print(dict(mode=mode,length=length,batch=batch,status=record['status']),flush=True)
                assert record['status']!='error',record
                (a.output/'e2e_progress.json').write_text(json.dumps([dict(mode=r['mode'],length=r['length'],batch=r['batch'],status=r['status']) for r in records],indent=2)+'\n')
    tag='_'.join(str(x) for x in a.lengths)
    save(a.output/f'e2e_grid_t{tag}.json',dict(status='complete',records=records,source_sha256=sources,
        lengths=a.lengths,grid=f'{len(a.lengths)} context lengths x4 batch sizes x4 systems; OOMs retained, other failures stop the scan'))


if __name__=='__main__':main()
