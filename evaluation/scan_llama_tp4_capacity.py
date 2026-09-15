"""Fresh torchrun per capacity trial; CUDA OOM is recorded, other failures stop the scan."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evaluation.v96kl_common import read_json,write_json,sha256

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['smoke','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sources={n:sha256(Path(n)) for n in ['basisserve/core/llama_tp4_k_offload.py','evaluation/llama_tp4_capacity_trial.py','basisserve/kernels/mapped_host_paged_attention.py','basisserve/kernels/csrc/mapped_host_paged_attention.cu','basisserve/kernels/csrc/conditional_router_page32.cu',__file__]}
    if a.stage=='run':assert read_json(a.output/'smoke.json')['source_sha256']==sources
    trials=[]
    def gpu_processes(own_pids=None):
        if own_pids is not None:
            processes=subprocess.run(['ps','-eo','pid=,ppid='],capture_output=True,text=True,check=True)
            parents={int(v[0]):int(v[1]) for line in processes.stdout.splitlines() if len(v:=line.split())==2}
            while True:
                children={pid for pid,parent in parents.items() if parent in own_pids}
                if children<=own_pids:break
                own_pids.update(children)
        probe=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        rows=[line.strip() for line in probe.stdout.splitlines() if line.strip()]
        return [line for line in rows if own_pids is None or int(line.split(',')[1]) not in own_pids]
    def trial(mode,length,batch):
        folder=a.output/f'{mode}_{length}_b{batch}';folder.mkdir(exist_ok=True)
        saved=folder/'result.json'
        if saved.exists():
            record=read_json(saved)
            assert record['source_sha256']==sources
            assert record['status'] in ['pass','gpu_oom']
            trials.append(record)
            return record['status']=='pass'
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4','-m','evaluation.llama_tp4_capacity_trial',
            '--root',str(a.root),'--output',str(folder),'--mode',mode,'--batch',str(batch),'--length',str(length)]
        if a.stage=='smoke':cmd+=['--smoke']
        external=[dict(time=time.time(),processes=gpu_processes())]
        print('EXTERNAL GPU OCCUPANCY (uuid, pid, MiB)',external[0]['processes'],flush=True)
        (a.output/'waiting.json').write_text(json.dumps(dict(status='running_with_existing_occupancy',trial=folder.name,time=time.time()))+'\n')
        print('COMMAND',' '.join(cmd),flush=True)
        with (folder/'run.log').open('w') as log:
            result=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            own_pids={result.pid}
            while result.poll() is None:
                observed=gpu_processes(own_pids)
                if observed!=external[-1]['processes']:
                    external.append(dict(time=time.time(),processes=observed))
                time.sleep(1)
        external.append(dict(time=time.time(),processes=gpu_processes(own_pids)))
        paths=[folder/f'rank{r}.json' for r in range(4)]
        if result.returncode==0:
            assert all(p.exists() for p in paths)
            reports=[read_json(p) for p in paths];assert all(d['status']=='complete' for d in reports)
            record=dict(mode=mode,length=length,batch=batch,status='pass',ranks=reports)
        else:
            text=(folder/'run.log').read_text(errors='replace')
            phases=[read_json(p) for p in sorted(folder.glob('phase*.json'))]
            oom='CUDA out of memory' in text or 'torch.OutOfMemoryError' in text
            record=dict(mode=mode,length=length,batch=batch,status='gpu_oom' if oom else 'error',returncode=result.returncode,phases=phases,log=str(folder/'run.log'))
        record['external_gpu_occupancy']=external
        record['capacity_condition']='Available GPU memory with existing external occupancy; not empty-card capacity'
        record['source_sha256']=sources
        if record['status']!='error':write_json(folder/'result.json',record)
        trials.append(record)
        print('RESULT',mode,length,batch,record['status'],flush=True)
        assert record['status']!='error',record
        return record['status']=='pass'
    if a.stage=='smoke':
        for mode in ['dense','offload']:assert trial(mode,4096,1)
        write_json(a.output/'smoke.json',dict(status='complete',source_sha256=sources,trials=trials));return
    boundaries=[]
    for length in [65536,131072]:
        for mode in ['dense','offload']:
            low=0;high=1
            while trial(mode,length,high):
                low=high;high*=2
                if high>64:break
            if high>64:
                boundaries.append(dict(mode=mode,length=length,max_pass=low,first_oom=None));continue
            while high-low>1:
                middle=(low+high)//2
                if trial(mode,length,middle):low=middle
                else:high=middle
            boundaries.append(dict(mode=mode,length=length,max_pass=low,first_oom=high))
            (a.output/'progress.json').write_text(json.dumps(dict(boundaries=boundaries),indent=2)+'\n')
    write_json(a.output/'summary.json',dict(status='complete',source_sha256=sources,boundaries=boundaries,trials=trials,
        protocol='Llama3.1-8B-Instruct TP4 BF16 Dense V128 original Wo; layerwise dense prefill QKV chunks2048, MLP and final norm chunks1024; offload B16R16 hard2048 sink32 recent64; 4 decode steps; fresh processes per trial; existing external GPU occupancy allowed and sampled every second'))
if __name__=='__main__':main()
