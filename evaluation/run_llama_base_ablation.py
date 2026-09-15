"""Four independent single-L40S workers and paired RULER aggregation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from evaluation.v96kl_common import read_json,write_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['evaluate','summarize'])
    p.add_argument('--root',type=Path,required=True);args=p.parse_args();r=args.root;o=r/'ruler_b0b4_b1024'
    def command(stage,rank,shard=0):
        return [sys.executable,'-u','-m','evaluation.eval_llama_base_ablation',stage,'--root',str(r),
            '--base-rank',str(rank),'--arms','ours','--arm','ours','--shard-index',str(shard)]
    if args.stage=='evaluate':
        devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==4 and all(x.isdigit() for x in devices)
        free={row.split(',')[0].strip():int(row.split(',')[1]) for row in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
        devices.sort(key=lambda d:free[d])
        # Pure R16 needs less VRAM; reserve the least-free GPU for that arm.
        assignments=[[(0,0),(0,1)],[(4,0),(4,1)],[(4,2),(0,2)],[(4,3),(0,3)]]
        write_json(o/'worker_assignment.json',dict(devices=devices,free_mib=free,assignments=assignments))
        def worker(item):
            device,tasks=item;env=dict(os.environ,CUDA_VISIBLE_DEVICES=device)
            with (o/'logs'/f'worker_gpu{device}.log').open('a') as log:
                for rank,shard in tasks:
                    result=subprocess.run(command('evaluate',rank,shard),env=env,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:return result.returncode
            return 0
        with ThreadPoolExecutor(max_workers=4) as executor:codes=list(executor.map(worker,zip(devices,assignments)))
        assert codes==[0]*4,codes
        return
    for rank in (0,4):subprocess.run(command('summarize',rank),check=True)
    summaries={f'b{rank}r16':read_json(o/f'b{rank}r16/summary.json') for rank in (0,4)}
    assert all(d['status']=='complete' and d['verified_predictions']==88 for d in summaries.values())
    paired=[]
    for index in range(88):
        records={arm:read_json(o/arm/'ours/evaluate'/f'sample_{index:03d}.json') for arm in summaries}
        assert records['b0r16']['sample']==records['b4r16']['sample']
        paired.append(dict(index=index,task=records['b0r16']['sample']['task'],
            **{arm:d['result']['score'] for arm,d in records.items()}))
    tasks={task:{arm:100*sum(row[arm] for row in paired if row['task']==task)/8 for arm in summaries}
        for task in dict.fromkeys(row['task'] for row in paired)}
    means={arm:d['means']['ours'] for arm,d in summaries.items()}
    write_json(o/'summary.json',dict(status='complete',verified_predictions=176,means=means,tasks=tasks,paired=paired,
        b4_wins=sum(row['b4r16']>row['b0r16'] for row in paired),b4_losses=sum(row['b4r16']<row['b0r16'] for row in paired),
        protocols={arm:d['protocol'] for arm,d in summaries.items()}))
    print(means,flush=True)


if __name__=='__main__':main()
