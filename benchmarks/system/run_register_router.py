"""Sequential real-query checks or matched ABBA decode on one allocated GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['trace','decode'],required=True)
    p.add_argument('--rank',type=int,default=16);p.add_argument('--warps',type=int,default=8)
    a=p.parse_args();root=Path('results/system_benchmarks/register_router')
    if a.phase=='trace':
        assert json.loads((root/'validation.json').read_text())['status']=='complete'
        cases=[('candidate',16,8,0,True),('trace',16,4,0,False),('trace',16,8,0,False),('trace',8,8,0,False)]
    else:cases=[(m,a.rank,a.warps,r,False) for m,r in [('baseline',0),('candidate',0),('candidate',1),('baseline',1)]]
    summaries=[]
    for mode,rank,warps,repeat,smoke in cases:
        args=[sys.executable,'-m','benchmarks.system.bench_register_router','--mode',mode,'--rank',str(rank),'--warps',str(warps),'--repeat',str(repeat)]
        if smoke:args.append('--smoke')
        tag=f'{mode}_b{rank}_w{warps}_r{repeat}'+('_smoke' if smoke else '')
        print('COMMAND',' '.join(args),flush=True)
        with (root/f'{tag}.log').open('w') as log:result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,tag
        folder=root if mode=='trace' else root/mode
        actual=('trace' if mode=='trace' else 'optimized')+tag[len(mode):]
        report=json.loads((folder/actual/'summary.json').read_text())
        report['command']=' '.join(args);report['register_router_mode']=mode
        (folder/actual/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        small={k:v for k,v in report.items() if k not in ['samples','router_samples','sources','decode_input_ids']}
        print('COMPLETE',small,flush=True)
        summaries.append(dict(path=str(folder/actual/'summary.json'),**small))
        progress=a.phase if a.phase=='trace' else f'decode_b{a.rank}'
        (root/f'{progress}_progress.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':main()
