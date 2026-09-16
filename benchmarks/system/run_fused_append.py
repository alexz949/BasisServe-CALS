"""Smoke and ABBA continuous-decode comparison for the fused append candidate."""
from pathlib import Path
import json
import subprocess
import sys
import statistics


def main():
    root=Path('results/system_benchmarks/fused_append');root.mkdir(parents=True,exist_ok=True)
    validation=json.loads((root/'validation.json').read_text());assert validation['status']=='complete'
    rows=[]
    for fused,repeat,smoke in [(True,10,True),(False,10,False),(True,10,False),(True,11,False),(False,11,False)]:
        args=[sys.executable,'-m','benchmarks.system.bench_two_stage','--mode','two','--window','64','--repeat',str(repeat)]
        if fused:args.append('--fused-append')
        if smoke:args.append('--smoke')
        tag=f'{"fused" if fused else "reference"}_{repeat}{"_smoke" if smoke else ""}'
        print('COMMAND',' '.join(args),flush=True)
        with (root/f'{tag}.log').open('w') as log:
            result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,tag
        path=Path('results/system_benchmarks/two_stage')/f'two{"_fused" if fused else ""}_w64_r{repeat}'/f'optimized_b16_w4_r0{"_smoke" if smoke else ""}'/'summary.json'
        summary=json.loads(path.read_text());rows.append(dict(fused=fused,repeat=repeat,smoke=smoke,path=str(path),summary=summary))
        print('COMPLETE',tag,summary.get('cuda_median_after10_ms'),flush=True)
    means={name:statistics.mean(r['summary']['cuda_median_after10_ms'] for r in rows if not r['smoke'] and r['fused']==flag) for name,flag in [('reference',False),('fused',True)]}
    payload=dict(status='complete',runs=rows,mean_cuda_medians_ms=means,reduction_percent=100*(1-means['fused']/means['reference']))
    (root/'decode.json').write_text(json.dumps(payload,indent=2)+'\n');print('RESULT',means,flush=True)

if __name__=='__main__':main()
