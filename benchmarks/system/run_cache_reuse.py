"""Run matched reuse/reload ablations in fresh GPU processes."""
import itertools
import json
from pathlib import Path
import subprocess
import sys


def main():
    root=Path('results/system_benchmarks/cache_reuse');root.mkdir(parents=True,exist_ok=True)
    arms=[(m,mode) for m in ['basis16','basis8'] for mode in ['native','reload','reuse']]+[('shadowkv_cpu',mode) for mode in ['native','reload']]
    records=[]
    for smoke,length,(method,mode) in [(True,8192,arm) for arm in arms]+[(False,length,arm) for length,arm in itertools.product([65536,131072],arms)]:
        folder=root/mode/f'{method}_t{length}_b1{"_smoke" if smoke else ""}';folder.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-m','benchmarks.system.bench_cache_reuse','--method',method,'--mode',mode,'--length',str(length)]
        if smoke:cmd.append('--smoke')
        print('COMMAND',' '.join(cmd),flush=True)
        with (folder/'run.log').open('w') as log:process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        assert process.returncode==0,dict(command=cmd,log=str(folder/'run.log'))
        d=json.loads((folder/'result.json').read_text());c=json.loads((folder/'cache.json').read_text())
        row=dict(method=method,mode=mode,length=length,smoke=smoke,command=cmd,
            ms_step=d['native_benchmark_ms_per_step'],details=d['details'])
        if c['layers']:
            # Formal native loop has100 measured steps plus one final extra inference.
            count=8 if smoke else 100
            hits=sum(sum(layer['hits_by_step'][:count]) for layer in c['layers'])
            valid=sum(sum(layer['valid_by_step'][:count]) for layer in c['layers'])
            row.update(hit_fraction=hits/valid,logical_host_K_bytes_per_step=(valid-hits)*128*2/count,
                cache_bytes=sum(layer['cache_bytes'] for layer in c['layers']))
        records.append(row);(root/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print(row,flush=True)
    (root/'summary.json').write_text(json.dumps(dict(status='complete',records=records),indent=2)+'\n')


if __name__=='__main__':main()
