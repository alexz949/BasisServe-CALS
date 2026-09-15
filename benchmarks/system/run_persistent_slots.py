"""Validate then benchmark persistent-slot K caching."""
import itertools
import json
from pathlib import Path
import subprocess
import sys


def main():
    root=Path('results/system_benchmarks/persistent_slots');root.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-m','benchmarks.system.validate_persistent_slots'],check=True)
    records=[]
    for smoke,length,rank,mode in [(True,8192,r,m) for r,m in itertools.product([16,8],['reload','reuse'])]+[(False,t,r,m) for t,r,m in itertools.product([65536,131072],[16,8],['reload','reuse'])]:
        folder=root/mode/f'basis{rank}_t{length}_b1{"_smoke" if smoke else ""}';folder.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-m','benchmarks.system.bench_persistent_slots','--rank',str(rank),'--mode',mode,'--length',str(length)]
        if smoke:cmd.append('--smoke')
        print('COMMAND',' '.join(cmd),flush=True)
        with (folder/'run.log').open('w') as log:process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        assert process.returncode==0,dict(command=cmd,log=str(folder/'run.log'))
        d=json.loads((folder/'result.json').read_text());c=json.loads((folder/'cache.json').read_text())
        record={k:v for k,v in c.items() if k!='layers'}
        record.update(command=cmd,ms_step=d['native_benchmark_ms_per_step'],details=d['details'])
        records.append(record);(root/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print(record,flush=True)
    (root/'summary.json').write_text(json.dumps(dict(status='complete',records=records),indent=2)+'\n')


if __name__=='__main__':main()
