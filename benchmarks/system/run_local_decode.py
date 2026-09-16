"""Paired baseline/optimized continuous-decode runs in one GPU allocation."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    root=Path('results/system_benchmarks/local_kernels');rows=[]
    cases=[('optimized',0,True),('baseline',0,False),('optimized',0,False),
           ('optimized',1,False),('baseline',1,False)]
    for mode,repeat,smoke in cases:
        command=[sys.executable,'-m','benchmarks.system.bench_local_kernels','--mode',mode,'--rank','16','--length','65536','--repeat',str(repeat)]
        if smoke:command.append('--smoke')
        tag=f'{mode}_b16_w4_r{repeat}{"_smoke" if smoke else ""}'
        print('COMMAND',' '.join(command),flush=True)
        with (root/f'{tag}.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,tag
        row=json.loads((root/tag/'summary.json').read_text())
        small={k:v for k,v in row.items() if k not in ['samples','router_samples','sources','decode_input_ids']}
        rows.append(small)
        (root/'decode_progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('COMPLETE',small,flush=True)
    formal=[]
    for mode,repeat,smoke in cases:
        if not smoke:formal.append(json.loads((root/f'{mode}_b16_w4_r{repeat}'/'summary.json').read_text()))
    same=all(row['decode_input_ids']==formal[0]['decode_input_ids'] for row in formal[1:])
    (root/'decode_summary.json').write_text(json.dumps(dict(status='complete',same_generated_token_sequence=same,runs=rows,
        order='optimized 8K smoke, then 64K baseline/optimized/optimized/baseline; each fresh process, same allocated GPU',
        scope='100-step continuous decode; no A/B timing probes. Logit finiteness and input token recording are identical in both variants.'),indent=2)+'\n')
    print('ALL_COMPLETE','same generated tokens',same,flush=True)


if __name__=='__main__':main()
