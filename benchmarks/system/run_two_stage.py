"""Sequential smoke, multi-window diagnostics, or matched serving comparisons."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['trace','decode'],required=True);a=p.parse_args()
    root=Path('results/system_benchmarks/two_stage')
    cases=[('two',64,0,True)]+[('trace',w,0,False) for w in [64,65,66]] if a.phase=='trace' else [(m,64,r,False) for m,r in [('full',0),('coarse',0),('two',0),('two',1),('coarse',1),('full',1)]]
    for mode,window,repeat,smoke in cases:
        args=[sys.executable,'-m','benchmarks.system.bench_two_stage','--mode',mode,'--window',str(window),'--repeat',str(repeat)]
        if smoke:args.append('--smoke')
        print('COMMAND',' '.join(args),flush=True)
        with (root/f'{mode}_w{window}_r{repeat}{"_smoke" if smoke else ""}.log').open('w') as log:
            result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(mode,window,repeat)
        print('COMPLETE',mode,window,repeat,flush=True)
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':main()
