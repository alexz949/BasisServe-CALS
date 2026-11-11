"""Prepare or execute the same-node TP4 smoke grid; --formal continues into the formal grid once every smoke passes."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

MODEL=Path('/u/azhang35/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137')
HUB=Path('/projects/bido/azhang35/hf/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots')
FACTORS={64:HUB/'0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-32b/c1/factor-banks/R64-S6',
         96:HUB/'bb3e548586ecd47cddd8201eb3e1f33077600205/ICLR-results/qwen3-32b/c1/factor-banks/R96-S6'}


def execute(job):
    """Run one planned command unless its result is already complete; return the exit code."""
    output=Path(job['output'])
    if output.exists() and json.loads(output.read_text()).get('status')=='complete':
        print('ALREADY COMPLETE',job['name'],flush=True);return 0
    log=output.with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
    print('START',job['name'],'log=',log,flush=True)
    with log.open('a') as handle:
        handle.write('\nCOMMAND '+shlex.join(job['command'])+'\n');handle.flush()
        result=subprocess.run(job['command'],stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:
        failed=json.loads(output.read_text()) if output.exists() else {}
        failed.update(status='failed',exit_code=result.returncode,log=str(log))
        output.write_text(json.dumps(failed,indent=2)+'\n')
        print('FAILED',job['name'],'exit_code=',result.returncode,'see',log,flush=True)
        return result.returncode
    assert json.loads(output.read_text())['status']=='complete'
    print('PASS',job['name'],flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results/rebuttal/gh200_tp4/tp4'))
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--formal',action='store_true')
    args=parser.parse_args()
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    python=sys.executable
    jobs=[]
    for rank,path in FACTORS.items():
        output=root/f'validate_v{rank}.json'
        command=[python,'-m','torch.distributed.run','--standalone','--nproc-per-node=4','-m',
                 'evaluation.validate_gh200_tp4','--factor-dir',str(path),'--output',str(output)]
        jobs.append(dict(name=f'validate_v{rank}',command=command,output=str(output)))
    formal=[]
    for batch in [1,8,32,64,128,256]:
        for arm in ['dense','v64','v96']:
            name=f'{arm}_b{batch}'
            common=[python,'-u','-m','evaluation.benchmark_gh200_tp4','--model',str(MODEL),
                    '--arm',arm,'--batch',str(batch),'--prefill-tokens','4096']
            if arm!='dense':common+=['--factor-dir',str(FACTORS[int(arm[1:])])]
            output=root/'formal'/f'{name}.json'
            formal.append(dict(name='formal_'+name,command=common+['--decode-tokens','128','--repeats','3','--output',str(output)],output=str(output)))
            if batch==1:
                output=root/f'{name}.json'
                jobs.append(dict(name=name,command=common+['--decode-tokens','8','--repeats','1','--smoke','--output',str(output)],output=str(output)))
    manifest=dict(status='prepared',same_node_gpus=4,model=str(MODEL),smoke=jobs,
                  formal_commands=[job['command'] for job in formal],formal_requires_smoke_and_user_approval=True,
                  factors={str(k):str(v) for k,v in FACTORS.items()})
    (root/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for job in jobs:print(shlex.join(job['command']),flush=True)
    if args.prepare_only:return
    assert os.environ.get('SLURM_NNODES')=='1'
    for job in jobs:
        code=execute(job)
        if code:sys.exit(code)
    print('ALL TP4 SMOKES PASSED',flush=True)
    if not args.formal:return
    # One failed formal configuration does not block the remaining grid.
    failures=[job['name'] for job in formal if execute(job)]
    print('FORMAL GRID FINISHED failed=',failures,flush=True)
    sys.exit(bool(failures))

if __name__=='__main__':
    main()
