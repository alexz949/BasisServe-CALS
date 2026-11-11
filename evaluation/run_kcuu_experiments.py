"""Same-allocation TP4 smoke gates and the 30-configuration KCuu matrix."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from basisserve.core.qwen3_8b_vllm_c1 import file_sha256
from evaluation.prepare_kcuu_llama import MODEL, FACTORS, OUTPUT, source_hashes
from evaluation.run_gh200_tp4_smoke import MODEL as QWEN_MODEL, FACTORS as QWEN_FACTORS

MODELS = {'llama': (MODEL, FACTORS), 'qwen': (QWEN_MODEL, QWEN_FACTORS[64])}


def prepare(root):
    python = sys.executable
    smoke, formal = [], []
    for model_name, (model, factors) in MODELS.items():
        common = ['--model', str(model), '--factor-dir', str(factors)]
        out = root / f'{model_name}_boundary.json'
        smoke.append(dict(name=f'{model_name}_boundary', output=str(out),
            command=[python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
                     '-m', 'evaluation.validate_kcuu_tp4', '--factor-dir', str(factors), '--output', str(out)]))
        for mode in ('ag', 'hybrid'):
            out = root / f'{model_name}_{mode}_smoke.json'
            smoke.append(dict(name=f'{model_name}_{mode}_smoke', output=str(out),
                command=[python, '-u', '-m', 'evaluation.smoke_kcuu_llama', *common,
                         '--mode', mode, '--output', str(out)]))
        out = root / f'{model_name}_logits_comparison.json'
        smoke.append(dict(name=f'{model_name}_logits', output=str(out),
            command=[python, '-m', 'evaluation.compare_kcuu_logits',
                     '--ag', str(root / f'{model_name}_ag_smoke.json'),
                     '--hybrid', str(root / f'{model_name}_hybrid_smoke.json'), '--output', str(out)]))
    for batch in (64, 16, 32, 128, 256):
        for model_name, (model, factors) in MODELS.items():
            for arm in ('dense', 'v64_ag', 'v64_hybrid'):
                name = f'{model_name}_{arm}_b{batch}'
                out = root / 'formal' / f'{name}.json'
                formal.append(dict(name=name, model=model_name, arm=arm, batch=batch, output=str(out),
                    command=[python, '-u', '-m', 'evaluation.benchmark_kcuu_tp4', '--model', str(model),
                        '--factor-dir', str(factors), '--arm', arm, '--batch', str(batch), '--output', str(out)]))
    assert len(formal) == 30
    return dict(source_sha256=source_hashes(), smoke=smoke, formal=formal,
        settings=dict(nodes=1, gpus_per_node=4, prefill_tokens=8192, output_tokens=128,
                      repeats=3, warmups=1, max_num_batched_tokens=8192,
                      gpu_memory_utilization=0.8, cudagraph_mode='FULL_DECODE_ONLY'),
        model_revisions={name:model.name for name,(model,_) in MODELS.items()},
        factor_manifest_sha256={name:file_sha256(factors/'results.json') for name,(_,factors) in MODELS.items()},
        authorization='User authorized smoke, fixes and the full matrix within the two-hour interactive allocation',
        deferred=['operator_breakdown', 'fixed_B64_decode_AB'])


def finished(job, hashes):
    path = Path(job['output'])
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    if result.get('status') not in ('passed', 'complete'):
        return False
    if result['source_sha256'] != hashes:
        return False
    if 'batch' in job:
        assert (result['batch'], result['arm'], result['prefill_tokens'], result['output_tokens'],
                result['repeats'], result['warmups']) == (job['batch'], job['arm'], 8192, 128, 3, 1)
        model, factors = MODELS[job['model']]
        assert result['model_revision'] == model.name
        assert result['factors_sha256'] == file_sha256(factors/'results.json')
        assert len(result['runs']) == 3
    return True


def execute(job, hashes, timeout_seconds):
    if finished(job, hashes):
        print('ALREADY COMPLETE', job['name'], flush=True)
        return 0
    out = Path(job['output'])
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix('.log')
    if out.exists():
        with out.with_suffix('.history.jsonl').open('a') as history:
            history.write(json.dumps(json.loads(out.read_text()))+'\n')
    print('START', job['name'], shlex.join(job['command']), 'LOG', log, flush=True)
    with log.open('a') as handle:
        handle.write('\nCOMMAND '+shlex.join(job['command'])+'\n'); handle.flush()
        code = subprocess.run(['timeout', '--signal=TERM', '--kill-after=30s', str(timeout_seconds),
                               *job['command']], stdout=handle, stderr=subprocess.STDOUT).returncode
    if code:
        result = json.loads(out.read_text()) if out.exists() else {}
        result.update(status='timeout' if code == 124 else 'failed', exit_code=code,
                      source_sha256=hashes, command=shlex.join(job['command']), log=str(log))
        out.write_text(json.dumps(result, indent=2)+'\n')
        print('FAILED', job['name'], code, flush=True)
    else:
        assert finished(job, hashes)
        print('PASS', job['name'], flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--stage', choices=('smoke', 'formal', 'all'), default='all')
    args = parser.parse_args()
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    plan = prepare(root)
    (root/'experiment_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    if args.prepare_only:
        for job in plan['smoke'] + plan['formal']:
            print(shlex.join(job['command']))
        return
    assert os.environ.get('SLURM_NNODES') == '1' and os.environ.get('CONDA_DEFAULT_ENV') == 'basis'
    for job in plan['smoke']:
        if args.stage == 'formal':
            assert finished(job, plan['source_sha256']), job['name']
        else:
            code = execute(job, plan['source_sha256'], 900)
            if code:
                sys.exit(code)
    if args.stage == 'smoke':
        return
    failures = [job['name'] for job in plan['formal'] if execute(job, plan['source_sha256'], 1200)]
    from evaluation.summarize_kcuu import summarize
    summarize(root)
    print('FORMAL MATRIX FINISHED failures=', failures, flush=True)
    sys.exit(bool(failures))


if __name__ == '__main__':
    main()
