"""Run and audit 1100-prompt original-V128 and C1-V96 RULER comparisons."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from evaluation.v96kl_common import read_json, write_json, sha256


ROOT = Path('/workspace/runs/l31-ruler100')
IDENTITY = Path('/workspace/runs/l31-cal128/identity.json')
BANK = Path('/workspace/runs/l31-dense128/router/ours_b16r16')
V96_BANK = Path('/workspace/runs/l31-cal128/router/ours_b16r16')
LOKI = Path('/workspace/runs/l31-loki-wiki/pca')
DENSE = 'evaluation.eval_llama_dense128'
ROUTING = 'evaluation.eval_llama_dense_v_routing128'
V96_CORE = 'evaluation.eval_llama_cal128'
V96_BASELINES = 'evaluation.eval_llama_baselines128'
ARMS = ('ours', 'shadowkv', 'lrqk', 'loki')
V96_ARMS = ('shadowkv', 'lrqk', 'loki')


def command(module, arguments, tp=False):
    launcher = ['torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=2', '-m'] if tp else []
    return [sys.executable, '-u', '-m', *launcher, module, *map(str, arguments)]


def run_jobs(jobs):
    workers = []
    for label, gpu, cmd in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2',
            HF_HOME='/workspace/hf-data', PYTORCH_ALLOC_CONF='expandable_segments:True',
            NCCL_P2P_DISABLE='1')
        print('RUN', label, 'GPU', gpu, shlex.join(cmd), flush=True)
        with (ROOT/'logs'/f'{label}.log').open('a') as log:
            log.write(shlex.join(cmd)+'\n')
            log.flush()
            workers.append((label, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
    codes = {label: worker.wait() for label, worker in workers}
    print('EXIT', codes, flush=True)
    assert all(code == 0 for code in codes.values()), codes


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'smoke', 'evaluate'))
    args = parser.parse_args()
    (ROOT/'logs').mkdir(parents=True, exist_ok=True)
    data = read_json(ROOT/'ruler/manifest.json')
    assert data['status'] == 'complete'
    assert data['protocol']['samples_per_task'] == 100 and data['protocol']['sequence_length'] == 131072
    dense_args = ['--identity', IDENTITY, '--data', ROOT/'ruler', '--bank', BANK,
                  '--output', ROOT/'v128-dense']
    route_args = ['--identity', IDENTITY, '--data', ROOT/'ruler', '--bank', BANK,
                  '--loki', LOKI, '--output', ROOT/'v128', '--dense-reference', ROOT/'v128-dense/full']
    v96_core_args = ['--identity', IDENTITY, '--data', ROOT/'ruler', '--bank', V96_BANK,
                     '--output', ROOT/'v96-core']
    v96_baseline_args = ['--identity', IDENTITY, '--data', ROOT/'ruler', '--loki', LOKI,
                         '--output', ROOT/'v96-baselines',
                         '--full-reference', ROOT/'v96-core/full']
    if args.stage == 'prepare':
        prompt_json = ROOT/'v128-dense/prompts.json'
        prompt_tensors = ROOT/'v128-dense/prompts.safetensors'
        if prompt_json.exists() and prompt_tensors.exists():
            prepared = read_json(prompt_json)
            assert prepared['status'] == 'complete' and len(prepared['rows']) == 1100
            assert prepared['identity_sha256'] == sha256(IDENTITY)
            assert prepared['data_sha256'] == sha256(ROOT/'ruler/manifest.json')
            assert prepared['tokens_sha256'] == sha256(prompt_tensors)
        else:
            run_jobs([('prepare', '', command(DENSE, ['prepare', *dense_args]))])
        for destination in (ROOT/'v128', ROOT/'v96-core', ROOT/'v96-baselines'):
            destination.mkdir(exist_ok=True)
            for name in ('prompts.json', 'prompts.safetensors'):
                source, target = ROOT/'v128-dense'/name, destination/name
                if target.exists():
                    assert sha256(source) == sha256(target)
                else:
                    target.hardlink_to(source)
        return
    if args.stage == 'smoke':
        run_jobs([('dense-smoke', '0,1', command(DENSE, ['smoke', *dense_args], tp=True))])
        run_jobs([('dense-audit-smoke', '', command(DENSE, ['audit-smoke', *dense_args]))])
        run_jobs([(f'{arm}-smoke', str(gpu), command(ROUTING, ['smoke', *route_args, '--arm', arm]))
                  for gpu, arm in enumerate(ARMS)])
        run_jobs([(f'{arm}-audit-smoke', '', command(ROUTING, ['audit-smoke', *route_args, '--arm', arm]))
                  for arm in ARMS])
        run_jobs([(f'v96-{arm}-smoke', str(gpu), command(V96_CORE,
            ['smoke', *v96_core_args, '--arm', arm])) for gpu, arm in enumerate(('full', 'ours'))])
        run_jobs([('v96-core-audit-smoke', '', command(V96_CORE,
            ['audit-smoke', *v96_core_args]))])
        run_jobs([(f'v96-{arm}-smoke', str(gpu), command(V96_BASELINES,
            ['smoke', *v96_baseline_args, '--arm', arm])) for gpu, arm in enumerate(V96_ARMS)])
        run_jobs([('v96-baselines-audit-smoke', '', command(V96_BASELINES,
            ['audit-smoke', *v96_baseline_args]))])
        print('V128 AND V96 SMOKE COMPLETE; formal evaluation has not started', flush=True)
        return
    run_jobs([(f'dense-evaluate-{shard}', gpu, command(DENSE,
        ['evaluate', *dense_args, '--shard', shard, '--shards', 2], tp=True))
        for shard, gpu in enumerate(('0,1', '2,3'))])
    run_jobs([('dense-summary', '', command(DENSE, ['summarize', *dense_args]))])
    for arm in ARMS:
        run_jobs([(f'{arm}-evaluate-{gpu}', str(gpu), command(ROUTING,
            ['evaluate', *route_args, '--arm', arm, '--shard', gpu, '--shards', 4])) for gpu in range(4)])
    run_jobs([('native-prefill-check', '0', command('evaluation.audit_llama_dense_prefill',
        ['--evaluation', ROOT/'v128', '--reference', ROOT/'v128-dense/full/evaluate',
         '--output', ROOT/'v128-prefill-check', '--identity', IDENTITY]))])
    run_jobs([('routing-summary', '', command('evaluation.summarize_llama_dense_v128',
        ['--evaluation', ROOT/'v128', '--reference', ROOT/'v128-dense/full/evaluate',
         '--prefill-check', ROOT/'v128-prefill-check', '--identity', IDENTITY,
         '--data', ROOT/'ruler', '--bank', BANK, '--loki', LOKI]))])
    dense_path, routing_path = ROOT/'v128-dense/summary.json', ROOT/'v128/summary.json'
    dense, routing = read_json(dense_path), read_json(routing_path)
    assert dense['verified_predictions'] == 1100 and routing['verified_predictions'] == 4400
    for key in ('identity_sha256', 'prompts_sha256', 'samples', 'samples_per_task', 'rank_schedule'):
        assert dense['protocol'][key] == routing['protocol'][key]
    tasks = {task: dict(dense=dense['tasks'][task]['full'], **values) for task, values in routing['tasks'].items()}
    arms = ('dense', *ARMS)
    means = {arm: sum(values[arm] for values in tasks.values())/11 for arm in arms}
    means10 = {arm: sum(values[arm] for task, values in tasks.items() if task != 'niah_single_3')/10 for arm in arms}
    write_json(ROOT/'v128-summary.json', dict(status='complete', samples_per_arm=1100,
        verified_predictions=5500, value_dimension=128, tasks=tasks, means=means,
        means_10_excluding_single_3=means10, prefill_audit=routing['prefill_audit'],
        protocol=routing['protocol'], source_summaries={str(p):sha256(p) for p in (dense_path, routing_path)}))
    lines = ['# Llama-3.1-8B-Instruct RULER 128k: original V128', '',
        '11 tasks × 100 identical prompts per arm; greedy generation with native EOS and official caps.',
        'Dense uses TP2. Routing arms use four independent GPUs. Native single-GPU Full-K checks cover all observed prefill first-token differences against TP2.', '',
        '| Task | Dense | B16R16 | ShadowKV | LRQK | Loki |', '|---|---:|---:|---:|---:|---:|']
    for task, values in tasks.items():
        lines.append('| '+task+' | '+' | '.join(f'{values[arm]:.4f}' for arm in arms)+' |')
    for label, values in [('11-task mean', means), ('10-task mean excluding single_3', means10)]:
        lines.append('| '+label+' | '+' | '.join(f'{values[arm]:.4f}' for arm in arms)+' |')
    lines += ['', 'B16R16: physical budget2048 including sink32/recent64. LRQK: top832+recent64. Loki: top856/recent0, Wikipedia PCA32.',
        'ShadowKV: official CPU-offload, routed2048 plus outliers/local/generated. Per-query top is not a hard physical KV-group cap.',
        'Environment: basis. Reproduce using `python -m evaluation.run_llama_ruler100 prepare`, then `smoke`, then `evaluate`. Full commands are in logs/.']
    (ROOT/'v128-summary.md').write_text('\n'.join(lines)+'\n')
    print('V128 COMPLETE', means, flush=True)

    for arm in ('full',):
        run_jobs([(f'v96-{arm}-evaluate-{gpu}', str(gpu), command(V96_CORE,
            ['evaluate', *v96_core_args, '--arm', arm, '--shard', gpu, '--shards', 4]))
            for gpu in range(4)])
    run_jobs([('v96-core-summary', '', command(V96_CORE, ['summarize', *v96_core_args]))])
    for arm in V96_ARMS:
        run_jobs([(f'v96-{arm}-evaluate-{gpu}', str(gpu), command(V96_BASELINES,
            ['evaluate', *v96_baseline_args, '--arm', arm, '--shard', gpu, '--shards', 4]))
            for gpu in range(4)])
    run_jobs([('v96-baselines-summary', '', command(V96_BASELINES,
        ['summarize', *v96_baseline_args]))])

    core_path, baseline_path = ROOT/'v96-core/summary.json', ROOT/'v96-baselines/summary.json'
    core, baselines = read_json(core_path), read_json(baseline_path)
    assert core['verified_predictions'] == 1100 and core['evaluated_arms'] == ['full']
    assert baselines['verified_predictions'] == 3300
    for key in ('identity_sha256', 'prompts_sha256', 'samples', 'samples_per_task', 'rank_schedule'):
        assert core['protocol'][key] == baselines['protocol'][key]
    v96_tasks = {task: dict(full=values['full'],
        **baselines['tasks'][task]) for task, values in core['tasks'].items()}
    v96_arms = ('full', *V96_ARMS)
    v96_means = {arm: sum(values[arm] for values in v96_tasks.values())/11 for arm in v96_arms}
    v96_means10 = {arm: sum(values[arm] for task, values in v96_tasks.items()
        if task != 'niah_single_3')/10 for arm in v96_arms}
    write_json(ROOT/'v96-summary.json', dict(status='complete', samples_per_arm=1100,
        verified_predictions=4400, value_dimension=96, tasks=v96_tasks, means=v96_means,
        means_10_excluding_single_3=v96_means10, protocol=dict(
            core=core['protocol'], baselines=baselines['protocol']),
        source_summaries={str(p): sha256(p) for p in (core_path, baseline_path)}))
    v96_lines = ['# Llama-3.1-8B-Instruct RULER 128k: C1 V96', '',
        '11 tasks × 100 identical prompts per arm; greedy generation with native EOS and official caps.',
        'All four arms use the same C1 V96 cache and output decoder. K/V96 remain resident on GPU.', '',
        '| Task | Full-K | ShadowKV | LRQK | Loki |',
        '|---|---:|---:|---:|---:|']
    for task, values in v96_tasks.items():
        v96_lines.append('| '+task+' | '+' | '.join(f'{values[arm]:.4f}' for arm in v96_arms)+' |')
    for label, values in [('11-task mean', v96_means),
                          ('10-task mean excluding single_3', v96_means10)]:
        v96_lines.append('| '+label+' | '+' | '.join(f'{values[arm]:.4f}' for arm in v96_arms)+' |')
    v96_lines += ['',
        'LRQK: top832+recent64. Loki: top856/recent0, Wikipedia PCA32.',
        'ShadowKV: resident rank160/chunk8/routed2048 with 48 outlier chunks; selected V96 is used directly without V128 reconstruction.',
        'Environment: basis. Reproduce using `python -m evaluation.run_llama_ruler100 prepare`, then `smoke`, then `evaluate`. Full commands are in logs/.']
    (ROOT/'v96-summary.md').write_text('\n'.join(v96_lines)+'\n')
    print('V96 COMPLETE', v96_means, flush=True)


if __name__ == '__main__':
    main()
