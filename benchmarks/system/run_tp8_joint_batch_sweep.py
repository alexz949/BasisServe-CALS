"""Single-trial Dense Flash/Basis Joint sweep for Llama-8B and Qwen-32B."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'benchmarks/system/bench_llama31_8b_tp8_combined.py'
PROMPTS = Path('/workspace/runs/l31-cal128/tp8-benchmark-prompts')
QWEN_SNAPSHOT = Path('/workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/3cd67a4d96070a1cb8ce6c15512359b17638cd64')
MODELS = ('llama', 'qwen')
ARMS = ('dense', 'basis_joint')
BATCHES = (1, 2, 4, 6, 8, 10, 12, 14, 16)


def command_for(model, arm, length, batch, output, conditioning, measured):
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=8', str(DRIVER), '--tp-size', '8', '--arm', arm,
               '--length', str(length), '--batch', str(batch), '--repeat', '0',
               '--conditioning-steps', str(conditioning), '--measure-steps', str(measured),
               '--tag', 'sweep', '--output-root', str(output)]
    if model == 'llama':
        command += ['--tokens', str(PROMPTS / f'p{length}_c0.safetensors'),
                    '--prompt-manifest', str(PROMPTS / f'p{length}_c0.json')]
    else:
        assert model == 'qwen'
        command += [
            '--model', '/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137',
            '--factor-root', '/workspace/runs/qwen3-32b-joint-v96/factors',
            '--router-root', str(QWEN_SNAPSHOT / 'checkpoints/qwen3-32b-128k/uniform96-b16r16-als40-pcg100'),
            '--tokens', '/workspace/runs/qwen3-32b-densev-ruler30/calibration/windows.safetensors',
        ]
    return command


def validate_rows(rows, arm, length, batch, conditioning, measured):
    assert len(rows) == 8 and {r['rank'] for r in rows} == set(range(8))
    for row in rows:
        assert row['status'] == 'complete' and row['arm'] == arm
        assert row['tp'] == 8 and row['batch'] == batch and row['prompt_tokens'] == length
        assert row['conditioning_steps'] == conditioning and row['measured_steps'] == measured
        assert len(row['decode_step_ms']) == measured and row['component_profile'] is None
        assert row['dtype'] == 'bfloat16' and not row['trace_enabled']
        assert len(row['generated_token_ids']) == batch
        assert row['decode_step_ms'] == rows[0]['decode_step_ms']
        if arm == 'dense':
            assert row['decode_attention_backend'] == 'torch_flash_sdpa'
            assert row['key_placement'] == 'gpu'
        else:
            assert row['routing_mode'] == 'full_scan_b16r16_persistent_slots'
            assert row['key_placement'] == 'pinned_host_with_gpu_slots' and row['value_rank'] == 96


def summarize_rows(rows):
    result = {
        'decode_ms': rows[0]['decode_step_mean_ms'],
        'tokens_per_second': rows[0]['decode_tokens_per_second'],
        'active_batch': rows[0]['batch'],
        'decode_attention_backend': rows[0]['decode_attention_backend'],
        'host_k_total_gib': sum(r['host_persistent_exact_key_bytes'] for r in rows) / 2**30,
        'sum_process_max_rss_gib': sum(r['process_max_rss_bytes'] for r in rows) / 2**30,
    }
    for field in ('prefill_peak_allocated_bytes', 'prefill_peak_reserved_bytes',
                  'decode_peak_allocated_bytes', 'decode_peak_reserved_bytes',
                  'decode_resident_allocated_bytes', 'decode_resident_reserved_bytes'):
        result[field.replace('_bytes', '_gib')] = max(r[field] for r in rows) / 2**30
    return result


def failure_status(log_text, folder):
    oom = 'CUDA out of memory' in log_text or 'torch.OutOfMemoryError' in log_text
    phases = {'starting': 'setup', 'extensions_ready': 'model_loading_or_cache_allocation',
              'model_ready': 'prefill', 'prefill_complete': 'conditioning_or_decode',
              'complete': 'completed_rank'}
    states = {}
    for path in folder.glob('rank*.log'):
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if entries:
            states[path.stem] = entries[-1]['status']
    return {'status': 'gpu_oom' if oom else 'failed', 'last_rank_states': states,
            'possible_failure_phases': sorted({phases.get(s, s) for s in states.values()})}


def save(manifest, output):
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    fields = ['model', 'length', 'batch', 'arm', 'status', 'active_batch', 'decode_ms',
              'tokens_per_second', 'prefill_peak_allocated_gib', 'decode_peak_allocated_gib',
              'decode_resident_allocated_gib', 'prefill_peak_reserved_gib',
              'decode_peak_reserved_gib', 'decode_resident_reserved_gib', 'host_k_total_gib',
              'sum_process_max_rss_gib', 'possible_failure_phases']
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(manifest['trials'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'formal'), required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/system_benchmarks/tp8_joint_batch_sweep')
    args = parser.parse_args()
    smoke = args.phase == 'smoke'
    lengths, batches = ([4096], [6]) if smoke else ([65536, 130048], list(BATCHES))
    conditioning, measured = (2, 8) if smoke else (16, 128)
    output = args.output.resolve() / args.phase
    output.mkdir(parents=True, exist_ok=True)
    config = {'models': list(MODELS), 'arms': list(ARMS), 'contexts': lengths,
              'batches': batches, 'conditioning_steps': conditioning, 'measure_steps': measured,
              'repeats': 1, 'stop_after_oom': False, 'phase': args.phase}
    path = output / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        assert manifest['config'] == config
    else:
        manifest = {'config': config, 'created_at': datetime.now(timezone.utc).isoformat(),
                    'status': 'running', 'trials': []}
    indexed = {(r['model'], r['length'], r['batch'], r['arm']): r for r in manifest['trials']}
    for model in MODELS:
        for length in lengths:
            for batch in batches:
                for arm in ARMS:
                    key = (model, length, batch, arm)
                    previous = indexed.get(key)
                    if previous and previous['status'] in ('complete', 'gpu_oom'):
                        continue
                    folder = output / model / f'p{length}_b{batch}' / arm
                    folder.mkdir(parents=True, exist_ok=True)
                    command = command_for(model, arm, length, batch, folder, conditioning, measured)
                    record = dict(model=model, length=length, batch=batch, arm=arm, status='running',
                                  command=command, started_at=datetime.now(timezone.utc).isoformat())
                    if previous:
                        manifest['trials'][manifest['trials'].index(previous)] = record
                    else:
                        manifest['trials'].append(record)
                    manifest['status'] = 'running'
                    save(manifest, output)
                    print('RUN', ' '.join(command), flush=True)
                    log_path = folder / 'run.log'
                    offset = log_path.stat().st_size if log_path.exists() else 0
                    with log_path.open('a') as log:
                        completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                                   stdout=log, stderr=subprocess.STDOUT)
                    record.update(returncode=completed.returncode, finished_at=datetime.now(timezone.utc).isoformat())
                    rank_root = folder / f'sweep_{arm}_p{length}_b{batch}_r0'
                    if completed.returncode == 0:
                        rows = [json.loads((rank_root / f'rank{i}.json').read_text()) for i in range(8)]
                        validate_rows(rows, arm, length, batch, conditioning, measured)
                        record.update(status='complete', **summarize_rows(rows))
                    else:
                        with log_path.open('rb') as log:
                            log.seek(offset)
                            text = log.read().decode(errors='replace')
                        record.update(failure_status(text, rank_root))
                    failed = record['status'] == 'failed' or (smoke and record['status'] != 'complete')
                    if failed:
                        manifest['status'] = 'stopped_on_failure'
                    save(manifest, output)
                    print(json.dumps(record), flush=True)
                    if failed:
                        return 1
    manifest['status'] = 'complete'
    manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
    save(manifest, output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
