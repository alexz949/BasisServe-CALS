"""Validate and summarize the single-trial TP8 Joint batch sweep."""

import argparse
import csv
import json
from pathlib import Path

from benchmarks.system.run_tp8_joint_batch_sweep import summarize_rows, validate_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/system_benchmarks/tp8_joint_batch_sweep'))
    args = parser.parse_args()
    root = args.output / 'formal'
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    config = manifest['config']
    assert config['phase'] == 'formal' and config['repeats'] == 1
    expected = {(m, n, b, a) for m in config['models'] for n in config['contexts']
                for b in config['context_batches'][str(n)] for a in config['arms']}
    indexed = {(r['model'], r['length'], r['batch'], r['arm']): r for r in manifest['trials']}
    assert set(indexed) == expected and len(manifest['trials']) == len(expected)
    prompts = {}
    validated = 0
    for key, record in indexed.items():
        model, length, batch, arm = key
        folder = root / model / f'p{length}_b{batch}' / arm
        if record['status'] == 'gpu_oom':
            assert record['returncode'] != 0
            log = (folder / 'run.log').read_text()
            assert 'CUDA out of memory' in log or 'torch.OutOfMemoryError' in log
            continue
        assert record['status'] == 'complete' and record['returncode'] == 0
        rank_folder = folder / f'sweep_{arm}_p{length}_b{batch}_r0'
        rows = [json.loads((rank_folder / f'rank{i}.json').read_text()) for i in range(8)]
        validate_rows(rows, arm, length, batch, config['conditioning_steps'], config['measure_steps'])
        assert all(record[field] == value for field, value in summarize_rows(rows).items())
        validated += len(rows)
        prompts[key] = (rows[0]['tokens'], rows[0]['prompt_cohort']['sample_ids'])
    pairs = []
    for model in config['models']:
        for length in config['contexts']:
            for batch in config['context_batches'][str(length)]:
                dense, basis = [indexed[(model, length, batch, a)] for a in config['arms']]
                pair = {'model': model, 'context': length, 'batch': batch}
                for label, record in [('dense', dense), ('basis', basis)]:
                    pair[f'{label}_status'] = record['status']
                    for field in ('decode_ms', 'tokens_per_second', 'prefill_peak_allocated_gib',
                                  'decode_peak_allocated_gib', 'decode_resident_allocated_gib'):
                        pair[f'{label}_{field}'] = record.get(field)
                success = dense['status'] == basis['status'] == 'complete'
                pair['latency_speedup'] = dense['decode_ms'] / basis['decode_ms'] if success else None
                pair['throughput_ratio'] = basis['tokens_per_second'] / dense['tokens_per_second'] if success else None
                pair['decode_peak_saved_gib'] = (dense['decode_peak_allocated_gib'] - basis['decode_peak_allocated_gib']) if success else None
                pair['decode_peak_saved_percent'] = (100 * pair['decode_peak_saved_gib'] / dense['decode_peak_allocated_gib']) if success else None
                if success:
                    assert prompts[(model, length, batch, 'dense')] == prompts[(model, length, batch, 'basis_joint')]
                pairs.append(pair)
    with (args.output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    completed = sum(r['status'] == 'complete' for r in indexed.values())
    lines = ['# TP8 Joint Batch Sweep Results', '',
             f'Completed {len(indexed)} attempts: {completed} successful, {len(indexed)-completed} GPU OOM.',
             f'Validated {validated} successful rank records and matched inputs for each successful pair.', '',
             'Environment: `basis`, 8 x L40S, BF16, TP8. One trial per point, not repeated estimates.',
             '16 conditioning forwards + 128 measured forwards; full-model steady decode, not request E2E.',
             'All points are newly measured in this sweep. No historical results are substituted.',
             'Dense is GPU-resident Flash SDPA; Basis is Joint V96 full-scan with historical K offload.',
             'Latency is mean per-step rank-max CUDA-event time; throughput is measured wall tokens/s.',
             'Memory is maximum single-rank PyTorch allocated GiB, not total device or KV-only memory.',
             '**Primary metric: peak decode allocated GPU memory versus batch.**',
             'Prefill OOM means no measured decode peak is available; it does not establish a decode-memory limit.',
             'Negative memory savings mean Basis uses more allocated GPU memory at that point.',
             'See [README](README.md) for the exact command, prompt sources and metric definitions.']
    lines += ['', '## Observed Memory Crossover', '',
              'Smallest jointly successful tested batch where Basis has lower allocated memory.',
              'These are observed grid points, not exact crossover thresholds; OOM pairs are excluded.', '',
              '| Model | Context | Decode peak | Prefill peak |',
              '| --- | ---: | ---: | ---: |']
    for model in config['models']:
        for length in config['contexts']:
            subset = [p for p in pairs if p['model'] == model and p['context'] == length
                      and p['dense_status'] == p['basis_status'] == 'complete']
            crossings = [min((p['batch'] for p in subset if p[f'basis_{metric}'] < p[f'dense_{metric}']),
                             default='not observed')
                         for metric in ('decode_peak_allocated_gib', 'prefill_peak_allocated_gib')]
            lines.append(f'| {model} | {length} | {crossings[0]} | {crossings[1]} |')
    def cell(value):
        return '-' if value is None else f'{value:.3f}'
    for model in config['models']:
        for length in config['contexts']:
            subset = [p for p in pairs if p['model'] == model and p['context'] == length]
            lines += ['', f'## {model} / {length}', '',
                      '| Batch | Dense decode peak GiB | Basis decode peak GiB | Saved GiB | Saved percent |',
                      '| ---: | ---: | ---: | ---: | ---: |']
            for p in subset:
                lines.append(f"| {p['batch']} | " + ' | '.join(cell(p[k]) for k in (
                    'dense_decode_peak_allocated_gib', 'basis_decode_peak_allocated_gib',
                    'decode_peak_saved_gib', 'decode_peak_saved_percent')) + ' |')
            lines += ['',
                      '| Batch | Dense status | Basis status | Dense ms | Basis ms | Speedup | Dense tokens/s | Basis tokens/s |',
                      '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
            for p in subset:
                lines.append(f"| {p['batch']} | {p['dense_status']} | {p['basis_status']} | "
                             + ' | '.join(cell(p[k]) for k in ('dense_decode_ms', 'basis_decode_ms',
                                 'latency_speedup', 'dense_tokens_per_second', 'basis_tokens_per_second')) + ' |')
            lines += ['', 'Auxiliary prefill peak allocated memory (GiB):', '',
                      '| Batch | Dense prefill peak | Basis prefill peak |',
                      '| ---: | ---: | ---: |']
            for p in subset:
                lines.append(f"| {p['batch']} | " + ' | '.join(cell(p[k]) for k in (
                    'dense_prefill_peak_allocated_gib', 'basis_prefill_peak_allocated_gib')) + ' |')
    lines += ['', '## Whole-Run Completion', '',
              'Largest successful tested batch through both prefill and decode, not a decode capacity bound.',
              'These points are constrained by this prefill implementation; do not infer that decode cannot fit.', '',
              '| Model | Context | Dense | Basis |', '| --- | ---: | ---: | ---: |']
    for model in config['models']:
        for length in config['contexts']:
            limits = [max((r['batch'] for r in indexed.values() if r['model'] == model
                           and r['length'] == length and r['arm'] == arm and r['status'] == 'complete'), default=0)
                      for arm in config['arms']]
            lines.append(f'| {model} | {length} | {limits[0]} | {limits[1]} |')
    lines += ['', '## OOM Evidence', '',
              'OOM is not a decode latency or successful capacity point. Stages below are bounded',
              'by the last emitted rank states; conditioning/decode are not separately instrumented.', '',
              '| Model | Context | Batch | Arm | Possible failure phases |',
              '| --- | ---: | ---: | --- | --- |']
    for r in indexed.values():
        if r['status'] == 'gpu_oom':
            lines.append(f"| {r['model']} | {r['length']} | {r['batch']} | {r['arm']} | "
                         + ', '.join(r['possible_failure_phases']) + ' |')
    lines += ['', '## Caveats and Artifacts', '',
              '- Single measurements do not provide run-to-run variance or quality equivalence.',
              '- Fixed serial order Dense then Basis; no graph/MLP/quantization changes.',
              '- Host NUMA placement is unverified; container set_mempolicy restrictions remain.',
              '- Peak RSS sums are historical process maxima, not simultaneous host-memory usage.',
              '- Source snapshot: `source.tar.gz`; existing experiment code was not changed during the sweep.',
              '- Per-trial commands/status/metrics: `formal/manifest.json` and `formal/summary.csv`.',
              '- Paired machine-readable table: `comparison.csv`; per-trial raw data and logs under `formal/`.',
              '- No SHA256 validation or generated-token equivalence check was performed.',
              '- See [README](README.md) for publication status, raw data links and reproduction requirements.']
    (args.output / 'RESULTS_SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'attempts': len(indexed), 'successes': completed, 'gpu_oom': len(indexed)-completed,
                      'validated_ranks': validated, 'summary': str(args.output / 'RESULTS_SUMMARY.md')}))


if __name__ == '__main__':
    main()
