"""Audit matched Qwen3.5 non-thinking task runs and summarize their scores."""

import argparse
import json
import math
from pathlib import Path
import re

from evaluation.qwen35_hybrid_common import atomic_save, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/q35_hybrid/hard_tasks'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prior = json.loads(Path('results/q35_hybrid/gsm8k_vllm/v128_g768_f512_wo_summary.json').read_text())
    assert prior['status'] == 'complete_and_audited'
    arm_names = {'dense': 'dense', 'dense_wo': 'dense_g768_f512_wo',
                 'v128_wo': 'twosided128_g768_f512_wo'}
    trusted = {}
    for arm, old_arm in arm_names.items():
        row = next(r for r in prior['rows'] if r['arm'] == old_arm)
        path = Path(row['args']['output'])
        assert sha256(path) == row['source_sha256']
        trusted[arm] = json.loads(path.read_text())['provenance']
    rows = []
    for task, count in [('minerva_math500', 500), ('mbpp_plus_full', 378), ('ifeval', 541)]:
        if args.smoke:
            count = 2
        shared = None
        for arm in arm_names:
            path = args.root / ('smoke' if args.smoke else 'full') / f'{arm}_{task}.json'
            result = json.loads(path.read_text())
            assert result['status'] == 'complete' and result['thinking'] is False
            assert result['args']['task'] == task
            assert result['args']['limit'] == (2 if args.smoke else None)
            assert result['provenance'] == trusted[arm]
            for key in ('bank', 'wo_bank'):
                if result['args'][key]:
                    hash_key = 'v_bank_sha256' if key == 'bank' else 'wo_bank_sha256'
                    assert sha256(result['args'][key]) == result['provenance'][hash_key]
            samples = sorted(result['evaluation']['samples'][task], key=lambda s: s['doc_id'])
            records = sorted(result['generation_records'], key=lambda r: r['doc_id'])
            assert len(samples) == len(records) == count
            assert {s['doc_id'] for s in samples} == {r['doc_id'] for r in records} == set(range(count))
            assert all(r['original_prompt_tokens'] == r['retained_prompt_tokens'] for r in records)
            identity = {
                'records': [{k: r[k] for k in ('task', 'doc_id', 'sample_index', 'prompt_sha256',
                    'seed', 'original_prompt_tokens', 'retained_prompt_tokens', 'max_gen_toks', 'stop_strings')}
                    for r in records],
                'documents': [(s['doc_id'], s['doc'], s['target']) for s in samples],
                # Harness serializes nested function reprs with process-specific addresses.
                # Actual rendered prompts and targets are compared independently above.
                'config': re.sub(r'(<function [^<>]+) at 0x[0-9a-f]+>', r'\1>',
                                 json.dumps(result['evaluation']['configs'][task], sort_keys=True)),
                'versions': result['versions'],
            }
            if shared is None:
                shared = identity
            assert identity == shared
            metrics = {k: v for k, v in result['evaluation']['results'][task].items()
                       if ',' in k and 'stderr' not in k and isinstance(v, (int, float))}
            assert metrics
            for key, value in metrics.items():
                metric = key.split(',')[0]
                values = [s[metric] for s in samples]
                if metric.startswith('inst_level_'):
                    values = [v for sequence in values for v in sequence]
                recomputed = sum(values) / len(values)
                assert math.isfinite(value) and math.isclose(value, recomputed, abs_tol=1e-12)
            capped = sum(r['finish_reason'] == 'length' for r in records)
            assert capped == result['length_capped']
            rows.append({'task': task, 'arm': arm, 'questions': count, 'metrics': metrics,
                         'length_capped': capped, 'closing_think_responses': result['closing_think_responses'],
                         'elapsed_seconds': result['elapsed_seconds'], 'source': str(path),
                         'source_sha256': sha256(path)})
    output = args.root / ('smoke_summary.json' if args.smoke else 'summary.json')
    atomic_save(output, {'status': 'complete_and_audited',
                        'identical_prompts_targets_and_protocol': True, 'rows': rows})
    print(json.dumps({'output': str(output), 'rows': rows}), flush=True)


if __name__ == '__main__':
    main()
