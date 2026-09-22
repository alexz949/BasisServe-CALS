#!/usr/bin/env python3
"""Audit and summarize the eight Qwen3.5 mixed-calibration eval shards."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.qwen35_hybrid_common import atomic_save


TASK_COUNTS = {
    'gsm8k': 1319,
    'minerva_math500': 500,
    'mbpp_plus_full': 378,
    'ifeval': 541,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def metric_values(rows, name, filter_name=None):
    values = []
    for row in rows:
        if filter_name is not None and row['filter'] != filter_name:
            continue
        value = row[name]
        if isinstance(value, list):
            values.extend(float(item) for item in value)
        else:
            values.append(float(value))
    assert values
    return values


def average(values):
    return sum(values) / len(values)


def summarize_task(task, payloads):
    count = TASK_COUNTS[task]
    expected = set(range(count))
    rows = []
    shard_records = []
    seen = set()
    for parity, payload in enumerate(payloads):
        assert payload['status'] == 'complete'
        assert payload['args']['task'] == task
        shard_ids = set(range(parity, count, 2))
        assert set(payload['args']['doc_ids']) == shard_ids
        samples = payload['evaluation']['samples'][task]
        sample_ids = {row['doc_id'] for row in samples}
        assert sample_ids == shard_ids and seen.isdisjoint(sample_ids)
        seen.update(sample_ids)
        rows.extend(samples)
        assert len(payload['generation_records']) == len(shard_ids)
        shard_records.append({
            'samples': len(shard_ids),
            'elapsed_seconds': payload['elapsed_seconds'],
            'length_capped': payload['length_capped'],
            'closing_think_responses': payload['closing_think_responses'],
        })
    assert seen == expected
    if task == 'gsm8k':
        metrics = {
            'exact_match_strict': average(metric_values(rows, 'exact_match', 'strict-match')),
            'exact_match_flexible': average(metric_values(rows, 'exact_match', 'flexible-extract')),
        }
    elif task == 'minerva_math500':
        metrics = {
            'exact_match': average(metric_values(rows, 'exact_match')),
            'math_verify': average(metric_values(rows, 'math_verify')),
        }
    elif task == 'mbpp_plus_full':
        metrics = {
            'base_pass_at_1': average(metric_values(rows, 'base_pass_at_1')),
            'plus_pass_at_1': average(metric_values(rows, 'plus_pass_at_1')),
        }
    else:
        metrics = {
            'prompt_level_strict_acc': average(metric_values(rows, 'prompt_level_strict_acc')),
            'instruction_level_strict_acc': average(metric_values(rows, 'inst_level_strict_acc')),
            'prompt_level_loose_acc': average(metric_values(rows, 'prompt_level_loose_acc')),
            'instruction_level_loose_acc': average(metric_values(rows, 'inst_level_loose_acc')),
        }
    return {
        'samples': count,
        'metrics': metrics,
        'length_capped': sum(record['length_capped'] for record in shard_records),
        'closing_think_responses': sum(record['closing_think_responses'] for record in shard_records),
        'parallel_elapsed_seconds': max(record['elapsed_seconds'] for record in shard_records),
        'shards': shard_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.input_dir)
    all_payloads = {}
    artifacts = {}
    provenance = None
    versions = None
    for task in TASK_COUNTS:
        payloads = []
        for parity_name in ('even', 'odd'):
            path = root / f'{task}_{parity_name}.json'
            payload = json.loads(path.read_text())
            payloads.append(payload)
            artifacts[path.name] = sha256(path)
            if provenance is None:
                provenance = payload['provenance']
                versions = dict(payload['versions'])
            assert payload['provenance'] == provenance
            for name, version in payload['versions'].items():
                if name in versions:
                    assert versions[name] == version
                else:
                    versions[name] = version
        all_payloads[task] = payloads
    summary = {
        'status': 'complete_and_audited',
        'protocol': {
            'model': 'Qwen/Qwen3.5-9B',
            'thinking': False,
            'seed': 20260909,
            'workers': 8,
            'sharding': 'even/odd original lm-eval doc_id per task',
            'tasks': TASK_COUNTS,
        },
        'provenance': provenance,
        'versions': versions,
        'results': {task: summarize_task(task, payloads)
                    for task, payloads in all_payloads.items()},
        'artifacts_sha256': artifacts,
    }
    atomic_save(args.output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
