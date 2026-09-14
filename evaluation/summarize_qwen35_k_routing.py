"""Audit all 7 x 88 paired RULER records before reporting task scores."""

import argparse
import json
from pathlib import Path

from evaluation.eval_qwen35_k_routing_ruler import ARMS, TASKS
from evaluation.qwen35_hybrid_common import atomic_save, sha256
from evaluation.ruler_v1 import TASK_BY_NAME, sample_score


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--loki-results', type=Path, help='Directory containing the corrected fixed-2048 Loki records')
    p.add_argument('--routing-results', type=Path, help='Root containing exact_sparse, b16r16 and b32r32 with recent64 and no pinned sink')
    p.add_argument('--output', type=Path)
    p.add_argument('--smoke', action='store_true', help='Audit the first paired question for all seven arms')
    args = p.parse_args()
    sample_count = 1 if args.smoke else 88
    tasks = TASKS[:1] if args.smoke else TASKS
    samples_per_task = 1 if args.smoke else 8
    records, hashes = {}, {}
    for arm in ARMS:
        rows = []
        for index in range(sample_count):
            directory = args.loki_results if arm == 'loki' and args.loki_results is not None else args.results/arm
            if arm in ('exact_sparse', 'b16r16', 'b32r32') and args.routing_results is not None:
                directory = args.routing_results/arm
            path = directory/f'{index:03d}.json'
            row = json.loads(path.read_text())
            assert row['status'] == 'complete' and row['index'] == index and row['protocol']['arm'] == arm
            assert row['task'] == TASKS[index//8] and row['ordinal'] == index%8
            task = TASK_BY_NAME[row['task']]
            assert row['input_tokens']+task.tokens_to_generate <= 65536
            assert 0 < len(row['generated_ids']) <= task.tokens_to_generate
            assert abs(row['score']-sample_score(row['prediction'], row['answers'], task.match_type)) < 1e-12
            if rows:
                assert row['protocol'] == rows[0]['protocol']
            rows.append(row)
            hashes[str(path)] = sha256(path)
        records[arm] = rows
    common_keys = ('v_bank_sha256', 'data_sha256', 'model_identity', 'prompt_format', 'thinking',
        'sequence_length', 'samples', 'wo_compression', 'prefill', 'selection')
    reference = records['full']
    # The fixed-budget Loki and no-sink/recent64 reruns change the evaluator and routing
    # dispatch module. Preserve their versions per arm; shared numerical
    # implementations must still match across the entire comparison.
    versioned_files = {'evaluation/eval_qwen35_k_routing_ruler.py',
        'basisserve/core/qwen35_k_routing_runtime.py'}
    assert records['loki'][0]['protocol']['loki']['topk'] == 2048
    for arm in ('exact_sparse', 'b16r16', 'b32r32'):
        protocol = records[arm][0]['protocol']
        assert protocol['ours_budget'] == 2048 and protocol['pinned_prefix_pages'] == 0
        assert protocol['recent_tokens'] == 64 and protocol['page_size'] == 32
    for arm, rows in records.items():
        for row, dense in zip(rows, reference, strict=True):
            assert all(row['protocol'][key] == dense['protocol'][key] for key in common_keys)
            code, dense_code = row['protocol']['code_sha256'], dense['protocol']['code_sha256']
            assert code.keys() == dense_code.keys()
            assert all(code[name] == dense_code[name] for name in code.keys()-versioned_files)
            assert row['input_ids_sha256'] == dense['input_ids_sha256'] and row['answers'] == dense['answers']
            assert row['first_logits_sha256'] == dense['first_logits_sha256'], (arm, row['index'])
    scores = {}
    for arm, rows in records.items():
        per_task = {task: 100*sum(row['score'] for row in rows if row['task'] == task)/samples_per_task for task in tasks}
        scores[arm] = dict(per_task=per_task, mean=sum(per_task.values())/len(tasks),
            total_seconds=sum(row['seconds'] for row in rows), peak_gib=max(row['peak_gib'] for row in rows))
    for arm in ARMS:
        scores[arm]['delta_vs_full_pp'] = scores[arm]['mean']-scores['full']['mean']
        scores[arm]['per_task_delta_vs_full_pp'] = {task: scores[arm]['per_task'][task]-scores['full']['per_task'][task] for task in tasks}
    report = dict(status='complete', scope='smoke' if args.smoke else 'formal',
        samples_per_arm=sample_count, records=sample_count*len(ARMS),
        protocols={arm: rows[0]['protocol'] for arm, rows in records.items()},
        code_audit='Shared numerical modules match; evaluator and routing dispatch versions recorded per arm after Loki budget and no-sink/recent64 corrections',
        first_logits_parity='bitwise for every paired sample across all arms', scores=scores, artifacts=hashes)
    output = args.output or args.results/'summary.json'
    if output.exists():
        assert json.loads(output.read_text()) == report
    else:
        atomic_save(output, report)
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == '__main__':
    main()
