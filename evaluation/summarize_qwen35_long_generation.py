"""Audit full V128+Wo task reruns with increased generation limits."""
import json
import math
from pathlib import Path

from evaluation.qwen35_hybrid_common import atomic_save, sha256


def main():
    root = Path('results/q35_hybrid/hard_tasks')
    output = root / 'long_generation_summary.json'
    assert not output.exists()
    trusted = json.loads((root / 'summary.json').read_text())
    summaries = []
    for task, count, cap, metric in [('minerva_math500', 500, 8192, 'math_verify'),
                                     ('mbpp_plus_full', 378, 4096, 'plus_pass_at_1')]:
        old_path = root / 'full' / f'v128_wo_{task}.json'
        new_path = root / 'full' / f'v128_wo_{task}_{cap}.json'
        trusted_row = next(r for r in trusted['rows'] if r['task'] == task and r['arm'] == 'v128_wo')
        assert sha256(old_path) == trusted_row['source_sha256']
        old, new = [json.loads(p.read_text()) for p in (old_path, new_path)]
        assert old['provenance'] == new['provenance'] and old['versions'] == new['versions']
        for key, value in old['args'].items():
            if key not in ('max_new_tokens', 'max_model_len', 'output'):
                assert new['args'][key] == value, key
        assert new['args']['max_new_tokens'] == cap
        arm_metrics, indexed = {}, {}
        for name, result in [('original', old), ('long', new)]:
            assert result['status'] == 'complete' and result['thinking'] is False
            samples = {s['doc_id']: s for s in result['evaluation']['samples'][task]}
            records = {r['doc_id']: r for r in result['generation_records']}
            assert len(result['generation_records']) == len(result['evaluation']['samples'][task]) == count
            assert set(samples) == set(records) == set(range(count))
            for i, record in records.items():
                assert record['original_prompt_tokens'] == record['retained_prompt_tokens']
                assert record['prompt_sha256'] == samples[i]['prompt_hash']
                assert record['max_gen_toks'] == result['args']['max_new_tokens']
            metrics = {k: v for k, v in result['evaluation']['results'][task].items()
                       if ',' in k and 'stderr' not in k and isinstance(v, (int, float))}
            for key, value in metrics.items():
                assert math.isclose(value, sum(s[key.split(',')[0]] for s in samples.values()) / count, abs_tol=1e-12)
            capped = sum(r['finish_reason'] == 'length' for r in records.values())
            assert capped == result['length_capped']
            arm_metrics[name] = dict(metrics=metrics, capped=capped,
                generated_tokens=sum(r['generated_tokens'] for r in records.values()),
                closing_think_responses=result['closing_think_responses'], elapsed_seconds=result['elapsed_seconds'])
            indexed[name] = (samples, records)
        transitions = []
        for i in range(count):
            a, ar = indexed['original'][0][i], indexed['original'][1][i]
            b, br = indexed['long'][0][i], indexed['long'][1][i]
            for key in ('doc', 'target', 'doc_hash', 'prompt_hash', 'target_hash'):
                assert a[key] == b[key]
            for key in ('task', 'doc_id', 'sample_index', 'prompt_sha256', 'seed',
                        'original_prompt_tokens', 'retained_prompt_tokens', 'stop_strings'):
                assert ar[key] == br[key]
            short_text, long_text = a['resps'][0][0], b['resps'][0][0]
            assert isinstance(short_text, str) and isinstance(long_text, str)
            transitions.append(dict(doc_id=i, original_correct=bool(a[metric]), long_correct=bool(b[metric]),
                original_capped=ar['finish_reason'] == 'length', long_capped=br['finish_reason'] == 'length',
                original_tokens=ar['generated_tokens'], long_tokens=br['generated_tokens'],
                exact_short_text_prefix=long_text.startswith(short_text), identical_text=long_text == short_text))
        improved = [t for t in transitions if not t['original_correct'] and t['long_correct']]
        regressed = [t for t in transitions if t['original_correct'] and not t['long_correct']]
        summaries.append(dict(task=task, questions=count, primary_metric=metric, arms=arm_metrics,
            original_source=str(old_path), original_sha256=sha256(old_path), long_source=str(new_path),
            long_sha256=sha256(new_path), improved=len(improved), regressed=len(regressed),
            improved_from_capped=sum(t['original_capped'] for t in improved),
            improved_from_capped_with_exact_prefix=sum(t['original_capped'] and t['exact_short_text_prefix'] for t in improved),
            transitions=transitions))
    atomic_save(output, dict(status='complete_and_audited', identical_prompts_targets_versions_and_factors=True,
        caveat='Fresh generation under a larger limit; altered batching or model-length settings can change trajectories. Inspect text-prefix identity before attributing individual changes to continuation.',
        rows=summaries))
    print(json.dumps([{k: v for k, v in r.items() if k != 'transitions'} for r in summaries]), flush=True)


if __name__ == '__main__':
    main()
