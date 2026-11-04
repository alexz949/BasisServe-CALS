"""Score the Qwen3-32B Table-2 RULER run on the first 10 prompts per task (100 prompts).

Full-K: prompts 0-83 and 85-87 from the device_map pipeline run (`ruler_n10/`), the remaining 13 from the head-sharded
TP2 run (`ruler_n10_tp2/`); BasisKV and the exact-QK oracle (exact_sparse): all 100 from the TP2 run; BasisKV with
per-query-head selection: all 100 from `ruler_n10_tp2_qh/`. The runs are separate protocols (TP size, evaluator hash,
selection rule), so the evaluator's own `summarize` cannot combine them; this script checks every record against the
frozen prompts and writes `ruler_n10_combined.{json,md}`. Arms whose runs are absent are skipped.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ('niah_single_1', 'niah_single_2', 'niah_multikey_1', 'niah_multikey_2', 'niah_multiquery', 'niah_multivalue',
         'qa_1', 'qa_2', 'vt', 'fwe')
LABELS = ('S1', 'S2', 'MK1', 'MK2', 'MQ', 'MV', 'QA1', 'QA2', 'VT', 'FWE')
RUNS = {'pipeline': ROOT / 'ruler_n10', 'tp2': ROOT / 'ruler_n10_tp2', 'tp2_qh': ROOT / 'ruler_n10_tp2_qh'}
# Reported arm -> (run, evaluator arm) sources; tp2_qh is BasisKV with per-query-head selection (--ours-selection head).
SOURCES = {'full': (('pipeline', 'full'), ('tp2', 'full')), 'exact_sparse': (('tp2', 'exact_sparse'),),
           'ours': (('tp2', 'ours'),), 'ours_qh': (('tp2_qh', 'ours'),)}


def records(arm):
    found = {}
    for source, evaluator_arm in SOURCES[arm]:
        for path in sorted((RUNS[source] / evaluator_arm / 'evaluate').glob('sample_*.json')):
            saved = json.loads(path.read_text())
            assert saved['status'] == 'complete'
            assert saved['protocol'].get('tensor_parallel', 1) == (1 if source == 'pipeline' else 2)
            index = saved['sample']['index']
            assert index not in found, f'{arm} prompt {index} appears in both runs'
            found[index] = dict(source=source, **saved)
    return found


def main():
    data = ROOT / 'ruler_data_10'
    rows = []
    for task in TASKS:
        lines = (data / task / 'validation.jsonl').read_text().splitlines()
        rows += [(task, ordinal, json.loads(line)) for ordinal, line in enumerate(lines)]
    names = {'full': 'Full-K', 'exact_sparse': 'Oracle exact-QK (Page4, GQA max)', 'ours': 'BasisKV (B16R16 Page4)',
             'ours_qh': 'BasisKV per query head (2048 each)'}
    arms = {arm: records(arm) for arm in names if all((RUNS[r] / a / 'evaluate').exists() for r, a in SOURCES[arm])}
    for arm, found in arms.items():
        assert sorted(found) == list(range(len(rows))), f'{arm}: {len(found)} of {len(rows)} prompts'
        for index, (task, ordinal, source) in enumerate(rows):
            sample = found[index]['sample']
            assert (sample['task'], sample['ordinal']) == (task, ordinal) and sample['answers'] == source['outputs']
    per_task = {arm: {task: 100 * sum(found[i]['result']['score'] for i, (t, _, _) in enumerate(rows) if t == task)
                      / sum(t == task for t, _, _ in rows) for task in TASKS} for arm, found in arms.items()}
    means = {arm: sum(values.values()) / len(TASKS) for arm, values in per_task.items()}

    def paired(arm, reference):
        diff = [arms[arm][i]['result']['score'] - arms[reference][i]['result']['score'] for i in range(len(rows))]
        ids = [(arms[arm][i]['result']['ids'], arms[reference][i]['result']['ids']) for i in range(len(rows))]
        return dict(delta=means[arm] - means[reference], wins=sum(d > 0 for d in diff), losses=sum(d < 0 for d in diff),
                    lost_prompts=[i for i, d in enumerate(diff) if d < 0], won_prompts=[i for i, d in enumerate(diff) if d > 0],
                    identical_generations=sum(a == b for a, b in ids), first_token_agreement=sum(a[0] == b[0] for a, b in ids))
    pairs = {f'{arm}_vs_{ref}': paired(arm, ref) for arm, ref in (('ours', 'full'), ('exact_sparse', 'full'), ('ours', 'exact_sparse'),
                                                                     ('ours_qh', 'full'), ('ours_qh', 'ours'))
             if arm in arms and ref in arms}
    sources = {arm: {s: sum(r['source'] == s for r in found.values()) for s in RUNS} for arm, found in arms.items()}
    seconds = {arm: {s: sum(r['result']['seconds'] for r in found.values() if r['source'] == s)
                     / max(sources[arm][s], 1) for s in RUNS if sources[arm][s]} for arm, found in arms.items()}
    # Physical KV-group support at the last decode step, mean over prompts and layers (recorded by the per-head runs).
    union = {arm: sum(sum(layer['physical_union_per_kv_group_mean'] for layer in r['result']['routing'])
                      / len(r['result']['routing']) for r in found.values()) / len(found)
             for arm, found in arms.items() if 'physical_union_per_kv_group_mean' in found[0]['result']['routing'][0]}
    summary = dict(per_task=per_task, ten_task_mean=means, paired=pairs, prompts=len(rows), sources=sources,
                   mean_seconds=seconds, physical_union_per_kv_group_mean=union)
    (ROOT / 'ruler_n10_combined.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Qwen3-32B RULER-128K, Table-2 protocol, first 10 prompts per task (100)', '',
             '| Method | ' + ' | '.join(LABELS) + ' | Avg. |', '|---|' + '---:|' * (len(TASKS) + 1)]
    for arm in arms:
        lines.append(f'| {names[arm]} | ' + ' | '.join(f'{per_task[arm][t]:.2f}' for t in TASKS) + f' | {means[arm]:.2f} |')
    lines.append('')
    for key, pair in pairs.items():
        lines.append(f"{key}: {pair['delta']:+.2f} (paired wins/losses {pair['wins']}/{pair['losses']}; identical generations "
                     f"{pair['identical_generations']}/{len(rows)}; first-token agreement {pair['first_token_agreement']}/{len(rows)}).")
    for arm, value in union.items():
        lines.append(f'{arm}: physical KV-group support {value:.1f} tokens (last decode step, mean over prompts and layers).')
    lines.append(f'Sources: {sources}; mean seconds per prompt {seconds}.')
    (ROOT / 'ruler_n10_combined.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
