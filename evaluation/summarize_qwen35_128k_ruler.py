"""Summarize Qwen3.5-9B 128K RULER arms from eval_qwen35_128k_ruler.py outputs and compare with the HF Page-Fisher release.

Per-task accuracy and the task-balanced mean for every arm present under ``--output``; paired bootstrap confidence
intervals between local arms on their common prompts; and, when ``--reference`` names the released
``ruler128k_summary.json`` (Page-Fisher B16R16, Full, Loki, LRQK, ShadowKV on the same 1100-prompt protocol), its
per-task numbers side by side. Reference arms are not paired with local arms: their per-sample outputs were not released.
"""
import argparse
import json
import random
from pathlib import Path

RULER_TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1', 'niah_multikey_2', 'niah_multiquery',
         'niah_multivalue', 'vt', 'fwe', 'qa_1', 'qa_2')


def load(root):
    out = {}
    for path in sorted(Path(root).glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('status') != 'complete':
            continue
        out[(record['task'], record['ordinal'])] = dict(score=float(record['score']), prefill=record['prefill_seconds'],
                                                        seconds=record['seconds'], peak=record['peak_gib'])
    return out


def task_mean(scores, keys):
    by_task = {}
    for task in TASKS:
        ks = [k for k in keys if k[0] == task]
        if ks:
            by_task[task] = 100 * sum(scores[k] for k in ks) / len(ks)
    return by_task, (sum(by_task.values()) / len(by_task) if by_task else float('nan'))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--bootstrap', type=int, default=4000)
    p.add_argument('--benchmark', choices=('ruler', 'longbench'), default='ruler')
    args = p.parse_args()
    global TASKS
    arms = {d.name: load(d) for d in sorted(args.output.iterdir()) if d.is_dir()}
    arms = {name: rows for name, rows in arms.items() if rows}
    TASKS = RULER_TASKS if args.benchmark == 'ruler' else tuple(dict.fromkeys(k[0] for rows in arms.values() for k in sorted(rows, key=lambda k: (0, k))))
    if args.benchmark == 'longbench':
        order = ('narrativeqa', 'multifieldqa_en', 'hotpotqa', 'musique', 'dureader', 'gov_report', 'samsum', 'passage_retrieval_en', 'lcc')
        TASKS = tuple(t for t in order if t in TASKS) + tuple(t for t in TASKS if t not in order)
    reference = json.loads(args.reference.read_text())['arms'] if args.reference else {}
    columns = [f'{name} (local)' for name in arms] + [f'{name} (HF PF release)' for name in reference]
    print(f"{'task':18s}" + ''.join(f'{c:>28s}' for c in columns))
    for task in TASKS:
        cells = []
        for name, rows in arms.items():
            ks = [k for k in rows if k[0] == task]
            cells.append(f'{100 * sum(rows[k]["score"] for k in ks) / len(ks):6.1f} (n={len(ks):3d})' if ks else '-')
        for name, arm in reference.items():
            cells.append(f'{100 * arm["tasks"][task]["accuracy"]:6.1f} (n={arm["tasks"][task]["samples"]:3d})')
        print(f'{task:18s}' + ''.join(f'{c:>28s}' for c in cells))
    cells = []
    for name, rows in arms.items():
        _, mean = task_mean({k: v['score'] for k, v in rows.items()}, list(rows))
        cells.append(f'{mean:6.2f} ({len(rows)} prompts)')
    for name, arm in reference.items():
        cells.append(f'{100 * arm["task_balanced_accuracy"]:6.2f} ({arm["samples"]} prompts)')
    print(f"{'task-balanced mean':18s}" + ''.join(f'{c:>28s}' for c in cells))
    if args.benchmark == 'longbench':
        six = [t for t in ('narrativeqa', 'multifieldqa_en', 'gov_report', 'samsum', 'passage_retrieval_en', 'lcc') if t in TASKS]
        cells = []
        for name, rows in arms.items():
            by, _ = task_mean({k: v['score'] for k, v in rows.items()}, list(rows))
            cells.append(f"{sum(by[t] for t in six) / len(six):6.2f} ({len(six)} tasks)" if all(t in by for t in six) else '-')
        print(f"{'LRQK 6-task mean':18s}" + ''.join(f'{c:>28s}' for c in cells))
    for name, rows in arms.items():
        prefill = sum(v['prefill'] for v in rows.values()) / len(rows)
        total = sum(v['seconds'] for v in rows.values()) / len(rows)
        peak = max(v['peak'] for v in rows.values())
        print(f'{name}: mean prefill {prefill:.1f} s, mean total {total:.1f} s, peak {peak:.1f} GiB')
    names = list(arms)
    random.seed(0)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            keys = sorted(set(arms[a]) & set(arms[b]))
            if not keys:
                continue
            sa = {k: arms[a][k]['score'] for k in keys}
            sb = {k: arms[b][k]['score'] for k in keys}
            _, ma = task_mean(sa, keys)
            _, mb = task_mean(sb, keys)
            diffs = []
            for _ in range(args.bootstrap):
                sample = [random.choice([k for k in keys if k[0] == t]) for t in TASKS for _ in range(sum(1 for k in keys if k[0] == t))]
                diffs.append(task_mean(sa, sample)[1] - task_mean(sb, sample)[1])
            diffs.sort()
            wins = sum(sa[k] > sb[k] for k in keys)
            losses = sum(sa[k] < sb[k] for k in keys)
            low, high = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
            print(f'{a} - {b}: {ma - mb:+.2f} 95% CI [{low:+.2f}, {high:+.2f}] win/loss {wins}/{losses} on {len(keys)} paired prompts')


if __name__ == '__main__':
    main()
