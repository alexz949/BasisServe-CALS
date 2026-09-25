"""Export evaluator sample records into the compact results layout used under results/: predictions/<arm>.jsonl (prompt token ids
stripped, protocol removed), protocols/<arm>.json (from the first record), README.md with a task-balanced summary table.
usage: export_eval_records.py <results_dir> <title> name=<.../arm/evaluate> [name=...] [--summary FILE ...] [--note TEXT]
Arms already present in <results_dir>/predictions are kept and included in the table."""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path


def compact(record):
    if 'sample' not in record:   # flat layout (Qwen3.5 evaluator): task/score at top level, no prompt ids stored
        return {k: v for k, v in record.items() if k != 'protocol'}
    sample = dict(record['sample'])
    ids = sample.pop('input_ids', None)
    if ids is not None:
        sample.setdefault('input_tokens', len(ids))
        sample.setdefault('input_sha256', hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    out = {k: v for k, v in record.items() if k not in ('protocol', 'sample')}
    out['sample'] = sample
    return out


def export_arm(results, name, source):
    paths = sorted(Path(source).glob('sample_*.json'), key=lambda p: int(p.stem.split('_')[1])) or sorted(Path(source).glob('[0-9]*.json'))
    records = [json.load(open(p)) for p in paths]
    assert records, source
    (results / 'protocols').mkdir(parents=True, exist_ok=True); (results / 'predictions').mkdir(parents=True, exist_ok=True)
    (results / 'protocols' / f'{name}.json').write_text(json.dumps(records[0]['protocol'], indent=1))
    with open(results / 'predictions' / f'{name}.jsonl', 'w') as f:
        for r in records: f.write(json.dumps(compact(r)) + '\n')


def table(results):
    rows = []
    for path in sorted((results / 'predictions').glob('*.jsonl')):
        by = defaultdict(list)
        for line in open(path):
            r = json.loads(line); s_, res = (r['sample'], r['result']) if 'sample' in r else (r, r)
            by[s_['task']].append(100 * res['score'])
        means = {t: sum(v) / len(v) for t, v in by.items()}
        rows.append((path.stem, sum(len(v) for v in by.values()), len(by), sum(means.values()) / len(means)))
    return rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('results', type=Path); p.add_argument('title'); p.add_argument('arms', nargs='*')
    p.add_argument('--summary', type=Path, nargs='*', default=[]); p.add_argument('--note', default='')
    a = p.parse_args()
    for arm in a.arms:
        name, source = arm.split('=', 1); export_arm(a.results, name, source)
    for s in a.summary:
        (a.results / 'summaries').mkdir(exist_ok=True); (a.results / 'summaries' / s.name).write_text(s.read_text())
    rows = table(a.results)
    first = json.load(open(a.results / 'protocols' / f'{rows[0][0]}.json'))
    excerpt = {k: first[k] for k in ('sequence_length', 'ours', 'samples_per_task', 'generation', 'benchmark', 'ours_budget') if k in first}
    lines = [f'# {a.title}', '', 'Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). '
             'Full protocol per arm is in `protocols/`; paired comparison tables in `summaries/`.', '']
    if a.note: lines += [a.note, '']
    lines += ['| arm | samples | tasks | task-balanced mean |', '|---|---:|---:|---:|'] + [f'| {n} | {s} | {t} | {m:.2f} |' for n, s, t, m in rows]
    lines += ['', 'Protocol excerpt (first arm):', '', '```json', json.dumps(excerpt, indent=1), '```', '']
    (a.results / 'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:4 + len(rows)]))


if __name__ == '__main__':
    main()
