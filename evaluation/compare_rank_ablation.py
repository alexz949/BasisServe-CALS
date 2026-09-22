"""Paired comparison of the rank-allocation ablation (B8R24, B0R32) against the main run's Full-K and B16R16 on the
same prompts, matched by (task, ordinal)."""
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/home/Ubuntu/nh8b_128k')
TASKS = ['niah_multikey_2', 'niah_multivalue', 'fwe', 'niah_multiquery']


def load(root):
    out = {}
    for p in root.glob('sample_*.json'):
        r = json.loads(p.read_text()); s = r['sample']
        if s['task'] in TASKS:
            out[(s['task'], s['ordinal'])] = float(r['result']['score'])
    return out


arms = {'Full-K': load(W / 'eval1100/full/evaluate'), 'B16R16': load(W / 'eval1100/ours/evaluate'),
        'LRQK': load(W / 'eval1100/lrqk/evaluate'), 'B8R24': load(W / 'eval_ablation/b8r24/ours/evaluate'),
        'B0R32': load(W / 'eval_ablation/b0r32/ours/evaluate')}
arms = {k: v for k, v in arms.items() if v}
common = sorted(set.intersection(*(set(v) for v in arms.values())))
print(f'配对样本 {len(common)}; arms {list(arms)}')
print(f"{'任务':16s}" + ''.join(f'{a:>9s}' for a in arms))
means = {a: [] for a in arms}
for t in TASKS:
    keys = [k for k in common if k[0] == t]
    row = [100 * np.mean([arms[a][k] for k in keys]) for a in arms]
    for a, v in zip(arms, row): means[a].append(v)
    print(f'{t:16s}' + ''.join(f'{v:9.1f}' for v in row) + f'   n={len(keys)}')
print(f"{'4 任务均分':16s}" + ''.join(f'{np.mean(means[a]):9.2f}' for a in arms))
rng = np.random.default_rng(0)
for a in ('B8R24', 'B0R32'):
    if a not in arms: continue
    for t in TASKS + ['all']:
        keys = [k for k in common if t == 'all' or k[0] == t]
        d = np.array([arms[a][k] - arms['B16R16'][k] for k in keys]); bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(5000)]
        print(f'  {a} − B16R16 [{t:16s}]: {100*d.mean():+.1f}  95% CI [{100*np.percentile(bs,2.5):+.1f}, {100*np.percentile(bs,97.5):+.1f}]  胜/负 {(d>0).sum()}/{(d<0).sum()}')
