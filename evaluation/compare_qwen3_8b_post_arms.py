"""Paired 11-task comparison on the Qwen3-8B (post-trained) 128K frozen prompts: main five arms plus the B8R24 rerun, matched by (task, ordinal)."""
import json
from pathlib import Path

import numpy as np

W = Path('/home/Ubuntu/q3_8b_post_128k')
ORDER = ['niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1', 'niah_multikey_2', 'niah_multiquery',
         'niah_multivalue', 'vt', 'fwe', 'qa_1', 'qa_2']


def load(root):
    out = {}
    for p in root.glob('sample_*.json'):
        r = json.loads(p.read_text()); s = r['sample']; out[(s['task'], s['ordinal'])] = float(r['result']['score'])
    return out


arms = {'Full-K': load(W / 'eval1100/full/evaluate'), 'B16R16': load(W / 'eval1100/ours/evaluate'),
        'B8R24': load(W / 'eval1100_b8r24/ours/evaluate'), 'LRQK': load(W / 'eval1100/lrqk/evaluate'),
        'ShadowKV': load(W / 'eval1100/shadowkv/evaluate'), 'Loki': load(W / 'eval1100/loki/evaluate')}
arms = {k: v for k, v in arms.items() if v}
common = sorted(set.intersection(*(set(v) for v in arms.values())))
print(f'配对样本 {len(common)}; arms {list(arms)}')
print(f"{'任务':16s}" + ''.join(f'{a:>9s}' for a in arms))
means = {a: [] for a in arms}
for t in ORDER:
    keys = [k for k in common if k[0] == t]
    if not keys: continue
    row = [100 * np.mean([arms[a][k] for k in keys]) for a in arms]
    for a, v in zip(arms, row): means[a].append(v)
    print(f'{t:16s}' + ''.join(f'{v:9.1f}' for v in row) + f'   n={len(keys)}')
print(f"{'RULER 均分':16s}" + ''.join(f'{np.mean(means[a]):9.2f}' for a in arms))
rng = np.random.default_rng(0)
def ci(a, b):
    d = np.array([arms[a][k] - arms[b][k] for k in common]); bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
    return f'{100*d.mean():+.2f}  95% CI [{100*np.percentile(bs,2.5):+.2f}, {100*np.percentile(bs,97.5):+.2f}]  胜/负 {(d>0).sum()}/{(d<0).sum()}'
for a, b in (('B8R24', 'B16R16'), ('B8R24', 'Full-K'), ('B8R24', 'LRQK'), ('B8R24', 'Loki'), ('B8R24', 'ShadowKV'), ('B16R16', 'Full-K'), ('LRQK', 'Full-K'), ('Loki', 'Full-K'), ('ShadowKV', 'Full-K')):
    if a in arms and b in arms: print(f'{a:8s} − {b:8s}: {ci(a, b)}')
