"""Plot captured exact attention mass versus log2(page size) for the router and the exact-QK oracle."""
import json, math, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
P, budgets, o = d['page_sizes'], d['budgets'], d['overall']
fig, axes = plt.subplots(1, len(budgets), figsize=(5.2 * len(budgets), 4), sharey=False)
for ax, b in zip(axes, budgets):
    for scorer, style in (('oracle', 'o-'), ('router', 's--')):
        ax.plot([math.log2(p) for p in P], [o[f'{scorer}|P{p}|B{b}'] for p in P], style,
                label=f'{scorer} (exact QK page scores)' if scorer == 'oracle' else 'B16R16 router (fixed, no refit)')
    ax.set_xticks([math.log2(p) for p in P]); ax.set_xticklabels([str(p) for p in P])
    ax.set_xlabel('page size P (K = B / P pages)'); ax.set_ylabel('captured exact attention mass')
    ax.set_title(f'B = {b} routed tokens (+ sink 32 + recent 64)'); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.suptitle(f"Llama-3.1-8B-Instruct V96, C4 64K held-out ({len(d['windows'])} windows x {d['queries_per_window']} queries, {len({k.split('|')[0] for k in d['per_layer']})} layers)", fontsize=10)
fig.tight_layout(); fig.savefig(dst, dpi=150)
print(dst)
