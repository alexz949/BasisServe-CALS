"""Plot complete, matched held-out router diagnostics for Qwen3 and Qwen3.5."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--qwen3', type=Path, required=True)
    parser.add_argument('--qwen35', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    series = [
        ('Qwen3-8B B16R16', args.qwen3, 16, list(range(36))),
        ('Qwen3.5-9B B16R16', args.qwen35, 16, list(range(3, 32, 4))),
        ('Qwen3.5-9B B32R32', args.qwen35, 32, list(range(3, 32, 4))),
    ]
    metrics = ('relative_mse', 'attention_mass', 'non_sink_attention_mass')
    datasets, sources, implementations = [], {}, set()
    for label, root, rank, layers in series:
        rows = []
        for layer in layers:
            path = root/f'b{rank}r{rank}'/f'l{layer:02d}.json'
            raw = path.read_bytes()
            row = json.loads(raw)
            assert row['status'] == 'complete' and row['layer'] == layer and row['rank'] == rank
            assert (row['budget'], row['page_size'], row['pinned_prefix_pages']) == (2048, 32, 1)
            assert [w['window'] for w in row['windows']] == list(range(64, 80))
            assert len(row['positions']) == 32
            for name in (*metrics, 'exact_attention_mass', 'exact_non_sink_attention_mass'):
                assert math.isfinite(row[name]) and row[name] >= 0
                if name != 'relative_mse':
                    assert row[name] <= 1 + 1e-6
            code = row['code_sha256'] if root == args.qwen3 else row['routing_code_sha256']
            implementations.add(tuple(code[name] for name in (
                'evaluation/routing_diagnostics.py',
                'basisserve/core/qwen35_k_routing_runtime.py',
                'basisserve/core/c1_conditional_page_attention.py',
                'basisserve/core/c1_v_k_index.py')))
            sources[str(path)] = hashlib.sha256(raw).hexdigest()
            rows.append(row)
        datasets.append((label, layers, rows))
    assert len(implementations) == 1
    outputs = [args.output.with_suffix(suffix) for suffix in ('.png', '.pdf', '.json')]
    assert not any(path.exists() for path in outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for color, (label, layers, rows) in zip(('C0', 'C1', 'C2'), datasets):
        for ax, metric in zip(axes, metrics):
            ax.plot(layers, [row[metric] for row in rows], marker='o', markersize=3,
                color=color, label=label)
            if metric != 'relative_mse':
                ax.plot(layers, [row['exact_'+metric] for row in rows], color=color,
                    linestyle='--', alpha=0.65)
    for ax, title in zip(axes, ('Full Base + Residual K rel-MSE',
            'Attention mass retained', 'Non-sink attention mass retained')):
        ax.set(title=title, xlabel='Full-attention layer index (0-based)')
        ax.grid(alpha=0.2)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    for ax in axes[1:]:
        ax.set_ylim(0, 1.02)
    note = ('16 held-out 32K windows; 32 queries/window; 2048-token budget, Page32. '
        'Dashed: Exact-K selection with the same page rule.\n'
        'V protocols differ: Qwen3 uniform V96, 32 fit / 4 held-out, ALS6; '
        'Qwen3.5 two-sided KL mean V192, 64 fit / 16 held-out, ALS12.')
    fig.suptitle(note, fontsize=9)
    fig.savefig(outputs[0], dpi=200)
    fig.savefig(outputs[1])
    plt.close(fig)
    report = dict(status='complete', source_sha256=sources, comparison_note=note,
        series=[dict(label=label, layers=layers,
            metrics={name: [r[name] for r in rows] for name in metrics})
            for label, layers, rows in datasets])
    outputs[2].write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(outputs=list(map(str, outputs)), source_files=len(sources))))


if __name__ == '__main__':
    main()
