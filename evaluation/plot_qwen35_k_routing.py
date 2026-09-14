"""Plot held-out Qwen3.5 K reconstruction and matched page recall."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--diagnostic', type=Path, required=True)
    args = p.parse_args()
    layers = (3, 7, 11, 15, 19, 23, 27, 31)
    data = {arm:[json.loads((args.diagnostic/arm/f'l{layer:02d}.json').read_text())
        for layer in layers] for arm in ('b16r16', 'b32r32')}
    for arm, rows in data.items():
        for layer, r in zip(layers, rows, strict=True):
            assert r['status'] == 'complete' and r['layer'] == layer
            assert r['budget'] == 2048 and r['recent_tokens'] == 64 and r['pinned_prefix_pages'] == 0
            assert [w['window'] for w in r['windows']] == list(range(64, 80))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for arm, label in (('b16r16', 'B16R16'), ('b32r32', 'B32R32')):
        axes[0].plot(layers, [r['relative_mse'] for r in data[arm]], 'o-', label=label)
        axes[1].plot(layers, [100*r['attention_mass'] for r in data[arm]], 'o-', label=label)
    axes[1].plot(layers, [100*r['exact_attention_mass'] for r in data['b16r16']],
        'k--', label='Exact-K, same page rule')
    for ax in axes:
        ax.set_xlabel('Full-attention layer (0-indexed)')
        ax.set_xticks(layers)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel('Held-out K relative MSE')
    axes[1].set_ylabel('Attention recall mass (%)')
    fig.suptitle('Qwen3.5-9B / allocated V192 / 16 x 32K held-out\nHard 2048 including recent64; no pinned sink')
    for suffix in ('png', 'pdf'):
        path = args.diagnostic/f'layer_curve.{suffix}'
        assert not path.exists()
        fig.savefig(path, dpi=180)
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
