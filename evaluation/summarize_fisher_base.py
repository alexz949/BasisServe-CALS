"""Pilot table of the Fisher-trained Value Base experiment: per layer, fit / held-out Fisher NMSE (loss / exact-score energy) of
every arm from the fitter's records, and the held-out routing diagnostics (routed-page recall, page KL, retained mass).
usage: summarize_fisher_base.py <output dir of fit_k_routing_fisher_base.py> --layers 0,13,33"""
import argparse
import json
from pathlib import Path

ARMS = [('MSE-B16', 'ours_base_b16r16', 'base_only'), ('Fisher-B16', 'ours_base_fisher_b16r16', 'base_only'),
        ('R16-Fisher', 'ours_zero_b0r16', 'total'), ('MSE-B16 + R16-Fisher', 'ours_base_b16r16', 'total'),
        ('Fisher-B16 + R16-Fisher', 'ours_base_fisher_b16r16', 'total'), ('Joint-Fisher B16R16', 'ours_joint_b16r16', 'final'),
        ('R32-Fisher', 'ours_zero_b0r32', 'total')]
DIAG = {'R16-Fisher': 'r16_fisher', 'MSE-B16 + R16-Fisher': 'mse_b16_r16_fisher', 'Fisher-B16 + R16-Fisher': 'fisher_b16_r16_fisher',
        'Joint-Fisher B16R16': 'joint_fisher_b16r16', 'R32-Fisher': 'r32_fisher', 'Exact-K': 'exact_k'}


def nmse(record, kind):
    losses = record['losses']
    if kind == 'base_only':
        return losses['total']['base_only']['fit'], losses['total']['base_only']['heldout']
    if kind == 'final':
        return losses['final']['fit']['nmse'], losses['final']['heldout']['nmse']
    return losses['total']['fit']['nmse'], losses['total']['heldout']['nmse']


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--layers', required=True)
    args = p.parse_args()
    layers = [int(x) for x in args.layers.split(',')]
    means = {}
    for layer in layers:
        print(f"\n== layer {layer}")
        print(f"{'arm':28s}{'fit Fisher':>12s}{'val Fisher':>12s}{'page recall':>13s}{'page KL':>10s}{'mass':>8s}")
        diag = args.output / 'diagnostics' / f'layer_{layer:03d}.json'
        diagnostics = json.load(open(diag))['arms'] if diag.exists() else {}
        for name, stage, kind in ARMS + [('Exact-K', None, None)]:
            fit = val = None
            if stage is not None:
                path = args.output / stage / f'layer_{layer:03d}.json'
                if path.exists():
                    fit, val = nmse(json.load(open(path)), kind)
            d = diagnostics.get(DIAG.get(name, ''), {})
            row = f"{name:28s}" + (f"{fit:12.4f}{val:12.4f}" if fit is not None else f"{'-':>12s}{'-':>12s}")
            row += (f"{d['page_recall']:13.4f}{d['page_kl']:10.4f}{d['mass']:8.4f}" if d else f"{'-':>13s}{'-':>10s}{'-':>8s}")
            print(row)
            for key, value in (('fit', fit), ('val', val), ('page_recall', d.get('page_recall')), ('page_kl', d.get('page_kl')), ('mass', d.get('mass'))):
                if value is not None:
                    means.setdefault(name, {}).setdefault(key, []).append(value)
    print(f"\n== mean over layers {layers}")
    print(f"{'arm':28s}{'fit Fisher':>12s}{'val Fisher':>12s}{'page recall':>13s}{'page KL':>10s}{'mass':>8s}")
    for name, _, _ in ARMS + [('Exact-K', None, None)]:
        m = means.get(name, {})
        cell = lambda key, width: (f"{sum(m[key]) / len(m[key]):{width}.4f}" if m.get(key) else f"{'-':>{width}s}")
        print(f"{name:28s}{cell('fit', 12)}{cell('val', 12)}{cell('page_recall', 13)}{cell('page_kl', 10)}{cell('mass', 8)}")


if __name__ == '__main__':
    main()
