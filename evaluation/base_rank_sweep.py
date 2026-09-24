"""How much Key energy the Base (V -> K affine reduced-rank map) explains as a function of Base rank.

Calibration-only statistic from the router fitter's saved FP64 moments (no RULER prompts involved): for each
layer and Base rank r, fit the affine reduced-rank map from V codes (the deployed V96 encoder, or the identity
for dense V) to pre-RoPE K on the fit windows and report the relative K reconstruction error on the held-out
calibration windows. 'centered explained' is the fraction of the mean-removed K energy the map explains.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.streaming_k_statistics import base_from_moments, base_mse


def split(payload, name):
    return {k.removeprefix(name + '_'): v for k, v in payload.items() if k.startswith(name + '_') and not k.startswith(name + '_post')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--router', type=Path, required=True, help='router fit dir with moments/ and base/ (base holds the V96 encoder)')
    p.add_argument('--ranks', type=int, nargs='+', default=[4, 8, 16, 24, 32, 48, 64, 96])
    p.add_argument('--label', required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    layers = sorted(int(f.stem.split('_')[1]) for f in (args.router / 'moments').glob('layer_*.safetensors'))
    report = {}
    for layer in layers:
        payload = load_file(str(args.router / 'moments' / f'layer_{layer:03d}.safetensors'))
        encoder = load_file(str(args.router / 'base' / f'layer_{layer:03d}.safetensors'))['encoder'].double()
        identity = torch.eye(encoder.shape[1], dtype=torch.float64).expand(encoder.shape[0], -1, -1).clone()
        fit, held = split(payload, 'fit'), split(payload, 'heldout')
        if int(held['count']) == 0:
            # Router fits with --diagnostic-count 0 replay no held-out windows; the Base has 96x16+16x128 parameters
            # per head fit on 32 x 128K tokens, so the in-sample statistic is the population value for practical purposes.
            held, evaluation_split = fit, 'fit (in-sample; no held-out moments captured)'
        else:
            evaluation_split = 'heldout'
        energy = float(held['kk'].diagonal(dim1=-2, dim2=-1).sum())
        centered = energy - float((held['sum_k'].square().sum(-1) / int(held['count'])).sum())
        row = dict(split=evaluation_split, key_energy=energy, mean_only_relative_mse=centered / energy)
        for name, enc, ranks in (('v96', encoder, [r for r in args.ranks if r <= encoder.shape[-1]]),
                                 ('dense_v', identity, [r for r in args.ranks if r <= identity.shape[-1]] + [identity.shape[-1]])):
            for r in sorted(set(ranks)):
                rel = base_mse(held, enc, base_from_moments(fit, enc, rank=r)[r])['relative_mse']
                row[f'{name}_r{r}'] = dict(relative_mse=rel, centered_explained=max(0.0, 1 - rel * energy / centered))
        report[str(layer)] = row
        print(f"{args.label} L{layer}: mean-only {row['mean_only_relative_mse']:.3f} | v96 r16 rel {row['v96_r16']['relative_mse']:.3f} "
              f"expl {row['v96_r16']['centered_explained']:.2f} | dense r16 expl {row['dense_v_r16']['centered_explained']:.2f} "
              f"dense r128 expl {row[f'dense_v_r{identity.shape[-1]}']['centered_explained']:.2f}", flush=True)
    keys = [k for k in next(iter(report.values())) if k.endswith(tuple(f'_r{r}' for r in args.ranks + [128]))]
    summary = {k: dict(mean_explained=statistics.mean(report[l][k]['centered_explained'] for l in report),
                       median_explained=statistics.median(report[l][k]['centered_explained'] for l in report),
                       mean_relative_mse=statistics.mean(report[l][k]['relative_mse'] for l in report)) for k in keys}
    print(f'\n== {args.label}: K energy explained by the Base (centered, {next(iter(report.values()))["split"]}), mean over {len(report)} layers')
    print('   rank :', ' '.join(f'{r:>5d}' for r in args.ranks))
    for name in ('v96', 'dense_v'):
        print(f'   {name:8s}', ' '.join(f"{100*summary[f'{name}_r{r}']['mean_explained']:5.1f}" if f'{name}_r{r}' in summary else '    -' for r in args.ranks), '%')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(label=args.label, router=str(args.router), layers=report, summary=summary), indent=1))


if __name__ == '__main__':
    main()
