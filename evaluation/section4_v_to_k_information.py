"""Section 4, experiment 1: how much of the pre-RoPE Key is affinely predictable from the Value representation.
The deployed model (identity checkpoint: compressed V and decoder installed, as for the router fit) is replayed on the
calibration windows; at every attention layer the dense Value v = W_v h (raw V, [tokens, groups x d]) and the pre-RoPE Key
k = W_k h (after k_norm where the model has one) are accumulated as FP64 sums and Grams of z = [v | k] over all tokens of every
window, separately for the fit and the held-out (validation) windows.  Stage `moments` writes one shard of windows; stage `solve`
merges the shards and, for every layer and KV group g, fits four affine predictors of K_g on the fit-window statistics:
    local compressed V   x = v_g E_g          (E_g: the checkpoint's value-coordinate encoder, d -> r)
    all compressed V     x = [v_1 E_1, ..., v_G E_G]
    local raw V          x = v_g
    all raw V            x = [v_1, ..., v_G]
and evaluates each strictly on the held-out windows as the centered explained energy
    explained = 1 - ||K_c - Khat_c||_F^2 / ||K_c||_F^2       (K_c centered with the held-out mean),
computed exactly from the held-out sums and Grams (the same statistic as evaluation/eval_v_key_information.py).
Outputs: result.json, summary.md, v_to_k_by_layer.pdf, v_to_k_by_layer_group.pdf."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from evaluation.fit_k_routing_streaming import encoder_for, install_deployed_teacher
from evaluation.k_routing_config import routing_config
from evaluation.v96kl_common import configure, read_json, sha256, write_json

VARIANTS = ('local_compressed_v', 'all_compressed_v', 'local_raw_v', 'all_raw_v')
LABELS = {'local_compressed_v': 'local compressed V', 'all_compressed_v': 'all-group compressed V',
          'local_raw_v': 'local raw V', 'all_raw_v': 'all-group raw V'}
CHUNK = 16384


def window_plan(manifest, smoke):
    fit_ids = [int(i) for i in manifest['fit_ids']]
    heldout_ids = [int(i) for i in manifest['validation_ids']]
    if smoke:
        fit_ids, heldout_ids = fit_ids[:1], heldout_ids[:1]
    return fit_ids, heldout_ids


@torch.inference_mode()
def moments(args, identity, manifest):
    config = routing_config(identity, rope=args.rope, sequence_length=args.sequence_length)
    assert config.model_type in ('llama', 'qwen3')
    model = AutoModelForCausalLM.from_pretrained(identity['model'], config=config, dtype=torch.bfloat16,
                                                 attn_implementation='sdpa', local_files_only=True).eval().cuda()
    dense_value_weights = {i: m.v_proj.weight.detach().clone() for i, m in c1_attention_layers(model)}
    install_deployed_teacher(model, SimpleNamespace(dense_v=False, native_audit=None, wo_bank=None, identity=args.identity), identity)
    modules = dict(c1_attention_layers(model))
    layers = [int(x) for x in args.layers.split(',')] if args.layers else sorted(modules)
    groups, dim = identity['hkv'], identity['head_dim']
    width = 2 * groups * dim
    fit_ids, heldout_ids = window_plan(manifest, args.smoke)
    plan = [('fit', i) for i in fit_ids] + [('heldout', i) for i in heldout_ids]
    plan = plan[args.window_shard::args.window_shards]
    windows = load_file(str(args.windows / 'windows.safetensors'))['input_ids']
    stats = {(split, layer): dict(gram=torch.zeros(width, width, dtype=torch.float64, device='cuda'),
                                  sum=torch.zeros(width, dtype=torch.float64, device='cuda'), count=0)
             for split in ('fit', 'heldout') for layer in layers}
    active = {}

    def capture(attention, positional, kwargs, layer=None):
        x = kwargs['hidden_states'] if 'hidden_states' in kwargs else positional[0]
        n = x.shape[1]
        k = attention.k_proj(x).view(1, n, groups, dim)
        if config.model_type == 'qwen3':
            k = attention.k_norm(k)
        v = torch.nn.functional.linear(x, dense_value_weights[layer]).view(1, n, groups, dim)
        z = torch.cat((v.reshape(n, groups * dim), k.reshape(n, groups * dim)), -1)      # [tokens, 2 G d], pre-RoPE
        s = stats[(active['split'], layer)]
        for start in range(0, n, CHUNK):
            c = z[start:start + CHUNK].float()
            s['gram'] += (c.mT @ c).double()
            s['sum'] += c.sum(0).double()
        s['count'] += n

    handles = [modules[layer].register_forward_pre_hook(lambda m, a, kw, layer=layer: capture(m, a, kw, layer=layer), with_kwargs=True)
               for layer in layers]
    for split, index in plan:
        active['split'] = split
        started = time.monotonic()
        out = model.model(windows[index:index + 1, :args.sequence_length].long().cuda(), use_cache=False)
        assert torch.isfinite(out.last_hidden_state).all()
        del out
        print(json.dumps(dict(stage='moments', shard=args.window_shard, split=split, window=index,
                              seconds=round(time.monotonic() - started, 1), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))), flush=True)
    for h in handles:
        h.remove()
    tensors = {}
    for (split, layer), s in stats.items():
        assert torch.isfinite(s['gram']).all()
        tensors[f'{split}_gram_{layer:03d}'] = s['gram'].cpu()
        tensors[f'{split}_sum_{layer:03d}'] = s['sum'].cpu()
        tensors[f'{split}_count_{layer:03d}'] = torch.tensor(s['count'], dtype=torch.int64)
    out = args.output / 'moments'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'shard{args.window_shard:02d}.safetensors'
    assert not path.exists(), path
    save_file(tensors, str(path), metadata=dict(json=json.dumps(dict(shard=args.window_shard, shards=args.window_shards, plan=plan,
        layers=layers, windows_sha256=sha256(args.windows / 'windows.safetensors'), identity_sha256=sha256(args.identity),
        sequence_length=args.sequence_length, smoke=args.smoke))))
    print(json.dumps(dict(stage='moments', shard=args.window_shard, windows=len(plan), file=str(path))), flush=True)


def merge(args, layers):
    parts = sorted((args.output / 'moments').glob('shard*.safetensors'))
    assert len(parts) == args.window_shards, [p.name for p in parts]
    from safetensors import safe_open
    merged, covered, meta, shards = {}, [], None, []
    for path in parts:
        for k, v in load_file(str(path)).items():
            merged[k] = merged.get(k, 0) + (v.double() if v.dtype != torch.int64 else v)
        with safe_open(str(path), 'pt') as s:
            m = json.loads(s.metadata()['json'])
        if meta is None:
            meta = {k: m[k] for k in ('windows_sha256', 'identity_sha256', 'sequence_length', 'smoke', 'layers')}
        assert all(m[k] == meta[k] for k in meta), path
        covered += [tuple(x) for x in m['plan']]
        shards.append(dict(file=path.name, sha256=sha256(path), shard=m['shard'], plan=m['plan']))
    assert sorted(set(covered)) == sorted(covered), 'duplicate windows across shards'
    return merged, meta, covered, shards


def solve_variant(gram_fit, sum_fit, n_fit, gram_ho, sum_ho, n_ho, feature_map, k_slice, ridge=1e-8):
    """feature_map M: [F, V]; x = M v.  Returns (fit explained, held-out explained) for the affine predictor of k[k_slice]."""
    def blocks(gram, s):
        vv, vk, kk = gram[:V, :V], gram[:V, k_slice], gram[k_slice, k_slice]
        return feature_map @ vv @ feature_map.mT, feature_map @ vk, kk, feature_map @ s[:V], s[k_slice]
    V = feature_map.shape[1]
    G_xx, G_xk, G_kk, s_x, s_k = blocks(gram_fit, sum_fit)
    mean_x, mean_k = s_x / n_fit, s_k / n_fit
    C_xx = G_xx - torch.outer(s_x, s_x) / n_fit
    C_xk = G_xk - torch.outer(s_x, s_k) / n_fit
    C_xx = C_xx + ridge * C_xx.diagonal().mean() * torch.eye(C_xx.shape[0], dtype=C_xx.dtype)
    W = torch.linalg.solve(C_xx, C_xk)                                   # [F, d]
    b = mean_k - W.mT @ mean_x

    def explained(gram, s, n):
        G_xx, G_xk, G_kk, s_x, s_k = blocks(gram, s)
        sse = (G_kk.trace() - 2 * (W.mT @ G_xk).trace() - 2 * b @ s_k + (W.mT @ G_xx @ W).trace()
               + 2 * b @ (W.mT @ s_x) + n * b @ b)
        energy = G_kk.trace() - (s_k @ s_k) / n
        return float(1 - sse / energy)
    return explained(gram_fit, sum_fit, n_fit), explained(gram_ho, sum_ho, n_ho)


def solve(args, identity, manifest):
    layers = [int(x) for x in args.layers.split(',')] if args.layers else list(identity['attention_layers'])
    merged, meta, covered, shards = merge(args, layers)
    assert meta['identity_sha256'] == sha256(args.identity) and meta['windows_sha256'] == sha256(args.windows / 'windows.safetensors')
    fit_ids, heldout_ids = window_plan(manifest, meta['smoke'])
    assert sorted(covered) == sorted([('fit', i) for i in fit_ids] + [('heldout', i) for i in heldout_ids])
    groups, dim = identity['hkv'], identity['head_dim']
    V = groups * dim
    per_layer = {}
    for layer in layers:
        E = encoder_for(identity, layer).double()                                    # [G, d, r]
        r = E.shape[-1]
        gf, sf, nf = merged[f'fit_gram_{layer:03d}'], merged[f'fit_sum_{layer:03d}'], int(merged[f'fit_count_{layer:03d}'])
        gh, sh, nh = merged[f'heldout_gram_{layer:03d}'], merged[f'heldout_sum_{layer:03d}'], int(merged[f'heldout_count_{layer:03d}'])
        all_compressed = torch.zeros(groups * r, V, dtype=torch.float64)
        for g in range(groups):
            all_compressed[g * r:(g + 1) * r, g * dim:(g + 1) * dim] = E[g].mT
        rows = {}
        for g in range(groups):
            k_slice = slice(V + g * dim, V + (g + 1) * dim)
            local_raw = torch.zeros(dim, V, dtype=torch.float64)
            local_raw[:, g * dim:(g + 1) * dim] = torch.eye(dim, dtype=torch.float64)
            maps = dict(local_compressed_v=all_compressed[g * r:(g + 1) * r], all_compressed_v=all_compressed,
                        local_raw_v=local_raw, all_raw_v=torch.eye(V, dtype=torch.float64))
            rows[g] = {name: dict(zip(('fit', 'heldout'), solve_variant(gf, sf, nf, gh, sh, nh, M, k_slice))) for name, M in maps.items()}
        per_layer[layer] = dict(groups=rows, v_rank=r, fit_tokens=nf, heldout_tokens=nh,
                                layer_mean={name: {s: sum(rows[g][name][s] for g in rows) / groups for s in ('fit', 'heldout')} for name in VARIANTS})
        print(json.dumps(dict(stage='solve', layer=layer, heldout={name: round(per_layer[layer]['layer_mean'][name]['heldout'], 4) for name in VARIANTS})), flush=True)
    global_mean = {name: {s: sum(per_layer[l]['layer_mean'][name][s] for l in layers) / len(layers) for s in ('fit', 'heldout')} for name in VARIANTS}
    thirds = {}
    for label, sel in (('first third', layers[:len(layers) // 3]), ('middle third', layers[len(layers) // 3:2 * len(layers) // 3]), ('last third', layers[2 * len(layers) // 3:])):
        if sel:
            thirds[label] = dict(layers=sel, **{name: sum(per_layer[l]['layer_mean'][name]['heldout'] for l in sel) / len(sel) for name in VARIANTS})
    checkpoint = Path(identity['checkpoint'])
    result = dict(status='complete', format='basisserve.section4.v_to_k_information.v1',
        model=dict(path=identity['model'], config_sha256=identity['model_config_sha256'], type='llama' if 'llama' in identity['model'].lower() else 'qwen3'),
        compressed_v=dict(checkpoint=str(checkpoint), manifest_sha256=identity['manifest_sha256'], identity_sha256=sha256(args.identity),
                          value_mode=identity['value_mode'], layer_ranks=identity['layer_ranks']),
        calibration=dict(windows=str(args.windows), windows_sha256=meta['windows_sha256'], manifest_sha256=sha256(args.windows / 'manifest.json'),
                         condition=manifest.get('condition'), fit_ids=fit_ids, heldout_ids=heldout_ids, tokens_per_window=args.sequence_length,
                         fit_tokens=per_layer[layers[0]]['fit_tokens'], heldout_tokens=per_layer[layers[0]]['heldout_tokens']),
        target='pre-RoPE K = W_k h (after k_norm for Qwen3), per KV group, all tokens of every window',
        teacher='deployed model (identity checkpoint: compressed V and decoder installed); raw V = dense W_v h on the same hidden states',
        predictors={name: LABELS[name] for name in VARIANTS},
        metric='centered explained energy 1 - ||K_c - Khat_c||_F^2 / ||K_c||_F^2 on the held-out windows (K_c centered with the held-out mean); affine predictors fitted on the fit windows only; fit-split values reported alongside',
        statistics='FP32 chunk Grams accumulated in FP64; ridge 1e-8 x mean diagonal on the centered feature Gram',
        seed='deterministic (no sampling)', dtype='bfloat16 replay, float64 solve', smoke=meta['smoke'],
        layers=layers, per_layer={str(l): per_layer[l] for l in layers},
        layer_averages={str(l): per_layer[l]['layer_mean'] for l in layers}, depth_thirds=thirds, global_average=global_mean,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip(),
        commands=dict(solve=shlex.join(sys.argv), moments_shards=shards),
        source_sha256={n: sha256(Path(__file__).resolve().parents[1] / n) for n in ('evaluation/section4_v_to_k_information.py', 'evaluation/fit_k_routing_streaming.py', 'evaluation/k_routing_config.py')})
    write_json(args.output / 'result.json', result)
    write_summary(args.output, result, layers, per_layer, global_mean, thirds)
    plots(args.output, layers, per_layer, groups)
    print(json.dumps(dict(stage='solve', global_heldout={name: round(global_mean[name]['heldout'], 4) for name in VARIANTS})), flush=True)


def write_summary(output, result, layers, per_layer, global_mean, thirds):
    lines = ['# Section 4 / Experiment 1: V -> pre-RoPE K affine predictability', '',
             f"Model `{result['model']['path'].split('/')[-3] if 'snapshots' in result['model']['path'] else result['model']['path']}` (config sha `{result['model']['config_sha256'][:12]}`), "
             f"compressed V `{Path(result['compressed_v']['checkpoint']).name}` (manifest sha `{result['compressed_v']['manifest_sha256'][:12]}`, {result['compressed_v']['value_mode']}). "
             f"Calibration `{Path(result['calibration']['windows']).name}` (sha `{result['calibration']['windows_sha256'][:12]}`): fit windows {result['calibration']['fit_ids']}, "
             f"held-out windows {result['calibration']['heldout_ids']}, {result['calibration']['tokens_per_window']} tokens each "
             f"({result['calibration']['fit_tokens']} fit / {result['calibration']['heldout_tokens']} held-out tokens per layer).", '',
             f"Target: {result['target']}. Metric: {result['metric']}.", '',
             '## Held-out explained K energy (mean over KV groups)', '',
             '| layers | ' + ' | '.join(LABELS[n] for n in VARIANTS) + ' |', '|---|' + '---:|' * len(VARIANTS)]
    for label, t in thirds.items():
        lines.append(f"| {label} ({t['layers'][0]}-{t['layers'][-1]}) | " + ' | '.join(f"{100 * t[n]:.2f}%" for n in VARIANTS) + ' |')
    lines.append(f"| all ({layers[0]}-{layers[-1]}) | " + ' | '.join(f"{100 * global_mean[n]['heldout']:.2f}%" for n in VARIANTS) + ' |')
    lines += ['', 'Fit-split values (same predictors, fit windows): ' + ', '.join(f"{LABELS[n]} {100 * global_mean[n]['fit']:.2f}%" for n in VARIANTS) + '.', '',
              '## Per layer (held-out, mean over groups; min-max over groups for local compressed V)', '',
              '| layer | ' + ' | '.join(LABELS[n] for n in VARIANTS) + ' | local compressed V min-max |', '|---:|' + '---:|' * (len(VARIANTS) + 1)]
    for l in layers:
        lm = per_layer[l]['layer_mean']
        vals = [per_layer[l]['groups'][g]['local_compressed_v']['heldout'] for g in per_layer[l]['groups']]
        lines.append(f'| {l} | ' + ' | '.join(f"{100 * lm[n]['heldout']:.2f}%" for n in VARIANTS) + f' | {100 * min(vals):.1f}-{100 * max(vals):.1f}% |')
    lines += ['', 'Plots: `v_to_k_by_layer.pdf` (held-out explained energy vs layer, four predictors), '
              '`v_to_k_by_layer_group.pdf` (layer x KV-group heatmaps, compressed V, local and all-group panels). Exact values in `result.json`.', '']
    (output / 'summary.md').write_text('\n'.join(lines))


def plots(output, layers, per_layer, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'local_compressed_v': ('tab:blue', '-', 'o'), 'all_compressed_v': ('tab:blue', '--', 's'),
              'local_raw_v': ('tab:gray', '-', 'o'), 'all_raw_v': ('tab:gray', '--', 's')}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name in VARIANTS:
        color, ls, marker = styles[name]
        ax.plot(layers, [100 * per_layer[l]['layer_mean'][name]['heldout'] for l in layers], color=color, linestyle=ls, marker=marker, markersize=3, linewidth=1.4, label=LABELS[name])
    ax.set_xlabel('layer'); ax.set_ylabel('held-out explained K energy (%)'); ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'v_to_k_by_layer.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, name, title in zip(axes, ('local_compressed_v', 'all_compressed_v'), ('local compressed V', 'all-group compressed V')):
        grid = [[100 * per_layer[l]['groups'][g][name]['heldout'] for g in range(groups)] for l in layers]
        im = ax.imshow(grid, aspect='auto', vmin=0, vmax=100, cmap='viridis', origin='lower', extent=(-0.5, groups - 0.5, layers[0] - 0.5, layers[-1] + 0.5))
        ax.set_title(title, fontsize=10); ax.set_xlabel('KV group'); ax.set_xticks(range(groups))
    axes[0].set_ylabel('layer')
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label='held-out explained K energy (%)')
    fig.savefig(output / 'v_to_k_by_layer_group.pdf', bbox_inches='tight'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=('moments', 'solve'))
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--windows', type=Path, required=True, help='calibration bank dir (manifest.json with fit_ids / validation_ids + windows.safetensors)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sequence-length', type=int, default=131072)
    p.add_argument('--rope', choices=('native', 'yarn2', 'yarn4'), default='native')
    p.add_argument('--layers', help='comma list (default: every attention layer)')
    p.add_argument('--window-shard', type=int, default=0)
    p.add_argument('--window-shards', type=int, default=8)
    p.add_argument('--smoke', action='store_true', help='one fit and one held-out window')
    args = p.parse_args()
    assert 0 <= args.window_shard < args.window_shards
    configure()
    identity = read_json(args.identity)
    assert identity['status'] == 'complete'
    assert sha256(Path(identity['model']) / 'config.json') == identity['model_config_sha256']
    manifest = read_json(args.windows / 'manifest.json')
    assert manifest['status'] == 'complete' and manifest['model_config_sha256'] == identity['model_config_sha256']
    assert sha256(args.windows / 'windows.safetensors') == manifest['sha256']
    (moments if args.stage == 'moments' else solve)(args, identity, manifest)


if __name__ == '__main__':
    main()
