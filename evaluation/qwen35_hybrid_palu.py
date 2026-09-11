"""Matched PaLU Fisher collection and grouped activation-aware V factors."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

from evaluation.qwen35_hybrid_common import atomic_save, full_layers, load_model, load_windows, sha256
from evaluation.collect_gqa_palu_fisher import _chunked_official_palu_loss_and_backward
from evaluation.reproduce_palu_paper_llama2_distributed import official_fisher_uniform_rank_map, _stable_cholesky
from basisserve.core.qwen35_gated_v_runtime import factor_hash


def fisher(args):
    model = load_model(args.model_path, args.device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    assert all(not isinstance(m, torch.nn.Dropout) or m.p == 0 for m in model.modules())
    selected = {i: a.v_proj for i, a in full_layers(model).items()}
    for module in selected.values():
        module.weight.requires_grad_(True)
    windows = load_windows(args.data, 'fit')
    indices = list(range(args.shard, len(windows), args.shards))
    if args.smoke_only:
        indices = indices[:1]
    records, total = [], {i: torch.zeros_like(m.weight, device='cpu', dtype=torch.float32) for i, m in selected.items()}
    recomputations = {i: 0 for i in selected}
    handles = []
    for i in selected:
        def hook(module, inputs, output, i=i):
            recomputations[i] += 1
        handles.append(model.model.layers[i].register_forward_hook(hook))
    for index in indices:
        start = time.monotonic()
        model.zero_grad(set_to_none=True)
        loss = _chunked_official_palu_loss_and_backward(model, windows[index:index + 1].to(args.device), chunk_size=128)
        assert torch.isfinite(loss)
        for i, module in selected.items():
            grad = module.weight.grad
            assert grad is not None and torch.isfinite(grad).all()
            total[i] += grad.detach().float().square().cpu()
        record = {'index': index, 'loss': float(loss), 'seconds': time.monotonic() - start,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
        records.append(record)
        print(json.dumps(record), flush=True)
    for handle in handles:
        handle.remove()
    atomic_save(args.output, {'format': 'basisserve.qwen35.palu_fisher_shard.v1', 'squared_gradient_sum': total,
        'indices': indices, 'records': records, 'layer_forward_calls': recomputations,
        'windows_sha256': sha256(Path(args.data) / 'windows.pt'), 'model_config_sha256': sha256(Path(args.model_path) / 'config.json'),
        'loss': 'official PaLU double-shift convention; targets tokens[2:]', 'no_optimizer': True,
        'statistic': 'elementwise mean squared window-mean-loss gradients, sqrt, then mean weight entries',
        'command': sys.argv})


@torch.no_grad()
def whiten(args):
    model = load_model(args.model_path, args.device)
    layers = full_layers(model)
    grams = {i: torch.zeros(a.v_proj.in_features, a.v_proj.in_features, device=args.device, dtype=torch.float64) for i, a in layers.items()}
    handles = []
    for i, a in layers.items():
        def hook(module, inputs, i=i):
            x = inputs[0].detach().flatten(0, 1).double()
            grams[i] += x.T @ x
        handles.append(a.v_proj.register_forward_pre_hook(hook))
    windows = load_windows(args.data, 'fit')
    for index, tokens in enumerate(windows):
        model.model(tokens[None].to(args.device), use_cache=False)
        if index % 16 == 0:
            print(json.dumps({'whitening_window': index}), flush=True)
    for handle in handles:
        handle.remove()
    atomic_save(args.output, {'input_grams': {i: g.cpu() for i, g in grams.items()},
        'v_weights': {i: a.v_proj.weight.detach().cpu() for i, a in layers.items()},
        'rows': windows.numel(), 'windows_sha256': sha256(Path(args.data) / 'windows.pt')})


@torch.no_grad()
def build(args):
    shards = [torch.load(p, weights_only=True, map_location='cpu') for p in sorted(Path(args.fisher_dir).glob('shard*.pt'))]
    assert sorted(i for s in shards for i in s['indices']) == list(range(256))
    assert all(s['windows_sha256'] == sha256(Path(args.data) / 'windows.pt') for s in shards)
    layers = sorted(shards[0]['squared_gradient_sum'])
    scalars = {str(i): float((sum(s['squared_gradient_sum'][i].double() for s in shards) / 256).sqrt().mean()) for i in layers}
    whitening = torch.load(args.whitening, weights_only=True, map_location='cpu')
    assert whitening['windows_sha256'] == sha256(Path(args.data) / 'windows.pt')
    rows = whitening['rows']
    assert rows == 256 * 2048
    grams = {}
    for i in layers:
        scale = _stable_cholesky(whitening['input_grams'][i].to(args.device))
        weighted = whitening['v_weights'][i].to(device=args.device, dtype=torch.float32) @ scale
        grams[i] = weighted.double() @ weighted.double().T
    for group_size, name in [(1, 'mlrd'), (2, 'glrd2'), (4, 'glrd4')]:
        for anchor in (64, 80, 96):
            rank_map, total_rank, dense_rank = official_fisher_uniform_rank_map(scalars,
                output_width=1024, num_heads=4, head_group_size=group_size, retained_ratio=anchor / 256, block_size=32)
            bank = {}
            for i in layers:
                encoders, decoders = [], []
                for group, rank in enumerate(rank_map[str(i)]):
                    width = group_size * 256
                    assert 0 < rank <= width
                    sl = slice(group * width, (group + 1) * width)
                    values, vectors = torch.linalg.eigh(grams[i][sl, sl])
                    u = vectors[:, -rank:].flip(-1)
                    signs = u.gather(0, u.abs().argmax(0)[None]).sign()
                    u *= signs
                    # Balanced singular-value factors, as in official PaLU.
                    root_sigma = values[-rank:].flip(0).clamp_min(1e-30).pow(0.25)
                    encoders.append((u / root_sigma).to(torch.bfloat16).cpu())
                    decoders.append((root_sigma[:, None] * u.T).to(torch.bfloat16).cpu())
                bank[i] = {'E_V': torch.stack(encoders), 'R_V': torch.stack(decoders)}
            payload = {'format': 'basisserve.qwen35.gated_v_als.v1', 'method': 'palu_' + name,
                'layers': bank, 'factor_sha256': factor_hash(bank), 'nominal_v_rank': anchor,
                'rank_map': rank_map, 'realized_v_retention': total_rank / dense_rank,
                'calibration_rows': rows, 'fisher_scalars': scalars, 'fisher_loss': shards[0]['loss'],
                'factorization': 'balanced SVD via W L left Gram; repository PaLU stable Cholesky',
                'whitening_sha256': sha256(args.whitening),
                'windows_sha256': sha256(Path(args.data) / 'windows.pt')}
            path = Path(args.output) / f'palu_{name}_v{anchor}.pt'
            atomic_save(path, payload)
            print(json.dumps({'file': str(path), 'rank_map': rank_map, 'realized_retention': total_rank / dense_rank}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['fisher', 'whiten', 'build'])
    p.add_argument('--model-path', default='results/q35_hybrid/model')
    p.add_argument('--data', default='results/q35_hybrid/data')
    p.add_argument('--capture-dir', default='results/q35_hybrid/capture')
    p.add_argument('--fisher-dir', default='results/q35_hybrid/fisher')
    p.add_argument('--whitening', default='results/q35_hybrid/whitening.pt')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=2)
    p.add_argument('--smoke-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
