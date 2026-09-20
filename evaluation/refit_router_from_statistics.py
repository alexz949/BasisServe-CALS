"""Refit the Base/Fisher residual router from persisted statistics; no teacher."""
import argparse
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.v96kl_common import (
    configure, read_json, write_json, save_tensors, sha256, code_hashes,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _fit_residual_grid
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    S80CompactSoftmaxFisherRouting, unpack_symmetric_fisher_grams,
)

SOURCES = (
    'evaluation/refit_router_from_statistics.py',
    'evaluation/calibrate_qwen3_32b_router_128k.py',
    'evaluation/streaming_k_statistics.py',
    'evaluation/eval_qwen3_8b_v80_conditional_residual_router.py',
    'basisserve/core/gqa_joint_routing_payload_s80_fisher.py',
)


def load_layer(root, layer, kv_groups):
    """Rebuild one layer's routing statistics in the order the capture wrote them."""
    path = root / f'layer_{layer:03d}.safetensors'
    record = read_json(path.with_suffix('.json'))
    assert record['status'] == 'complete' and record['layer'] == layer
    assert record['sha256'] == sha256(path)
    payload = load_file(str(path))
    dim = int(record['key_dim'])
    heads = int(record['heads'])
    grams = unpack_symmetric_fisher_grams(payload['packed'], dimension=dim)
    queries = payload['queries']
    assert queries.shape == (heads, record['examples'], dim)
    assert grams.shape == (heads, record['examples'], dim, dim)
    statistics = S80CompactSoftmaxFisherRouting(
        queries, grams, torch.arange(heads) // (heads // kv_groups),
        0, dim, float(record['scaling']), float(record['teacher_fisher_energy']))
    bases = {name: payload[f'base_{name}'] for name in ('left', 'right', 'bias')}
    return record, statistics, bases


def refit(args):
    layers = sorted(int(p.stem.split('_')[1]) for p in args.statistics.glob('layer_*.safetensors'))
    assert layers, f'no statistics under {args.statistics}'
    selected = [layer for position, layer in enumerate(layers)
                if position % args.num_shards == args.shard_index]
    args.bank.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        source='refit from persisted statistics; no teacher replay',
        statistics_root=str(args.statistics.resolve()),
        base_rank=args.base_rank, residual_rank=args.residual_rank,
        sweeps=args.sweeps, relative_damping=1e-5,
        pcg_tolerance=1e-5, pcg_iterations=args.pcg_iterations,
        validation_scope='in-sample: the capture carries no held-out windows',
        code_sha256=code_hashes(list(SOURCES)))
    for layer in selected:
        started = time.monotonic()
        record, statistics, bases = load_layer(args.statistics, layer, args.kv_groups)
        assert record['base_rank'] == args.base_rank
        factors, losses = _fit_residual_grid(
            {args.base_rank: statistics}, {args.base_rank: statistics},
            residual_ranks=(args.residual_rank,), sweeps=args.sweeps,
            relative_damping=1e-5, iterative_tolerance=1e-5,
            iterative_max_iterations=args.pcg_iterations, device=torch.device('cuda'))
        key = (args.base_rank, args.residual_rank)
        suffix = f'b{args.base_rank}_r{args.residual_rank}'
        tensors = {f'base_{name}_b{args.base_rank}': bases[name].float()
                   for name in ('left', 'right', 'bias')}
        tensors[f'residual_encoder_{suffix}'] = factors[key][0].cpu().float()
        tensors[f'residual_query_{suffix}'] = factors[key][1].cpu().float()
        assert all(torch.isfinite(t).all() for t in tensors.values())
        path = args.bank / f'layer_{layer:03d}.safetensors'
        save_tensors(path, tensors)
        write_json(path.with_suffix('.json'), dict(status='complete', layer=layer,
            protocol=protocol, sha256=sha256(path), losses=losses,
            statistics_sha256=record['sha256'],
            capture_protocol=record['protocol'],
            elapsed_seconds=time.monotonic() - started))
        print('refit complete layer', layer, 'losses', losses,
              'seconds', time.monotonic() - started, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--statistics', type=Path, required=True)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--kv-groups', type=int, default=8)
    p.add_argument('--base-rank', type=int, default=16)
    p.add_argument('--residual-rank', type=int, default=16)
    p.add_argument('--sweeps', type=int, default=40)
    p.add_argument('--pcg-iterations', type=int, default=100)
    args = p.parse_args()
    configure()
    assert 0 <= args.shard_index < args.num_shards
    torch.backends.cuda.matmul.allow_tf32 = True
    refit(args)


if __name__ == '__main__':
    main()
