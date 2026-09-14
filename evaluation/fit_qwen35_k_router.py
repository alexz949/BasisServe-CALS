"""Fit one Qwen3.5 Base/Residual capacity from completed native captures."""

import argparse
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file

from basisserve.core.c1_v_conditional_k_router import fit_affine_reduced_rank_map
from basisserve.core.query_position_sampling import candidate_positions, select_stratified_query_positions
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (
    _value_codes, _apply_base, _stack_base_map, _fit_residual_grid,
)
from evaluation.fit_qwen3_8b_q8_fisher_residual import build_multi_query_statistics
from evaluation.k_routing_capture_windows import CapturedWindows
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, sha256


def fit_base(rows, keys, encoder, rank):
    groups, dim, width = encoder.shape
    sums = torch.zeros(groups, width, dtype=torch.float64)
    targets = torch.zeros(groups, dim, dtype=torch.float64)
    gram = torch.zeros(groups, width, width, dtype=torch.float64)
    cross = torch.zeros(groups, width, dim, dtype=torch.float64)
    for index in range(len(rows)):
        z = _value_codes(rows[index, ..., :dim].to(encoder).float(), encoder).transpose(0, 1)
        k = keys[index].to(encoder).float().transpose(0, 1)
        sums += z.sum(1).double().cpu()
        targets += k.sum(1).double().cpu()
        gram += torch.bmm(z.mT, z).double().cpu()
        cross += torch.bmm(z.mT, k).double().cpu()
        print(json.dumps(dict(base_rank=rank, base_window=index)), flush=True)
    return {rank: tuple(fit_affine_reduced_rank_map(row_count=len(rows)*rows.shape[1],
        input_sum=sums[group], target_sum=targets[group], input_gram=gram[group],
        input_target_gram=cross[group], rank=rank, fit_bias=True) for group in range(groups))}


def base_mse(rows, keys, encoder, base, indices):
    error = energy = 0.0
    for index in indices:
        z = _value_codes(rows[index, ..., :256].cuda().float(), encoder)
        target = keys[index].cuda().float()
        prediction = _apply_base(z, base)
        error += (prediction-target).double().square().sum().item()
        energy += target.double().square().sum().item()
    assert energy > 0
    return error/energy


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--v-bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--rank', type=int, choices=(16, 32), required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    bank = load_bank(args.v_bank)
    assert bank['status'] == 'complete' and bank['method'] == 'twosided'
    assert bank['nominal_v_rank'] == 192 and bank['encoder_sweeps'] == 12 and bank['encoder_cg'] == 16
    assert bank['wo_compression'] is False and args.layer in bank['schedule']
    assert sum(bank['schedule'].values()) == 1536
    source_hashes, locations, rope_hash = {}, [], None
    grid = candidate_positions(32768)
    for index in range(80):
        root = args.capture/f'w{index:03d}'
        row = json.loads((root/'manifest.json').read_text())
        assert row['status'] == 'complete' and row['index'] == index and row['length'] == 32768
        assert row['split'] == ('fit' if index < 64 else 'diagnostic')
        assert row['windows_sha256'] == bank['windows_sha256'] and row['wo_compression'] is False
        assert row['model_config_sha256'] == bank['model_identity']['config_sha256']
        assert row['candidate_positions'] == grid
        path = root/f'l{args.layer:02d}.safetensors'
        digest = sha256(path)
        assert digest == row['files'][path.name]
        source_hashes[str(path)] = digest
        locations.append((path, 0))
        rope_hash = rope_hash or row['files']['rope.safetensors']
        assert row['files']['rope.safetensors'] == rope_hash
    rope_path = args.capture/'w000'/'rope.safetensors'
    assert sha256(rope_path) == rope_hash
    rope = load_file(str(rope_path))
    cos, sin = rope['cos'].cuda().float(), rope['sin'].cuda().float()
    shapes = dict(rows=(32768, 4, 512), pre_rope_keys=(32768, 4, 256), candidate_queries=(len(grid), 16, 256))
    captures = {key: CapturedWindows(tuple(locations), key, shape) for key, shape in shapes.items()}
    rows, keys = captures['rows'], captures['pre_rope_keys']
    queries = torch.stack(list(captures['candidate_queries']))
    v_rank = bank['schedule'][args.layer]
    encoder = (torch.eye(256).repeat(4, 1, 1) if v_rank == 256
        else bank['layers'][args.layer]['E_V']).cuda().float()
    assert encoder.shape == (4, 256, v_rank)
    destination = args.output/f'b{args.rank}r{args.rank}'/f'l{args.layer:02d}.pt'
    protocol = dict(v_bank_sha256=sha256(args.v_bank), capture_hashes=source_hashes,
        windows_sha256=bank['windows_sha256'], rope_sha256=rope_hash, base_rank=args.rank,
        residual_rank=args.rank, fit_windows=64, diagnostic_windows=16, sequence_length=32768,
        fit_queries=64, diagnostic_queries=32, bcd_sweeps=40, pcg_iterations=100,
        pcg_damping=1e-5, pcg_tolerance=1e-5, page_size=32, excluded_prefix_pages=1,
        routing_budget=2048, wo_compression=False, base_objective='affine pre-RoPE closed-form RRR',
        residual_objective='causal non-sink Page-Fisher', source_sha256=sha256(Path(__file__)))
    if destination.exists():
        saved = torch.load(destination, map_location='cpu', weights_only=True)
        assert saved['status'] == 'complete' and saved['protocol'] == protocol
        return
    started = time.monotonic()
    selections, selected = {}, {}
    for split, count, section in [('fit', 16, slice(0, 64)), ('diagnostic', 8, slice(64, 80))]:
        selection, _, _ = select_stratified_query_positions(queries[:64].cuda(), grid,
            context_length=32768, num_bins=4, queries_per_bin=count)
        selections[split] = selection
        selected[split] = queries[section, [grid.index(position) for position in selection['selected_positions']]].contiguous()
    bases = fit_base(rows[:64], keys[:64], encoder, args.rank)
    stacked = _stack_base_map(bases[args.rank], device='cuda')
    metrics = {split: dict(pre_rope_base_relative_mse=base_mse(rows, keys, encoder, stacked, indices))
        for split, indices in [('fit', range(64)), ('diagnostic', range(64, 80))]}
    statistics = {}
    for split, section in [('fit', slice(0, 64)), ('diagnostic', slice(64, 80))]:
        statistics[split], metrics[split]['causal_queries'] = build_multi_query_statistics(
            selected[split], rows[section], query_positions=selections[split]['selected_positions'],
            cos=cos, sin=sin, value_encoder=encoder, base_maps=bases,
            page_size=32, excluded_prefix_pages=1, device='cuda')
    factors, losses = _fit_residual_grid(statistics['fit'], statistics['diagnostic'],
        residual_ranks=(args.rank,), sweeps=40, relative_damping=1e-5,
        iterative_tolerance=1e-5, iterative_max_iterations=100, device=torch.device('cuda'))
    tensors = {f'base_{name}_b{args.rank}': torch.stack([getattr(base, name) for base in bases[args.rank]]).float().cpu()
        for name in ('left', 'right', 'bias')}
    for name, value in zip(('encoder', 'query'), factors[args.rank, args.rank], strict=True):
        tensors[f'residual_{name}_b{args.rank}_r{args.rank}'] = value.float().cpu()
    assert all(torch.isfinite(value).all() for value in tensors.values())
    atomic_save(destination, dict(status='complete', layer=args.layer, v_rank=v_rank,
        protocol=protocol, tensors=tensors, selections=selections, metrics=metrics,
        losses=losses, seconds=time.monotonic()-started))
    print(json.dumps(dict(complete=True, layer=args.layer, rank=args.rank, losses=losses)), flush=True)


if __name__ == '__main__':
    main()
