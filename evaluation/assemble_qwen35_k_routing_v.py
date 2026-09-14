"""Assemble the ALS12/32K V192 bank from two-sided terminal-KL probes."""

import argparse
import json
import math
from pathlib import Path

import torch

from basisserve.core.qwen35_gated_v_runtime import factor_hash
from evaluation.build_qwen3_8b_c1_two_sided_factorized_kl_schedule import (
    allocate_layer_schedule, predict_two_sided_factorized_costs,
)
from evaluation.qwen35_hybrid_common import atomic_save, sha256


LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
RANKS = (128, 160, 192, 224, 256)
FIT_RANKS = (128, 160, 192, 224)


def validate_profile(row, schedule, factors, windows_hash):
    assert row['status'] == 'complete' and row['schedule'] == schedule
    assert row['windows_sha256'] == windows_hash and row['wo_compression'] is False
    assert row['profile_indices'] == list(range(16))
    bank, hashes = {}, {}
    for layer, rank in zip(LAYERS, schedule, strict=True):
        payload, path = factors[layer, rank]
        bank[layer] = {key: payload[key] for key in ('E_V', 'R_V')}
        hashes[path.name] = sha256(path)
    assert row['source_factors'] == hashes and row['factor_sha256'] == factor_hash(bank)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--factors', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    factors, errors, windows_hash = {}, [], None
    for layer in LAYERS:
        curve = {256: 0.0}
        capture_hashes = None
        for rank in FIT_RANKS:
            path = args.factors/f'l{layer:02d}_r{rank:03d}.pt'
            row = torch.load(path, map_location='cpu', weights_only=True)
            assert row['status'] == 'complete' and row['layer'] == layer and row['rank'] == rank
            assert row['encoder_sweeps'] == 12 and row['encoder_cg'] == 16
            fast_fit = rank <= 128 or (layer >= 19 and rank in (160, 224))
            assert row['decoder_cg'] == (50 if fast_fit else 200)
            if fast_fit:
                assert row['matmul_allow_tf32'] is True
            assert row['wo_compression'] is False
            assert row['E_V'].shape == (4, 256, rank) and row['R_V'].shape == (4, rank, 256)
            assert all(row[key].dtype == torch.bfloat16 and torch.isfinite(row[key]).all() for key in ('E_V', 'R_V'))
            capture_hashes = capture_hashes or row['capture_hashes']
            assert row['capture_hashes'] == capture_hashes
            windows_hash = windows_hash or row['windows_sha256']
            assert row['windows_sha256'] == windows_hash
            curve[rank] = row['metrics']['fit_export_relative_mse']
            assert math.isfinite(curve[rank]) and curve[rank] >= 0
            factors[layer, rank] = (row, path)
        errors.append(curve)
    anchor = json.loads((args.profile/'anchor.json').read_text())
    validate_profile(anchor, [192]*8, factors, windows_hash)
    deltas, profile_hashes = {}, {'anchor.json': sha256(args.profile/'anchor.json')}
    for rank in (160, 224):
        deltas[rank] = []
        for offset, layer in enumerate(LAYERS):
            path = args.profile/f'l{layer:02d}_r{rank:03d}.json'
            row = json.loads(path.read_text())
            expected = [192]*8
            expected[offset] = rank
            validate_profile(row, expected, factors, windows_hash)
            assert row['model_identity'] == anchor['model_identity']
            assert row['teacher_hashes'] == anchor['teacher_hashes']
            deltas[rank].append(row['mean_kl']-anchor['mean_kl'])
            profile_hashes[path.name] = sha256(path)
    costs, minus, plus = predict_two_sided_factorized_costs(errors, deltas[160], deltas[224],
        candidate_ranks=RANKS, anchor_rank=192, compression_probe_rank=160, expansion_probe_rank=224, exponent=1.25)
    schedule, cost = allocate_layer_schedule(costs, candidate_ranks=RANKS, anchor_rank=192, target_average_rank=192)
    assert sum(schedule) == 1536
    bank, sources = {}, {}
    for layer, rank in zip(LAYERS, schedule, strict=True):
        if rank == 256:
            continue
        row, path = factors[layer, rank]
        bank[layer] = {key: row[key] for key in ('E_V', 'R_V')}
        sources[layer] = dict(path=str(path), sha256=sha256(path))
    atomic_save(args.output, dict(format='basisserve.qwen35.gated_v_als.v1', status='complete',
        method='twosided', layers=bank, factor_sha256=factor_hash(bank), nominal_v_rank=192,
        schedule=dict(zip(LAYERS, schedule)), windows_sha256=windows_hash, encoder_sweeps=12,
        encoder_cg=16,
        decoder_cg_by_layer_rank={layer: {rank: factors[layer, rank][0]['decoder_cg']
            for rank in FIT_RANKS} for layer in LAYERS},
        wo_compression=False, source_factors=sources, profile_hashes=profile_hashes,
        model_identity=anchor['model_identity'], compression_sensitivity=minus, expansion_sensitivity=plus,
        predicted_cost=cost, candidate_ranks=RANKS, kl_exponent=1.25,
        local_curve_split='fit', profile_indices=anchor['profile_indices']))
    print(json.dumps(dict(schedule=dict(zip(LAYERS, schedule)), average_rank=sum(schedule)/8)), flush=True)


if __name__ == '__main__':
    main()
