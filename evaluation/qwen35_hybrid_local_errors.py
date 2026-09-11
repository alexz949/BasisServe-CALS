"""Common-input local error decomposition, separate from trajectory PPL."""

import argparse
import json
from pathlib import Path

import torch

from evaluation.qwen35_hybrid_common import atomic_save, load_bank, sha256


def error_terms(native, v_only, composed):
    assert native.shape == v_only.shape == composed.shape
    native, v_only, composed = native.double(), v_only.double(), composed.double()
    v_error, ag_error = native - v_only, v_only - composed
    return {
        'native_energy': float(native.square().sum()),
        'v_output_energy': float(v_only.square().sum()),
        'v_error_energy': float(v_error.square().sum()),
        'ag_error_energy': float(ag_error.square().sum()),
        'composed_error_energy': float((native - composed).square().sum()),
        'twice_error_inner_product': float(2 * (v_error * ag_error).sum()),
    }


def normalized_errors(terms):
    reconstructed = terms['v_error_energy'] + terms['ag_error_energy'] + terms['twice_error_inner_product']
    scale = max(terms['v_error_energy'] + terms['ag_error_energy'] + abs(terms['twice_error_inner_product']), 1.)
    assert abs(terms['composed_error_energy'] - reconstructed) < 1e-10 * scale
    return {**terms,
        'err_v': terms['v_error_energy'] / max(terms['native_energy'], 1e-300),
        'err_ag_given_v': terms['ag_error_energy'] / max(terms['v_output_energy'], 1e-300),
        'err_composed': terms['composed_error_energy'] / max(terms['native_energy'], 1e-300),
        'normalized_cross_term': terms['twice_error_inner_product'] / max(terms['native_energy'], 1e-300)}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', required=True)
    p.add_argument('--wo-bank', required=True)
    p.add_argument('--capture-dir', default='results/q35_hybrid/capture')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--chunk-rows', type=int, default=2048)
    args = p.parse_args()
    torch.set_num_threads(2)
    v = load_bank(args.bank)
    wo = torch.load(args.wo_bank, weights_only=True, map_location='cpu')
    assert wo['upstream_v_factor_sha256'] == v['factor_sha256']
    weights = torch.load(Path(args.capture_dir) / 'weights.pt', weights_only=True, map_location='cpu')
    rows = []
    for record in wo['full_attention']['layers']:
        layer = record['layer_index']
        weight = weights[layer].to(device=args.device, dtype=torch.float32)
        output_encoders = record['private_encoders'].to(device=args.device, dtype=torch.float32)
        output_decoder = record['joint_decoder_weight'].to(device=args.device, dtype=torch.float32)
        tp, _, rank = output_encoders.shape
        approximated_weight = (output_encoders @ output_decoder.T.reshape(tp, rank, -1)).flatten(0, 1)
        flat_weight = weight.flatten(0, 1)
        for split in ('fit', 'heldout'):
            totals, count = {}, 0
            for path in sorted(Path(args.capture_dir).glob(f'{split}_*.pt')):
                captured = torch.load(path, weights_only=True, map_location='cpu', mmap=True)['layers'][layer]
                for start in range(0, len(captured['z']), args.chunk_rows):
                    z = captured['z'][start:start + args.chunk_rows].to(device=args.device, dtype=torch.float32)
                    gate = captured['gate'][start:start + args.chunk_rows].to(device=args.device, dtype=torch.float32)
                    if layer in v['layers']:
                        encoder = v['layers'][layer]['E_V'].to(z)
                        decoder = v['layers'][layer]['R_V'].to(z)
                        mapping = torch.arange(z.shape[1], device=z.device) // (z.shape[1] // encoder.shape[0])
                        restored = torch.einsum('nhd,hdr,hre->nhe', z, encoder[mapping], decoder[mapping])
                    else:
                        restored = z
                    native_output = (z * gate).flatten(1) @ flat_weight
                    post = (restored * gate).flatten(1)
                    values = error_terms(native_output, post @ flat_weight, post @ approximated_weight)
                    totals = {key: totals.get(key, 0.) + value for key, value in values.items()}
                    count += len(z)
            assert count == {'fit': 256, 'heldout': 64}[split] * 2048
            row = {'layer': layer, 'split': split, 'rows': count, **normalized_errors(totals)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    atomic_save(args.output, {'format': 'basisserve.qwen35.hybrid_common_input_errors.v1', 'rows': rows,
        'scope': 'full-attention local operators on common native-dense pre-gate captures',
        'native_target': 'FP32 (Z * native_sigmoid_gate) @ original_output_weight',
        'distinct_from': 'frozen-V recaptured Wo fitting statistics and whole-model PPL trajectories',
        'v_factor_sha256': v['factor_sha256'], 'wo_bank_sha256': sha256(args.wo_bank),
        'capture_manifest_sha256': sha256(Path(args.capture_dir) / 'manifest.json')})


if __name__ == '__main__':
    main()
