"""Assemble the uniform gated-V bank (basisserve.qwen35.gated_v_als.v1) from the per-layer 128K fits of ``fit_qwen35_128k_v.py``.

Mirrors ``qwen35_hybrid_banks.save_bank`` for the uniform schedule, with the model identity verified against the snapshot
(config sha256 and every model safetensors file) and the encoder sweep count taken from the factors."""
import argparse
import json
from pathlib import Path

import torch

from basisserve.core.qwen35_gated_v_runtime import factor_hash
from evaluation.qwen35_hybrid_common import atomic_save, sha256

LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--model-revision', required=True)
    p.add_argument('--factors', type=Path, required=True)
    p.add_argument('--rank', type=int, default=192)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = json.loads((args.model / 'config.json').read_text())['text_config']
    layers = tuple(i for i, kind in enumerate(config['layer_types']) if kind == 'full_attention')
    assert layers == LAYERS
    bank, sources, sweeps, windows_hash, metrics = {}, {}, None, None, {}
    for layer in layers:
        path = args.factors / f'l{layer:02d}_r{args.rank:03d}.pt'
        payload = torch.load(path, weights_only=True, map_location='cpu')
        assert payload['status'] == 'complete' and payload['rank'] == args.rank and payload['layer'] == layer
        sweeps = sweeps or payload['encoder_sweeps']; windows_hash = windows_hash or payload['windows_sha256']
        assert payload['encoder_sweeps'] == sweeps and payload['windows_sha256'] == windows_hash
        bank[layer] = {key: payload[key] for key in ('E_V', 'R_V')}
        sources[layer] = {'file': str(path), 'sha256': sha256(path)}
        metrics[layer] = payload['metrics']
    model_files = {p.name: sha256(p) for p in sorted(args.model.glob('*.safetensors'))}
    identity = {'model_revision': args.model_revision, 'config_sha256': sha256(args.model / 'config.json'), 'model_files_sha256': model_files}
    schedule = {layer: args.rank for layer in layers}
    atomic_save(args.output, {
        'format': 'basisserve.qwen35.gated_v_als.v1', 'method': 'uniform', 'layers': bank, 'status': 'complete',
        'factor_sha256': factor_hash(bank), 'nominal_v_rank': args.rank, 'schedule': schedule,
        'value_retention': args.rank / 256, 'windows_sha256': windows_hash, 'encoder_sweeps': sweeps,
        'encoder_cg': payload['encoder_cg'], 'decoder_cg': payload['decoder_cg'], 'row_stride': payload['row_stride'],
        'fit_windows': payload['fit_windows'], 'rows_per_window': payload['rows_per_window'], 'sequence_length': 131072,
        'wo_compression': False, 'source_sha256': sha256(Path(__file__)),
        'model_identity': identity, 'source_factors': sources, 'fit_export_relative_mse': {l: m['fit_export_relative_mse'] for l, m in metrics.items()},
        'diagnostic_export_relative_mse': {l: m.get('diagnostic_export_relative_mse') for l, m in metrics.items()},
        'work_dtype': 'float32', 'factor_dtype': 'bfloat16', 'capture_trajectory': 'native_dense',
        'objective': 'joint_gated_attention_output_mse', 'target_kind': 'native_attention_output_excluding_output_bias',
        'initialization': 'group_pooled_pre_gate_pca', 'v_cache_heads': config['num_key_value_heads'],
        'query_heads': config['num_attention_heads'], 'value_head_dim': config['head_dim']})
    print(json.dumps(dict(output=str(args.output), layers=list(layers), rank=args.rank, sweeps=sweeps,
                          fit_mse={str(l): m['fit_export_relative_mse'] for l, m in metrics.items()})), flush=True)


if __name__ == '__main__':
    main()
