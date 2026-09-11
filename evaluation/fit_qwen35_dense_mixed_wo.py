"""Fit GDN rank 768 and reuse Dense-calibrated full-attention rank 512."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

from basisserve.core.qwen35_gdn_private_ag import fit_qwen35_private_ag_joint_factors
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, sha256


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('fit', 'assemble'))
    p.add_argument('--bank', default='results/q35_hybrid/banks/c1_uniform_v256.pt')
    p.add_argument('--moments', default='results/q35_hybrid/wo_dense_moments')
    p.add_argument('--source-wo', default='results/q35_hybrid/wo_dense/wo_bank.pt')
    p.add_argument('--output', default='results/q35_hybrid/wo_dense_g768_f512')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    torch.set_num_threads(2)
    assert 0 <= a.shard_index < a.num_shards
    print(json.dumps({'command': sys.argv, 'args': vars(a), 'python': sys.executable}), flush=True)
    bank = load_bank(a.bank)
    assert not bank['layers'] and set(bank['schedule'].values()) == {256}
    root, output = Path(a.moments), Path(a.output)
    manifest_sha = sha256(root / 'manifest.json')
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['trajectory'] == 'native_dense'
    assert manifest['upstream_v_factor_sha256'] == bank['factor_sha256']
    assert manifest['windows_sha256'] == bank['windows_sha256']
    assert manifest['verified_model_identity'] == bank['model_identity']
    source_sha = sha256(a.source_wo)
    source = torch.load(a.source_wo, map_location='cpu', weights_only=True)
    assert source['upstream_v_factor_sha256'] == bank['factor_sha256']
    assert source['moment_manifest_sha256'] == manifest_sha and source['tp_size'] == 4
    gdn_ids = sorted(r['layer_index'] for r in source['gdn']['layers'])
    full_ids = sorted(r['layer_index'] for r in source['full_attention']['layers'])
    assert len(gdn_ids) == 24 and full_ids == [3, 7, 11, 15, 19, 23, 27, 31]
    assert sorted(gdn_ids + full_ids) == list(range(32))
    if a.stage == 'fit':
        for i in gdn_ids[a.shard_index::a.num_shards]:
            path = output / f'wo_l{i:02d}.pt'
            if path.exists():
                record = torch.load(path, map_location='cpu', weights_only=True)
                assert record['source_wo_sha256'] == source_sha
                assert record['private_encoders'].shape == (4, 1024, 768)
                assert record['moment_manifest_sha256'] == manifest_sha
                continue
            train = torch.load(root / f'fit_l{i:02d}.pt', map_location=a.device, weights_only=True)
            heldout = torch.load(root / f'heldout_l{i:02d}.pt', map_location=a.device, weights_only=True)
            for moment in (train, heldout):
                assert moment['layer_type'] == 'gdn'
                assert moment['upstream_v_factor_sha256'] == bank['factor_sha256']
                assert moment['windows_sha256'] == bank['windows_sha256']
                assert moment['weight'].shape == (4096, 4096)
            started = time.monotonic()
            fitted = fit_qwen35_private_ag_joint_factors(
                train['weight'], train['second_moment'], heldout['second_moment'],
                tp_size=4, local_rank=768, encoder_sweeps=6, minimum_encoder_sweeps=6,
                work_dtype=torch.float64, factor_dtype=torch.bfloat16)
            record = {'layer_index': i, 'layer_type': 'gdn',
                      'private_encoders': fitted.private_encoders,
                      'joint_decoder_weight': fitted.joint_decoder_weight, 'metrics': fitted.metrics,
                      'work_dtype': 'float64', 'upstream_v_factor_sha256': bank['factor_sha256'],
                      'moment_manifest_sha256': manifest_sha, 'source_wo_sha256': source_sha,
                      'seconds': time.monotonic() - started}
            atomic_save(path, record)
            print(json.dumps({'layer': i, 'seconds': record['seconds'], 'metrics': fitted.metrics}), flush=True)
        return
    gdn = []
    audit = []
    for i in gdn_ids:
        record = torch.load(output / f'wo_l{i:02d}.pt', map_location='cpu', weights_only=True)
        assert record['layer_index'] == i and record['layer_type'] == 'gdn'
        assert record['source_wo_sha256'] == source_sha and record['moment_manifest_sha256'] == manifest_sha
        assert record['upstream_v_factor_sha256'] == bank['factor_sha256']
        assert record['private_encoders'].shape == (4, 1024, 768)
        assert record['joint_decoder_weight'].shape == (4096, 3072)
        assert record['work_dtype'] == 'float64'
        assert record['metrics']['diagnostics']['encoder_sweeps_completed'] == 6
        for key in ('private_encoders', 'joint_decoder_weight'):
            assert record[key].dtype == torch.bfloat16 and torch.isfinite(record[key]).all()
        gdn.append(record)
        previous = next(r for r in source['gdn']['layers'] if r['layer_index'] == i)
        audit.append({'layer': i, 'rank': 768, 'selected_sweep': record['metrics']['selected_sweep'],
                      'heldout_relative_mse': record['metrics']['quantized_heldout_relative_output_mse'],
                      'previous_rank512_heldout_relative_mse': previous['metrics']['quantized_heldout_relative_output_mse'],
                      'maximum_encoder_solve_residual': record['metrics']['diagnostics']['encoder_maximum_relative_residual']})
    for record in source['full_attention']['layers']:
        assert record['private_encoders'].shape == (4, 1024, 512)
        assert record['joint_decoder_weight'].shape == (4096, 2048)
    mixed = {**source, 'gdn': {**source['gdn'], 'layers': gdn},
             'retained_output_input_fraction': 0.6875,
             'retained_output_input_fraction_by_type': {'gdn': 0.75, 'full_attention': 0.5},
             'source_rank_by_type': {'gdn': 768, 'full_attention': 512},
             'source_wo_sha256': source_sha, 'source_wo_path': a.source_wo,
             'construction': 'refit GDN only on identical Dense moments; reuse full-attention tensors exactly'}
    destination = output / 'wo_bank.pt'
    atomic_save(destination, mixed)
    exported = torch.load(destination, map_location='cpu', weights_only=True)
    for original, saved in zip(source['full_attention']['layers'], exported['full_attention']['layers'], strict=True):
        assert original['layer_index'] == saved['layer_index']
        for key in ('private_encoders', 'joint_decoder_weight'):
            assert torch.equal(original[key], saved[key])
    atomic_save(output / 'audit.json', {'status': 'complete', 'bank_sha256': sha256(destination),
                                      'source_wo_sha256': source_sha, 'full_attention_exact_reuse': True,
                                      'moment_manifest_sha256': manifest_sha, 'gdn_rows': audit})
    print(json.dumps({'output': str(destination), 'gdn_layers': 24, 'full_attention_exact_reuse': True}), flush=True)


if __name__ == '__main__':
    main()
