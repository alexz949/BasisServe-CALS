#!/usr/bin/env python3
"""Verify all Qwen3-32B Dense-V Base16/Residual16 128K routing factors."""

import argparse
from pathlib import Path

import torch

from evaluation.fit_k_routing_streaming import encoder_for, verified
from evaluation.v96kl_common import configure, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--windows', type=Path, required=True)
    args = parser.parse_args()
    configure()
    identity = read_json(args.identity)
    assert identity['status'] == 'complete' and identity['attention_layers'] == list(range(64))
    manifest = read_json(args.windows.with_name('manifest.json'))
    assert manifest['sha256'] == sha256(args.windows)
    assert manifest['fit_ids'] == list(range(32))
    assert manifest['validation_ids'] == list(range(32, 48))
    hashes, metrics = {}, {}
    for layer in identity['attention_layers']:
        path = args.root / 'ours_b16r16' / f'layer_{layer:03d}.safetensors'
        tensors, meta = verified(path)
        protocol = meta['protocol']
        assert protocol['format'] == 'basisserve.k_router.streaming.v1'
        assert protocol['base_rank'] == protocol['residual_rank'] == 16
        assert protocol['sequence_length'] == 131072 and protocol['rope'] == 'yarn4'
        assert protocol['runtime_config']['max_position_embeddings'] == 131072
        assert protocol['runtime_config']['rope_parameters']['factor'] == 4.0
        assert protocol['dense_v'] and not protocol['smoke']
        assert protocol['fit_ids'] == list(range(32))
        assert protocol['diagnostic_ids'] == list(range(32, 48))
        assert protocol['fit_queries'] == 64 and protocol['diagnostic_queries'] == 32
        assert protocol['windows_sha256'] == manifest['sha256']
        assert protocol['windows_manifest_sha256'] == sha256(args.windows.with_name('manifest.json'))
        assert meta['identity_sha256'] == sha256(args.identity)
        assert meta['v_rank'] == identity['head_dim'] == 128
        assert meta['sweeps'] == 40 and meta['pcg_iterations'] == 100
        assert len(meta['losses']['b16_r16']['sweeps']) == 40
        base_path = args.root / 'base' / path.name
        base, base_meta = verified(base_path)
        assert meta['base_sha256'] == sha256(base_path)
        assert base_meta['protocol'] == protocol and base_meta['identity_sha256'] == meta['identity_sha256']
        assert torch.equal(base['encoder'], encoder_for(identity, layer, dense_v=True))
        for name in ('left', 'right', 'bias'):
            assert torch.equal(tensors[f'base_{name}_b16'], base[name].float())
        assert tensors['residual_encoder_b16_r16'].shape == (8, 128, 16)
        assert tensors['residual_query_b16_r16'].shape == (64, 128, 16)
        assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in tensors.values())
        for name, digest in protocol['source_sha256'].items():
            assert sha256(Path(name)) == digest
        hashes[str(layer)] = meta['sha256']
        metrics[str(layer)] = meta['losses']['b16_r16']
    write_json(args.root / 'manifests' / 'fit_audit.json', dict(
        status='complete', layers=64, identity_sha256=sha256(args.identity),
        bank_sha256=hashes, metrics=metrics,
        scope='Dense-V Base16/Residual16 tensors, YaRN4 runtime, 32x128K fit plus 16x128K held-out'))
    print('Verified all 64 Qwen3-32B Dense-V B16R16 factors', flush=True)


if __name__ == '__main__':
    main()
