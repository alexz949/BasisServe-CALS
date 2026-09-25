"""Wrap the payload of scripts/build_qwen35_gdn_private_ag_joint_factors.py into the released bank layout
(top-level status + protocol, per-layer status) that prepare_qwen35_128k_identity.py and the HF checkpoints use."""
import argparse
from pathlib import Path
import torch
from evaluation.qwen35_hybrid_common import sha256


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--condition', default='c4_retrieval_50_50 (mix48 fit windows 0-31: 16 C4 + 16 synthetic retrieval, 128K)')
    args = p.parse_args()
    raw = torch.load(args.raw, map_location='cpu', weights_only=False)
    assert raw['format'] == 'basisserve.qwen35.gdn_private_ag_joint_factors.v2' and int(raw['schema_version']) == 1
    layers = []
    for layer in raw['layers']:
        assert layer['private_encoders'].ndim == 3 and layer['joint_decoder_weight'].ndim == 2
        layers.append(dict(status='complete', **layer))
    assert len(layers) == 24
    fit = raw['fit_collection']; als = raw['als']
    protocol = dict(tp_size=int(raw['tp_size']), local_rank=int(raw['local_rank']), gathered_width=int(raw['total_private_rank']),
                    rank_allocation='uniform', encoder_sweeps=int(als['encoder_sweeps']), minimum_encoder_sweeps=int(als['minimum_encoder_sweeps']),
                    encoder_solver='exact_single_source_two_sided_cholesky', work_dtype=str(als.get('work_dtype', 'float64')),
                    trajectory='native_dense', upstream_compression=None, fit_windows=int(fit['num_samples']),
                    sequence_length=int(fit['sequence_length']), windows_sha256=fit['windows_sha256'], calibration_condition=args.condition,
                    selection_data='fitting moments only (passed as held-out too); fixed 6 encoder sweeps', factor_dtype=raw['factor_dtype'],
                    objective=raw['objective'], builder_command=raw.get('command'), raw_bank_sha256=sha256(args.raw))
    torch.save(dict(format=raw['format'], schema_version=raw['schema_version'], status='complete', protocol=protocol, layers=layers,
                    geometry=raw['geometry'], als=als, environment=raw.get('environment')), args.output)
    print(dict(output=str(args.output), layers=len(layers), windows_sha256=protocol['windows_sha256'][:16], tp=protocol['tp_size'], rank=protocol['local_rank']))


if __name__ == '__main__':
    main()
