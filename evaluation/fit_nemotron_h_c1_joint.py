"""Fit the seven-rank Nemotron attention V bank with ALS12 and fixed CG16."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import fit_llama2_mha_c1_joint as fitter
from evaluation.v96kl_common import configure, read_json, sha256


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='command_name', required=True)
    fit = stages.add_parser('fit-shard')
    fitter._add_common(fit)
    fit.add_argument('--audit', type=Path, required=True)
    fit.add_argument('--layer-shard-index', type=int, required=True)
    fit.add_argument('--layer-shard-count', type=int, default=4)
    fit.add_argument('--device', default='cuda:0')
    fit.add_argument('--torch-num-threads', type=int, default=2)
    fit.add_argument('--resume', action='store_true')
    fit.set_defaults(fit_windows=256, validation_windows=64, work_dtype='float32', factor_dtype='bfloat16')
    merge = stages.add_parser('merge')
    merge.add_argument('--audit', type=Path, required=True)
    merge.add_argument('--output-dir', required=True)
    merge.add_argument('--layers', default='all')
    args = parser.parse_args()
    audit = read_json(args.audit)
    assert audit['status'] == 'complete'
    model_path = Path(audit['model'])
    assert sha256(model_path / 'config.json') == audit['config_sha256']
    config = read_json(model_path / 'config.json')
    head_dim = config.get('head_dim') or config['hidden_size'] // config['num_attention_heads']
    assert (config['model_type'], config['num_hidden_layers'], config['hidden_size'],
        config['num_attention_heads'], config['num_key_value_heads'], head_dim) == (
            'nemotron_h', 98, 8192, 64, 8, 128)
    layers = [row['layer'] for row in audit['layers'] if row['kind'] == 'full_attention']
    fitter.activate_nemotron_h_profile(layers)
    assert set(fitter._parse_layers(args.layers)) <= set(layers)
    if args.command_name == 'fit-shard':
        snapshot = read_json(Path(args.snapshot_dir) / 'manifest.json')
        assert snapshot['status'] == 'complete' and snapshot['dense_teacher']
        assert snapshot['layer_kind'] == 'full_attention' and snapshot['layers'] == layers
        assert snapshot['audit_sha256'] == sha256(args.audit)
        assert snapshot['model']['config_sha256'] == audit['config_sha256']
        assert args.decoder_objective == 'full_layer'
        assert args.encoder_cg_mode == 'fixed' and args.encoder_cg_fixed_iterations == 16
        assert args.encoder_cg_max_iterations in (None, 16)
        assert args.cache_rank in (32, 48, 64, 80, 96, 112, 128)
        assert 0 <= args.layer_shard_index < args.layer_shard_count <= len(layers)
        fitter._fit_shard(args)
    else:
        fitter._merge(args)


if __name__ == '__main__':
    main()
