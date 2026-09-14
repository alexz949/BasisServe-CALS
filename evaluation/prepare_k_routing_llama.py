"""Fetch and audit frozen allocated V96 payloads, optionally prepare C4 windows."""
import argparse
from pathlib import Path
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, sha256
from evaluation.prepare_v96kl_data import windows

MODEL = Path('/deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b')
REVISION = 'f1a6253b5d5c747a2475cbf9e704a67d97930b31'
PREFIX = 'ICLR-results/llama31-8b/checkpoints/L31-8B-C1-R96'
OUTPUT = ROOT / 'results/k_routing_fit/llama31_8b'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-only', action='store_true')
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--prefix', default=PREFIX)
    p.add_argument('--output', type=Path, default=OUTPUT)
    args = p.parse_args()
    configure()
    snapshot = Path(snapshot_download('alexz949/BasisServe-CALS', revision=REVISION,
                                     allow_patterns=[args.prefix + '/*'], max_workers=2))
    checkpoint = snapshot / args.prefix
    manifest = read_json(checkpoint / 'manifest.json')
    config = read_json(args.model / 'config.json')
    compression = manifest['compression']
    assert manifest['status'] == 'complete'
    assert compression['method'] == 'c1-two-sided-kl'
    assert compression['allocation'] == 'two_sided_factorized_terminal_kl_alpha1'
    assert compression['equivalent_rank_target'] == 96
    assert manifest['model']['config_sha256'] == sha256(args.model / 'config.json')
    assert manifest['model']['safetensors_index_sha256'] == sha256(args.model / 'model.safetensors.index.json')
    index = read_json(args.model / 'model.safetensors.index.json')
    assert all((args.model / name).is_file() for name in set(index['weight_map'].values()))
    assert sha256(checkpoint / manifest['artifact']['file']) == manifest['artifact']['sha256']
    layers, hq, hkv = config['num_hidden_layers'], config['num_attention_heads'], config['num_key_value_heads']
    hidden = config['hidden_size']
    dim = config.get('head_dim', hidden // hq)
    ranks = compression['layer_ranks']
    allocation = read_json(checkpoint / manifest['artifact']['file'])['selection']
    assert allocation['selected_candidate'] == 'two_sided_factorized_kl'
    assert allocation['factorized_method']['exponent'] == 1
    assert allocation['selected_schedule'] == ranks
    assert len(ranks) == len(manifest['layers']) == layers
    assert sum(map(sum, ranks)) == layers * hkv * 96
    for i, record in enumerate(manifest['layers']):
        assert record['layer'] == i and record['ranks'] == ranks[i]
        assert len(ranks[i]) == hkv and len(set(ranks[i])) == 1
        path = checkpoint / record['file']
        assert sha256(path) == record['sha256']
        tensors = load_file(str(path))
        rank = ranks[i][0]
        assert tensors['source_ranks'].tolist() == ranks[i]
        assert tensors['value_coordinate_encoders'].shape == (hkv, dim, rank)
        assert tensors['head_output_decoders'].shape == (hq, rank, hidden)
        assert all(torch.isfinite(t).all() for t in tensors.values())
        print('verified layer', i, 'V rank', rank, flush=True)
    write_json(args.output / 'manifests/v96.json', dict(status='complete', checkpoint=str(checkpoint),
        model=str(args.model), repository_revision=REVISION, manifest_sha256=sha256(checkpoint / 'manifest.json'),
        model_config_sha256=sha256(args.model / 'config.json'), attention_layers=list(range(layers)),
        layer_ranks=[r[0] for r in ranks], mean_rank=96,
        hq=hq, hkv=hkv, head_dim=dim, hidden_size=hidden))
    if not args.checkpoint_only:
        windows(argparse.Namespace(model=args.model, calibration=args.output / 'calibration'))


if __name__ == '__main__':
    main()
