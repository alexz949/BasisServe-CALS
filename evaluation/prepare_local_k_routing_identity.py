"""Verify a local two-sided KL checkpoint and bind it to a routing experiment."""
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from evaluation.v96kl_common import configure, read_json, write_json, sha256


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    configure()
    model, checkpoint = args.model.resolve(), args.checkpoint.resolve()
    config = read_json(model / 'config.json')
    manifest = read_json(checkpoint / 'manifest.json')
    assert manifest['status'] == 'complete'
    assert manifest['model']['config_sha256'] == sha256(model / 'config.json')
    assert manifest['model']['safetensors_index_sha256'] == sha256(model / 'model.safetensors.index.json')
    artifact = checkpoint / manifest['artifact']['file']
    assert sha256(artifact) == manifest['artifact']['sha256']
    selection = read_json(artifact)['selection']
    assert selection['selected_candidate'] == 'two_sided_factorized_kl'
    assert selection['forced_selected_candidate'] == 'two_sided_factorized_kl'
    ranks = selection['selected_schedule']
    layers, hq, hkv = config['num_hidden_layers'], config['num_attention_heads'], config['num_key_value_heads']
    hidden = config['hidden_size']
    dim = hidden // hq
    assert len(ranks) == len(manifest['layers']) == layers
    assert sum(map(sum, ranks)) == layers * hkv * 96
    assert manifest['compression']['layer_ranks'] == ranks
    for i, record in enumerate(manifest['layers']):
        rank = ranks[i][0]
        assert record['layer'] == i and record['ranks'] == ranks[i]
        assert ranks[i] == [rank] * hkv and rank in (64, 80, 96, 112, 128)
        path = checkpoint / record['file']
        assert sha256(path) == record['sha256']
        tensors = load_file(str(path))
        assert tensors['source_ranks'].tolist() == ranks[i]
        assert tensors['value_coordinate_encoders'].shape == (hkv, dim, rank)
        assert tensors['head_output_decoders'].shape == (hq, rank, hidden)
        assert all(torch.isfinite(t).all() for t in tensors.values())
    write_json(args.output, dict(status='complete', checkpoint=str(checkpoint), model=str(model),
        manifest_sha256=sha256(checkpoint / 'manifest.json'),
        model_config_sha256=sha256(model / 'config.json'), attention_layers=list(range(layers)),
        layer_ranks=[r[0] for r in ranks], mean_rank=96, hq=hq, hkv=hkv,
        head_dim=dim, hidden_size=hidden, source='local two-sided KL allocation'))


if __name__ == '__main__':
    main()
