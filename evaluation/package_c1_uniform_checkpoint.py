"""Package a merged uniform-rank C1 ALS factor bank into the K-routing evaluator's checkpoint contract.

evaluation/eval_k_routing_ruler.py install() needs only:
  - checkpoint/manifest.json: {status: complete, layers: [{layer, file, ranks, sha256}]}
  - each layer file: value_coordinate_encoders (hkv,head_dim,rank), head_output_decoders
    (hq,rank,hidden), source_ranks (hkv,) int64 equal to that layer's `ranks`.
Layers are matched by index, so hybrid models with non-contiguous attention layers (Nemotron-H) package
the same way as dense ones. Also writes the identity JSON the evaluators and router fitters consume.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--factor-bank-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--model', required=True, type=Path, help='base model snapshot (config.json is hashed into the identity)')
    p.add_argument('--identity', required=True, type=Path, help='identity JSON to write')
    p.add_argument('--rank', type=int, default=96)
    p.add_argument('--format', default='basisserve.c1_uniform_manual_package.v1')
    args = p.parse_args()
    config = json.loads((args.model / 'config.json').read_text())
    hq, hkv, hidden = config['num_attention_heads'], config['num_key_value_heads'], config['hidden_size']
    head_dim = config.get('attention_head_dim') or config.get('head_dim') or hidden // hq
    results = json.loads((args.factor_bank_dir / 'results.json').read_text())
    assert results['status'] == 'complete'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranks = [args.rank] * hkv
    layers = []
    for record in sorted(results['records'], key=lambda r: int(r['layer'])):
        layer_index = int(record['layer'])
        src = args.factor_bank_dir / record['artifact']['file']
        assert sha256(src) == record['artifact']['sha256']
        tensors = load_file(str(src))
        assert tensors['value_coordinate_encoders'].shape == (hkv, head_dim, args.rank)
        assert tensors['head_output_decoders'].shape == (hq, args.rank, hidden)
        tensors['source_ranks'] = torch.tensor(ranks, dtype=torch.int64)
        dest = args.output_dir / f'layer_{layer_index:03d}.safetensors'
        save_file(tensors, str(dest))
        layers.append(dict(layer=layer_index, file=dest.name, ranks=ranks, sha256=sha256(dest)))
    assert layers and len({entry['layer'] for entry in layers}) == len(layers)
    manifest_path = args.output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(dict(
        status='complete', format=args.format, model=str(args.model.resolve()),
        model_config_sha256=sha256(args.model / 'config.json'),
        source_results_sha256=sha256(args.factor_bank_dir / 'results.json'), layers=layers), indent=1))
    args.identity.parent.mkdir(parents=True, exist_ok=True)
    args.identity.write_text(json.dumps(dict(
        status='complete', checkpoint=str(args.output_dir.resolve()), model=str(args.model.resolve()),
        model_config_sha256=sha256(args.model / 'config.json'), manifest_sha256=sha256(manifest_path),
        attention_layers=[entry['layer'] for entry in layers], layer_ranks=[args.rank] * len(layers),
        mean_rank=args.rank, hq=hq, hkv=hkv, head_dim=head_dim, hidden_size=hidden,
        value_mode=f'C1 uniform V{args.rank} encoder and decoder'), indent=1))
    print('packaged checkpoint', args.output_dir, len(layers), 'layers; identity', args.identity, flush=True)


if __name__ == '__main__':
    main()
