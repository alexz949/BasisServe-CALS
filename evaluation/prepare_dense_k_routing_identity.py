#!/usr/bin/env python3
"""Freeze a dense-V model identity for K-routing fitting and evaluation."""

import argparse
from pathlib import Path

from transformers import AutoConfig

from evaluation.v96kl_common import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    config_path = args.model / 'config.json'
    index_path = args.model / 'model.safetensors.index.json'
    manifest_path = args.checkpoint / 'manifest.json'
    assert config_path.is_file() and index_path.is_file() and manifest_path.is_file()
    index = read_json(index_path)
    shards = sorted(set(index['weight_map'].values()))
    assert shards and all((args.model / shard).is_file() for shard in shards)
    manifest = read_json(manifest_path)
    assert manifest['status'] == 'complete' and manifest['compression']['method'] == 'dense'
    assert not manifest['layers'] and manifest['model']['config_sha256'] == sha256(config_path)

    config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    assert config.model_type in ('llama', 'qwen3')
    head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    layers = list(range(config.num_hidden_layers))
    write_json(args.output, dict(
        status='complete', format='basisserve.dense_v_k_routing_identity.v1',
        model=str(args.model.resolve()), checkpoint=str(args.checkpoint.resolve()),
        manifest_sha256=sha256(manifest_path), model_config_sha256=sha256(config_path),
        model_index_sha256=sha256(index_path), model_shards=shards,
        attention_layers=layers, layer_ranks=[head_dim] * len(layers),
        hq=config.num_attention_heads, hkv=config.num_key_value_heads,
        head_dim=head_dim, hidden_size=config.hidden_size,
        value_mode='dense original V and Wo'))
    print('Prepared dense-V routing identity', args.output, flush=True)


if __name__ == '__main__':
    main()
