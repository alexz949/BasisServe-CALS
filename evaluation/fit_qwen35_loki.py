"""Centered pre-RoPE Key PCA for Loki, from the same 64 native C4 windows."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from evaluation.build_qwen3_8b_loki_pca import _fit_pca
from evaluation.assemble_qwen35_k_routing_v import LAYERS
from evaluation.qwen35_hybrid_common import atomic_save, sha256


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert 0 <= args.shard_index < args.num_shards
    for layer in LAYERS[args.shard_index::args.num_shards]:
        destination = args.output/f'l{layer:02d}.pt'
        sums = torch.zeros(1, 4, 256, dtype=torch.float64)
        grams = torch.zeros(1, 4, 256, 256, dtype=torch.float64)
        hashes, windows_hash, config_hash = {}, None, None
        for index in range(64):
            root = args.capture/f'w{index:03d}'
            manifest = json.loads((root/'manifest.json').read_text())
            assert manifest['status'] == 'complete' and manifest['index'] == index and manifest['split'] == 'fit'
            assert manifest['wo_compression'] is False and manifest['length'] == 32768
            windows_hash = windows_hash or manifest['windows_sha256']
            config_hash = config_hash or manifest['model_config_sha256']
            assert manifest['windows_sha256'] == windows_hash and manifest['model_config_sha256'] == config_hash
            path = root/f'l{layer:02d}.safetensors'
            hashes[str(path)] = sha256(path)
            assert hashes[str(path)] == manifest['files'][path.name]
            with safe_open(str(path), framework='pt', device='cpu') as stream:
                keys = stream.get_tensor('pre_rope_keys')[0].cuda().float().transpose(0, 1)
            assert keys.shape == (4, 32768, 256) and torch.isfinite(keys).all()
            sums[0] += keys.sum(1).double().cpu()
            grams[0] += (keys.mT@keys).double().cpu()
            print(json.dumps(dict(layer=layer, window=index)), flush=True)
        projector, mean, spectrum, retained = _fit_pca(torch.tensor([64*32768]), sums, grams, rank=32)
        record = dict(status='complete', format='basisserve.qwen35.loki_key_pca.v1', layer=layer,
            rank=32, projector=projector[0].bfloat16(), mean=mean[0], spectrum=spectrum[0],
            retained=retained[0], fit_windows=64, sequence_length=32768, windows_sha256=windows_hash,
            model_config_sha256=config_hash, capture_hashes=hashes, wo_compression=False,
            pca_coordinate='pre-RoPE after K RMSNorm', runtime_coordinate='post-RoPE Q/K without mean subtraction')
        if destination.exists():
            old = torch.load(destination, map_location='cpu', weights_only=True)
            assert old['status'] == 'complete' and old['capture_hashes'] == hashes
            torch.testing.assert_close(old['projector'], record['projector'], atol=0, rtol=0)
        else:
            atomic_save(destination, record)


if __name__ == '__main__':
    main()
