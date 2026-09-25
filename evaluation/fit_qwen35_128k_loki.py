"""Qwen3.5-9B 128K Loki Key PCA (rank 32) on the dense model, streamed from the router's calibration windows.
Loki is calibrated on the dense model, as the method defines it: keys are taken pre-RoPE after K RMSNorm (the k_norm
output of every full-attention layer), the PCA is centered, and the runtime projects post-RoPE Q/K with the basis
without mean subtraction (basisserve.core.qwen35_k_routing_runtime, arm 'loki'). Nothing but per-layer, per-head
key sums and Grams touches the disk. Stage `moments` accumulates one window shard; stage `fit` merges the shards and
writes one bank record per layer in the basisserve.qwen35.loki_key_pca.v1 format that the evaluator loads."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from evaluation.assemble_qwen35_k_routing_v import LAYERS
from evaluation.build_qwen3_8b_loki_pca import _fit_pca
from evaluation.qwen35_hybrid_common import atomic_save, load_model, sha256

FORMAT = 'basisserve.qwen35.loki_key_pca.v1'
MOMENTS_FORMAT = 'basisserve.qwen35.loki_key_moments.v1'
RANK, KV_HEADS, HEAD_DIM = 32, 4, 256
PCA_COORDINATE = 'centered pre-RoPE K PCA after K RMSNorm (k_norm output), dense model'
RUNTIME_COORDINATE = 'post-RoPE Q/K projection without mean subtraction'


def read_windows(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['artifact']['tensor'] == 'input_ids'
    path = root / manifest['artifact']['file']
    digest = sha256(path)
    assert digest == manifest['artifact']['sha256']
    with safe_open(str(path), framework='pt', device='cpu') as stream:
        windows = stream.get_tensor('input_ids')
    assert list(windows.shape) == manifest['shape']
    return windows, [int(i) for i in manifest['fit_ids']], digest, manifest['model_config_sha256']


class KeyMoments:
    """Forward hooks on k_norm: sums [layers, heads, dim] and Grams [layers, heads, dim, dim] in float64 on the GPU."""

    def __init__(self, model, layers, device):
        self.rows = 0
        self.sums = torch.zeros(len(layers), KV_HEADS, HEAD_DIM, dtype=torch.float64, device=device)
        self.grams = torch.zeros(len(layers), KV_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float64, device=device)
        self.handles = [model.model.layers[layer].self_attn.k_norm.register_forward_hook(self.hook(slot))
                        for slot, layer in enumerate(layers)]

    def hook(self, slot):
        def accumulate(module, inputs, output):
            keys = output[0].double().transpose(0, 1)  # [heads, tokens, dim]; k_norm output is [1, tokens, heads, dim]
            assert keys.shape[0] == KV_HEADS and keys.shape[2] == HEAD_DIM and torch.isfinite(keys).all()
            self.sums[slot] += keys.sum(1)
            self.grams[slot] += keys.mT @ keys
            if slot == 0:
                self.rows += keys.shape[1]
        return accumulate

    def remove(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def moments(args):
    windows, fit_ids, windows_hash, config_hash = read_windows(args.windows)
    selected = fit_ids[args.shard_index::args.num_shards]
    destination = args.output / 'moments' / f'shard{args.shard_index:02d}.pt'
    assert not destination.exists(), destination
    model = load_model(str(args.model), 'cuda:0')
    assert sha256(args.model / 'config.json') == config_hash
    collector = KeyMoments(model, LAYERS, 'cuda:0')
    for index in selected:
        model.model(windows[index:index + 1].long().cuda(), use_cache=False)
        print(json.dumps(dict(shard=args.shard_index, window=index, rows=collector.rows)), flush=True)
    collector.remove()
    assert collector.rows == len(selected) * windows.shape[1]
    atomic_save(destination, dict(status='complete', format=MOMENTS_FORMAT, layers=list(LAYERS), windows=selected,
        rows=collector.rows, sums=collector.sums.cpu(), grams=collector.grams.cpu(), windows_sha256=windows_hash,
        model_config_sha256=config_hash, sequence_length=int(windows.shape[1]), shard_index=args.shard_index,
        num_shards=args.num_shards))


def fit(args):
    _, fit_ids, windows_hash, config_hash = read_windows(args.windows)
    parts = sorted((args.output / 'moments').glob('shard*.pt'))
    assert len(parts) == args.num_shards, [p.name for p in parts]
    rows, covered, hashes = 0, [], {}
    sums = torch.zeros(len(LAYERS), KV_HEADS, HEAD_DIM, dtype=torch.float64)
    grams = torch.zeros(len(LAYERS), KV_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float64)
    for path in parts:
        part = torch.load(path, map_location='cpu', weights_only=True)
        assert part['status'] == 'complete' and part['format'] == MOMENTS_FORMAT and part['layers'] == list(LAYERS)
        assert part['windows_sha256'] == windows_hash and part['model_config_sha256'] == config_hash
        assert part['num_shards'] == args.num_shards
        rows += part['rows']
        covered += part['windows']
        sums += part['sums']
        grams += part['grams']
        hashes[path.name] = sha256(path)
    assert sorted(covered) == fit_ids
    sequence_length = part['sequence_length']
    assert rows == len(fit_ids) * sequence_length
    projector, mean, spectrum, retained = _fit_pca(torch.tensor([rows] * len(LAYERS), dtype=torch.float64), sums, grams, rank=RANK)
    for slot, layer in enumerate(LAYERS):
        record = dict(status='complete', format=FORMAT, layer=layer, rank=RANK, projector=projector[slot].bfloat16(),
            mean=mean[slot], spectrum=spectrum[slot], retained=retained[slot], fit_windows=len(fit_ids),
            fit_ids=fit_ids, sequence_length=sequence_length, windows_sha256=windows_hash, model_config_sha256=config_hash,
            moments_sha256=hashes, wo_compression=False, pca_coordinate=PCA_COORDINATE, runtime_coordinate=RUNTIME_COORDINATE)
        atomic_save(args.output / 'bank' / f'l{layer:02d}.pt', record)
        print(json.dumps(dict(layer=layer, retained=[round(float(x), 4) for x in retained[slot]],
                              top_eigenvalue=[round(float(x), 3) for x in spectrum[slot, :, 0]])), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=('moments', 'fit'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--windows', type=Path, required=True, help='calibration window directory (manifest.json + windows.safetensors)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=8)
    args = p.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (moments if args.stage == 'moments' else fit)(args)


if __name__ == '__main__':
    main()
