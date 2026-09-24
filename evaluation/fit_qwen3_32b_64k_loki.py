"""Fit Qwen3-32B Loki Key PCA (rank 32) on the dense model over the 32 fit windows of a calibration bank.

Loki is calibrated on the dense model, as the method defines it; this is the one baseline whose calibration
does not see the deployed V96. Keys are taken after Qwen3's k_norm, either before RoPE (`--coordinate pre_rope`,
the formal Llama protocol) or after the YaRN RoPE the evaluator deploys (`--coordinate post_rope`); the runtime
projects post-RoPE Q/K with the basis without mean subtraction. The RoPE geometry comes from
`k_routing_config.routing_config` so it is identical to the evaluator's. Layerwise replay keeps one layer and one
window on GPU.
"""
import argparse
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

PKGROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKGROOT))
from evaluation.build_qwen3_8b_loki_pca import _fit_pca
from evaluation.k_routing_config import routing_config
from evaluation.v96kl_common import configure, read_json, save_tensors, sha256, write_json

FIT_WINDOWS, RANK = 32, 32
LOKI_FORMAT = 'basisserve.qwen3_32b.loki_key_pca_r32.v1'
COORDINATES = dict(pre_rope='centered pre-RoPE K PCA after Qwen3 k_norm (formal Llama protocol coordinate)',
                   post_rope='centered post-RoPE K PCA after Qwen3 k_norm and the deployed YaRN RoPE')
LOKI_RUNTIME_COORDINATE = 'post-RoPE Q/K projection without mean subtraction'
SOURCES = ('evaluation/fit_qwen3_32b_64k_loki.py', 'evaluation/build_qwen3_8b_loki_pca.py', 'evaluation/k_routing_config.py')


def calibration_windows(calibration, identity, sequence_length):
    path = calibration / 'windows.safetensors'
    manifest = read_json(calibration / 'manifest.json')
    assert manifest['status'] == 'complete' and manifest['sha256'] == sha256(path)
    assert manifest['model_config_sha256'] == identity['model_config_sha256']
    assert list(manifest['fit_ids']) == list(range(FIT_WINDOWS))
    windows = load_file(str(path))['input_ids'][:FIT_WINDOWS]
    assert windows.shape == (FIT_WINDOWS, sequence_length)
    return windows, manifest['sha256']


@torch.inference_mode()
def replay(model, sequences, sequence_length, post_rope, visit):
    hidden = [model.model.embed_tokens(tokens.long()) for tokens in sequences]
    positions = torch.arange(sequence_length, device='cuda')[None]
    cos, sin = model.model.rotary_emb.to('cuda')(torch.empty(1, device='cuda', dtype=torch.bfloat16), positions)
    current = {}

    def collect(attention, positional, kwargs):
        x = kwargs['hidden_states']
        n = x.shape[1]
        q = attention.q_norm(attention.q_proj(x).view(1, n, -1, attention.head_dim)).transpose(1, 2)
        k = attention.k_norm(attention.k_proj(x).view(1, n, -1, attention.head_dim)).transpose(1, 2)
        if post_rope:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        visit(current['layer'], current['index'], k[0])

    for layer, module in enumerate(model.model.layers):
        started = time.monotonic()
        module.to('cuda')
        handle = module.self_attn.register_forward_pre_hook(collect, with_kwargs=True)
        for index, state in enumerate(hidden):
            current.update(layer=layer, index=index)
            output = module(state[None].cuda(), attention_mask=None, position_ids=positions,
                            position_embeddings=(cos, sin), use_cache=False)
            state.copy_(output[0].cpu())
            del output
        handle.remove()
        module.to('meta')
        print('layer', layer, 'sequences', len(hidden), 'seconds', round(time.monotonic() - started, 1), flush=True)


def capture(args, identity, config):
    windows, windows_sha256 = calibration_windows(args.calibration, identity, args.sequence_length)
    ids = list(range(args.shard_index, FIT_WINDOWS, args.num_shards))
    path = args.output / 'moments' / f'shard_{args.shard_index:02d}.safetensors'
    if path.with_suffix('.json').exists():
        print('shard complete', path, flush=True)
        return
    model = AutoModelForCausalLM.from_pretrained(identity['model'], config=config, dtype=torch.bfloat16,
                                                 local_files_only=True, attn_implementation='sdpa').eval()
    layers, kv_heads, head_dim = config.num_hidden_layers, identity['hkv'], identity['head_dim']
    assert model.config.model_type == 'qwen3' and layers == len(identity['attention_layers'])
    protocol = dict(model_config_sha256=identity['model_config_sha256'], windows_sha256=windows_sha256,
                    fit_windows=FIT_WINDOWS, sequence_length=args.sequence_length, rope=args.rope,
                    rope_parameters=dict(model.config.rope_parameters), max_position_embeddings=model.config.max_position_embeddings,
                    teacher='dense BF16 Qwen3-32B (no V96); SDPA causal; layerwise replay of complete windows',
                    key=COORDINATES[args.coordinate], coordinate=args.coordinate,
                    accumulation='FP64 per-KV-head sums and uncentered Grams',
                    num_shards=args.num_shards, source_sha256={n: sha256(PKGROOT / n) for n in SOURCES})
    sums = torch.zeros(layers, kv_heads, head_dim, dtype=torch.float64, device='cuda')
    grams = torch.zeros(layers, kv_heads, head_dim, head_dim, dtype=torch.float64, device='cuda')
    counts = torch.zeros(layers, dtype=torch.int64)

    def accumulate(layer, index, key):
        key = key.double()
        sums[layer].add_(key.sum(1))
        grams[layer].add_(key.mT @ key)
        counts[layer] += key.shape[1]

    replay(model, [windows[i] for i in ids], args.sequence_length, args.coordinate == 'post_rope', accumulate)
    assert (counts == len(ids) * args.sequence_length).all()
    assert torch.isfinite(sums).all() and torch.isfinite(grams).all()
    save_tensors(path, dict(sums=sums.cpu(), grams=grams.cpu(), counts=counts))
    write_json(path.with_suffix('.json'), dict(status='complete', shard_index=args.shard_index,
                                               window_ids=ids, protocol=protocol, sha256=sha256(path)))
    print('capture complete', path, flush=True)


def fit(args, identity, config):
    _, windows_sha256 = calibration_windows(args.calibration, identity, args.sequence_length)
    records = [read_json(args.output / 'moments' / f'shard_{i:02d}.json') for i in range(args.num_shards)]
    protocol = records[0]['protocol']
    assert all(r['status'] == 'complete' and r['protocol'] == protocol for r in records)
    assert protocol['num_shards'] == args.num_shards and protocol['windows_sha256'] == windows_sha256
    assert protocol['coordinate'] == args.coordinate and protocol['rope'] == args.rope
    assert sorted(i for r in records for i in r['window_ids']) == list(range(FIT_WINDOWS))
    layers, kv_heads, head_dim = config.num_hidden_layers, identity['hkv'], identity['head_dim']
    sums = torch.zeros(layers, kv_heads, head_dim, dtype=torch.float64)
    grams = torch.zeros(layers, kv_heads, head_dim, head_dim, dtype=torch.float64)
    counts = torch.zeros(layers, dtype=torch.int64)
    moments = {}
    for i, record in enumerate(records):
        path = args.output / 'moments' / f'shard_{i:02d}.safetensors'
        assert sha256(path) == record['sha256']
        moments[path.name] = record['sha256']
        payload = load_file(str(path))
        sums += payload['sums']
        grams += payload['grams']
        counts += payload['counts']
    assert (counts == FIT_WINDOWS * args.sequence_length).all()
    projector, mean, spectrum, retained = _fit_pca(counts, sums, grams, rank=RANK)
    root = args.output / 'pca'
    entries = []
    for layer in range(layers):
        path = root / f'layer_{layer:03d}.safetensors'
        save_tensors(path, dict(projector=projector[layer].bfloat16().contiguous(),
                                mean=mean[layer].contiguous(), spectrum=spectrum[layer].contiguous()))
        entries.append(dict(layer=layer, file=path.name, sha256=sha256(path),
                            retained=retained[layer].tolist(), mean_retained=float(retained[layer].mean())))
    write_json(root / 'manifest.json', dict(
        format=LOKI_FORMAT, status='complete', rank=RANK, fit_windows=FIT_WINDOWS, sequence_length=args.sequence_length,
        fit_ids=list(range(FIT_WINDOWS)), smoke=False, windows_sha256=windows_sha256, rope=args.rope,
        model_config_sha256=identity['model_config_sha256'], coordinate=COORDINATES[args.coordinate],
        runtime=LOKI_RUNTIME_COORDINATE, capture_protocol=protocol, moments_sha256=moments,
        source_sha256={n: sha256(PKGROOT / n) for n in SOURCES}, layers=entries))
    print('fit complete', root, 'mean retained', float(retained.mean()), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('capture', 'fit'))
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--coordinate', choices=tuple(COORDINATES), required=True)
    p.add_argument('--rope', choices=('native', 'yarn2', 'yarn4'), required=True)
    p.add_argument('--sequence-length', type=int, default=65536)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=8)
    args = p.parse_args()
    configure()
    assert 0 <= args.shard_index < args.num_shards
    identity = read_json(args.identity)
    assert identity['status'] == 'complete'
    config = routing_config(identity, rope=args.rope, sequence_length=args.sequence_length)
    {'capture': capture, 'fit': fit}[args.stage](args, identity, config)


if __name__ == '__main__':
    main()
