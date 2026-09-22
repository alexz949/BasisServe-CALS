"""Fit Llama-3.1-8B-Instruct Loki Key PCA (rank 32) on post-RoPE Keys over 32 x 128K C4 windows.

Loki is calibrated on the dense model, as the method defines it; this is the one
baseline whose calibration does not see the deployed V96. Keys are taken after
RoPE (Llama has no k_norm), and the runtime projects post-RoPE Q/K with the basis
without mean subtraction. Layerwise replay keeps one layer and one window on GPU.
"""
import argparse
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

PKGROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKGROOT))
from evaluation.build_qwen3_8b_loki_pca import _fit_pca
from evaluation.v96kl_common import configure, read_json, save_tensors, sha256, write_json

FIT_WINDOWS, RANK, LAYERS, KV_HEADS, HEAD_DIM, SEQUENCE_LENGTH = 32, 32, 32, 8, 128, 131072
LOKI_FORMAT = 'basisserve.llama31_8b_instruct.loki_key_pca_r32_128k.v1'
LOKI_FIT_COORDINATE = 'centered pre-RoPE K PCA (Llama-3.1; no k_norm); matches the formal l31-v96-ruler128k Loki protocol'
LOKI_RUNTIME_COORDINATE = 'post-RoPE Q/K projection without mean subtraction'
SOURCES = ('evaluation/fit_llama31_8b_128k_loki.py', 'evaluation/build_qwen3_8b_loki_pca.py')


def calibration_windows(calibration, model_path):
    path = calibration / 'windows.safetensors'
    manifest = read_json(calibration / 'manifest.json')
    assert manifest['status'] == 'complete' and manifest['sha256'] == sha256(path)
    assert manifest['model_config_sha256'] == sha256(model_path / 'config.json')
    fit_ids = list(manifest['fit_ids'])
    assert fit_ids == list(range(FIT_WINDOWS))
    windows = load_file(str(path))['input_ids'][:FIT_WINDOWS]
    assert windows.shape == (FIT_WINDOWS, SEQUENCE_LENGTH)
    return windows, manifest['sha256']


@torch.inference_mode()
def replay(model, sequences, visit):
    hidden = [model.model.embed_tokens(tokens.long()) for tokens in sequences]
    positions = torch.arange(SEQUENCE_LENGTH, device='cuda')[None]
    cos, sin = model.model.rotary_emb.to('cuda')(torch.empty(1, device='cuda', dtype=torch.bfloat16), positions)
    current = {}

    def collect(attention, positional, kwargs):
        x = kwargs['hidden_states']
        n = x.shape[1]
        q = attention.q_proj(x).view(1, n, -1, attention.head_dim).transpose(1, 2)
        k = attention.k_proj(x).view(1, n, -1, attention.head_dim).transpose(1, 2)
        # Formal protocol: the PCA basis is fit on centered pre-RoPE keys; the runtime projects post-RoPE Q/K.
        visit(current['layer'], current['index'], q[0], k[0])

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


def capture(args):
    windows, windows_sha256 = calibration_windows(args.calibration, args.model)
    ids = list(range(args.shard_index, FIT_WINDOWS, args.num_shards))
    path = args.output / 'moments' / f'shard_{args.shard_index:02d}.safetensors'
    if path.with_suffix('.json').exists():
        print('shard complete', path, flush=True)
        return
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation='sdpa').eval()
    assert model.config.model_type == 'llama' and model.config.num_hidden_layers == LAYERS
    protocol = dict(model_config_sha256=sha256(args.model / 'config.json'), windows_sha256=windows_sha256,
                    fit_windows=FIT_WINDOWS, sequence_length=SEQUENCE_LENGTH,
                    rope_parameters=dict(model.config.rope_parameters), max_position_embeddings=model.config.max_position_embeddings,
                    teacher='dense BF16 Llama-3.1-8B-Instruct (no V96); SDPA causal; layerwise replay of complete windows',
                    key=LOKI_FIT_COORDINATE, accumulation='FP64 per-KV-head sums and uncentered Grams',
                    num_shards=args.num_shards, source_sha256={n: sha256(PKGROOT / n) for n in SOURCES})
    sums = torch.zeros(LAYERS, KV_HEADS, HEAD_DIM, dtype=torch.float64, device='cuda')
    grams = torch.zeros(LAYERS, KV_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float64, device='cuda')
    counts = torch.zeros(LAYERS, dtype=torch.int64)

    def accumulate(layer, index, query, key):
        key = key.double()
        sums[layer].add_(key.sum(1))
        grams[layer].add_(key.mT @ key)
        counts[layer] += key.shape[1]

    replay(model, [windows[i] for i in ids], accumulate)
    assert (counts == len(ids) * SEQUENCE_LENGTH).all()
    assert torch.isfinite(sums).all() and torch.isfinite(grams).all()
    save_tensors(path, dict(sums=sums.cpu(), grams=grams.cpu(), counts=counts))
    write_json(path.with_suffix('.json'), dict(status='complete', shard_index=args.shard_index,
                                               window_ids=ids, protocol=protocol, sha256=sha256(path)))
    print('capture complete', path, flush=True)


def fit(args):
    _, windows_sha256 = calibration_windows(args.calibration, args.model)
    records = [read_json(args.output / 'moments' / f'shard_{i:02d}.json') for i in range(args.num_shards)]
    protocol = records[0]['protocol']
    assert all(r['status'] == 'complete' and r['protocol'] == protocol for r in records)
    assert protocol['num_shards'] == args.num_shards and protocol['windows_sha256'] == windows_sha256
    assert sorted(i for r in records for i in r['window_ids']) == list(range(FIT_WINDOWS))
    sums = torch.zeros(LAYERS, KV_HEADS, HEAD_DIM, dtype=torch.float64)
    grams = torch.zeros(LAYERS, KV_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float64)
    counts = torch.zeros(LAYERS, dtype=torch.int64)
    moments = {}
    for i, record in enumerate(records):
        path = args.output / 'moments' / f'shard_{i:02d}.safetensors'
        assert sha256(path) == record['sha256']
        moments[path.name] = record['sha256']
        payload = load_file(str(path))
        sums += payload['sums']
        grams += payload['grams']
        counts += payload['counts']
    assert (counts == FIT_WINDOWS * SEQUENCE_LENGTH).all()
    projector, mean, spectrum, retained = _fit_pca(counts, sums, grams, rank=RANK)
    root = args.output / 'pca'
    layers = []
    for layer in range(LAYERS):
        path = root / f'layer_{layer:03d}.safetensors'
        save_tensors(path, dict(projector=projector[layer].bfloat16().contiguous(),
                                mean=mean[layer].contiguous(), spectrum=spectrum[layer].contiguous()))
        layers.append(dict(layer=layer, file=path.name, sha256=sha256(path),
                           retained=retained[layer].tolist(), mean_retained=float(retained[layer].mean())))
    write_json(root / 'manifest.json', dict(
        format=LOKI_FORMAT, status='complete', rank=RANK, fit_windows=FIT_WINDOWS, sequence_length=SEQUENCE_LENGTH,
        fit_ids=list(range(FIT_WINDOWS)), smoke=False, windows_sha256=windows_sha256,
        model_config_sha256=protocol['model_config_sha256'], coordinate=LOKI_FIT_COORDINATE,
        runtime=LOKI_RUNTIME_COORDINATE, capture_protocol=protocol, moments_sha256=moments,
        source_sha256={n: sha256(PKGROOT / n) for n in SOURCES}, layers=layers))
    print('fit complete', root, 'mean retained', float(retained.mean()), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('capture', 'fit'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=8)
    args = p.parse_args()
    configure()
    assert 0 <= args.shard_index < args.num_shards
    {'capture': capture, 'fit': fit}[args.stage](args)


if __name__ == '__main__':
    main()
