"""Fit Nemotron-H Loki Key PCA (rank 32) on the dense model's attention Keys over the 128K fit windows.

Loki is calibrated on the dense model, as the method defines it; this is the one baseline whose
calibration does not see the deployed V96 + Wo. Nemotron-H attention has no RoPE and no k_norm, so the
fit coordinate is the raw k_proj output, and the runtime projects raw Q/K with the basis without mean
subtraction. The whole model runs on one GPU (four attention layers only), so full forwards replace the
layerwise replay used for Llama; the Mamba chunk scan uses the Triton drop-in.
"""
import argparse
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.build_qwen3_8b_loki_pca import _fit_pca
from evaluation.v96kl_common import configure, read_json, save_tensors, sha256, write_json
from evaluation import nemotron_h_triton_mamba as triton_mamba
from basisserve.checkpoint.c1_attention_layers import c1_attention_layers

RANK = 32
LOKI_FORMAT = 'basisserve.nemotron_h.loki_key_pca_r32_128k.v1'
LOKI_FIT_COORDINATE = 'centered raw K PCA (Nemotron-H attention: no RoPE, no k_norm)'
LOKI_RUNTIME_COORDINATE = 'raw Q/K projection without mean subtraction'
SOURCES = ('evaluation/fit_nemotron_h_128k_loki.py', 'evaluation/build_qwen3_8b_loki_pca.py',
           'evaluation/nemotron_h_triton_mamba.py')


def calibration_windows(calibration, model_path):
    path = calibration / 'windows.safetensors'
    manifest = read_json(calibration / 'manifest.json')
    assert manifest['status'] == 'complete' and manifest['sha256'] == sha256(path)
    assert manifest['model_config_sha256'] == sha256(model_path / 'config.json')
    fit_ids = list(manifest['fit_ids'])
    assert fit_ids == list(range(len(fit_ids)))
    windows = load_file(str(path))['input_ids'][fit_ids]
    return windows, manifest['sha256'], fit_ids


@torch.inference_mode()
def capture(args):
    windows, windows_sha256, fit_ids = calibration_windows(args.calibration, args.model)
    sequence_length = int(windows.shape[1])
    ids = fit_ids[args.shard_index::args.num_shards]
    path = args.output / 'moments' / f'shard_{args.shard_index:02d}.safetensors'
    if path.with_suffix('.json').exists():
        print('shard complete', path, flush=True)
        return
    triton_mamba.install()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation='sdpa', trust_remote_code=False).cuda().eval()
    assert model.config.model_type == 'nemotron_h'
    triton_mamba.restore_dt_limit(model)
    attention = c1_attention_layers(model)
    layers = [index for index, _ in attention]
    kv_heads, head_dim = model.config.num_key_value_heads, attention[0][1].head_dim
    protocol = dict(model_config_sha256=sha256(args.model / 'config.json'), windows_sha256=windows_sha256,
                    fit_windows=len(fit_ids), sequence_length=sequence_length, attention_layers=layers,
                    teacher='dense BF16 Nemotron-H (no V96, no Wo); SDPA causal; full forwards on one GPU; '
                            'Mamba chunk scan via vLLM Triton drop-in',
                    key=LOKI_FIT_COORDINATE, accumulation='FP64 per-KV-head sums and uncentered Grams', dt_limit=list(map(str, triton_mamba.DT_LIMIT)),
                    num_shards=args.num_shards, source_sha256={n: sha256(ROOT / n) for n in SOURCES})
    sums = torch.zeros(len(layers), kv_heads, head_dim, dtype=torch.float64, device='cuda')
    grams = torch.zeros(len(layers), kv_heads, head_dim, head_dim, dtype=torch.float64, device='cuda')
    counts = torch.zeros(len(layers), dtype=torch.int64)
    handles = []
    for slot, (index, module) in enumerate(attention):
        def collect(attention, positional, kwargs, slot=slot):
            x = kwargs['hidden_states'] if 'hidden_states' in kwargs else positional[0]
            key = attention.k_proj(x).view(x.shape[1], kv_heads, head_dim).transpose(0, 1).double()
            sums[slot].add_(key.sum(1))
            grams[slot].add_(key.mT @ key)
            counts[slot] += key.shape[1]
        handles.append(module.register_forward_pre_hook(collect, with_kwargs=True))
    started = time.monotonic()
    for offset, index in enumerate(ids):
        output = model.model(input_ids=windows[offset:offset + 1].long().cuda(), use_cache=False)
        assert torch.isfinite(output.last_hidden_state).all()
        del output
        print('window', index, f'{time.monotonic() - started:.0f}s', flush=True)
    for handle in handles:
        handle.remove()
    assert (counts == len(ids) * sequence_length).all()
    assert torch.isfinite(sums).all() and torch.isfinite(grams).all()
    save_tensors(path, dict(sums=sums.cpu(), grams=grams.cpu(), counts=counts))
    write_json(path.with_suffix('.json'), dict(status='complete', shard_index=args.shard_index,
                                               window_ids=ids, protocol=protocol, sha256=sha256(path)))
    print('capture complete', path, flush=True)


def fit(args):
    _, windows_sha256, fit_ids = calibration_windows(args.calibration, args.model)
    records = [read_json(args.output / 'moments' / f'shard_{i:02d}.json') for i in range(args.num_shards)]
    protocol = records[0]['protocol']
    assert all(r['status'] == 'complete' and r['protocol'] == protocol for r in records)
    assert protocol['num_shards'] == args.num_shards and protocol['windows_sha256'] == windows_sha256
    assert sorted(i for r in records for i in r['window_ids']) == fit_ids
    layers = protocol['attention_layers']
    sums = grams = counts = None
    moments = {}
    for i, record in enumerate(records):
        path = args.output / 'moments' / f'shard_{i:02d}.safetensors'
        assert sha256(path) == record['sha256']
        moments[path.name] = record['sha256']
        payload = load_file(str(path))
        sums = payload['sums'] if sums is None else sums + payload['sums']
        grams = payload['grams'] if grams is None else grams + payload['grams']
        counts = payload['counts'] if counts is None else counts + payload['counts']
    assert (counts == len(fit_ids) * protocol['sequence_length']).all()
    projector, mean, spectrum, retained = _fit_pca(counts, sums, grams, rank=RANK)
    root = args.output / 'pca'
    entries = []
    for slot, layer in enumerate(layers):
        path = root / f'layer_{layer:03d}.safetensors'
        save_tensors(path, dict(projector=projector[slot].bfloat16().contiguous(),
                                mean=mean[slot].contiguous(), spectrum=spectrum[slot].contiguous()))
        entries.append(dict(layer=layer, file=path.name, sha256=sha256(path),
                            retained=retained[slot].tolist(), mean_retained=float(retained[slot].mean())))
    write_json(root / 'manifest.json', dict(
        format=LOKI_FORMAT, status='complete', rank=RANK, fit_windows=len(fit_ids),
        sequence_length=protocol['sequence_length'], fit_ids=fit_ids, smoke=False, windows_sha256=windows_sha256,
        model_config_sha256=protocol['model_config_sha256'], coordinate=LOKI_FIT_COORDINATE,
        runtime=LOKI_RUNTIME_COORDINATE, capture_protocol=protocol, moments_sha256=moments,
        source_sha256={n: sha256(ROOT / n) for n in SOURCES}, layers=entries))
    print('fit complete', root, 'mean retained', float(retained.mean()), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('capture', 'fit'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    args = p.parse_args()
    configure()
    assert 0 <= args.shard_index < args.num_shards
    {'capture': capture, 'fit': fit}[args.stage](args)


if __name__ == '__main__':
    main()
