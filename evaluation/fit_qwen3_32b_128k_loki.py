#!/usr/bin/env python3
"""Fit Qwen3-32B Loki Key PCA on post-RoPE Keys over the 32 x 128K C4 windows.

Loki's default (``--rotary-type postrotary``) fits the PCA on the Keys attention
actually scores, after k_norm and RoPE, and projects post-RoPE Q/K with that basis
at runtime. The runtime skips mean subtraction: q.(k - mu) differs from q.k by a
per-query constant, so the Top-K support is unchanged.

  capture  replays the dense BF16 teacher one decoder layer at a time over this
           shard's windows and accumulates per-layer post-RoPE Key sums and Grams
           in FP64.
  fit      merges every shard and writes the rank-32 checkpoint that
           eval_qwen3_32b_v96_ruler.py loads.
  overlap  smoke diagnostic on frozen RULER prompts: Top-K support chosen by this
           PCA and by a baseline PCA, scored against exact-QK Top-K.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.build_qwen3_8b_loki_pca import _fit_pca
from evaluation.eval_qwen3_32b_v96_ruler import (
    LOKI_FIT_COORDINATE,
    LOKI_FORMAT,
    LOKI_RUNTIME_COORDINATE,
    LOKI_TOPK,
    SEQUENCE_LENGTH,
    effective_config,
)
from evaluation.v96kl_common import code_hashes, configure, read_json, save_tensors, sha256, write_json


FIT_WINDOWS = 32
RANK = 32
LAYERS = 64
KV_HEADS = 8
HEAD_DIM = 128
SOURCES = [
    'evaluation/fit_qwen3_32b_128k_loki.py',
    'evaluation/eval_qwen3_32b_v96_ruler.py',
    'evaluation/build_qwen3_8b_loki_pca.py',
]


def load_teacher(model_path):
    """Dense BF16 Qwen3-32B in host RAM under the evaluator's static YaRN 4.0 rope."""
    return AutoModelForCausalLM.from_pretrained(
        model_path, config=effective_config(model_path), dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').eval()


@torch.inference_mode()
def replay(model, sequences, visit):
    """Run the teacher one decoder layer at a time over host-resident token sequences.

    ``visit(layer, index, query, key)`` receives the post-RoPE Q ``[heads, tokens, dim]``
    and K ``[kv heads, tokens, dim]`` that the layer's attention scores for
    ``sequences[index]``. Hidden states stay in host RAM and each layer is released
    after its last sequence, so one GPU holds a single layer and one sequence.
    """
    hidden = [model.model.embed_tokens(tokens.long()) for tokens in sequences]
    longest = max(len(tokens) for tokens in sequences)
    positions = torch.arange(longest, device='cuda')[None]
    cos, sin = model.model.rotary_emb.to('cuda')(
        torch.empty(1, device='cuda', dtype=torch.bfloat16), positions)
    current = {}

    def collect(attention, positional, kwargs):
        x = kwargs['hidden_states']
        n = x.shape[1]
        q = attention.q_norm(attention.q_proj(x).view(1, n, -1, attention.head_dim)).transpose(1, 2)
        k = attention.k_norm(attention.k_proj(x).view(1, n, -1, attention.head_dim)).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
        visit(current['layer'], current['index'], q[0], k[0])

    for layer, module in enumerate(model.model.layers):
        started = time.monotonic()
        module.to('cuda')
        handle = module.self_attn.register_forward_pre_hook(collect, with_kwargs=True)
        for index, state in enumerate(hidden):
            current.update(layer=layer, index=index)
            n = state.shape[0]
            output = module(state[None].cuda(), attention_mask=None, position_ids=positions[:, :n],
                            position_embeddings=(cos[:, :n], sin[:, :n]), use_cache=False)
            state.copy_(output[0].cpu())
            del output
        handle.remove()
        module.to('meta')
        print('layer', layer, 'sequences', len(hidden), 'seconds', round(time.monotonic() - started, 1), flush=True)


def calibration_windows(calibration, model_path):
    path = calibration / 'windows.safetensors'
    manifest = read_json(calibration / 'manifest.json')
    assert manifest['status'] == 'complete' and manifest['sha256'] == sha256(path)
    assert manifest['model_config_sha256'] == sha256(model_path / 'config.json')
    assert manifest['fit_ids'] == list(range(FIT_WINDOWS)) and manifest['validation_ids'] == []
    windows = load_file(str(path))['input_ids']
    assert windows.shape == (FIT_WINDOWS, SEQUENCE_LENGTH)
    return windows, manifest['sha256']


def capture(args):
    windows, windows_sha256 = calibration_windows(args.calibration, args.model)
    ids = list(range(args.shard_index, FIT_WINDOWS, args.num_shards))
    path = args.output / 'moments' / f'shard_{args.shard_index:02d}.safetensors'
    if path.with_suffix('.json').exists():
        print('shard complete', path, flush=True)
        return
    model = load_teacher(args.model)
    protocol = dict(
        model_config_sha256=sha256(args.model / 'config.json'),
        windows_sha256=windows_sha256,
        fit_windows=FIT_WINDOWS,
        sequence_length=SEQUENCE_LENGTH,
        rope_parameters=dict(model.config.rope_parameters),
        max_position_embeddings=model.config.max_position_embeddings,
        teacher='dense BF16 Qwen3-32B; SDPA causal; layerwise replay of complete windows',
        key=LOKI_FIT_COORDINATE,
        accumulation='FP64 per-KV-head sums and uncentered Grams',
        num_shards=args.num_shards,
        source_sha256=code_hashes(SOURCES),
    )
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
    assert protocol['num_shards'] == args.num_shards
    assert protocol['windows_sha256'] == windows_sha256
    assert protocol['model_config_sha256'] == sha256(args.model / 'config.json')
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
        format=LOKI_FORMAT, status='complete', rank=RANK, fit_windows=FIT_WINDOWS,
        sequence_length=SEQUENCE_LENGTH, fit_ids=list(range(FIT_WINDOWS)), smoke=False,
        windows_sha256=windows_sha256, model_config_sha256=protocol['model_config_sha256'],
        coordinate=LOKI_FIT_COORDINATE, runtime=LOKI_RUNTIME_COORDINATE,
        capture_protocol=protocol, moments_sha256=moments, source_sha256=code_hashes(SOURCES),
        layers=layers))
    print('fit complete', root, 'mean retained', float(retained.mean()), flush=True)


def support_metrics(query, key, projectors, top_k):
    """Top-K overlap with exact-QK Top-K and captured exact attention mass per query row."""
    heads, count, dim = query.shape
    groups, tokens, _ = key.shape
    grouped = query.reshape(groups, heads // groups * count, dim)
    positions = tokens - count + torch.arange(count, device=key.device)
    future = (torch.arange(tokens, device=key.device)[None] > positions[:, None]).repeat(heads // groups, 1)
    exact = (grouped.float() @ key.float().mT).mul_(dim ** -0.5).masked_fill_(future, -torch.inf)
    probability = exact.softmax(-1)
    reference = exact.topk(top_k, dim=-1).indices
    chosen = torch.zeros_like(exact, dtype=torch.bool).scatter_(-1, reference, True)
    metrics = dict(exact_mass=probability.gather(-1, reference).sum(-1))
    for name, projector in projectors.items():
        # Routing codes are BF16 as in the runtime sidecar; scores accumulate in FP32.
        query_code = torch.einsum('gqd,gdr->gqr', grouped, projector)
        key_code = torch.einsum('gtd,gdr->gtr', key, projector)
        approximate = (query_code.float() @ key_code.float().mT).masked_fill_(future, -torch.inf)
        selected = approximate.topk(top_k, dim=-1).indices
        metrics[f'{name}_overlap'] = chosen.gather(-1, selected).float().mean(-1)
        metrics[f'{name}_mass'] = probability.gather(-1, selected).sum(-1)
    return {name: float(value.mean()) for name, value in metrics.items()}


def overlap(args):
    prompts = read_json(args.prompts / 'prompts.json')
    assert prompts['status'] == 'complete'
    assert prompts['tokens_sha256'] == sha256(args.prompts / 'prompts.safetensors')
    tokens = load_file(str(args.prompts / 'prompts.safetensors'))
    rows = [prompts['rows'][index] for index in args.indices]
    sequences = []
    for row in rows:
        tensor = tokens[str(row['index'])]
        assert len(tensor) == row['input_tokens']
        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == row['input_sha256']
        sequences.append(tensor)
    sources = {'post_rope': args.output / 'pca', 'baseline': args.baseline}
    manifests = {name: read_json(root / 'manifest.json') for name, root in sources.items()}
    assert manifests['post_rope']['coordinate'] == LOKI_FIT_COORDINATE
    projectors = [{name: load_file(str(root / manifests[name]['layers'][layer]['file']))['projector'].cuda()
                   for name, root in sources.items()} for layer in range(LAYERS)]
    results = [[None] * LAYERS for _ in rows]

    def measure(layer, index, query, key):
        results[index][layer] = support_metrics(query[:, -args.queries:], key, projectors[layer], args.top_k)

    replay(load_teacher(args.model), sequences, measure)
    fields = list(results[0][0])
    per_prompt = [dict(index=row['index'], task=row['task'], input_tokens=row['input_tokens'],
                       **{f: sum(r[f] for r in layers) / LAYERS for f in fields})
                  for row, layers in zip(rows, results)]
    per_layer = [{f: sum(results[p][layer][f] for p in range(len(rows))) / len(rows) for f in fields}
                 for layer in range(LAYERS)]
    overall = {f: sum(p[f] for p in per_prompt) / len(rows) for f in fields}
    write_json(args.output / 'overlap.json', dict(
        status='complete', top_k=args.top_k, queries_per_prompt=args.queries,
        query_positions='last prompt tokens, each with its own causal support',
        teacher='dense BF16 Qwen3-32B (no V96), YaRN 4.0',
        baseline=str(args.baseline), baseline_coordinate=manifests['baseline']['coordinate'],
        post_rope_manifest_sha256=sha256(sources['post_rope'] / 'manifest.json'),
        baseline_manifest_sha256=sha256(sources['baseline'] / 'manifest.json'),
        overall=overall, per_prompt=per_prompt, per_layer=per_layer, layers=results,
        source_sha256=code_hashes(SOURCES)))
    lines = [f'# Qwen3-32B Loki Top-{args.top_k} support vs exact QK', '',
             f'Last {args.queries} prompt positions per prompt, all 64 layers and 64 query heads. '
             'Overlap: fraction of exact Top-K recovered. Mass: exact softmax mass on the chosen support.', '',
             '| Prompt | Task | Pre-RoPE overlap | Post-RoPE overlap | Pre-RoPE mass | Post-RoPE mass | Exact Top-K mass |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for p in per_prompt + [dict(index='all', task='mean', **overall)]:
        lines.append(f"| {p['index']} | {p['task']} | {p['baseline_overlap']:.4f} | {p['post_rope_overlap']:.4f} | "
                     f"{p['baseline_mass']:.4f} | {p['post_rope_mass']:.4f} | {p['exact_mass']:.4f} |")
    worst = sorted(range(LAYERS), key=lambda layer: per_layer[layer]['post_rope_overlap'])[:5]
    lines += ['', 'Lowest post-RoPE overlap layers: ' + ', '.join(
        f"{layer} ({per_layer[layer]['post_rope_overlap']:.3f} vs pre-RoPE {per_layer[layer]['baseline_overlap']:.3f})"
        for layer in worst)]
    (args.output / 'overlap.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('capture', 'fit', 'overlap'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=8)
    parser.add_argument('--prompts', type=Path, help='directory with the frozen prompts.json/prompts.safetensors')
    parser.add_argument('--baseline', type=Path, help='baseline Loki checkpoint compared by the overlap stage')
    parser.add_argument('--indices', type=int, nargs='+', help='prompt indices for the overlap stage')
    parser.add_argument('--queries', type=int, default=16)
    parser.add_argument('--top-k', type=int, default=LOKI_TOPK)
    args = parser.parse_args()
    configure()
    assert 0 <= args.shard_index < args.num_shards
    {'capture': capture, 'fit': fit, 'overlap': overlap}[args.stage](args)


if __name__ == '__main__':
    main()
