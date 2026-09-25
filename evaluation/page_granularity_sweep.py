"""Captured exact attention mass versus routing page size, with the router factors fixed (no refit).

For every attention layer, every held-out window and a grid of query positions, the deployed model is replayed once and,
at each query, pages of size P are ranked either by the exact QK scores (oracle: still limited by the page granularity)
or by an approximate page router's scores (one `router:<name>` scorer per bank given). With a fixed routed-token budget B
(K = B / P pages, taken from the historical region that excludes the pinned sink tokens (--sink; 0 for the no-sink
protocol) and the exact recent 64), the metrics, averaged over query heads, groups, queries, windows and layers, are

    mass        AMR(P, B) = sum_{t in support(P, B)} p_t,   p = softmax(exact scores over the whole prefix),
                support = sink + recent + the K routed pages;
    recall      router only: fraction of the oracle's K routed pages (same P and B) that the router also selected;
    dispersion  fraction of the 256-token segments of the routable region touched by the K routed pages.

Page scores follow the runtime's `_selected_pages`: per-head log-sum-exp mass inside the page, softmax over pages,
GQA max over the group's heads. P = 1 is per-token retrieval. Output: JSON with per-layer and overall tables.
"""
import argparse
import math
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from evaluation.calibration_routing_quality import load_bank, sidecar_for
from evaluation.fit_k_routing_streaming import install_deployed_teacher
from evaluation.k_routing_config import routing_config
from evaluation.v96kl_common import configure, read_json, sha256, write_json

SINK, RECENT, SEGMENT = 32, 64, 256
METRICS = ('mass', 'recall', 'dispersion')


def routed_pages(scores, page_size, budget, historical):
    """scores: [groups, group_heads, length]; returns the K = budget / P routed page indices [groups, K] over [SINK, historical)."""
    groups, group_heads, length = scores.shape
    region = scores[..., SINK:historical].float()
    count = region.shape[-1]
    pages = math.ceil(count / page_size)
    padding = pages * page_size - count
    if padding:
        region = torch.nn.functional.pad(region, (0, padding), value=-torch.inf)
    log_mass = torch.logsumexp(region.reshape(groups, group_heads, pages, page_size), dim=-1)   # [g, c, pages]
    group_mass = torch.softmax(log_mass, dim=-1).max(dim=1).values                                # GQA max -> [g, pages]
    k = min(budget // page_size, pages)
    return group_mass.topk(k, dim=-1).indices                                                      # [g, k]


def support_ids(top, page_size, historical):
    """Routed page indices [groups, K] -> token ids [groups, K * P] (ids beyond the routable region are -1)."""
    ids = (top[..., None] * page_size + torch.arange(page_size, device=top.device)).flatten(-2) + SINK
    return ids.masked_fill(ids >= historical, -1)


def captured_mass(probs, ids, historical, length):
    """probs: [groups, group_heads, length] exact mass; support = sink + recent + routed ids (ids < 0 ignored)."""
    groups, group_heads, _ = probs.shape
    base = probs[..., :SINK].sum(-1) + probs[..., historical:length].sum(-1)                       # [g, c]
    valid = ids >= 0
    gathered = probs.gather(2, ids.clamp_min(0)[:, None, :].expand(groups, group_heads, -1)) * valid[:, None, :]
    return (base + gathered.sum(-1)).mean().item()


def page_recall(top, reference):
    """top, reference: [groups, K] page indices; mean over groups of |top & reference| / K."""
    hits = (top[:, :, None] == reference[:, None, :]).any(-1).float().sum(-1)                     # [g]
    return (hits / reference.shape[-1]).mean().item()


def dispersion(ids, historical):
    """Fraction of SEGMENT-token segments of [SINK, historical) holding at least one routed token, mean over groups."""
    segments = max(1, math.ceil((historical - SINK) / SEGMENT))
    valid = ids >= 0
    segment = ((ids.clamp_min(SINK) - SINK) // SEGMENT).clamp_max(segments - 1)
    touched = torch.zeros(ids.shape[0], segments, dtype=torch.bool, device=ids.device)
    touched.scatter_(1, segment.masked_fill(~valid, 0), valid)
    return touched.float().mean().item()


def main():
    global SINK
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--windows', type=Path, required=True, help='calibration bank dir; its validation_ids are replayed')
    p.add_argument('--banks', required=True, help='name=dir[,name=dir...]: router banks (ours_b*r*) scored at every page size')
    p.add_argument('--sink', type=int, default=32, help='pinned sink tokens always kept (0 for the no-sink protocol)')
    p.add_argument('--sequence-length', type=int, default=65536)
    p.add_argument('--rope', choices=('native', 'yarn2', 'yarn4'), default='native')
    p.add_argument('--queries-per-window', type=int, default=32)
    p.add_argument('--page-sizes', default='1,2,4,8,16,32')
    p.add_argument('--budgets', default='256,2048')
    p.add_argument('--windows-limit', type=int)
    p.add_argument('--layers', help='comma list (default: every attention layer)')
    p.add_argument('--chunk', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.sink >= 0 and not args.output.exists(), args.output
    SINK = args.sink
    configure()
    identity = read_json(args.identity)
    config = routing_config(identity, rope=args.rope, sequence_length=args.sequence_length)
    assert config.model_type in ('llama', 'qwen3')
    model = AutoModelForCausalLM.from_pretrained(identity['model'], config=config, dtype=torch.bfloat16,
                                                 attn_implementation='sdpa', local_files_only=True).eval().cuda()
    install_deployed_teacher(model, SimpleNamespace(dense_v=False, native_audit=None, wo_bank=None, identity=args.identity), identity)
    modules = dict(c1_attention_layers(model))
    layers = [int(x) for x in args.layers.split(',')] if args.layers else sorted(modules)
    heads, groups, dim = config.num_attention_heads, config.num_key_value_heads, identity['head_dim']
    group_heads = heads // groups
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_rope
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as qwen3_rope
    apply_rotary_pos_emb = llama_rope if config.model_type == 'llama' else qwen3_rope
    bank_dirs = dict(item.split('=', 1) for item in args.banks.split(','))
    banks = {name: load_bank(Path(path), layers) for name, path in bank_dirs.items()}
    bank_records = {}
    for name, path in bank_dirs.items():
        record = read_json(Path(path) / 'layer_000.json')
        bank_records[name] = dict(path=path, layer_000_sha256=sha256(Path(path) / 'layer_000.safetensors'), identity_sha256=record.get('identity_sha256'),
            protocol={k: record['protocol'].get(k) for k in ('format', 'objective', 'page_size', 'excluded_prefix_pages', 'excluded_recent_tokens',
                                                            'base_rank', 'residual_rank', 'fit_ids', 'windows_sha256', 'sequence_length')})
        assert record.get('identity_sha256') == sha256(args.identity), name
    page_sizes = [int(x) for x in args.page_sizes.split(',')]
    budgets = [int(x) for x in args.budgets.split(',')]
    assert all(b % ps == 0 for b in budgets for ps in page_sizes)
    wm = read_json(args.windows / 'manifest.json')
    windows = load_file(str(args.windows / 'windows.safetensors'))['input_ids']
    assert windows.shape[1] >= args.sequence_length
    ids = [int(i) for i in wm['validation_ids']][:args.windows_limit]
    grid = torch.linspace(4096, args.sequence_length - 1, args.queries_per_window).long().tolist()
    scorers = ['oracle'] + [f'router:{name}' for name in banks]
    print(dict(layers=len(layers), windows=ids, queries_per_window=len(grid), page_sizes=page_sizes, budgets=budgets, sink=SINK, banks=list(banks)), flush=True)
    sums = {}   # (layer, scorer, P, B, metric) -> [sum, count]

    def add(key, value):
        s = sums.setdefault(key, [0.0, 0])
        s[0] += value
        s[1] += 1

    def capture(attention, positional, kwargs, layer=None):
        x = kwargs['hidden_states'] if 'hidden_states' in kwargs else positional[0]
        n = x.shape[1]
        k = attention.k_proj(x).view(1, n, groups, dim)
        q = attention.q_proj(x).view(1, n, heads, dim)
        v_codes = attention.v_proj(x).view(1, n, groups, -1).transpose(1, 2)
        if config.model_type == 'qwen3':
            q, k = attention.q_norm(q), attention.k_norm(k)
        cos, sin = kwargs['position_embeddings']
        q, k_post = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        scaling = attention.scaling
        sidecars = {name: sidecar_for(banks[name][layer], v_codes, k_post, cos, sin)[0].float() for name in banks}   # [groups, n, r]
        k_rep = k_post.repeat_interleave(group_heads, dim=1)
        for start in range(0, len(grid), args.chunk):
            pos = grid[start:start + args.chunk]
            q_sel = q[:, :, pos].float()                                                          # [1, heads, Q, dim]
            exact = (torch.einsum('bhqd,bhtd->bhqt', q_sel, k_rep.float()) * scaling)[0].view(groups, group_heads, len(pos), n)
            routers = {}
            for name in banks:
                codes = torch.einsum('bhqd,hdr->bhqr', q_sel, banks[name][layer]['projector'].float())[0].view(groups, group_heads, len(pos), -1)
                routers[name] = torch.einsum('gcqr,gtr->gcqt', codes, sidecars[name]) * scaling
            for j, position in enumerate(pos):
                length = position + 1
                historical = length - RECENT
                ex = exact[:, :, j, :length]
                probs = torch.softmax(ex, dim=-1)
                for ps in page_sizes:
                    for b in budgets:
                        oracle_top = routed_pages(ex, ps, b, historical)
                        oracle_ids = support_ids(oracle_top, ps, historical)
                        add((layer, 'oracle', ps, b, 'mass'), captured_mass(probs, oracle_ids, historical, length))
                        add((layer, 'oracle', ps, b, 'dispersion'), dispersion(oracle_ids, historical))
                        for name in banks:
                            top = routed_pages(routers[name][:, :, j, :length], ps, b, historical)
                            sel = support_ids(top, ps, historical)
                            add((layer, f'router:{name}', ps, b, 'mass'), captured_mass(probs, sel, historical, length))
                            add((layer, f'router:{name}', ps, b, 'recall'), page_recall(top, oracle_top))
                            add((layer, f'router:{name}', ps, b, 'dispersion'), dispersion(sel, historical))
            del exact, routers, q_sel
        del sidecars, k_rep
        torch.cuda.empty_cache()

    handles = [modules[layer].register_forward_pre_hook(lambda m, a, kw, layer=layer: capture(m, a, kw, layer=layer), with_kwargs=True)
               for layer in layers]
    with torch.inference_mode():
        for w in ids:
            out = model.model(windows[w, :args.sequence_length][None].long().cuda(), use_cache=False)
            del out
            print(dict(window=w, queries=len(grid)), flush=True)
    for h in handles:
        h.remove()
    per_layer = {f'{metric}|{layer}|{scorer}|P{ps}|B{b}': s[0] / s[1] for (layer, scorer, ps, b, metric), s in sums.items()}
    overall = {}
    for (layer, scorer, ps, b, metric) in sums:
        key = f'{metric}|{scorer}|P{ps}|B{b}'
        if key not in overall:
            vals = [sums[(l, scorer, ps, b, metric)][0] / sums[(l, scorer, ps, b, metric)][1] for l in layers]
            overall[key] = sum(vals) / len(vals)
    lines = []
    for b in budgets:
        lines.append(f'budget B={b} (routed tokens; sink {SINK} + recent {RECENT} always included)')
        lines.append(f"{'P':26s}" + ''.join(f'{ps:>8d}' for ps in page_sizes))
        for metric in METRICS:
            for scorer in scorers:
                if f'{metric}|{scorer}|P{page_sizes[0]}|B{b}' in overall:
                    lines.append(f'{metric + " " + scorer:26s}' + ''.join(f"{overall[f'{metric}|{scorer}|P{ps}|B{b}']:8.4f}" for ps in page_sizes))
    text = '\n'.join(lines)
    print(text, flush=True)
    write_json(args.output, dict(status='complete', identity=str(args.identity), identity_sha256=sha256(args.identity),
                                 model_config_sha256=identity['model_config_sha256'], checkpoint_manifest_sha256=identity['manifest_sha256'],
                                 windows=ids, windows_sha256=sha256(args.windows / 'windows.safetensors'), banks=bank_records,
                                 sink=SINK, recent=RECENT, segment=SEGMENT, sequence_length=args.sequence_length,
                                 queries_per_window=len(grid), query_positions=grid, page_sizes=page_sizes, budgets=budgets,
                                 layers=layers, scorers=scorers, metrics=list(METRICS), overall=overall, per_layer=per_layer, table=text,
                                 gpu=torch.cuda.get_device_name(), dtype='bfloat16 replay, float32 scores', command=shlex.join(sys.argv)))


if __name__ == '__main__':
    main()
