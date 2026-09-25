"""Captured exact attention mass versus routing page size, with the current router fixed (no refit).

For every attention layer, every held-out window and a grid of query positions, the deployed model is replayed once and,
at each query, pages of size P in {1, 2, 4, 8, 16, 32} are ranked either by the exact QK scores (oracle: still limited
by the page granularity) or by the approximate page router's scores (router). With a fixed routed-token budget B
(K = B / P pages, taken from the historical region that excludes the pinned 32-token sink and the exact recent 64),
the metric is the exact softmax attention mass captured by sink + recent + the K routed pages, averaged over query
heads, groups, queries, windows and layers:

    AMR(P, B) = sum_{t in support(P, B)} p_t,   p = softmax(exact scores over the whole prefix).

Page scores follow the runtime's `_selected_pages`: per-head log-sum-exp mass inside the page, softmax over pages,
GQA max over the group's heads. P = 1 is per-token retrieval. Output: JSON with per-layer and overall tables.
"""
import argparse
import math
from pathlib import Path
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
from evaluation.v96kl_common import configure, read_json, write_json

SINK, RECENT = 32, 64


def routed_support(scores, page_size, budget, historical):
    """scores: [groups, group_heads, length]; returns token ids [groups, budget] chosen from [SINK, historical)."""
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
    top = group_mass.topk(k, dim=-1).indices                                                       # [g, k]
    ids = (top[..., None] * page_size + torch.arange(page_size, device=scores.device)).flatten(-2) + SINK
    return ids.masked_fill(ids >= historical, -1)


def captured_mass(probs, ids, historical, length):
    """probs: [groups, group_heads, length] exact mass; support = sink + recent + routed ids (ids < 0 ignored)."""
    groups, group_heads, _ = probs.shape
    base = probs[..., :SINK].sum(-1) + probs[..., historical:length].sum(-1)                       # [g, c]
    valid = ids >= 0
    gathered = probs.gather(2, ids.clamp_min(0)[:, None, :].expand(groups, group_heads, -1)) * valid[:, None, :]
    return (base + gathered.sum(-1)).mean().item()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--windows', type=Path, required=True, help='calibration bank dir; its validation_ids are replayed')
    p.add_argument('--bank', type=Path, required=True, help='router bank dir (ours_b16r16)')
    p.add_argument('--sequence-length', type=int, default=65536)
    p.add_argument('--rope', choices=('native', 'yarn2', 'yarn4'), default='native')
    p.add_argument('--queries-per-window', type=int, default=32)
    p.add_argument('--page-sizes', default='1,2,4,8,16,32')
    p.add_argument('--budgets', default='256,2048')
    p.add_argument('--windows-limit', type=int)
    p.add_argument('--chunk', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    configure()
    identity = read_json(args.identity)
    config = routing_config(identity, rope=args.rope, sequence_length=args.sequence_length)
    assert config.model_type in ('llama', 'qwen3')
    model = AutoModelForCausalLM.from_pretrained(identity['model'], config=config, dtype=torch.bfloat16,
                                                 attn_implementation='sdpa', local_files_only=True).eval().cuda()
    install_deployed_teacher(model, SimpleNamespace(dense_v=False, native_audit=None, wo_bank=None, identity=args.identity), identity)
    modules = dict(c1_attention_layers(model))
    layers = sorted(modules)
    heads, groups, dim = config.num_attention_heads, config.num_key_value_heads, identity['head_dim']
    group_heads = heads // groups
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_rope
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as qwen3_rope
    apply_rotary_pos_emb = llama_rope if config.model_type == 'llama' else qwen3_rope
    bank = load_bank(args.bank, layers)
    page_sizes = [int(x) for x in args.page_sizes.split(',')]
    budgets = [int(x) for x in args.budgets.split(',')]
    assert all(b % ps == 0 for b in budgets for ps in page_sizes)
    wm = read_json(args.windows / 'manifest.json')
    windows = load_file(str(args.windows / 'windows.safetensors'))['input_ids']
    assert windows.shape[1] >= args.sequence_length
    ids = [int(i) for i in wm['validation_ids']][:args.windows_limit]
    grid = torch.linspace(4096, args.sequence_length - 1, args.queries_per_window).long().tolist()
    print(dict(layers=len(layers), windows=ids, queries_per_window=len(grid), page_sizes=page_sizes, budgets=budgets), flush=True)
    sums = {}   # (layer, scorer, P, B) -> [sum, count]

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
        sidecar = sidecar_for(bank[layer], v_codes, k_post, cos, sin)[0].float()                # [groups, n, r]
        k_rep = k_post.repeat_interleave(group_heads, dim=1)
        for start in range(0, len(grid), args.chunk):
            pos = grid[start:start + args.chunk]
            q_sel = q[:, :, pos].float()                                                          # [1, heads, Q, dim]
            exact = (torch.einsum('bhqd,bhtd->bhqt', q_sel, k_rep.float()) * scaling)[0].view(groups, group_heads, len(pos), n)
            codes = torch.einsum('bhqd,hdr->bhqr', q_sel, bank[layer]['projector'].float())[0].view(groups, group_heads, len(pos), -1)
            router = torch.einsum('gcqr,gtr->gcqt', codes, sidecar) * scaling
            for j, position in enumerate(pos):
                length = position + 1
                historical = length - RECENT
                ex = exact[:, :, j, :length]
                probs = torch.softmax(ex, dim=-1)
                rt = router[:, :, j, :length]
                for scorer, sc in (('oracle', ex), ('router', rt)):
                    for ps in page_sizes:
                        for b in budgets:
                            support = routed_support(sc, ps, b, historical)
                            m = captured_mass(probs, support, historical, length)
                            key = (layer, scorer, ps, b)
                            s = sums.setdefault(key, [0.0, 0])
                            s[0] += m; s[1] += 1
            del exact, router, codes, q_sel
        del sidecar, k_rep
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
    per_layer = {f'{layer}|{scorer}|P{ps}|B{b}': s[0] / s[1] for (layer, scorer, ps, b), s in sums.items()}
    overall = {}
    for scorer in ('oracle', 'router'):
        for b in budgets:
            for ps in page_sizes:
                vals = [sums[(layer, scorer, ps, b)][0] / sums[(layer, scorer, ps, b)][1] for layer in layers]
                overall[f'{scorer}|P{ps}|B{b}'] = sum(vals) / len(vals)
    lines = []
    for b in budgets:
        lines.append(f'budget B={b} (routed tokens; sink 32 + recent 64 always included)')
        lines.append('P       ' + ''.join(f'{ps:>8d}' for ps in page_sizes))
        for scorer in ('oracle', 'router'):
            lines.append(f'{scorer:8s}' + ''.join(f"{overall[f'{scorer}|P{ps}|B{b}']:8.4f}" for ps in page_sizes))
    text = '\n'.join(lines)
    print(text, flush=True)
    write_json(args.output, dict(status='complete', identity=str(args.identity), bank=str(args.bank), windows=ids,
                                 sequence_length=args.sequence_length, queries_per_window=len(grid), page_sizes=page_sizes,
                                 budgets=budgets, overall=overall, per_layer=per_layer, table=text))


if __name__ == '__main__':
    main()
