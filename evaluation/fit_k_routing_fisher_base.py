"""Fisher-trained Value Base for the B16R16 K router: sequential (Fisher-B16 then Fisher-R16) and joint / alternating
(Fisher B16R16 on the COMBINED score), on top of the window-sharded fitter's records (moments, selections, MSE-RRR base).

Runtime representation is unchanged (evaluation/eval_k_routing_ruler*.py, basisserve/core/c1_v_conditional_k_router.py):
    base_code = Vcode @ base_left_b16 ; base_pre = base_code @ base_right_b16 + base_bias_b16 ; base_post = RoPE(base_pre)
    residual  = K_post - base_post     ; residual_code = residual @ residual_encoder_b16_r16
    score     = d^-1/2 [ q . base_post + (q U_h) . residual_code ]
The Base is a function of the resident Value latent only; K never enters the Base at inference.

Objective (all stages): exact-K page-Fisher loss of the score error with the exact teacher, page size P, the deployed
exclusions (pinned sink pages, recent tokens, causal prefix) and GQA head/group handling of the canonical fitter:
    delta_t = shat_t - s_t = d^-1/2 q_eff . (base_post_t - k_t),  q_eff = q - E_g U_h^T q      (U = E = 0: Base alone)
    L = 1/2 sum_p m_p (dbar_p - sum_p' m_p' dbar_p')^2 ,  m_p / dbar_p from the exact softmax of s = d^-1/2 q . k_exact
This is the same quadratic the canonical residual solver minimizes (compact_softmax_fisher_loss over the residual Grams of
basisserve.core.c1_v_conditional_k_router.residual_page_fisher_gram); every loss reported here is evaluated through it.

Base step (exact block-coordinate minimization, no gradient descent): with B fixed the Base score is linear in (A, b) through
the RoPE-lifted per-token features  phi_t = [ c_t (x) (B R_t^T q_eff) , R_t^T q_eff ]  (R_t^T q = q*cos_t - rotate_half(q)*sin_t);
with A fixed it is linear in (B, b) through  phi_t = [ (c_t A) (x) (R_t^T q_eff) , R_t^T q_eff ].  The page-Fisher normal
equations of the rows [phi_t, y_t] (y_t = q_eff . k_t) are built by the same residual_page_fisher_gram (head-specific rows)
and solved in FP64; each half-step minimizes the exact objective over its block, so the loss is non-increasing.

Stages
  capture   window-sharded replay of the deployed teacher; per (layer, window) feature cache: dense value rows, post-RoPE keys,
            queries at the canonical selected positions (from the source fitter's moments record)
  basefit   sequential Fisher-B16 from the MSE-RRR Base (U = E = 0), --half-steps alternations of A and B
  residual  residual page-Fisher statistics from the cache for a Base record, canonical solver (40 sweeps / PCG 100)
  joint     alternating Fisher B16R16 from a sequential bank: per outer sweep, A and B half-steps on the combined score
            with U/E fixed, then residual statistics REBUILT from the updated Base and U/E refit (warm-started canonical
            solver, fixed inner sweeps); final canonical residual polish
  diagnose  held-out comparison of banks and of Exact-K: total Fisher NMSE (loss / exact-score energy), routed-page
            recall, page KL, retained exact attention mass, with the deployed page selector
"""
import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from basisserve.core.c1_v_conditional_k_router import (AffineReducedRankMap, _rotate_half, build_conditional_routing_sidecar,
                                                      conditional_routing_query_projector, page_fisher_gram_from_page_rows,
                                                      page_fisher_teacher, residual_page_fisher_gram)
from basisserve.core.gqa_joint_routing_payload_s80_ablation import fit_page_fisher_router
from basisserve.core.gqa_joint_routing_payload_s80_fisher import S80CompactSoftmaxFisherRouting, compact_softmax_fisher_loss
from evaluation import fit_k_routing_streaming as S
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (_apply_base, _fit_residual_grid, _post_rope_rows, _value_codes)
from evaluation.k_routing_config import routing_config, routing_position_embeddings
from evaluation.llama_sink_recent_routing_p16 import page_support
from evaluation.v96kl_common import read_json, sha256, write_json

FORMAT = 'basisserve.k_router.fisher_base.v1'
SOURCES = ('evaluation/fit_k_routing_fisher_base.py', 'evaluation/fit_k_routing_windowed.py', 'evaluation/fit_k_routing_streaming.py',
           'evaluation/fit_qwen3_8b_q8_fisher_residual.py', 'evaluation/eval_qwen3_8b_v80_conditional_residual_router.py',
           'basisserve/core/c1_v_conditional_k_router.py', 'basisserve/core/gqa_joint_routing_payload_s80_fisher.py',
           'basisserve/core/gqa_joint_routing_payload_s80_ablation.py', 'evaluation/llama_sink_recent_routing_p16.py')


# ----------------------------------------------------------------------------------------------------------------- math
def inverse_rope(queries, cos, sin):
    """R_t^T q for every token t: (R_t x) . q == x . inverse_rope(q)_t.  queries [heads, d]; cos/sin [tokens, d] -> [heads, tokens, d]."""
    q = queries[:, None, :]
    return q * cos[None] - _rotate_half(q) * sin[None]


def effective_queries(queries, residual_encoder, residual_query, heads_per_group):
    """q_eff = q - E_g U_h^T q: the query the Base error is measured against once the residual code corrects it."""
    if residual_encoder is None:
        return queries
    heads = queries.shape[0]
    groups = torch.arange(heads, device=queries.device) // heads_per_group
    encoders = residual_encoder.index_select(0, groups)                       # [heads, d, r]
    codes = torch.einsum('hd,hdr->hr', queries, residual_query)                # q U_h
    return queries - torch.einsum('hr,hdr->hd', codes, encoders)


def base_step_rows(mode, codes, q_tilde, left, right):
    """Head-specific feature rows of the Base score for one KV group.  codes [tokens, rV]; q_tilde [heads, tokens, d];
    left [rV, r]; right [r, d].  mode 'A': rows [c (x) (B q~), q~]; mode 'B': rows [(c A) (x) q~, q~]."""
    if mode == 'A':
        w = torch.einsum('rd,htd->htr', right, q_tilde)                        # B R_t^T q_eff   [heads, tokens, r]
        outer = torch.einsum('tv,htr->htvr', codes, w)                         # c_t (x) w_t     [heads, tokens, rV, r]
    else:
        assert mode == 'B'
        a = codes @ left                                                       # c_t A           [tokens, r]
        outer = torch.einsum('tr,htd->htrd', a, q_tilde)                       # (c_t A) (x) q~_t [heads, tokens, r, d]
    return torch.cat((outer.reshape(*outer.shape[:2], -1), q_tilde), -1)


def base_from_solution(mode, x, left, right, rank_v, rank, dim):
    """Unpack the block solution x = [vec(block), bias] into (left, right, bias)."""
    if mode == 'A':
        return x[:rank_v * rank].reshape(rank_v, rank), right, x[rank_v * rank:]
    return left, x[:rank * dim].reshape(rank, dim), x[rank * dim:]


def solve_normal_equations(gram, relative_damping, previous):
    """gram [F+1, F+1] FP64 = page-Fisher Gram of the rows [phi, y].  Proximal exact block step: minimize
    1/2 (x^T G_pp x - 2 x^T g_py + g_yy) + 1/2 lambda |x - previous|^2 with lambda = relative_damping * mean diag(G_pp).
    The Base normal equations are extremely ill-conditioned (directions the teacher never looks at); the proximal term keeps
    those coordinates at their previous (MSE-RRR initialized) values instead of letting them run off, and guarantees that the
    undamped loss never increases (L(x) <= L(x) + prox(x) <= L(previous))."""
    n = gram.shape[0] - 1
    system, rhs = gram[:n, :n], gram[:n, n]
    damping = float(relative_damping) * float(system.diagonal().mean().clamp_min(torch.finfo(gram.dtype).tiny))
    factor = torch.linalg.cholesky(system + damping * torch.eye(n, dtype=gram.dtype, device=gram.device))
    x = torch.cholesky_solve((rhs + damping * previous.to(rhs))[:, None], factor)[:, 0]
    return x, damping


def quadratic_loss(gram, x, scaling):
    n = gram.shape[0] - 1
    return 0.5 * float(scaling) ** 2 * float(x @ gram[:n, :n] @ x - 2 * x @ gram[:n, n] + gram[n, n])


def base_post_for(codes, left, right, bias, cos, sin):
    """[tokens, groups, d] post-RoPE Base prediction from value codes [tokens, groups, rV]; cos/sin [tokens, d]
    (the same helpers the canonical residual statistics use)."""
    return _post_rope_rows(_apply_base(codes, (left, right, bias)), cos[None], sin[None])


def maps_of(left, right, bias):
    return tuple(AffineReducedRankMap(left[g], right[g], bias[g]) for g in range(left.shape[0]))


def approximate_scores(queries, v_codes, k_post, factors, cos, sin, scaling):
    """Deployed proxy score for all heads over a prefix: queries [heads, d], v_codes [tokens, groups, rV], k_post [tokens, groups, d].
    Uses the runtime sidecar construction so this is the score the evaluators route with."""
    heads, groups = queries.shape[0], k_post.shape[1]
    codes = v_codes.transpose(0, 1)[None]                                      # [1, groups, tokens, rV]
    keys = k_post.transpose(0, 1)[None]                                        # [1, groups, tokens, d]
    if factors['base_rank'] == 0:
        sidecar = torch.einsum('bgtd,gdr->bgtr', keys, factors['residual_encoder'].to(keys.dtype))
        projector = factors['residual_query']
    else:
        sidecar = build_conditional_routing_sidecar(codes, keys, base_left=factors['base_left'], base_right=factors['base_right'],
                                                    base_bias=factors['base_bias'], residual_encoder=factors['residual_encoder'],
                                                    cos=cos[None], sin=sin[None])
        projector = conditional_routing_query_projector(factors['residual_query'])
    query_codes = torch.einsum('hd,hdw->hw', queries, projector.to(queries.dtype)).reshape(groups, heads // groups, -1)
    return (torch.einsum('ghw,gtw->ght', query_codes, sidecar[0].to(queries.dtype)) * float(scaling)).reshape(heads, -1)


def page_group_scores(scores, historical, page_size, pinned_pages):
    """Replicates the deployed selector's page ranking: per-head log page mass over the historical prefix, softmax over the
    routable pages, GQA max over the group's heads.  scores [groups, group_heads, length] -> [groups, pages]."""
    proxy = scores[..., :historical].float()
    page_count = math.ceil(historical / page_size)
    padding = page_count * page_size - historical
    if padding:
        proxy = torch.nn.functional.pad(proxy, (0, padding), value=-torch.inf)
    log_mass = torch.logsumexp(proxy.reshape(*proxy.shape[:2], page_count, page_size), dim=-1)
    log_mass[..., :pinned_pages] = -torch.inf
    return torch.softmax(log_mass, dim=-1).max(dim=1).values


def support_metrics(scores, exact, page_size, pinned_pages, budget, recent):
    """scores/exact [groups, group_heads, length] for one query: mass on the routed support, routed-page recall vs the exact-K
    ranking, KL(exact page distribution || router page distribution) over the routable pages."""
    groups, group_heads, length = exact.shape
    assert recent == 64                                                        # page_support keeps exactly the recent 64 tokens
    historical = length - recent
    ids, valid = page_support(scores[None], budget, page_size, pinned_pages)
    ids, valid = ids[0], valid[0]
    routed_pages = (budget - recent) // page_size
    routed = ids[:, :budget - recent] // page_size                              # page id of every routed token
    probs = torch.softmax(exact, dim=-1)
    gathered = probs.gather(2, ids.clamp_min(0)[:, None, :].expand(groups, group_heads, -1)) * valid[:, None, :]
    mass = gathered.sum(-1).mean().item()
    exact_group = page_group_scores(exact, historical, page_size, pinned_pages)
    router_group = page_group_scores(scores, historical, page_size, pinned_pages)
    k = min(routed_pages - pinned_pages, exact_group.shape[1] - pinned_pages)
    exact_top = exact_group.topk(k, dim=-1).indices
    recall = sum(len(set(routed[g].tolist()) & set(exact_top[g].tolist())) / k for g in range(groups)) / groups
    p = exact_group / exact_group.sum(-1, keepdim=True).clamp_min(1e-12)
    q = router_group / router_group.sum(-1, keepdim=True).clamp_min(1e-12)
    kl = (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(-1).mean().item()
    return dict(mass=mass, page_recall=recall, page_kl=kl)


# ---------------------------------------------------------------------------------------------------------------- cache
def cache_file(root, layer, window):
    return root / 'cache' / f'l{layer:03d}' / f'w{window:03d}.safetensors'


def window_ids(protocol, split):
    return list(protocol['fit_ids'] if split == 'fit' else protocol['diagnostic_ids'])


def source_records(args, layer):
    """Moments (selections) and Base records of the source window-sharded fit: the canonical query positions and the MSE-RRR Base."""
    _, moments_meta = S.verified(S.layer_file(args.source, 'moments', layer))
    base, base_meta = S.verified(S.layer_file(args.source, 'base', layer))
    assert moments_meta['protocol'] == base_meta['protocol'] and base_meta['identity_sha256'] == sha256(args.identity)
    protocol = base_meta['protocol']
    assert protocol['format'] == 'basisserve.k_router.streaming.v2' and protocol['objective'] == 'page_fisher'
    assert protocol['fit_queries'] == 64 and protocol['diagnostic_ids'] and protocol['diagnostic_queries'] == 32
    assert protocol['base_rank'] == args.base_rank and protocol['page_size'] == args.page_size
    return moments_meta['selections'], base, base_meta, protocol


def load_teacher(args, identity):
    config = routing_config(identity, rope=args.rope, sequence_length=args.sequence_length)
    assert config.model_type in ('llama', 'qwen3')
    model = AutoModelForCausalLM.from_pretrained(identity['model'], config=config, dtype=torch.bfloat16, attn_implementation='sdpa',
                                                 local_files_only=True, trust_remote_code=False).eval().cuda()
    dense_value_weights = {i: m.v_proj.weight.detach().clone() for i, m in c1_attention_layers(model)}
    S.install_deployed_teacher(model, args, identity)
    if config.model_type == 'llama':
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    else:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    return model, dict(c1_attention_layers(model)), dense_value_weights, apply_rotary_pos_emb


def capture(args, identity, windows, layers, protocol, selections):
    """Write the feature cache of this window shard: fit windows and held-out windows alike (split recorded per file)."""
    model, attention_modules, dense_value_weights, apply_rotary_pos_emb = load_teacher(args, identity)
    config = model.config
    heads, groups, dim = config.num_attention_heads, config.num_key_value_heads, identity['head_dim']
    positions = {layer: {split: torch.tensor(selections[layer][split]['selected_positions'], dtype=torch.long)
                         for split in ('fit', 'heldout')} for layer in layers}
    active, handles = {}, []
    for layer in layers:
        module = attention_modules[layer]

        def hook(attention, positional, kwargs, layer=layer):
            x = kwargs['hidden_states'] if 'hidden_states' in kwargs else positional[0]
            n = x.shape[1]
            k = attention.k_proj(x).view(1, n, groups, dim)
            v = torch.nn.functional.linear(x, dense_value_weights[layer]).view(1, n, groups, dim)
            q = attention.q_proj(x).view(1, n, heads, dim)
            if config.model_type == 'qwen3':
                q, k = attention.q_norm(q), attention.k_norm(k)
            q, k_post = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), *kwargs['position_embeddings'])
            tensors = dict(v=v[0].to(torch.bfloat16).contiguous(), k_post=k_post[0].transpose(0, 1).to(torch.bfloat16).contiguous())
            for split in ('fit', 'heldout'):
                tensors[f'queries_{split}'] = q[0][:, positions[layer][split].to(q.device)].transpose(0, 1).to(torch.bfloat16).contiguous()
            path = cache_file(args.output, layer, active['index'])
            path.parent.mkdir(parents=True, exist_ok=True)
            S.save_record(path, tensors, dict(protocol=protocol, layer=layer, window_id=active['index'], split=active['split'],
                                              positions={s: positions[layer][s].tolist() for s in ('fit', 'heldout')}, dtype='bfloat16'))

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    shard = {split: [w for i, w in enumerate(window_ids(protocol, split)) if i % args.window_shards == args.window_shard]
             for split in ('fit', 'heldout')}
    print(dict(stage='capture', shard=args.window_shard, shards=args.window_shards, windows=shard, layers=layers), flush=True)
    with torch.inference_mode():
        for split in ('fit', 'heldout'):
            for index in shard[split]:
                active.update(split=split, index=index)
                started = time.time()
                model.model(input_ids=windows[index][None].cuda(), use_cache=False)
                print(dict(stage='capture', window=index, split=split, seconds=round(time.time() - started, 1),
                           peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)), flush=True)
    for h in handles:
        h.remove()


class LayerCache:
    """Cached windows of one layer, restored to FP32 on the GPU one window at a time.  Each cache file's sha256 is verified on its
    first read in this process; the exact-score page-Fisher energy of a split (Base independent) is computed once and reused."""

    def __init__(self, args, layer, protocol, encoder):
        self.args, self.layer, self.protocol = args, layer, protocol
        self.encoder = encoder.cuda().float()
        self.positions = None
        self.verified_paths = set()
        self.exact_energies = {}

    def windows(self, split):
        for index in window_ids(self.protocol, split):
            path = cache_file(self.args.output, self.layer, index)
            if path in self.verified_paths:
                payload, meta = load_file(str(path)), read_json(path.with_suffix('.json'))
            else:
                payload, meta = S.verified(path)
                self.verified_paths.add(path)
            assert meta['layer'] == self.layer and meta['window_id'] == index and meta['split'] == split
            assert meta['protocol']['windows_sha256'] == self.protocol['windows_sha256']
            if self.positions is None:
                self.positions = meta['positions']
            assert meta['positions'] == self.positions
            v = payload['v'].cuda().float()
            yield index, dict(v=v, codes=_value_codes(v, self.encoder), k_post=payload['k_post'].cuda().float(),
                              queries=payload[f'queries_{split}'].cuda().float())


# ------------------------------------------------------------------------------------------------------- base half-steps
def _pad_tokens(x, tokens):
    """Zero-pad the token axis (second to last) to ``tokens`` rows."""
    return torch.nn.functional.pad(x, (0, 0, 0, tokens - x.shape[-2])) if x.shape[-2] < tokens else x


def base_gram_at_position(mode, w, slot, position, base, residual, cos, sin, *, page_size, excluded_prefix_pages, excluded_recent_tokens,
                          scaling, heads_per_group):
    """Page-Fisher normal-equation Grams of one query position for every KV group: [groups, F+1, F+1] (summed over the group's heads).
    Identical to stacking the per-token rows of base_step_rows and calling residual_page_fisher_gram head by head (the teacher
    distribution comes from page_fisher_teacher, the Gram from page_fisher_gram_from_page_rows), but the page representatives of
    the Kronecker features are formed page by page from the weighted per-token factors, so the [heads, tokens, F] rows are never
    materialized and the Gram of all groups is one batched matmul."""
    left, right, bias = base
    groups, rank_v, rank = left.shape
    dim = right.shape[-1]
    prefix = position + 1 - excluded_recent_tokens
    queries = w['queries'][slot]                                                                            # exact teacher queries
    q_eff = queries if residual is None else effective_queries(queries, residual[0], residual[1], heads_per_group)
    q_tilde = inverse_rope(q_eff, cos[:prefix], sin[:prefix])                                             # [heads, prefix, d]
    masses, rows = [], []
    for group in range(groups):
        first, last = group * heads_per_group, (group + 1) * heads_per_group
        keys = w['k_post'][:prefix, group]
        probabilities, mass, first_token = page_fisher_teacher(queries[first:last], keys, scaling=scaling, page_size=page_size,
                                                               excluded_prefix_pages=excluded_prefix_pages)
        pages = int(mass.shape[-1])
        padded = pages * page_size
        codes = _pad_tokens(w['codes'][first_token:prefix, group], padded).reshape(pages, page_size, rank_v)
        lifted = _pad_tokens(q_tilde[first:last, first_token:], padded).reshape(heads_per_group, pages, page_size, dim)
        target = _pad_tokens(torch.einsum('hd,td->ht', q_eff[first:last], keys[first_token:])[..., None], padded)[..., 0]
        target = target.reshape(heads_per_group, pages, page_size)
        if mode == 'A':
            weights = torch.einsum('rd,hpsd->hpsr', right[group], lifted)                                 # B R_t^T q_eff
            kron = torch.einsum('hps,psv,hpsr->hpvr', probabilities, codes, weights).flatten(-2)          # sum_s p c_s (x) w_s
        else:
            assert mode == 'B'
            projected = codes @ left[group]                                                               # c_t A
            kron = torch.einsum('hps,psr,hpsd->hprd', probabilities, projected, lifted).flatten(-2)       # sum_s p (c_s A) (x) q~_s
        linear = torch.einsum('hps,hpsd->hpd', probabilities, lifted)
        response = torch.einsum('hps,hps->hp', probabilities, target)
        numerator = torch.cat((kron, linear, response[..., None]), -1)
        rows.append(numerator / mass.clamp_min(torch.finfo(mass.dtype).tiny)[..., None])
        masses.append(mass)
    grams = page_fisher_gram_from_page_rows(torch.cat(masses), torch.cat(rows))                            # [groups*heads, F+1, F+1]
    return grams.reshape(groups, heads_per_group, *grams.shape[-2:]).sum(1)


def accumulate_base_gram(cache, split, mode, base, residual, cos, sin, *, page_size, excluded_prefix_pages, excluded_recent_tokens,
                         scaling, heads_per_group):
    """Page-Fisher normal equations of one half-step, summed over the split's windows, query positions and the heads of each KV
    group (FP64).  base = (left [G, rV, r], right [G, r, d], bias [G, d]); residual = (encoder [G, d, r], query [H, d, r]) or None."""
    left, right, bias = base
    groups, rank_v, rank = left.shape
    dim = right.shape[-1]
    width = (rank_v * rank if mode == 'A' else rank * dim) + dim + 1
    gram = torch.zeros(groups, width, width, dtype=torch.float64, device=left.device)
    positions_key = 'fit' if split == 'fit' else 'heldout'
    for index, w in cache.windows(split):
        positions = cache.positions[positions_key]
        for slot, position in enumerate(positions):
            gram += base_gram_at_position(mode, w, slot, position, base, residual, cos, sin, page_size=page_size,
                                          excluded_prefix_pages=excluded_prefix_pages, excluded_recent_tokens=excluded_recent_tokens,
                                          scaling=scaling, heads_per_group=heads_per_group).double()
        del w
    return gram


def balanced(base):
    """Rebalance left/right per group so that A B = (U sqrt(S)) (sqrt(S) V^T): the Base map is unchanged (same runtime numbers up to
    float rounding) but the next block step is better conditioned.  Recorded as a no-op on the objective."""
    left, right, bias = base
    new_left, new_right = torch.empty_like(left), torch.empty_like(right)
    for group in range(left.shape[0]):
        u, sv, vh = torch.linalg.svd((left[group].double() @ right[group].double()), full_matrices=False)
        rank = left.shape[-1]
        root = sv[:rank].sqrt()
        new_left[group] = (u[:, :rank] * root).to(left.dtype)
        new_right[group] = (root[:, None] * vh[:rank]).to(right.dtype)
    return new_left, new_right, bias


def solution_of(mode, base, group):
    left, right, bias = base
    block = left[group].reshape(-1) if mode == 'A' else right[group].reshape(-1)
    return torch.cat((block, bias[group])).double()


def base_half_step(cache, mode, base, residual, cos, sin, *, relative_damping, **support):
    """One exact block-coordinate step of the Base on the fit split; returns the new base and the per-group losses before/after."""
    gram = accumulate_base_gram(cache, 'fit', mode, base, residual, cos, sin, **support)
    left, right, bias = (t.clone() for t in base)
    groups, rank_v, rank = left.shape
    dim = right.shape[-1]
    before, after, dampings, norms, steps = [], [], [], [], []
    for group in range(groups):
        x0 = solution_of(mode, base, group)
        before.append(quadratic_loss(gram[group], x0, support['scaling']))
        x, damping = solve_normal_equations(gram[group], relative_damping, x0)
        after.append(quadratic_loss(gram[group], x, support['scaling']))
        assert torch.isfinite(x).all() and after[-1] <= before[-1] + 1e-9 * max(1.0, abs(before[-1]))
        l, r, b = base_from_solution(mode, x.float(), left[group], right[group], rank_v, rank, dim)
        left[group], right[group], bias[group] = l, r, b
        dampings.append(damping)
        norms.append(float(x.norm()))
        steps.append(float((x - x0).norm()))
    del gram
    torch.cuda.empty_cache()
    return (left, right, bias), dict(mode=mode, loss_before=sum(before), loss_after=sum(after), damping=dampings,
                                     solution_norm=norms, step_norm=steps)


# ------------------------------------------------------------------------------------------------- residual statistics
def residual_statistics(cache, split, base, cos, sin, *, page_size, excluded_prefix_pages, excluded_recent_tokens):
    """Residual page-Fisher statistics of a Base over the split's cached windows: the same per-(query, group) Grams as the canonical
    build_multi_query_statistics (identical teacher, page grouping, exclusions and query-major / window-minor order; verified by
    tests/test_fisher_base_router.py), accumulated on the GPU and copied once per window.  Also returns the Base-independent
    page-Fisher energy of the exact score (computed once per split and cached)."""
    left, right, bias = base
    payloads, exact_energy = [], 0.0
    first_pass = split not in cache.exact_energies
    positions = None
    scaling = None
    for index, w in cache.windows(split):
        positions = cache.positions['fit' if split == 'fit' else 'heldout']
        stop = max(positions) + 1
        tokens, groups, dim = w['k_post'].shape
        heads = w['queries'].shape[1]
        heads_per_group = heads // groups
        scaling = dim ** -0.5
        base_post = base_post_for(w['codes'][:stop], left, right, bias, cos[:stop], sin[:stop])
        residual = w['k_post'][:stop] - base_post
        grams = torch.empty(heads, len(positions), dim, dim, dtype=torch.float32, device=residual.device)
        energy = 0.0
        for slot, position in enumerate(positions):
            prefix = position + 1 - excluded_recent_tokens
            for group in range(groups):
                first, last = group * heads_per_group, (group + 1) * heads_per_group
                keys = w['k_post'][:prefix, group]
                gram, teacher_energy = residual_page_fisher_gram(w['queries'][slot, first:last], keys, residual[:prefix, group],
                                                                 scaling=scaling, page_size=page_size, excluded_prefix_pages=excluded_prefix_pages)
                grams[first:last, slot] = gram
                energy += teacher_energy
                if first_pass:
                    _, exact = residual_page_fisher_gram(w['queries'][slot, first:last], keys, keys, scaling=scaling, page_size=page_size,
                                                         excluded_prefix_pages=excluded_prefix_pages)
                    exact_energy += exact
        payloads.append(dict(queries=w['queries'].permute(1, 0, 2).cpu(), grams=grams.cpu(), energy=energy))
        del base_post, residual, grams, w
    if first_pass:
        cache.exact_energies[split] = exact_energy
    heads, dim = payloads[0]['queries'].shape[0], payloads[0]['queries'].shape[-1]
    n, q = len(payloads), len(positions)
    queries = torch.stack([p['queries'] for p in payloads], 2).reshape(heads, q * n, dim)
    grams = torch.stack([p['grams'] for p in payloads], 2).reshape(heads, q * n, dim, dim)
    groups = left.shape[0]
    stat = S80CompactSoftmaxFisherRouting(queries, grams, torch.arange(heads) // (heads // groups), 0, dim, scaling,
                                          sum(p['energy'] for p in payloads))
    return stat, cache.exact_energies[split]


def total_loss(stat, residual):
    """Fisher loss of the combined score through the residual Grams; residual None = Base alone (U = E = 0)."""
    if residual is None:
        heads, dim = stat.queries_by_head.shape[0], stat.key_dim
        zero_e = torch.zeros(int(stat.head_to_kv_group.max()) + 1, dim, 0, device=stat.queries_by_head.device)
        zero_u = torch.zeros(heads, dim, 0, device=stat.queries_by_head.device)
        return compact_softmax_fisher_loss(stat, routing_payload_encoders=zero_e, routing_query_factors=zero_u)
    return compact_softmax_fisher_loss(stat, routing_payload_encoders=residual[0].to(stat.queries_by_head.device),
                                       routing_query_factors=residual[1].to(stat.queries_by_head.device))


def to_device(stat, device):
    return S80CompactSoftmaxFisherRouting(stat.queries_by_head.to(device), stat.fisher_grams_by_head.to(device),
                                          stat.head_to_kv_group.to(device), 0, stat.key_dim, stat.scaling, stat.teacher_fisher_energy)


# --------------------------------------------------------------------------------------------------------------- records
def base_tensors(base, encoder):
    left, right, bias = base
    return dict(left=left.cpu().float(), right=right.cpu().float(), bias=bias.cpu().float(), encoder=encoder.cpu().float())


def bank_tensors(base, residual, base_rank, residual_rank):
    left, right, bias = base
    tensors = {f'base_left_b{base_rank}': left, f'base_right_b{base_rank}': right, f'base_bias_b{base_rank}': bias,
               f'residual_encoder_b{base_rank}_r{residual_rank}': residual[0], f'residual_query_b{base_rank}_r{residual_rank}': residual[1]}
    tensors = {k: v.cpu().float().contiguous() for k, v in tensors.items()}
    assert all(torch.isfinite(t).all() for t in tensors.values())
    return tensors


def new_protocol(args, source_protocol, **extra):
    protocol = dict(source_protocol)
    protocol.update(format=FORMAT, source_protocol_format=source_protocol['format'], base_objective=extra.pop('base_objective'),
                    fisher_page_size=args.page_size, base_relative_damping=args.base_relative_damping,
                    source_sha256={name: sha256(ROOT / name) for name in SOURCES}, **extra)
    return json.loads(json.dumps(protocol, allow_nan=False))


def read_bank(path, base_rank, residual_rank):
    payload, meta = S.verified(path)
    base = (payload[f'base_left_b{base_rank}'], payload[f'base_right_b{base_rank}'], payload[f'base_bias_b{base_rank}'])
    residual = (payload[f'residual_encoder_b{base_rank}_r{residual_rank}'], payload[f'residual_query_b{base_rank}_r{residual_rank}'])
    return base, residual, meta


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------------------------------------------- stages
def stage_basefit(args, identity, layer, protocol, base_record, base_meta, cos, sin, support):
    """Sequential Fisher-B16: exact A/B alternation from the MSE-RRR Base with no residual (q_eff = q)."""
    started = time.time()
    cache = LayerCache(args, layer, protocol, base_record['encoder'])
    base = tuple(base_record[k].cuda().float() for k in ('left', 'right', 'bias'))
    log = []

    def losses(current):
        out = {}
        for split in ('fit', 'heldout'):
            stat, exact_energy = residual_statistics(cache, split, current, cos, sin, **support)
            loss = total_loss(to_device(stat, torch.device('cuda')), None)
            out[split] = dict(loss=loss, exact_energy=exact_energy, nmse=loss / exact_energy)
            del stat
        return out

    initial = losses(base)
    print(dict(stage='basefit', layer=layer, step=0, losses=initial), flush=True)
    for step in range(1, args.half_steps + 1):
        mode = 'A' if step % 2 == 1 else 'B'
        base, record = base_half_step(cache, mode, base, None, cos, sin, relative_damping=args.base_relative_damping,
                                      scaling=base[1].shape[-1] ** -0.5, heads_per_group=args.heads_per_group, **support)
        if mode == 'B':
            base = balanced(base)
        # The fit loss after the step is the exact quadratic value (loss_after); only the held-out loss needs a pass.
        stat, exact_energy = residual_statistics(cache, 'heldout', base, cos, sin, **support)
        val = total_loss(to_device(stat, torch.device('cuda')), None)
        record.update(step=step, losses=dict(fit=dict(loss=record['loss_after'], exact_energy=initial['fit']['exact_energy'],
                                                      nmse=record['loss_after'] / initial['fit']['exact_energy']),
                                             heldout=dict(loss=val, exact_energy=exact_energy, nmse=val / exact_energy)), balanced=mode == 'B')
        del stat
        log.append(record)
        print(dict(stage='basefit', layer=layer, **{k: v for k, v in record.items() if k not in ('damping', 'solution_norm', 'step_norm')},
                   max_solution_norm=max(record['solution_norm']), max_step_norm=max(record['step_norm'])), flush=True)
    final = losses(base)
    print(dict(stage='basefit', layer=layer, step='final', losses=final), flush=True)
    assert all(torch.isfinite(t).all() for t in base)
    S.save_record(S.layer_file(args.output, 'base_fisher', layer), base_tensors(base, base_record['encoder']),
                  dict(protocol=new_protocol(args, protocol, base_objective='page_fisher_sequential', base_half_steps=args.half_steps),
                       layer=layer, identity_sha256=sha256(args.identity), initialization_base_sha256=base_meta['sha256'],
                       initialization='closed-form MSE-RRR Base16 of the source fit (base_from_moments)', query_positions=cache.positions,
                       losses=dict(initial=initial, half_steps=log, final=final), wall_time_seconds=round(time.time() - started, 1), gpu=gpu_name()))


def stage_residual(args, identity, layer, protocol, base_stage, cos, sin, support):
    """Residual statistics for a Base record (base / base_fisher / zero) and the canonical residual solver; writes a bank."""
    started = time.time()
    _, base_record, base_meta, _ = source_records(args, layer)
    if base_stage == 'zero':
        groups, dim, rank_v = base_record['encoder'].shape[0], base_record['encoder'].shape[1], base_record['encoder'].shape[2]
        base = (torch.zeros(groups, rank_v, 0), torch.zeros(groups, 0, dim), torch.zeros(groups, dim))
        base_sha, base_objective, base_rank = None, 'none (pure residual routing on the post-RoPE key)', 0
    else:
        payload, meta = S.verified(S.layer_file(args.output if base_stage == 'base_fisher' else args.source, base_stage, layer))
        base = (payload['left'], payload['right'], payload['bias'])
        base_sha = meta['sha256']
        base_objective = 'mse_rrr' if base_stage == 'base' else meta['protocol']['base_objective']
        base_rank = base[0].shape[-1]
    cache = LayerCache(args, layer, protocol, base_record['encoder'])
    base = tuple(t.cuda().float() for t in base)
    fit, fit_exact = residual_statistics(cache, 'fit', base, cos, sin, **support)
    val, val_exact = residual_statistics(cache, 'heldout', base, cos, sin, **support)
    factors, losses = _fit_residual_grid({base_rank: fit}, {base_rank: val}, residual_ranks=(args.residual_rank,), sweeps=args.sweeps,
                                         relative_damping=1e-5, iterative_tolerance=1e-5, iterative_max_iterations=args.pcg_iterations,
                                         device=torch.device('cuda'))
    residual = factors[(base_rank, args.residual_rank)]
    key = f'b{base_rank}_r{args.residual_rank}'
    fit_loss = losses[key]['fit_page_fisher_nmse'] * fit.teacher_fisher_energy
    val_loss = losses[key]['validation_page_fisher_nmse'] * val.teacher_fisher_energy
    losses['total'] = dict(fit=dict(loss=fit_loss, exact_energy=fit_exact, nmse=fit_loss / fit_exact),
                           heldout=dict(loss=val_loss, exact_energy=val_exact, nmse=val_loss / val_exact),
                           base_only=dict(fit=total_loss(to_device(fit, torch.device('cuda')), None) / fit_exact,
                                          heldout=total_loss(to_device(val, torch.device('cuda')), None) / val_exact))
    tensors = bank_tensors(base, residual, base_rank, args.residual_rank)
    S.save_record(S.layer_file(args.output, f'ours_{base_stage}_b{base_rank}r{args.residual_rank}', layer), tensors,
                  dict(protocol=new_protocol(args, protocol, base_objective=base_objective, residual_objective='page_fisher_sequential',
                                             base_rank=base_rank, residual_rank=args.residual_rank),
                       layer=layer, v_rank=base_record['encoder'].shape[-1], identity_sha256=sha256(args.identity), losses=losses,
                       base_sha256=base_sha, base_stage=base_stage, query_positions=cache.positions,
                       residual_statistics='rebuilt from the cache for this Base (never reused across Base records)',
                       sweeps=args.sweeps, pcg_iterations=args.pcg_iterations, wall_time_seconds=round(time.time() - started, 1), gpu=gpu_name()))
    print(dict(stage='residual', layer=layer, base_stage=base_stage, total=losses['total']), flush=True)


def stage_joint(args, identity, layer, protocol, cos, sin, support):
    """Alternating Fisher B16R16 from the sequential bank: per outer sweep, A and B half-steps on the combined score with U/E fixed,
    then the residual statistics are rebuilt from the updated Base and U/E refit (warm start, fixed inner sweeps)."""
    started = time.time()
    _, base_record, _, _ = source_records(args, layer)
    init_path = S.layer_file(args.output, f'ours_base_fisher_b{args.base_rank}r{args.residual_rank}', layer)
    base, residual, init_meta = read_bank(init_path, args.base_rank, args.residual_rank)
    cache = LayerCache(args, layer, protocol, base_record['encoder'])
    base = tuple(t.cuda().float() for t in base)
    residual = tuple(t.cuda().float() for t in residual)
    scaling = base[1].shape[-1] ** -0.5
    device = torch.device('cuda')

    def evaluate(current_base, current_residual):
        out, stats = {}, {}
        for split in ('fit', 'heldout'):
            stat, exact = residual_statistics(cache, split, current_base, cos, sin, **support)
            stats[split] = to_device(stat, device)
            loss = total_loss(stats[split], current_residual)
            out[split] = dict(loss=loss, exact_energy=exact, nmse=loss / exact)
        return out, stats

    history = []
    initial, stats = evaluate(base, residual)
    print(dict(stage='joint', layer=layer, sweep=0, losses=initial), flush=True)
    rebuilds = 0
    for sweep in range(1, args.outer_sweeps + 1):
        record = dict(sweep=sweep, half_steps=[])
        for mode in ('A', 'B'):
            base, step = base_half_step(cache, mode, base, residual, cos, sin, relative_damping=args.base_relative_damping,
                                        scaling=scaling, heads_per_group=args.heads_per_group, **support)
            record['half_steps'].append({k: v for k, v in step.items() if k != 'damping'})
            if mode == 'B':
                base = balanced(base)
            print(dict(stage='joint', layer=layer, sweep=sweep, half_step=mode, loss_before=step['loss_before'], loss_after=step['loss_after'],
                       max_solution_norm=max(step['solution_norm']), max_step_norm=max(step['step_norm'])), flush=True)
        # The Base changed: every residual statistic built from the previous Base is stale and is rebuilt here.
        del stats
        after_base, stats = evaluate(base, residual)
        rebuilds += 1
        record['after_base_steps'] = after_base
        fitted = fit_page_fisher_router(stats['fit'], initial_routing_encoders=residual[0], initial_query_factors=residual[1],
                                        active_joint_rows=torch.arange(stats['fit'].key_dim, device=device), sweeps=args.inner_sweeps,
                                        relative_damping=1e-5, relative_tolerance=1e-5, max_iterations=args.pcg_iterations)
        residual = (fitted.routing_encoders.float(), fitted.routing_query_factors.float())
        record['after_residual_step'] = {split: dict(loss=total_loss(stats[split], residual), exact_energy=after_base[split]['exact_energy'],
                                                     nmse=total_loss(stats[split], residual) / after_base[split]['exact_energy'])
                                         for split in ('fit', 'heldout')}
        record['residual_sweeps'] = [dict(sweep=s.sweep, loss_before=s.loss_before, loss_after=s.loss_after_encoder) for s in fitted.sweeps]
        history.append(record)
        print(dict(stage='joint', layer=layer, sweep=sweep, after_base=after_base, after_residual=record['after_residual_step']), flush=True)
    assert rebuilds == args.outer_sweeps
    # Final canonical residual polish on the final Base (statistics rebuilt once more).
    del stats
    final_stats = {split: to_device(residual_statistics(cache, split, base, cos, sin, **support)[0], device) for split in ('fit', 'heldout')}
    fitted = fit_page_fisher_router(final_stats['fit'], initial_routing_encoders=residual[0], initial_query_factors=residual[1],
                                    active_joint_rows=torch.arange(final_stats['fit'].key_dim, device=device), sweeps=args.sweeps,
                                    relative_damping=1e-5, relative_tolerance=1e-5, max_iterations=args.pcg_iterations)
    residual = (fitted.routing_encoders.float(), fitted.routing_query_factors.float())
    final, _ = evaluate(base, residual)
    tensors = bank_tensors(base, residual, args.base_rank, args.residual_rank)
    S.save_record(S.layer_file(args.output, f'ours_joint_b{args.base_rank}r{args.residual_rank}', layer), tensors,
                  dict(protocol=new_protocol(args, protocol, base_objective='page_fisher_joint', residual_objective='page_fisher_joint',
                                             joint_outer_sweeps=args.outer_sweeps, residual_inner_sweeps=args.inner_sweeps,
                                             final_residual_sweeps=args.sweeps),
                       layer=layer, v_rank=base_record['encoder'].shape[-1], identity_sha256=sha256(args.identity),
                       initialization_bank_sha256=init_meta['sha256'], initialization=str(init_path.name),
                       losses=dict(initial=initial, sweeps=history, final=final,
                                   final_residual_sweeps=[dict(sweep=s.sweep, loss_before=s.loss_before, loss_after=s.loss_after_encoder)
                                                          for s in fitted.sweeps]),
                       residual_statistics_rebuilds=rebuilds + 1, query_positions=cache.positions,
                       sweeps=args.sweeps, pcg_iterations=args.pcg_iterations, wall_time_seconds=round(time.time() - started, 1), gpu=gpu_name()))
    print(dict(stage='joint complete', layer=layer, final=final), flush=True)


def stage_diagnose(args, identity, layer, protocol, cos, sin, support):
    """Held-out comparison of banks and Exact-K on identical observations (heldout windows x heldout positions)."""
    _, base_record, _, _ = source_records(args, layer)
    cache = LayerCache(args, layer, protocol, base_record['encoder'])
    arms = {}
    for spec in args.arms:
        name, path = spec.split('=', 1)
        payload, meta = S.verified(Path(path) / f'layer_{layer:03d}.safetensors')
        base_tag = next(k for k in payload if k.startswith('base_left_')).removeprefix('base_left_')
        res_tag = next(k for k in payload if k.startswith('residual_encoder_')).removeprefix('residual_encoder_')
        f = {n: payload[f'{n}_{base_tag}'].cuda().float() for n in ('base_left', 'base_right', 'base_bias')}
        f.update({n: payload[f'{n}_{res_tag}'].cuda().float() for n in ('residual_encoder', 'residual_query')})
        f['base_rank'] = int(base_tag[1:])
        f['sha256'] = meta['sha256']
        arms[name] = f
    scaling = identity['head_dim'] ** -0.5
    heads_per_group = args.heads_per_group
    totals = {name: dict(loss=0.0, mass=[], page_recall=[], page_kl=[]) for name in list(arms) + ['exact_k']}
    exact_energy, observations = 0.0, 0
    budget = protocol['deployment_page_budget_tokens'] + support['excluded_recent_tokens']
    for index, w in cache.windows('heldout'):
        positions = cache.positions['heldout']
        for slot, position in enumerate(positions):
            prefix = position + 1 - support['excluded_recent_tokens']
            queries = w['queries'][slot]
            heads, groups = queries.shape[0], w['k_post'].shape[1]
            exact = torch.einsum('ghd,tgd->ght', queries.reshape(groups, heads_per_group, -1), w['k_post'][:position + 1]) * scaling
            for group in range(groups):
                keys = w['k_post'][:prefix, group]
                _, energy = residual_page_fisher_gram(queries[group * heads_per_group:(group + 1) * heads_per_group], keys, keys,
                                                      scaling=scaling, page_size=support['page_size'],
                                                      excluded_prefix_pages=support['excluded_prefix_pages'])
                exact_energy += energy
            observations += 1
            for name, f in arms.items():
                scores = approximate_scores(queries, w['codes'][:position + 1], w['k_post'][:position + 1], f, cos[:position + 1],
                                            sin[:position + 1], scaling).reshape(groups, heads_per_group, -1)
                m = support_metrics(scores, exact, support['page_size'], support['excluded_prefix_pages'], budget,
                                    support['excluded_recent_tokens'])
                for k in ('mass', 'page_recall', 'page_kl'):
                    totals[name][k].append(m[k])
                base = (f['base_left'], f['base_right'], f['base_bias'])
                base_post = (base_post_for(w['codes'][:prefix], *base, cos[:prefix], sin[:prefix]) if f['base_rank']
                             else torch.zeros_like(w['k_post'][:prefix]))
                for group in range(groups):
                    first, last = group * heads_per_group, (group + 1) * heads_per_group
                    keys = w['k_post'][:prefix, group]
                    g, _ = residual_page_fisher_gram(queries[first:last], keys, keys - base_post[:, group], scaling=scaling,
                                                     page_size=support['page_size'], excluded_prefix_pages=support['excluded_prefix_pages'])
                    proxy = torch.eye(keys.shape[-1], device=keys.device) - f['residual_query'][first:last] @ f['residual_encoder'][group].mT
                    err = torch.einsum('hd,hde->he', queries[first:last], proxy)
                    totals[name]['loss'] += 0.5 * scaling ** 2 * float(torch.einsum('he,hef,hf->', err, g, err))
            m = support_metrics(exact, exact, support['page_size'], support['excluded_prefix_pages'], budget, support['excluded_recent_tokens'])
            for k in ('mass', 'page_recall', 'page_kl'):
                totals['exact_k'][k].append(m[k])
        del w
    report = {name: dict(total_fisher_nmse=(t['loss'] / exact_energy if name != 'exact_k' else 0.0),
                         mass=sum(t['mass']) / len(t['mass']), page_recall=sum(t['page_recall']) / len(t['page_recall']),
                         page_kl=sum(t['page_kl']) / len(t['page_kl']), bank_sha256=arms.get(name, {}).get('sha256'))
              for name, t in totals.items()}
    out = args.output / 'diagnostics' / f'layer_{layer:03d}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, dict(layer=layer, observations=observations, exact_score_energy=exact_energy, budget_tokens=budget,
                         page_size=support['page_size'], pinned_pages=support['excluded_prefix_pages'], recent=support['excluded_recent_tokens'],
                         heldout_windows=window_ids(protocol, 'heldout'), heldout_positions=cache.positions['heldout'], arms=report))
    print(f"layer {layer}: {observations} held-out observations")
    print(f"{'arm':28s}{'Fisher NMSE':>13s}{'page recall':>13s}{'page KL':>10s}{'mass':>8s}")
    for name, r in report.items():
        print(f"{name:28s}{r['total_fisher_nmse']:13.4f}{r['page_recall']:13.4f}{r['page_kl']:10.4f}{r['mass']:8.4f}")


def main():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('capture', 'basefit', 'residual', 'joint', 'diagnose'))
    for name in ('identity', 'windows', 'source', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--layers', required=True, help='comma list of attention layers')
    p.add_argument('--rope', choices=('native', 'yarn2', 'yarn4'), required=True)
    p.add_argument('--sequence-length', type=int, default=131072)
    p.add_argument('--base-rank', type=int, default=16)
    p.add_argument('--residual-rank', type=int, default=16)
    p.add_argument('--page-size', type=int, default=8, choices=(1, 2, 4, 8, 16, 32), help='Fisher page size (must match the source fit)')
    p.add_argument('--window-shard', type=int, default=0)
    p.add_argument('--window-shards', type=int, default=8)
    p.add_argument('--half-steps', type=int, default=6, help='basefit: alternations A, B, A, ... (exact block steps)')
    p.add_argument('--outer-sweeps', type=int, default=4, help='joint: outer alternating sweeps (base half-steps + residual step)')
    p.add_argument('--inner-sweeps', type=int, default=10, help='joint: residual solver sweeps per outer sweep (warm-started)')
    p.add_argument('--sweeps', type=int, default=40, help='canonical residual solver sweeps (residual stage, joint final polish)')
    p.add_argument('--pcg-iterations', type=int, default=100)
    p.add_argument('--base-relative-damping', type=float, default=1e-6, help='proximal damping of each Base half-step, relative to the mean diagonal of its normal equations')
    p.add_argument('--base-stage', choices=('base', 'base_fisher', 'zero'), default='base', help='residual: which Base record')
    p.add_argument('--arms', nargs='*', default=[], help='diagnose: name=<bank dir> ...')
    p.add_argument('--teacher', choices=('deployed',), default='deployed')
    p.add_argument('--dense-v', action='store_true')
    args = p.parse_args()
    assert not args.dense_v
    identity = read_json(args.identity)
    assert identity['status'] == 'complete'
    layers = [int(x) for x in args.layers.split(',')]
    args.heads_per_group = identity['hq'] // identity['hkv']
    selections, base_record, base_meta, protocol = {}, {}, {}, None
    for layer in layers:
        selections[layer], base_record[layer], base_meta[layer], protocol = source_records(args, layer)
    support = dict(page_size=args.page_size, excluded_prefix_pages=protocol['excluded_prefix_pages'],
                   excluded_recent_tokens=protocol['excluded_recent_tokens'])
    assert protocol['sequence_length'] == args.sequence_length and protocol['rope'] == args.rope
    if args.stage == 'capture':
        windows = load_file(str(args.windows))['input_ids']
        assert sha256(args.windows) == protocol['windows_sha256']
        capture(args, identity, windows[:, :args.sequence_length], layers, protocol, selections)
        return
    config = routing_config(identity, rope=args.rope, sequence_length=args.sequence_length)
    cos, sin = routing_position_embeddings(config, args.sequence_length, torch.device('cuda'))
    cos, sin = cos[0].float(), sin[0].float()
    for layer in layers:
        if args.stage == 'basefit':
            stage_basefit(args, identity, layer, protocol, base_record[layer], base_meta[layer], cos, sin, support)
        elif args.stage == 'residual':
            stage_residual(args, identity, layer, protocol, args.base_stage, cos, sin, support)
        elif args.stage == 'joint':
            stage_joint(args, identity, layer, protocol, cos, sin, support)
        else:
            stage_diagnose(args, identity, layer, protocol, cos, sin, support)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
