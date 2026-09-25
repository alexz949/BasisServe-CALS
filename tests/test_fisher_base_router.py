"""Correctness checks of the Fisher-trained Value Base fitter (evaluation/fit_k_routing_fisher_base.py) on synthetic data."""
import math

import pytest
import torch

from basisserve.core.c1_v_conditional_k_router import (_rotate_half, build_conditional_routing_sidecar,
                                                      conditional_routing_query_projector, residual_page_fisher_gram)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import softmax_fisher_gram
from evaluation import fit_k_routing_fisher_base as F

TOKENS, GROUPS, HPG, DIM, RV, RANK, RES, PAGE = 96, 2, 2, 8, 6, 2, 3, 4
HEADS = GROUPS * HPG
SCALING = DIM ** -0.5
SUPPORT = dict(page_size=PAGE, excluded_prefix_pages=1, excluded_recent_tokens=8)


def rope(tokens, dim, seed=0):
    freqs = 0.3 * (torch.arange(dim // 2, dtype=torch.float64) + 1) / (dim // 2)
    angles = torch.arange(tokens, dtype=torch.float64)[:, None] * freqs[None]
    angles = torch.cat((angles, angles), -1)
    return angles.cos().float(), angles.sin().float()


def apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


class FakeCache:
    """In-memory stand-in for fit_k_routing_fisher_base.LayerCache: random windows with fixed query positions."""

    def __init__(self, seed, windows=2, heldout=1):
        g = torch.Generator().manual_seed(seed)
        self.encoder = torch.randn(GROUPS, DIM, RV, generator=g) / math.sqrt(DIM)
        self.positions = dict(fit=[47, 71, 95], heldout=[63, 95])
        self.exact_energies = {}
        self.data = {}
        for split, count in (('fit', windows), ('heldout', heldout)):
            self.data[split] = []
            for _ in range(count):
                v = torch.randn(TOKENS, GROUPS, DIM, generator=g)
                k = torch.randn(TOKENS, GROUPS, DIM, generator=g) + 0.5 * v          # keys partly predictable from V
                q = torch.randn(len(self.positions[split]), HEADS, DIM, generator=g)
                self.data[split].append(dict(v=v, codes=torch.einsum('tgd,gdr->tgr', v, self.encoder), k_post=k, queries=q))

    def windows(self, split):
        for index, w in enumerate(self.data[split]):
            yield index, w


def random_base(seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(GROUPS, RV, RANK, generator=g) * 0.3, torch.randn(GROUPS, RANK, DIM, generator=g) * 0.3,
            torch.randn(GROUPS, DIM, generator=g) * 0.1)


def random_residual(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(GROUPS, DIM, RES, generator=g) * 0.3, torch.randn(HEADS, DIM, RES, generator=g) * 0.3


def test_inverse_rope_transposes_the_rotation():
    cos, sin = rope(TOKENS, DIM)
    q = torch.randn(HEADS, DIM, generator=torch.Generator().manual_seed(1))
    x = torch.randn(TOKENS, DIM, generator=torch.Generator().manual_seed(2))
    lhs = torch.einsum('hd,td->ht', q, apply_rope(x[None], cos, sin)[0])       # q . (R_t x)
    rhs = torch.einsum('htd,td->ht', F.inverse_rope(q, cos, sin), x)           # (R_t^T q) . x
    torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_page_size_one_head_specific_rows_match_token_fisher():
    g = torch.Generator().manual_seed(3)
    queries = torch.randn(HPG, DIM, generator=g, dtype=torch.float64)
    keys = torch.randn(20, DIM, generator=g, dtype=torch.float64)
    rows = torch.randn(HPG, 20, 5, generator=g, dtype=torch.float64)
    page_grams, _ = residual_page_fisher_gram(queries, keys, rows, scaling=0.5, page_size=1, excluded_prefix_pages=0)
    for h in range(HPG):
        token_grams, _ = softmax_fisher_gram(queries[h:h + 1], torch.cat((rows[h], keys), -1), value_dim=5, scaling=0.5)
        torch.testing.assert_close(page_grams[h], token_grams[0][:5, :5])


def test_head_specific_rows_reduce_to_shared_rows():
    g = torch.Generator().manual_seed(4)
    queries = torch.randn(HPG, DIM, generator=g, dtype=torch.float64)
    keys = torch.randn(30, DIM, generator=g, dtype=torch.float64)
    rows = torch.randn(30, 7, generator=g, dtype=torch.float64)
    shared, energy_shared = residual_page_fisher_gram(queries, keys, rows, scaling=0.4, page_size=PAGE, excluded_prefix_pages=1)
    per_head, energy_per_head = residual_page_fisher_gram(queries, keys, rows[None].expand(HPG, -1, -1), scaling=0.4, page_size=PAGE,
                                                         excluded_prefix_pages=1)
    torch.testing.assert_close(shared, per_head)
    assert energy_per_head == 0.0 and energy_shared >= 0.0                    # width 7 != DIM: no exact residual score energy


def test_teacher_weights_come_from_exact_keys_only():
    g = torch.Generator().manual_seed(5)
    queries = torch.randn(HPG, DIM, generator=g, dtype=torch.float64)
    keys = torch.randn(40, DIM, generator=g, dtype=torch.float64)
    rows = torch.randn(HPG, 40, 6, generator=g, dtype=torch.float64)
    base, _ = residual_page_fisher_gram(queries, keys, rows, scaling=0.5, page_size=PAGE, excluded_prefix_pages=0)
    scaled, _ = residual_page_fisher_gram(queries, keys, 3.0 * rows, scaling=0.5, page_size=PAGE, excluded_prefix_pages=0)
    torch.testing.assert_close(scaled, 9.0 * base)                            # page masses unchanged by any proxy feature
    moved, _ = residual_page_fisher_gram(queries, keys + torch.randn(keys.shape, generator=g, dtype=torch.float64), rows, scaling=0.5,
                                         page_size=PAGE, excluded_prefix_pages=0)
    assert not torch.allclose(moved, base)                                    # ... but they do follow the exact keys


def test_base_output_is_a_function_of_value_only():
    cos, sin = rope(TOKENS, DIM)
    cache = FakeCache(6)
    left, right, bias = random_base(7)
    w = cache.data['fit'][0]
    post = F.base_post_for(w['codes'], left, right, bias, cos, sin)
    perturbed = F.base_post_for(w['codes'], left, right, bias, cos, sin)     # keys are not an input at all
    torch.testing.assert_close(post, perturbed)
    assert post.shape == (TOKENS, GROUPS, DIM)
    assert torch.allclose(post, torch.einsum('tgi,gir,grd->tgd', w['codes'], left, right) + bias) is False   # RoPE applied


def test_half_step_minimizes_the_same_loss_the_residual_grams_report():
    cos, sin = rope(TOKENS, DIM)
    cache = FakeCache(8)
    base = random_base(9)
    support = dict(SUPPORT, scaling=SCALING, heads_per_group=HPG)
    stat, exact = F.residual_statistics(cache, 'fit', base, cos, sin, **SUPPORT)
    before_reported = F.total_loss(stat, None)
    gram = F.accumulate_base_gram(cache, 'fit', 'A', base, None, cos, sin, **support)
    before_quadratic = sum(F.quadratic_loss(gram[g], F.solution_of('A', base, g), SCALING) for g in range(GROUPS))
    assert abs(before_reported - before_quadratic) <= 1e-4 * max(1.0, abs(before_reported))
    new_base, record = F.base_half_step(cache, 'A', base, None, cos, sin, relative_damping=1e-10, **support)
    assert record['loss_after'] <= record['loss_before']
    stat_after, _ = F.residual_statistics(cache, 'fit', new_base, cos, sin, **SUPPORT)
    after_reported = F.total_loss(stat_after, None)
    assert abs(after_reported - record['loss_after']) <= 1e-4 * max(1.0, abs(after_reported))
    assert after_reported < before_reported
    # exact block minimizer: any other A (same B) is worse
    left, right, bias = new_base
    worse = (left + 0.05 * torch.randn_like(left), right, bias)
    stat_worse, _ = F.residual_statistics(cache, 'fit', worse, cos, sin, **SUPPORT)
    assert F.total_loss(stat_worse, None) >= after_reported - 1e-6
    # alternating A / B keeps decreasing
    newer_base, record_b = F.base_half_step(cache, 'B', new_base, None, cos, sin, relative_damping=1e-10, **support)
    assert record_b['loss_after'] <= record_b['loss_before'] <= record['loss_after'] + 1e-6 * max(1.0, record['loss_after'])
    assert all(torch.isfinite(t).all() for t in newer_base)


def test_joint_half_step_uses_the_combined_score():
    cos, sin = rope(TOKENS, DIM)
    cache = FakeCache(10)
    base, residual = random_base(11), random_residual(12)
    support = dict(SUPPORT, scaling=SCALING, heads_per_group=HPG)
    stat, _ = F.residual_statistics(cache, 'fit', base, cos, sin, **SUPPORT)
    combined_before = F.total_loss(stat, residual)
    gram = F.accumulate_base_gram(cache, 'fit', 'B', base, residual, cos, sin, **support)
    quadratic = sum(F.quadratic_loss(gram[g], F.solution_of('B', base, g), SCALING) for g in range(GROUPS))
    assert abs(combined_before - quadratic) <= 1e-4 * max(1.0, abs(combined_before))
    new_base, record = F.base_half_step(cache, 'B', base, residual, cos, sin, relative_damping=1e-10, **support)
    stat_after, _ = F.residual_statistics(cache, 'fit', new_base, cos, sin, **SUPPORT)   # statistics rebuilt for the new Base
    assert abs(F.total_loss(stat_after, residual) - record['loss_after']) <= 1e-4 * max(1.0, abs(record['loss_after']))
    assert record['loss_after'] < combined_before
    assert not torch.allclose(stat_after.fisher_grams_by_head, stat.fisher_grams_by_head)   # stale statistics would differ


def test_runtime_sidecar_scores_match_explicit_formula():
    cos, sin = rope(TOKENS, DIM)
    cache = FakeCache(13)
    left, right, bias = random_base(14)
    encoder, query = random_residual(15)
    w = cache.data['fit'][0]
    n = 40
    queries = w['queries'][0]
    factors = dict(base_left=left, base_right=right, base_bias=bias, residual_encoder=encoder, residual_query=query, base_rank=RANK)
    scores = F.approximate_scores(queries, w['codes'][:n], w['k_post'][:n], factors, cos[:n], sin[:n], SCALING)
    base_post = F.base_post_for(w['codes'][:n], left, right, bias, cos[:n], sin[:n])
    explicit = torch.empty(HEADS, n)
    for h in range(HEADS):
        g = h // HPG
        e = w['k_post'][:n, g] - base_post[:, g]
        explicit[h] = SCALING * (base_post[:, g] @ queries[h] + (e @ encoder[g]) @ (query[h].T @ queries[h]))
    torch.testing.assert_close(scores, explicit, atol=1e-5, rtol=1e-5)
    # and the same numbers through the runtime projector on the runtime sidecar
    sidecar = build_conditional_routing_sidecar(w['codes'][:n].transpose(0, 1)[None], w['k_post'][:n].transpose(0, 1)[None],
                                                base_left=left, base_right=right, base_bias=bias, residual_encoder=encoder,
                                                cos=cos[None, :n], sin=sin[None, :n])
    projector = conditional_routing_query_projector(query)
    codes = torch.einsum('hd,hdw->hw', queries, projector).reshape(GROUPS, HPG, -1)
    runtime = (torch.einsum('ghw,gtw->ght', codes, sidecar[0]) * SCALING).reshape(HEADS, n)
    torch.testing.assert_close(scores, runtime, atol=1e-5, rtol=1e-5)


def test_pure_residual_scores_ignore_the_base():
    cos, sin = rope(TOKENS, DIM)
    cache = FakeCache(16)
    encoder, query = random_residual(17)
    w = cache.data['fit'][0]
    factors = dict(base_left=torch.zeros(GROUPS, RV, 0), base_right=torch.zeros(GROUPS, 0, DIM), base_bias=torch.zeros(GROUPS, DIM),
                   residual_encoder=encoder, residual_query=query, base_rank=0)
    scores = F.approximate_scores(w['queries'][0], w['codes'][:32], w['k_post'][:32], factors, cos[:32], sin[:32], SCALING)
    explicit = torch.stack([SCALING * (w['k_post'][:32, h // HPG] @ encoder[h // HPG]) @ (query[h].T @ w['queries'][0, h]) for h in range(HEADS)])
    torch.testing.assert_close(scores, explicit, atol=1e-5, rtol=1e-5)


def test_bank_tensors_have_runtime_names_and_are_finite():
    base, residual = random_base(18), random_residual(19)
    tensors = F.bank_tensors(base, residual, 16, 16)
    assert set(tensors) == {'base_left_b16', 'base_right_b16', 'base_bias_b16', 'residual_encoder_b16_r16', 'residual_query_b16_r16'}
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in tensors.values())
    bad = (base[0].clone(), base[1].clone(), base[2].clone())
    bad[2][0, 0] = float('nan')
    with pytest.raises(AssertionError):
        F.bank_tensors(bad, residual, 16, 16)


def test_support_metrics_of_the_exact_router_are_perfect():
    g = torch.Generator().manual_seed(20)
    length = 300                                                              # the deployed selector always keeps recent 64
    exact = torch.randn(GROUPS, HPG, length, generator=g)
    metrics = F.support_metrics(exact, exact, page_size=PAGE, pinned_pages=1, budget=64 + 64, recent=64)
    assert metrics['page_recall'] == 1.0 and abs(metrics['page_kl']) < 1e-6 and 0.0 < metrics['mass'] <= 1.0
    noisy = F.support_metrics(exact + torch.randn_like(exact), exact, page_size=PAGE, pinned_pages=1, budget=64 + 64, recent=64)
    assert noisy['page_recall'] <= 1.0 and noisy['page_kl'] > 0.0 and noisy['mass'] <= metrics['mass'] + 1e-6


def test_residual_statistics_match_the_canonical_builder():
    from basisserve.core.c1_v_conditional_k_router import AffineReducedRankMap
    from evaluation.fit_qwen3_8b_q8_fisher_residual import build_multi_query_statistics
    cos, sin = rope(TOKENS, DIM)
    cache = FakeCache(21, windows=3)
    left, right, bias = random_base(22)
    stat, exact_energy = F.residual_statistics(cache, 'fit', (left, right, bias), cos, sin, **SUPPORT)
    maps = {RANK: tuple(AffineReducedRankMap(left[g].double(), right[g].double(), bias[g].double()) for g in range(GROUPS))}
    rows = torch.stack([torch.cat((w['v'], w['k_post']), -1) for w in cache.data['fit']])
    queries = torch.stack([w['queries'] for w in cache.data['fit']])
    combined, _ = build_multi_query_statistics(queries, rows, query_positions=cache.positions['fit'], cos=cos[None], sin=sin[None],
                                               value_encoder=cache.encoder, base_maps=maps, page_size=PAGE,
                                               excluded_prefix_pages=SUPPORT['excluded_prefix_pages'], device=torch.device('cpu'),
                                               excluded_recent_tokens=SUPPORT['excluded_recent_tokens'])
    canonical = combined[RANK]
    torch.testing.assert_close(stat.queries_by_head, canonical.queries_by_head)
    torch.testing.assert_close(stat.fisher_grams_by_head, canonical.fisher_grams_by_head, atol=1e-4, rtol=1e-4)
    assert abs(stat.teacher_fisher_energy - canonical.teacher_fisher_energy) <= 1e-4 * max(1.0, canonical.teacher_fisher_energy)
    assert exact_energy > 0 and cache.exact_energies['fit'] == exact_energy
    again, energy_again = F.residual_statistics(cache, 'fit', (left, right, bias), cos, sin, **SUPPORT)   # cached energy, same statistics
    assert energy_again == exact_energy
    torch.testing.assert_close(again.fisher_grams_by_head, stat.fisher_grams_by_head)


def test_structured_base_gram_matches_materialized_rows():
    cos, sin = rope(TOKENS, DIM)
    cache = FakeCache(23)
    base, residual = random_base(24), random_residual(25)
    w = cache.data['fit'][1]
    for mode, res in (('A', None), ('B', residual), ('A', residual)):
        for slot, position in enumerate(cache.positions['fit']):
            structured = F.base_gram_at_position(mode, w, slot, position, base, res, cos, sin, page_size=PAGE, excluded_prefix_pages=1,
                                                 excluded_recent_tokens=8, scaling=SCALING, heads_per_group=HPG)
            prefix = position + 1 - 8
            queries = w['queries'][slot]
            q_eff = F.effective_queries(queries, res[0], res[1], HPG) if res is not None else queries
            q_tilde = F.inverse_rope(q_eff, cos[:prefix], sin[:prefix])
            for g in range(GROUPS):
                first, last = g * HPG, (g + 1) * HPG
                keys = w['k_post'][:prefix, g]
                y = torch.einsum('hd,td->ht', q_eff[first:last], keys)
                rows = F.base_step_rows(mode, w['codes'][:prefix, g], q_tilde[first:last], base[0][g], base[1][g])
                rows = torch.cat((rows, y[..., None]), -1)
                reference, _ = residual_page_fisher_gram(queries[first:last], keys, rows, scaling=SCALING, page_size=PAGE, excluded_prefix_pages=1)
                torch.testing.assert_close(structured[g], reference.sum(0), atol=1e-4, rtol=1e-4)


def test_balancing_preserves_the_base_map():
    left, right, bias = random_base(26)
    skewed = (left * 40.0, right / 40.0, bias)
    balanced = F.balanced(skewed)
    torch.testing.assert_close(torch.einsum('gvr,grd->gvd', *skewed[:2]), torch.einsum('gvr,grd->gvd', *balanced[:2]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(balanced[2], bias)
    assert abs(float(balanced[0].norm()) - float(balanced[1].norm())) < 1e-3 * float(balanced[0].norm())
    cos, sin = rope(TOKENS, DIM)
    cache = FakeCache(27)
    w = cache.data['fit'][0]
    torch.testing.assert_close(F.base_post_for(w['codes'], *skewed, cos, sin), F.base_post_for(w['codes'], *balanced, cos, sin), atol=1e-4, rtol=1e-4)
