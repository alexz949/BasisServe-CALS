from __future__ import annotations

import torch

from basisserve.core.c1_k_output_closure import conjugate_gradient
from basisserve.core.c1_k_proxy_fit import (
    KProxyPairSamples,
    canonicalize_gqa_k_proxy_factors,
    fit_gqa_k_proxy,
    raw_score_objective,
    sample_valid_causal_pairs,
)
from basisserve.core.c1_k_refine import GQAKProxyFactors


def _samples(*, pairs_per_head: int = 64, head_dim: int = 4) -> KProxyPairSamples:
    generator = torch.Generator().manual_seed(20260826)
    query_heads = 4
    query_head = torch.arange(query_heads).repeat_interleave(pairs_per_head)
    return KProxyPairSamples(
        query=torch.randn(
            query_heads * pairs_per_head,
            head_dim,
            generator=generator,
            dtype=torch.float64,
        ),
        key=torch.randn(
            query_heads * pairs_per_head,
            head_dim,
            generator=generator,
            dtype=torch.float64,
        ),
        query_head=query_head,
    )


def test_pca_factor_shapes_and_gqa_ownership() -> None:
    result = fit_gqa_k_proxy(
        _samples(),
        num_query_heads=4,
        num_kv_heads=2,
        proxy_rank=2,
        initialization="pca_shared",
        accumulation_dtype=torch.float64,
    )
    assert result.factors.key_encoder.shape == (2, 4, 2)
    assert result.factors.query_encoders.shape == (4, 4, 2)
    torch.testing.assert_close(
        result.factors.query_encoders[0], result.factors.query_encoders[1]
    )
    torch.testing.assert_close(
        result.factors.query_encoders[2], result.factors.query_encoders[3]
    )


def test_alternating_fit_does_not_increase_raw_score_objective() -> None:
    samples = _samples(pairs_per_head=96)
    result = fit_gqa_k_proxy(
        samples,
        num_query_heads=4,
        num_kv_heads=2,
        proxy_rank=2,
        als_sweeps=2,
        ridge=1.0e-6,
        cg_iterations=64,
        cg_tolerance=1.0e-10,
        accumulation_dtype=torch.float64,
    )
    assert result.objective_history[-1] <= result.objective_history[0]
    assert all(
        after <= before * (1 + 1e-10)
        for before, after in zip(result.objective_history, result.objective_history[1:])
    )
    assert len(result.cg_diagnostics) == 2 * (4 + 2)


def test_full_rank_pca_shared_recovers_exact_qk_scores() -> None:
    samples = _samples(head_dim=4)
    result = fit_gqa_k_proxy(
        samples,
        num_query_heads=4,
        num_kv_heads=2,
        proxy_rank=4,
        initialization="pca_shared",
        ridge=0,
        accumulation_dtype=torch.float64,
    )
    objective = raw_score_objective(
        samples, result.factors, num_kv_heads=2, ridge=0
    )
    assert objective < 1.0e-20


def test_qr_canonicalization_preserves_scores_and_orthonormalizes_key() -> None:
    generator = torch.Generator().manual_seed(9)
    factors = GQAKProxyFactors(
        torch.randn(2, 5, 3, generator=generator),
        torch.randn(4, 5, 3, generator=generator),
    )
    query = torch.randn(7, 5, generator=generator)
    key = torch.randn(7, 5, generator=generator)
    heads = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    groups = heads // 2

    def score(current: GQAKProxyFactors) -> torch.Tensor:
        q = torch.einsum(
            "nd,ndr->nr", query, current.query_encoders.index_select(0, heads)
        )
        k = torch.einsum(
            "nd,ndr->nr", key, current.key_encoder.index_select(0, groups)
        )
        return (q * k).sum(-1)

    canonical = canonicalize_gqa_k_proxy_factors(factors)
    torch.testing.assert_close(score(canonical), score(factors), rtol=2e-5, atol=2e-5)
    gram = canonical.key_encoder.mT @ canonical.key_encoder
    torch.testing.assert_close(
        gram, torch.eye(3).expand(2, 3, 3), rtol=2e-5, atol=2e-5
    )


def test_matrix_free_cg_matches_explicit_spd_solve() -> None:
    generator = torch.Generator().manual_seed(10)
    matrix = torch.randn(12, 12, generator=generator, dtype=torch.float64)
    matrix = matrix.mT @ matrix + 0.5 * torch.eye(12, dtype=torch.float64)
    right = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    observed, diagnostics = conjugate_gradient(
        lambda value: (matrix @ value.flatten()).reshape_as(value),
        right,
        max_iterations=24,
        relative_tolerance=1.0e-12,
    )
    expected = torch.linalg.solve(matrix, right.flatten()).reshape_as(right)
    torch.testing.assert_close(observed, expected, rtol=1e-9, atol=1e-9)
    assert diagnostics["converged"]


def test_causal_pair_sampling_excludes_future_and_padding() -> None:
    generator = torch.Generator().manual_seed(12)
    query = torch.randn(2, 4, 6, 3, generator=generator)
    key = torch.randn(2, 2, 6, 3, generator=generator)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0]], dtype=torch.bool
    )
    capture = sample_valid_causal_pairs(
        query,
        key,
        attention_mask=mask,
        queries_per_sequence=4,
        keys_per_query=3,
        recent_pair_fraction=1 / 3,
        top_score_pair_fraction=1 / 3,
        random_seed=7,
    )
    assert torch.all(capture.key_position <= capture.query_position)
    assert torch.all(mask[capture.sequence_id, capture.query_position])
    assert torch.all(mask[capture.sequence_id, capture.key_position])
    capture.samples.validate(num_query_heads=4)


def test_causal_pair_sampling_is_reproducible_and_respects_gqa_keys() -> None:
    generator = torch.Generator().manual_seed(13)
    query = torch.randn(1, 4, 5, 3, generator=generator)
    key = torch.randn(1, 2, 5, 3, generator=generator)
    kwargs = dict(
        queries_per_sequence=5,
        keys_per_query=4,
        recent_pair_fraction=0.25,
        top_score_pair_fraction=0.25,
        random_seed=9,
    )
    first = sample_valid_causal_pairs(query, key, **kwargs)
    second = sample_valid_causal_pairs(query, key, **kwargs)
    torch.testing.assert_close(first.query_position, second.query_position)
    torch.testing.assert_close(first.key_position, second.key_position)
    groups = first.samples.query_head // 2
    expected_key = key[
        first.sequence_id,
        groups,
        first.key_position,
    ]
    torch.testing.assert_close(first.samples.key, expected_key)
