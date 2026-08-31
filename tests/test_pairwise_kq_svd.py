from __future__ import annotations

import torch

from basisserve.core.kq_svd import kq_svd_projectors
from basisserve.core.pairwise_kq_svd import (
    block_diagonal_pair_factors,
    fit_pairwise_kq_svd,
    pairwise_score_squared_errors,
)


def _activations(
    *,
    groups: int = 3,
    rows: int = 97,
    query_rows: int = 83,
    head_dim: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260828)
    key0 = torch.randn(groups, rows, head_dim, generator=generator, dtype=torch.float64)
    key1 = torch.randn(groups, rows, head_dim, generator=generator, dtype=torch.float64)
    query0 = torch.randn(
        groups,
        query_rows,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    query1 = torch.randn(
        groups,
        query_rows,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    return key0, key1, query0, query1


def _grams(
    key0: torch.Tensor,
    key1: torch.Tensor,
    query0: torch.Tensor,
    query1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_key = torch.cat((key0, key1), dim=-1)
    return (
        pair_key.mT @ pair_key,
        torch.stack((query0.mT @ query0, query1.mT @ query1)),
    )


def _direct_errors(
    key0: torch.Tensor,
    key1: torch.Tensor,
    query0: torch.Tensor,
    query1: torch.Tensor,
    key_projector: torch.Tensor,
    query_projector: torch.Tensor,
) -> torch.Tensor:
    pair_key = torch.cat((key0, key1), dim=-1)
    code = pair_key @ key_projector
    errors = []
    for key, query, query_factor in zip(
        (key0, key1),
        (query0, query1),
        query_projector,
        strict=True,
    ):
        exact = key @ query.mT
        approximate = code @ (query @ query_factor).mT
        errors.append((exact - approximate).square().sum(dim=(-2, -1)))
    return torch.stack(errors)


def test_gram_error_matches_materialized_scores() -> None:
    activations = _activations()
    pair_key_gram, query_gram = _grams(*activations)
    result = fit_pairwise_kq_svd(pair_key_gram, query_gram, rank=9)
    gram_error, _, _ = pairwise_score_squared_errors(
        pair_key_gram,
        query_gram,
        result.key_projector,
        result.query_projector,
    )
    direct_error = _direct_errors(
        *activations,
        result.key_projector,
        result.query_projector,
    )
    torch.testing.assert_close(gram_error, direct_error, rtol=2e-11, atol=2e-9)


def test_joint_rank_beats_matched_storage_independent_ranks() -> None:
    key0, key1, query0, query1 = _activations()
    pair_key_gram, query_gram = _grams(key0, key1, query0, query1)
    independent_key = []
    independent_query = []
    head_dim = key0.shape[-1]
    for slot in (0, 1):
        start = slot * head_dim
        stop = start + head_dim
        key_factor, query_factor, _ = kq_svd_projectors(
            pair_key_gram[:, start:stop, start:stop],
            query_gram[slot],
            rank=4,
        )
        independent_key.append(key_factor)
        independent_query.append(query_factor)
    block_key, block_query = block_diagonal_pair_factors(
        torch.stack(independent_key),
        torch.stack(independent_query),
    )
    independent_error, _, _ = pairwise_score_squared_errors(
        pair_key_gram,
        query_gram,
        block_key,
        block_query,
    )
    joint = fit_pairwise_kq_svd(pair_key_gram, query_gram, rank=8)
    joint_error, _, _ = pairwise_score_squared_errors(
        pair_key_gram,
        query_gram,
        joint.key_projector,
        joint.query_projector,
    )
    assert torch.all(joint_error.sum(dim=0) <= independent_error.sum(dim=0) + 1e-8)
    assert float(joint_error.sum()) < float(independent_error.sum())


def test_full_pair_rank_recovers_both_score_matrices() -> None:
    activations = _activations(groups=2, head_dim=8)
    pair_key_gram, query_gram = _grams(*activations)
    result = fit_pairwise_kq_svd(pair_key_gram, query_gram, rank=16)
    error, energy, relative = pairwise_score_squared_errors(
        pair_key_gram,
        query_gram,
        result.key_projector,
        result.query_projector,
    )
    assert float(error.max() / energy.max()) < 1e-24
    assert float(relative.max()) < 1e-24


def test_block_diagonal_factors_preserve_independent_scores() -> None:
    key0, key1, query0, query1 = _activations(groups=2, head_dim=10)
    independent_key = torch.randn(2, 2, 10, 4, dtype=torch.float64)
    independent_query = torch.randn(2, 2, 10, 4, dtype=torch.float64)
    pair_key, pair_query = block_diagonal_pair_factors(
        independent_key,
        independent_query,
    )
    pair_code = torch.cat((key0, key1), dim=-1) @ pair_key
    torch.testing.assert_close(
        pair_code @ (query0 @ pair_query[0]).mT,
        (key0 @ independent_key[0]) @ (query0 @ independent_query[0]).mT,
    )
    torch.testing.assert_close(
        pair_code @ (query1 @ pair_query[1]).mT,
        (key1 @ independent_key[1]) @ (query1 @ independent_query[1]).mT,
    )
