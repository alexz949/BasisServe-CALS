from __future__ import annotations

import torch

from basisserve.core.kq_svd import (
    key_svd_projector,
    kq_svd_projectors,
    project_grouped_queries_and_keys,
    relative_score_frobenius_error,
)


def test_kq_svd_matches_direct_score_objective_and_beats_key_svd() -> None:
    generator = torch.Generator().manual_seed(17)
    key = torch.randn(97, 12, generator=generator, dtype=torch.float64)
    query = torch.randn(181, 12, generator=generator, dtype=torch.float64)
    key[:, :4] *= 4.0
    query[:, 4:8] *= 3.0
    key_gram = key.mT @ key
    query_gram = query.mT @ query

    key_basis, _ = key_svd_projector(key_gram, rank=5)
    kq_key, kq_query, _ = kq_svd_projectors(key_gram, query_gram, rank=5)
    key_error = relative_score_frobenius_error(
        key_gram,
        query_gram,
        key_basis,
        key_basis,
    )
    kq_error = relative_score_frobenius_error(
        key_gram,
        query_gram,
        kq_key,
        kq_query,
    )
    dense_scores = key @ query.mT
    compressed_scores = (key @ kq_key) @ (query @ kq_query).mT
    direct_error = (dense_scores - compressed_scores).square().sum()
    direct_error /= dense_scores.square().sum()

    torch.testing.assert_close(kq_error, direct_error, rtol=1e-10, atol=1e-12)
    assert kq_error <= key_error + 1e-12


def test_full_rank_kq_svd_reconstructs_identity_with_balanced_factors() -> None:
    generator = torch.Generator().manual_seed(23)
    key = torch.randn(80, 9, generator=generator, dtype=torch.float64)
    query = torch.randn(120, 9, generator=generator, dtype=torch.float64)
    key_projector, query_projector, _ = kq_svd_projectors(
        key.mT @ key,
        query.mT @ query,
        rank=9,
    )

    torch.testing.assert_close(
        key_projector @ query_projector.mT,
        torch.eye(9, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(key_projector, dim=0),
        torch.linalg.vector_norm(query_projector, dim=0),
        rtol=1e-10,
        atol=1e-10,
    )


def test_grouped_projection_reuses_one_query_projector_per_kv_group() -> None:
    generator = torch.Generator().manual_seed(29)
    query = torch.randn(2, 4, 7, 6, generator=generator, dtype=torch.float64)
    key = torch.randn(2, 2, 7, 6, generator=generator, dtype=torch.float64)
    key_projector = torch.randn(
        2,
        6,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    query_projector = torch.randn(
        2,
        6,
        3,
        generator=generator,
        dtype=torch.float64,
    )

    compressed_query, compressed_key = project_grouped_queries_and_keys(
        query,
        key,
        key_projector,
        query_projector,
    )
    torch.testing.assert_close(
        compressed_query[:, :2],
        torch.einsum("bhld,dr->bhlr", query[:, :2], query_projector[0]),
    )
    torch.testing.assert_close(
        compressed_query[:, 2:],
        torch.einsum("bhld,dr->bhlr", query[:, 2:], query_projector[1]),
    )
    torch.testing.assert_close(
        compressed_key[:, 1],
        torch.einsum("bld,dr->blr", key[:, 1], key_projector[1]),
    )


def test_grouped_projection_accepts_one_query_projector_per_query_head() -> None:
    generator = torch.Generator().manual_seed(31)
    query = torch.randn(2, 4, 7, 6, generator=generator, dtype=torch.float64)
    key = torch.randn(2, 2, 7, 6, generator=generator, dtype=torch.float64)
    key_projector = torch.randn(2, 6, 3, generator=generator, dtype=torch.float64)
    query_projector = torch.randn(4, 6, 3, generator=generator, dtype=torch.float64)

    compressed_query, compressed_key = project_grouped_queries_and_keys(
        query,
        key,
        key_projector,
        query_projector,
    )
    torch.testing.assert_close(
        compressed_query[:, 3],
        torch.einsum("bld,dr->blr", query[:, 3], query_projector[3]),
    )
    torch.testing.assert_close(
        compressed_key[:, 0],
        torch.einsum("bld,dr->blr", key[:, 0], key_projector[0]),
    )
