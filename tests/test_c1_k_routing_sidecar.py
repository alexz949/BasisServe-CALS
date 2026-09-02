from __future__ import annotations

import math

import pytest
import torch

from basisserve.core.c1_k_routing_sidecar import (
    RoutingDynamicCache,
    build_routing_sidecar,
    routing_proxy_scores,
    routing_storage_ratio,
    select_routing_pages,
    truncate_routing_projectors,
)


def test_incremental_routing_cache_matches_full_projection_and_crop() -> None:
    generator = torch.Generator().manual_seed(20260830)
    key = torch.randn(2, 3, 7, 5, generator=generator)
    value = torch.randn(2, 3, 7, 4, generator=generator)
    projector = torch.randn(3, 5, 2, generator=generator)
    cache = RoutingDynamicCache()

    for start, stop in ((0, 3), (3, 7)):
        cache.update(key[:, :, start:stop], value[:, :, start:stop], 0)
        cache.update_routing_sidecar(
            key[:, :, start:stop], projector, layer_idx=0
        )

    torch.testing.assert_close(
        cache.routing_sidecar(0),
        build_routing_sidecar(key, projector),
    )
    assert cache.get_seq_length() == 7

    cache.crop(4)
    assert cache.get_seq_length() == 4
    torch.testing.assert_close(
        cache.routing_sidecar(0),
        build_routing_sidecar(key[:, :, :4], projector),
    )

    cache.update(key[:, :, 4:5], value[:, :, 4:5], 0)
    cache.update_routing_sidecar(key[:, :, 4:5], projector, layer_idx=0)
    assert cache.get_seq_length() == 5
    torch.testing.assert_close(
        cache.routing_sidecar(0),
        build_routing_sidecar(key[:, :, :5], projector),
    )


def test_full_rank_identity_sidecar_matches_exact_gqa_scores() -> None:
    generator = torch.Generator().manual_seed(20260828)
    query = torch.randn(4, 5, generator=generator, dtype=torch.float64)
    key = torch.randn(2, 11, 5, generator=generator, dtype=torch.float64)
    projector = torch.eye(5, dtype=torch.float64).expand(2, -1, -1).clone()

    sidecar = build_routing_sidecar(key, projector)
    observed = routing_proxy_scores(
        query,
        sidecar,
        projector,
        head_dim=5,
    )

    expanded_key = key.repeat_interleave(2, dim=0)
    expected = torch.einsum("hd,htd->ht", query, expanded_key) / math.sqrt(5)
    torch.testing.assert_close(observed, expected)


def test_asymmetric_query_head_factors_match_explicit_rank_operator() -> None:
    generator = torch.Generator().manual_seed(91)
    query = torch.randn(4, 6, generator=generator, dtype=torch.float64)
    key = torch.randn(2, 9, 6, generator=generator, dtype=torch.float64)
    key_projector = torch.randn(2, 6, 3, generator=generator, dtype=torch.float64)
    query_projector = torch.randn(
        4, 6, 3, generator=generator, dtype=torch.float64
    )

    sidecar = build_routing_sidecar(key, key_projector)
    observed = routing_proxy_scores(
        query,
        sidecar,
        query_projector,
        head_dim=6,
    )

    kv_index = torch.tensor([0, 0, 1, 1])
    expected = torch.empty_like(observed)
    for head in range(4):
        transformed = query[head] @ (
            query_projector[head]
            @ key_projector[kv_index[head]].mT
        )
        expected[head] = transformed @ key[kv_index[head]].mT / math.sqrt(6)
    torch.testing.assert_close(observed, expected)


def test_page_selection_uses_one_fixed_group_max_budget() -> None:
    query = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    key = torch.tensor(
        [
            [[7.0, 0.0], [6.0, 0.0], [0.0, 8.0], [0.0, 7.0]],
            [[0.0, 9.0], [0.0, 8.0], [10.0, 0.0], [9.0, 0.0]],
        ]
    )
    projector = torch.eye(2).expand(2, -1, -1).clone()
    selection = select_routing_pages(
        query,
        build_routing_sidecar(key, projector),
        projector,
        head_dim=2,
        page_size=2,
        nominal_token_budget=2,
    )

    assert selection.page_mask.tolist() == [[False, True], [False, True]]
    assert selection.token_mask.tolist() == [
        [False, False, True, True],
        [False, False, True, True],
    ]


def test_truncate_and_storage_accounting() -> None:
    key = torch.randn(2, 8, 6)
    query = torch.randn(4, 8, 6)
    truncated_key, truncated_query = truncate_routing_projectors(
        key,
        query,
        rank=3,
    )
    assert truncated_key.shape == (2, 8, 3)
    assert truncated_query.shape == (4, 8, 3)
    assert routing_storage_ratio(
        value_rank=64,
        routing_rank=32,
        key_width=128,
        value_width=128,
    ) == pytest.approx(0.375)


@pytest.mark.parametrize("rank", (0, 7))
def test_truncate_rejects_invalid_rank(rank: int) -> None:
    with pytest.raises(ValueError, match="routing rank"):
        truncate_routing_projectors(
            torch.randn(2, 8, 6),
            torch.randn(2, 8, 6),
            rank=rank,
        )
