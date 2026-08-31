from __future__ import annotations

import math

import torch

from basisserve.core.layer_ahead_prefetch import (
    apply_headwise_linear,
    fit_headwise_linear,
    group_query_heads,
    prefetch_set_statistics,
    ungroup_query_heads,
)


def test_headwise_linear_fit_recovers_exact_map() -> None:
    generator = torch.Generator().manual_seed(43)
    source = torch.randn(2, 3, 23, 5, generator=generator, dtype=torch.float64)
    weight = torch.randn(3, 5, 7, generator=generator, dtype=torch.float64)
    target = torch.einsum("bhsi,hio->bhso", source, weight)

    factors = fit_headwise_linear(source, target)
    predicted = apply_headwise_linear(source, factors)

    torch.testing.assert_close(predicted.double(), target, rtol=1.0e-5, atol=1.0e-5)


def test_grouped_query_round_trip() -> None:
    query = torch.arange(2 * 8 * 5 * 3).reshape(2, 8, 5, 3)
    grouped = group_query_heads(query, kv_heads=2)
    restored = ungroup_query_heads(grouped, query_heads=8)

    assert grouped.shape == (2, 2, 5, 12)
    torch.testing.assert_close(restored, query)


def test_prefetch_set_statistics_reports_late_and_wasted_pages() -> None:
    actual = torch.tensor([[[True, True, False, False], [True, False, True, False]]])
    prefetched = torch.tensor([[[True, False, True, False], [True, True, True, False]]])

    metrics = prefetch_set_statistics(actual, prefetched)

    assert metrics["prefetch_page_recall"] == 0.75
    assert math.isclose(
        metrics["prefetch_page_precision"],
        (0.5 + 2.0 / 3.0) / 2.0,
        rel_tol=1.0e-6,
    )
    assert metrics["mean_late_pages_per_kv_head"] == 0.5
    assert metrics["mean_wasted_pages_per_kv_head"] == 1.0
    assert metrics["fully_covered_kv_head_fraction"] == 0.5
