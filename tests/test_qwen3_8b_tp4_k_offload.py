from __future__ import annotations

import math

import torch

from basisserve.core.qwen3_8b_tp4_k_offload import (
    PinnedCPUPageMajorKeyCache,
    conditional_router_page_log_mass,
    quest_page_scores,
    select_fixed_group_max_pages,
    shadowkv_page_scores,
)


def _apply_rope_reference(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def test_page_major_cache_appends_and_packs_selected_pages() -> None:
    cache = PinnedCPUPageMajorKeyCache(
        batch_size=1,
        kv_heads=2,
        capacity=7,
        head_dim=3,
        page_size=2,
        dtype=torch.float32,
    )
    first = torch.arange(1 * 2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3)
    second = torch.arange(1 * 2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 3) + 100
    cache.append(first)
    cache.append(second)
    selected = torch.tensor([[[0, 3], [1, 2]]])

    observed = cache.fetch(selected, device=torch.device("cpu"))
    complete = torch.cat((first, second), dim=2)
    padded = torch.nn.functional.pad(complete, (0, 0, 0, 1)).reshape(1, 2, 4, 2, 3)
    expected = torch.stack(
        (
            torch.stack((padded[0, 0, 0], padded[0, 0, 3])),
            torch.stack((padded[0, 1, 1], padded[0, 1, 2])),
        )
    ).unsqueeze(0)

    torch.testing.assert_close(observed, expected)
    assert cache.length == 7
    assert cache.last_requested_bytes == observed.numel() * observed.element_size()


def test_group_max_selection_keeps_prefix_and_current_with_fixed_budget() -> None:
    page_log_mass = torch.tensor(
        [[[[100.0, 1.0, 8.0, 7.0, -5.0], [90.0, 9.0, 2.0, 6.0, -4.0]]]]
    )
    selected = select_fixed_group_max_pages(
        page_log_mass,
        pages_per_kv_head=3,
        pinned_prefix_pages=1,
        force_current_page=True,
    )

    assert selected.tolist() == [[[0, 1, 4]]]


def test_group_max_selection_normalizes_each_query_head() -> None:
    page_log_mass = torch.tensor(
        [[[[0.0, 10.0, 0.0], [102.0, 101.0, 100.0]]]]
    )
    selected = select_fixed_group_max_pages(
        page_log_mass,
        pages_per_kv_head=1,
        pinned_prefix_pages=0,
        force_current_page=False,
    )

    assert selected.tolist() == [[[1]]]


def test_conditional_page_lse_matches_explicit_base_plus_residual_scores() -> None:
    generator = torch.Generator().manual_seed(20260902)
    batch = 1
    kv_heads = 2
    heads_per_kv = 4
    query_heads = kv_heads * heads_per_kv
    tokens = 7
    value_rank = 5
    base_rank = 3
    residual_rank = 2
    page_size = 2
    query = torch.randn(batch, query_heads, 1, 128, generator=generator)
    value = torch.randn(batch, kv_heads, tokens, value_rank, generator=generator)
    residual_code = torch.randn(
        batch, kv_heads, tokens, residual_rank, generator=generator
    )
    base_left = torch.randn(kv_heads, value_rank, base_rank, generator=generator)
    base_right = torch.randn(kv_heads, base_rank, 128, generator=generator)
    base_bias = torch.randn(kv_heads, 128, generator=generator)
    residual_query = torch.randn(
        query_heads, 128, residual_rank, generator=generator
    )
    angles = torch.randn(tokens, 64, generator=generator)
    cos = angles.cos()
    sin = angles.sin()
    scale = 128**-0.5

    observed = conditional_router_page_log_mass(
        query,
        value,
        residual_code,
        base_left=base_left,
        base_right=base_right,
        base_bias=base_bias,
        residual_query=residual_query,
        rope_cos=cos,
        rope_sin=sin,
        page_size=page_size,
        page_chunk=2,
        scale=scale,
    )

    base_pre = torch.einsum(
        "bgtv,gvr,grd->bgtd", value, base_left, base_right
    ) + base_bias[None, :, None]
    base_post = _apply_rope_reference(
        base_pre,
        cos[None, None],
        sin[None, None],
    )
    grouped_query = query[:, :, 0].reshape(batch, kv_heads, heads_per_kv, 128)
    grouped_projector = residual_query.reshape(
        kv_heads, heads_per_kv, 128, residual_rank
    )
    query_code = torch.einsum("bghd,ghdr->bghr", grouped_query, grouped_projector)
    scores = scale * (
        torch.einsum("bghd,bgtd->bght", grouped_query, base_post)
        + torch.einsum("bghr,bgtr->bght", query_code, residual_code)
    )
    padded = torch.nn.functional.pad(
        scores,
        (0, math.ceil(tokens / page_size) * page_size - tokens),
        value=-torch.inf,
    )
    expected = torch.logsumexp(
        padded.reshape(batch, kv_heads, heads_per_kv, -1, page_size),
        dim=-1,
    )

    torch.testing.assert_close(observed, expected, rtol=1.0e-5, atol=1.0e-6)


def test_quest_page_scores_match_coordinate_bounds() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(1, 8, 1, 128, generator=generator)
    minimum = torch.randn(1, 2, 3, 128, generator=generator)
    maximum = minimum + torch.rand(1, 2, 3, 128, generator=generator)
    scale = 128**-0.5

    observed = quest_page_scores(
        query,
        minimum,
        maximum,
        scale=scale,
    )
    grouped = query[:, :, 0].reshape(1, 2, 4, 128)
    expected = scale * torch.maximum(
        grouped[:, :, :, None] * minimum[:, :, None],
        grouped[:, :, :, None] * maximum[:, :, None],
    ).sum(dim=-1)

    torch.testing.assert_close(observed, expected)


def test_shadowkv_page_scores_match_mean_landmark_dot_products() -> None:
    generator = torch.Generator().manual_seed(19)
    query = torch.randn(1, 8, 1, 128, generator=generator)
    means = torch.randn(1, 2, 5, 128, generator=generator)
    scale = 128**-0.5

    observed = shadowkv_page_scores(query, means, scale=scale)
    grouped = query[:, :, 0].reshape(1, 2, 4, 128)
    expected = scale * torch.einsum("bghd,bgpd->bghp", grouped, means)

    torch.testing.assert_close(observed, expected)
