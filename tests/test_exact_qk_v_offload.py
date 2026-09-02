from __future__ import annotations

import torch

from basisserve.core.exact_qk_v_offload import (
    full_gqa_value_attention,
    gqa_group_max_page_mass_mask,
    gqa_union_adaptive_page_mass_mask,
    gqa_union_page_mass_mask,
    gqa_union_token_topk_mask,
    sparse_gqa_value_attention,
)


def test_group_max_page_mass_uses_one_fixed_physical_budget() -> None:
    scores = torch.tensor(
        [
            [8.0, 7.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 8.0],
            [6.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 9.0],
        ]
    )

    token_mask, page_mask = gqa_group_max_page_mass_mask(
        scores,
        num_kv_heads=2,
        page_size=2,
        pages_per_kv_head=1,
    )

    assert page_mask.tolist() == [[False, True], [False, True]]
    assert token_mask.tolist() == [
        [False, False, True, True],
        [False, False, True, True],
    ]


def test_adaptive_page_mass_refines_only_uncertain_query_heads() -> None:
    scores = torch.tensor(
        [
            [0.0, -2.0, -3.0, -4.0],
            [0.0, -0.1, -3.0, -4.0],
        ]
    )

    selection = gqa_union_adaptive_page_mass_mask(
        scores,
        num_kv_heads=2,
        page_size=1,
        base_pages_per_query_head=1,
        max_pages_per_query_head=2,
        tail_mass_ratio_threshold=0.5,
    )

    assert selection.page_mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
    ]
    assert selection.token_mask.tolist() == selection.page_mask.tolist()
    assert selection.eligible_query_heads == 2
    assert selection.refined_query_heads == 1
    torch.testing.assert_close(
        torch.tensor(selection.tail_mass_ratio_sum),
        torch.exp(torch.tensor(-2.0)) + torch.exp(torch.tensor(-0.1)),
    )


def test_adaptive_page_mass_unions_refined_pages_within_gqa_group() -> None:
    scores = torch.tensor(
        [
            [0.0, -2.0, -3.0, -4.0],
            [0.0, -0.1, -3.0, -4.0],
        ]
    )

    selection = gqa_union_adaptive_page_mass_mask(
        scores,
        num_kv_heads=1,
        page_size=1,
        base_pages_per_query_head=1,
        max_pages_per_query_head=2,
        tail_mass_ratio_threshold=0.5,
    )

    assert selection.page_mask.tolist() == [[True, True, False, False]]


def test_token_topk_unions_query_heads_within_each_gqa_group() -> None:
    scores = torch.tensor(
        [
            [9.0, 8.0, 0.0, 0.0],
            [0.0, 7.0, 6.0, 0.0],
            [0.0, 0.0, 5.0, 4.0],
            [3.0, 0.0, 0.0, 2.0],
        ]
    )

    mask = gqa_union_token_topk_mask(scores, num_kv_heads=2, top_k=1)

    assert mask.tolist() == [
        [True, True, False, False],
        [True, False, True, False],
    ]


def test_page_mass_uses_logsumexp_and_handles_short_final_page() -> None:
    scores = torch.tensor(
        [
            [5.5, 5.5, 0.0, 0.0, 6.0],
            [0.0, 0.0, 4.0, 4.0, 0.0],
        ]
    )

    token_mask, page_mask = gqa_union_page_mass_mask(
        scores,
        num_kv_heads=1,
        page_size=2,
        pages_per_query_head=1,
    )

    assert page_mask.tolist() == [[True, True, False]]
    assert token_mask.tolist() == [[True, True, True, True, False]]


def test_full_union_sparse_attention_matches_dense_attention() -> None:
    generator = torch.Generator().manual_seed(83)
    scores = torch.randn(4, 17, generator=generator)
    value = torch.randn(2, 17, 7, generator=generator)
    full_mask = torch.ones(2, 17, dtype=torch.bool)

    reference = full_gqa_value_attention(scores, value, heads_per_group=2)
    sparse = sparse_gqa_value_attention(
        scores,
        value,
        full_mask,
        heads_per_group=2,
    )

    torch.testing.assert_close(sparse.output, reference)
    torch.testing.assert_close(sparse.selected_teacher_mass, torch.ones(4))
