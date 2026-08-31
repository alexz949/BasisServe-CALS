from __future__ import annotations

import math

import torch

from basisserve.core.pairwise_k_sparse import (
    PairwiseQuestConfig,
    pairwise_quest_select_pages,
    pairwise_quest_sparse_attention,
)


def test_quest_shared_page_score_bounds_every_gqa_head_and_token() -> None:
    generator = torch.Generator().manual_seed(20260828)
    projected_query = torch.randn(1, 4, 3, 5, generator=generator)
    history_code = torch.randn(1, 2, 7, 5, generator=generator)
    scaling = 0.37
    config = PairwiseQuestConfig(
        page_size=3,
        historical_token_budget=3,
        landmark_dtype="float32",
    )

    selection = pairwise_quest_select_pages(
        projected_query,
        history_code,
        config,
        scaling=scaling,
    )

    assert selection.page_scores.shape == (1, 2, 3, 3)
    assert selection.selected_page_indices.shape == (1, 2, 3, 1)
    for group in range(2):
        for head_in_group in range(2):
            head = 2 * group + head_in_group
            for query_position in range(3):
                for token in range(7):
                    page = token // config.page_size
                    exact_score = scaling * torch.dot(
                        projected_query[0, head, query_position],
                        history_code[0, group, token],
                    )
                    assert selection.page_scores[0, group, query_position, page] >= (
                        exact_score - 1e-6
                    )


def test_full_budget_sparse_attention_matches_dense_mixed_coordinate_attention() -> None:
    generator = torch.Generator().manual_seed(20260829)
    query = torch.randn(1, 4, 3, 6, generator=generator)
    projected_query = torch.randn(1, 4, 3, 5, generator=generator)
    history_code = torch.randn(1, 2, 7, 5, generator=generator)
    current_key = torch.randn(1, 2, 3, 6, generator=generator)
    value_cache = torch.randn(1, 2, 10, 4, generator=generator)
    scaling = 1.0 / math.sqrt(6)
    config = PairwiseQuestConfig(
        page_size=3,
        historical_token_budget=99,
        landmark_dtype="float32",
    )

    result = pairwise_quest_sparse_attention(
        query,
        projected_query,
        history_code,
        current_key,
        value_cache,
        config,
        scaling=scaling,
    )

    repeated_history = history_code.repeat_interleave(2, dim=1)
    historical_scores = torch.matmul(
        projected_query,
        repeated_history.transpose(2, 3),
    ) * scaling
    repeated_current_key = current_key.repeat_interleave(2, dim=1)
    current_scores = torch.matmul(
        query,
        repeated_current_key.transpose(2, 3),
    ) * scaling
    causal = torch.ones(3, 3, dtype=torch.bool).tril()
    current_scores.masked_fill_(~causal.view(1, 1, 3, 3), -torch.inf)
    probabilities = torch.softmax(
        torch.cat((historical_scores, current_scores), dim=-1),
        dim=-1,
    )
    repeated_value = value_cache.repeat_interleave(2, dim=1)
    expected = torch.matmul(probabilities, repeated_value)

    torch.testing.assert_close(result.output, expected, rtol=2e-6, atol=2e-6)
    assert result.statistics["selected_physical_tokens"] == 1 * 2 * 3 * 7
    assert result.statistics["selected_token_fraction"] == 1.0


def test_sparse_selection_obeys_physical_page_budget() -> None:
    generator = torch.Generator().manual_seed(20260830)
    projected_query = torch.randn(2, 8, 3, 7, generator=generator)
    history_code = torch.randn(2, 2, 10, 7, generator=generator)
    config = PairwiseQuestConfig(
        page_size=4,
        historical_token_budget=5,
        landmark_dtype="float32",
    )

    selection = pairwise_quest_select_pages(
        projected_query,
        history_code,
        config,
        scaling=1.0,
    )

    assert config.page_budget == 2
    assert selection.selected_page_indices.shape == (2, 2, 3, 2)
    assert selection.selected_token_indices.shape == (2, 2, 3, 8)
    assert torch.all(selection.selected_token_valid.sum(dim=-1) <= 8)
    assert selection.selected_page_indices.ndim == 4
