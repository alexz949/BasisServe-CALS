from __future__ import annotations

import math

import pytest
import torch

from basisserve.core.c1_conditional_page_attention import (
    c1_conditional_page_topk_attention,
)
from basisserve.core.c1_k_reverse_shadow import (
    ReverseShadowConfig,
    c1_k_reverse_shadow_block_attention,
)


@pytest.mark.parametrize("query_block_size", (1, 2, 4))
def test_full_query_matches_single_query_page_reference(
    query_block_size: int,
) -> None:
    torch.manual_seed(5)
    query = torch.randn(1, 4, 7, 8)
    key = torch.randn(1, 2, 7, 8)
    value = torch.randn(1, 2, 7, 5)
    sidecar = torch.randn(1, 2, 7, 3)
    query_projector = torch.randn(4, 8, 3)
    config = ReverseShadowConfig(
        page_size=2,
        exact_token_budget=4,
        selector="kq_svd",
        quest_support="physical_shared",
        pinned_prefix_pages=1,
    )
    expected, _ = c1_k_reverse_shadow_block_attention(
        query,
        key,
        value,
        config,
        routing_query_projector=query_projector,
        routing_sidecar=sidecar,
    )
    observed = c1_conditional_page_topk_attention(
        query,
        key,
        value,
        sidecar,
        query_projector,
        page_size=2,
        exact_token_budget=4,
        pinned_prefix_pages=1,
        scale=8**-0.5,
        query_block_size=query_block_size,
    )
    torch.testing.assert_close(observed.output, expected, atol=2e-6, rtol=2e-6)


def test_full_budget_matches_dense_causal_gqa_attention() -> None:
    torch.manual_seed(11)
    query = torch.randn(1, 4, 7, 8)
    key = torch.randn(1, 2, 7, 8)
    value = torch.randn(1, 2, 7, 5)
    sidecar = torch.randn(1, 2, 7, 3)
    query_projector = torch.randn(4, 8, 3)
    observed = c1_conditional_page_topk_attention(
        query,
        key,
        value,
        sidecar,
        query_projector,
        page_size=2,
        exact_token_budget=8,
        pinned_prefix_pages=1,
        scale=8**-0.5,
        query_block_size=3,
    )

    head_to_kv = torch.arange(2).repeat_interleave(2)
    expanded_key = key.index_select(1, head_to_kv)
    expanded_value = value.index_select(1, head_to_kv)
    scores = torch.matmul(query, expanded_key.transpose(-1, -2)) / math.sqrt(8)
    causal = torch.ones(7, 7, dtype=torch.bool).tril()
    probability = torch.softmax(
        scores.float().masked_fill(~causal, -torch.inf),
        dim=-1,
    )
    expected = torch.matmul(probability, expanded_value.float())
    torch.testing.assert_close(observed.output, expected, atol=2e-6, rtol=2e-6)
    assert observed.statistics["selected_tokens"] == 56.0
    assert observed.statistics["query_selected_tokens"] == 112.0
