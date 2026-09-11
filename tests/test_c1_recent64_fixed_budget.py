import torch

from basisserve.core.c1_conditional_recent_attention import (
    fixed_budget_recent_support, c1_conditional_page_recent64_attention,
)


def test_budget_sink_recent_and_disjoint_pages():
    for length in [1, 31, 32, 63, 64, 95, 2047, 2048, *range(2049, 2081), 32768]:
        scores = torch.arange(length).float().view(1, 1, 1, -1).expand(1, 4, 1, -1)
        valid = torch.ones_like(scores, dtype=torch.bool)
        ids, mask = fixed_budget_recent_support(scores, valid, kv_heads=1)
        selected = ids[mask].tolist()
        assert len(selected) == len(set(selected)) == min(length, 2048)
        assert set(range(min(32, length))) <= set(selected)
        assert set(range(max(0, length-64), length)) <= set(selected)
        if length > 2048:
            history = set(selected) - set(range(32)) - set(range(length-64, length))
            assert len(history) == 61*32
            assert all(set(range(i//32*32, i//32*32+32)) <= history for i in history)


def test_fixed_attention_matches_explicit_support():
    torch.manual_seed(5)
    length = 133
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 1, length, 8)
    v = torch.randn(1, 1, length, 6)
    side = torch.randn(1, 1, length, 4)
    projector = torch.randn(4, 8, 4)
    result = c1_conditional_page_recent64_attention(
        q, k, v, side, projector, page_size=32, exact_token_budget=96,
        pinned_prefix_pages=1, scale=8**-.5, query_block_size=1,
        recent_within_budget=True,
    )
    scores = q @ k.repeat_interleave(4, 1).mT * 8**-.5
    scores[..., 32:length-64] = -torch.inf
    expected = scores.softmax(-1) @ v.repeat_interleave(4, 1)
    torch.testing.assert_close(result.output, expected)
    assert result.statistics['selected_tokens'] == 96
