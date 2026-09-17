import torch

from evaluation.eval_llama_chunk8_recent64_ruler import (
    chunk8_routed_plus_recent_support,
    page32_routed_plus_recent_support,
)


def _check_complete_recent(ids: torch.Tensor, total_tokens: int) -> None:
    recent = torch.arange(total_tokens - 64, total_tokens)
    for group in range(ids.shape[1]):
        assert torch.isin(recent, ids[0, group].cpu()).all()
        assert int(ids[0, group].unique().numel()) == int(ids.shape[-1])


def test_chunk8_aligned_budget_is_2048_historical_plus_recent64() -> None:
    torch.manual_seed(7)
    total_tokens = 4096
    chunks = (total_tokens - 64) // 8
    logits = torch.randn(1, 2, 4, chunks)
    ids, stats = chunk8_routed_plus_recent_support(logits, total_tokens)
    assert ids.shape == (1, 2, 2112)
    assert stats == {
        "historical_budget_tokens": 2048,
        "recent_outside_tokens": 64,
        "logical_support_cap_tokens": 2112,
        "actual_support_tokens": 2112,
        "page_size": 8,
        "equivalent_page_slots": 264,
        "historical_pages": 256,
        "pinned_sink_pages": 4,
        "routed_historical_pages": 252,
        "tail_pages": 8,
        "exact_tail_tokens": 64,
        "unused_token_capacity": 0,
    }
    _check_complete_recent(ids, total_tokens)


def test_chunk8_ragged_budget_keeps_recent64_below_2112() -> None:
    torch.manual_seed(8)
    total_tokens = 4099
    chunks = (total_tokens - 64) // 8
    logits = torch.randn(1, 2, 4, chunks)
    ids, stats = chunk8_routed_plus_recent_support(logits, total_tokens)
    assert ids.shape == (1, 2, 2107)
    assert stats["historical_pages"] == 255
    assert stats["routed_historical_pages"] == 251
    assert stats["tail_pages"] == 9
    assert stats["exact_tail_tokens"] == 67
    assert stats["unused_token_capacity"] == 5
    _check_complete_recent(ids, total_tokens)


def test_page32_aligned_budget_is_2048_historical_plus_recent64() -> None:
    torch.manual_seed(9)
    total_tokens = 4096
    scores = torch.randn(1, 2, 4, total_tokens)
    ids, valid = page32_routed_plus_recent_support(scores)
    assert ids.shape == valid.shape == (1, 2, 2112)
    assert valid.all()
    _check_complete_recent(ids, total_tokens)


def test_page32_ragged_budget_keeps_recent64_below_2112() -> None:
    torch.manual_seed(10)
    total_tokens = 4101
    scores = torch.randn(1, 2, 4, total_tokens)
    ids, valid = page32_routed_plus_recent_support(scores)
    assert ids.shape == valid.shape == (1, 2, 2085)
    assert valid.all()
    _check_complete_recent(ids, total_tokens)
