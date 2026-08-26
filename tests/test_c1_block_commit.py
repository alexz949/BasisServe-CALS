from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from basisserve.core.c1_block_commit import (
    block_scheduled_teacher_forced_nll,
    compare_exact_block_and_sequential_schedules,
    transactional_exact_c1_greedy_trace,
)
from basisserve.core.c1_shadow_kv import C1ShadowKeyValueCache, ShadowKeyConfig


class ScheduleSensitiveCacheLM(nn.Module):
    def __init__(
        self, *, block_perturbation: float = 0.0, vocab_size: int = 17
    ) -> None:
        super().__init__()
        self.block_perturbation = float(block_perturbation)
        self.vocab_size = int(vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values,
        use_cache: bool,
        **kwargs,
    ) -> SimpleNamespace:
        del kwargs
        assert use_cache
        values = input_ids.to(torch.float32)
        query_length = int(input_ids.shape[1])
        schedule_offset = torch.arange(
            query_length,
            device=input_ids.device,
            dtype=torch.float32,
        )
        schedule_offset = schedule_offset * self.block_perturbation
        keys = torch.stack(
            (
                0.1 * values + schedule_offset,
                0.03 * values.square() + schedule_offset,
                0.07 * values + 0.2,
                0.11 * values + 0.4,
            ),
            dim=-1,
        ).unsqueeze(1)
        c1_values = torch.stack(
            (
                0.2 * values + schedule_offset,
                0.05 * values - schedule_offset,
            ),
            dim=-1,
        ).unsqueeze(1)
        visible_keys, _ = past_key_values.update(keys, c1_values, 0)
        base = int(visible_keys.shape[-2]) - query_length
        vocabulary = torch.arange(
            self.vocab_size,
            device=input_ids.device,
            dtype=torch.float32,
        )
        logits = []
        for index in range(query_length):
            prefix_sum = visible_keys[..., : base + index + 1, :].sum()
            center = torch.remainder(values[:, index] + prefix_sum, self.vocab_size)
            logits.append(
                -0.25 * (vocabulary.unsqueeze(0) - center.unsqueeze(1)).square()
            )
        return SimpleNamespace(logits=torch.stack(logits, dim=1))


def _cache() -> C1ShadowKeyValueCache:
    return C1ShadowKeyValueCache(
        num_layers=1,
        config=ShadowKeyConfig(bits=16, group_size=2, recent_exact_window=0),
    )


def test_block_scheduled_teacher_forcing_scores_every_post_prefill_token() -> None:
    model = ScheduleSensitiveCacheLM()
    tokens = torch.tensor([[2, 5, 1, 7, 3, 4, 8, 6]])
    cache = _cache()

    result = block_scheduled_teacher_forced_nll(
        model,
        tokens,
        cache=cache,
        prefill_length=2,
        block_length=4,
        capture_prediction_logits=True,
    )

    assert result.evaluated_tokens == 6
    assert result.block_count == 2
    assert len(result.token_nlls) == 6
    assert len(result.top1_token_ids) == 6
    assert result.prediction_logits is not None
    assert tuple(result.prediction_logits.shape[:2]) == (1, 6)
    assert cache.committed_length == tokens.shape[1]


def test_exact_block_oracle_reports_zero_drift_for_schedule_invariant_model() -> None:
    model = ScheduleSensitiveCacheLM(block_perturbation=0.0)
    tokens = torch.tensor([[2, 5, 1, 7, 3, 4, 8, 6]])

    result = compare_exact_block_and_sequential_schedules(
        model,
        tokens,
        sequential_cache=_cache(),
        block_cache=_cache(),
        prefill_length=2,
        block_length=3,
    )

    assert result.top1_agreement == 1.0
    assert result.first_top1_disagreement is None
    assert result.sequential_nll_sum == pytest.approx(result.block_nll_sum, abs=1e-6)
    assert result.mean_kl_sequential_to_block == pytest.approx(0.0, abs=1e-7)
    assert result.mean_top5_overlap == pytest.approx(1.0)
    assert result.cache_drift
    assert all(record.key_relative_l2 == 0.0 for record in result.cache_drift)
    assert all(record.c1_value_relative_l2 == 0.0 for record in result.cache_drift)


def test_exact_block_oracle_detects_schedule_dependent_kv_and_logits() -> None:
    model = ScheduleSensitiveCacheLM(block_perturbation=0.125)
    tokens = torch.tensor([[2, 5, 1, 7, 3, 4, 8, 6]])

    result = compare_exact_block_and_sequential_schedules(
        model,
        tokens,
        sequential_cache=_cache(),
        block_cache=_cache(),
        prefill_length=2,
        block_length=4,
    )

    assert any(record.key_relative_l2 > 0.0 for record in result.cache_drift)
    assert any(record.c1_value_relative_l2 > 0.0 for record in result.cache_drift)
    assert any(value > 0.0 for value in result.logit_maximum_absolute_error_by_block)
    assert result.block_nll_sum != pytest.approx(result.sequential_nll_sum, abs=1e-6)
    assert result.mean_kl_sequential_to_block > 0.0


def test_exact_proposal_oracle_reuses_the_generating_sequential_branch() -> None:
    model = ScheduleSensitiveCacheLM(block_perturbation=0.0)
    prompt = torch.tensor([[2, 5, 1]])
    sequential_cache = _cache()
    trace = transactional_exact_c1_greedy_trace(
        model,
        prompt,
        cache=sequential_cache,
        max_new_tokens=6,
    )

    result = compare_exact_block_and_sequential_schedules(
        model,
        trace.token_ids,
        sequential_cache=sequential_cache,
        block_cache=_cache(),
        prefill_length=prompt.shape[1],
        block_length=3,
        sequential_reference=trace.schedule,
    )

    assert result.evaluated_tokens == 6
    assert result.sequential_label_top1_matches == 6
    assert result.top1_agreement == 1.0
