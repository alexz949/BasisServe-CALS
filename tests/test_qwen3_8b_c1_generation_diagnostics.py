from __future__ import annotations

import torch
from torch.nn import functional as F

from basisserve.core.sampled_terminal_kl import terminal_kl_from_statistics
from evaluation.diagnose_qwen3_8b_c1_generation import (
    classify_divergence,
    position_ranges,
    requested_position_ranges,
    response_event_masks,
    sample_positions_by_fine_bucket,
    select_replay_documents,
    streaming_token_metrics,
    teacher_statistics_from_logsumexp,
)


def test_streamed_token_metrics_and_kl_match_dense_softmax() -> None:
    generator = torch.Generator().manual_seed(11)
    teacher_hidden = torch.randn(7, 5, generator=generator)
    student_hidden = torch.randn(7, 5, generator=generator)
    weight = torch.randn(13, 5, generator=generator)
    targets = torch.tensor([0, 2, 4, 6, 8, 10, 12])

    teacher_metrics = streaming_token_metrics(
        teacher_hidden,
        weight,
        targets,
        vocab_chunk_size=4,
    )
    student_metrics = streaming_token_metrics(
        student_hidden,
        weight,
        targets,
        vocab_chunk_size=4,
    )
    teacher_logits = F.linear(teacher_hidden, weight)
    student_logits = F.linear(student_hidden, weight)
    direct_values, direct_indices = teacher_logits.topk(2, dim=-1)
    direct_target_logits = teacher_logits.gather(1, targets[:, None]).squeeze(1)
    direct_target_ranks = 1 + (
        teacher_logits > direct_target_logits[:, None]
    ).sum(dim=-1)
    direct_best_other = torch.where(
        direct_indices[:, 0] == targets,
        direct_values[:, 1],
        direct_values[:, 0],
    )
    assert torch.allclose(
        teacher_metrics.logsumexp,
        torch.logsumexp(teacher_logits, dim=-1),
        atol=1e-6,
    )
    assert torch.equal(teacher_metrics.top_token, direct_indices[:, 0])
    assert torch.equal(teacher_metrics.target_rank, direct_target_ranks)
    assert torch.allclose(
        teacher_metrics.top1_margin,
        direct_values[:, 0] - direct_values[:, 1],
        atol=1e-6,
    )
    assert torch.allclose(
        teacher_metrics.target_margin,
        direct_target_logits - direct_best_other,
        atol=1e-6,
    )

    statistics = teacher_statistics_from_logsumexp(
        teacher_hidden,
        weight,
        teacher_metrics.logsumexp,
        vocab_chunk_size=4,
    )
    streamed_kl = terminal_kl_from_statistics(
        statistics,
        student_hidden,
        student_metrics.logsumexp,
    )
    direct_log_teacher = teacher_logits.log_softmax(dim=-1)
    direct_kl = (
        direct_log_teacher.exp()
        * (direct_log_teacher - student_logits.log_softmax(dim=-1))
    ).sum(dim=-1)
    assert torch.allclose(streamed_kl, direct_kl, atol=2e-6)


def test_position_buckets_include_requested_2048_tail() -> None:
    fine = position_ranges(4096)
    requested = requested_position_ranges(4096)
    assert len(fine) == 16
    assert fine[0] == (0, 256)
    assert fine[-1] == (3840, 4096)
    assert requested[:2] == ((0, 256), (256, 512))
    assert requested[-1] == (2048, 4096)
    assert len(requested) == 9


def test_c4_sampling_is_balanced_reproducible_and_never_targets_past_window() -> None:
    left = sample_positions_by_fine_bucket(
        sequence_length=4096,
        positions_per_bucket=32,
        seed=7,
        window_index=512,
    )
    right = sample_positions_by_fine_bucket(
        sequence_length=4096,
        positions_per_bucket=32,
        seed=7,
        window_index=512,
    )
    assert torch.equal(left, right)
    assert len(left) == 512
    assert int(left.min()) >= 0
    assert int(left.max()) <= 4094
    counts = torch.bincount(left // 256, minlength=16)
    assert counts.tolist() == [32] * 16


def test_response_event_masks_distinguish_number_marker_and_gold_answer() -> None:
    response = "We use 12 + 6. #### 18"
    offsets = [(index, index + 1) for index in range(len(response))]
    masks = response_event_masks(response, offsets)
    assert int(masks["numeric_token"].sum()) == 5
    assert int(masks["answer_marker_token"].sum()) == 4
    assert int(masks["gold_answer_token"].sum()) == 2


def test_divergence_classifier_separates_planning_calculation_and_control() -> None:
    response = "Plan the operation and use 42.\n#### 42"
    assert classify_divergence(
        response=response,
        offset=(0, 4),
        token_index=2,
        terminated=False,
    ) == ("planning", "early_reasoning_0_19")
    assert classify_divergence(
        response=response,
        offset=(27, 29),
        token_index=21,
        terminated=False,
    ) == ("calculation", "arithmetic_or_numeric")
    assert classify_divergence(
        response=response,
        offset=(31, 35),
        token_index=30,
        terminated=False,
    ) == ("control", "answer_marker")
    assert classify_divergence(
        response=response,
        offset=None,
        token_index=40,
        terminated=True,
    ) == ("control", "termination_length")


def test_replay_selection_uses_only_dense_correct_and_preserves_both_outcomes() -> None:
    dense = {
        index: {"exact_match": float(index < 8)} for index in range(10)
    }
    c1 = {
        index: {"exact_match": float(index % 2 == 0)} for index in range(10)
    }
    selected = select_replay_documents(
        dense,
        c1,
        maximum=6,
        seed=3,
    )
    assert len(selected) == 6
    assert all(dense[index]["exact_match"] for index in selected)
    assert any(c1[index]["exact_match"] for index in selected)
    assert any(not c1[index]["exact_match"] for index in selected)
    assert sum(not c1[index]["exact_match"] for index in selected) == 3
    assert sum(bool(c1[index]["exact_match"]) for index in selected) == 3
