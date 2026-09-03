"""Exact terminal-KL statistics on nested samples of token positions.

The vocabulary is always evaluated exactly.  Only the expectation over causal
prediction positions is sampled.  A dense teacher distribution is represented
by its log normalizer, expected output embedding, and expected log probability,
so subsequent paired comparisons never need to retain teacher vocabulary
logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class TeacherTerminalStatistics:
    """Sufficient statistics for ``KL(teacher || student)`` at selected tokens."""

    logsumexp: Tensor
    expected_output_weight: Tensor
    expected_log_probability: Tensor


def parse_position_counts(raw: str) -> tuple[int, ...]:
    """Parse distinct increasing terminal-position sample counts."""

    counts = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
    assert counts
    assert counts == tuple(sorted(set(counts)))
    assert min(counts) > 0
    return counts


def nested_prediction_positions(
    *,
    num_windows: int,
    sequence_length: int,
    position_counts: Sequence[int],
    seed: int,
) -> Tensor:
    """Return per-window nested uniform samples of causal logit positions.

    Position ``t`` predicts token ``t + 1``.  Consequently a length-``T``
    window has ``T - 1`` valid positions, numbered ``0`` through ``T - 2``.
    Every smaller sample is a prefix of the largest sample's random ordering.
    """

    counts = tuple(map(int, position_counts))
    assert num_windows > 0 and sequence_length > 1
    assert counts
    assert counts == tuple(sorted(set(counts)))
    assert min(counts) > 0
    available = sequence_length - 1
    assert max(counts) <= available
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.stack(
        [
            torch.randperm(available, generator=generator)[: max(counts)]
            for _ in range(num_windows)
        ],
        dim=0,
    )


def gather_prediction_hidden(hidden_states: Tensor, positions: Tensor) -> Tensor:
    """Gather independent positions from each batch row of ``[B,T,H]`` states."""

    assert hidden_states.ndim == 3 and positions.ndim == 2
    assert hidden_states.shape[0] == positions.shape[0]
    assert positions.numel() > 0
    work = positions.to(device=hidden_states.device, dtype=torch.long)
    assert int(work.min()) >= 0 and int(work.max()) < hidden_states.shape[1]
    return hidden_states.gather(
        1,
        work.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]),
    )


def _validate_output_head(hidden_states: Tensor, output_weight: Tensor) -> None:
    assert hidden_states.ndim >= 2 and output_weight.ndim == 2
    assert hidden_states.shape[-1] == output_weight.shape[-1]
    assert output_weight.shape[0] > 0


@torch.no_grad()
def streaming_output_logsumexp(
    hidden_states: Tensor,
    output_weight: Tensor,
    *,
    vocab_chunk_size: int,
) -> Tensor:
    """Compute exact full-vocabulary FP32 log normalizers without full logits."""

    _validate_output_head(hidden_states, output_weight)
    assert vocab_chunk_size > 0
    original_shape = hidden_states.shape[:-1]
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    accumulator: Tensor | None = None
    for start in range(0, output_weight.shape[0], vocab_chunk_size):
        stop = min(output_weight.shape[0], start + vocab_chunk_size)
        weight = output_weight[start:stop].to(device=flat.device, dtype=torch.float32)
        chunk = F.linear(flat.float(), weight)
        chunk_lse = torch.logsumexp(chunk, dim=-1)
        accumulator = (
            chunk_lse
            if accumulator is None
            else torch.logaddexp(accumulator, chunk_lse)
        )
        del weight, chunk, chunk_lse
    assert accumulator is not None
    return accumulator.reshape(original_shape)


@torch.no_grad()
def teacher_terminal_statistics(
    teacher_hidden: Tensor,
    output_weight: Tensor,
    *,
    vocab_chunk_size: int,
) -> TeacherTerminalStatistics:
    """Compress a bias-free dense teacher softmax into exact KL statistics.

    The output head is streamed twice: once for the global normalizer and once
    for normalized moments.  Peak vocabulary storage is therefore bounded by
    ``num_selected_positions * vocab_chunk_size``.
    """

    _validate_output_head(teacher_hidden, output_weight)
    logsumexp = streaming_output_logsumexp(
        teacher_hidden,
        output_weight,
        vocab_chunk_size=vocab_chunk_size,
    )
    original_shape = teacher_hidden.shape[:-1]
    hidden_size = teacher_hidden.shape[-1]
    flat_hidden = teacher_hidden.reshape(-1, hidden_size)
    flat_lse = logsumexp.reshape(-1)
    expected_weight = torch.zeros(
        flat_hidden.shape[0],
        hidden_size,
        device=flat_hidden.device,
        dtype=torch.float32,
    )
    expected_log_probability = torch.zeros(
        flat_hidden.shape[0],
        device=flat_hidden.device,
        dtype=torch.float32,
    )
    for start in range(0, output_weight.shape[0], vocab_chunk_size):
        stop = min(output_weight.shape[0], start + vocab_chunk_size)
        weight = output_weight[start:stop].to(
            device=flat_hidden.device,
            dtype=torch.float32,
        )
        logits = F.linear(flat_hidden.float(), weight)
        log_probabilities = logits - flat_lse.unsqueeze(-1)
        probabilities = log_probabilities.exp()
        expected_weight.add_(probabilities @ weight.float())
        expected_log_probability.add_(
            (probabilities * log_probabilities).sum(dim=-1)
        )
        del weight, logits, log_probabilities, probabilities
    return TeacherTerminalStatistics(
        logsumexp=logsumexp.float(),
        expected_output_weight=expected_weight.reshape(
            *original_shape, hidden_size
        ),
        expected_log_probability=expected_log_probability.reshape(original_shape),
    )


@torch.no_grad()
def terminal_kl_from_statistics(
    statistics: TeacherTerminalStatistics,
    student_hidden: Tensor,
    student_logsumexp: Tensor,
) -> Tensor:
    """Return per-position exact ``KL(teacher || student)`` for a bias-free head."""

    assert tuple(statistics.expected_output_weight.shape) == tuple(student_hidden.shape)
    expected_shape = student_hidden.shape[:-1]
    assert tuple(student_logsumexp.shape) == tuple(expected_shape)
    expected_student_logit = (
        statistics.expected_output_weight.float() * student_hidden.float()
    ).sum(dim=-1)
    return (
        statistics.expected_log_probability.float()
        - expected_student_logit
        + student_logsumexp.float()
    )


@torch.no_grad()
def paired_terminal_kl_delta(
    statistics: TeacherTerminalStatistics,
    *,
    anchor_hidden: Tensor,
    candidate_hidden: Tensor,
    anchor_logsumexp: Tensor,
    candidate_logsumexp: Tensor,
) -> Tensor:
    """Return exact per-position ``KL(T||P) - KL(T||A)``."""

    assert tuple(anchor_hidden.shape) == tuple(candidate_hidden.shape)
    assert tuple(statistics.expected_output_weight.shape) == tuple(anchor_hidden.shape)
    expected_shape = anchor_hidden.shape[:-1]
    assert tuple(anchor_logsumexp.shape) == tuple(expected_shape)
    assert tuple(candidate_logsumexp.shape) == tuple(expected_shape)
    linear_delta = (
        statistics.expected_output_weight.float()
        * (anchor_hidden.float() - candidate_hidden.float())
    ).sum(dim=-1)
    return linear_delta + candidate_logsumexp.float() - anchor_logsumexp.float()


@torch.no_grad()
def selected_token_nll(
    hidden_states: Tensor,
    logsumexp: Tensor,
    output_weight: Tensor,
    labels: Tensor,
) -> Tensor:
    """Return per-position NLL without constructing vocabulary logits."""

    assert tuple(labels.shape) == tuple(hidden_states.shape[:-1])
    assert tuple(logsumexp.shape) == tuple(labels.shape)
    label_weight = F.embedding(
        labels.to(device=output_weight.device, dtype=torch.long),
        output_weight,
    ).to(hidden_states.device)
    label_logits = (hidden_states.float() * label_weight.float()).sum(dim=-1)
    return logsumexp.float() - label_logits
