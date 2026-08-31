"""End-to-end teacher metrics and pairwise rank-schedule moves."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class PairwiseRankSwap:
    """A budget-preserving adjacent rank transfer between two layers."""

    receiver_layer: int
    donor_layer: int
    receiver_rank_before: int
    receiver_rank_after: int
    donor_rank_before: int
    donor_rank_after: int
    predicted_delta: float

    def apply(self, schedule: Sequence[int]) -> tuple[int, ...]:
        updated = list(int(rank) for rank in schedule)
        if updated[self.receiver_layer] != self.receiver_rank_before:
            raise ValueError("receiver rank does not match the candidate swap")
        if updated[self.donor_layer] != self.donor_rank_before:
            raise ValueError("donor rank does not match the candidate swap")
        updated[self.receiver_layer] = self.receiver_rank_after
        updated[self.donor_layer] = self.donor_rank_after
        if sum(updated) != sum(schedule):
            raise AssertionError("pairwise rank swap changed the total rank budget")
        return tuple(updated)


def _chunked_logsumexp(logits: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must include token and vocabulary dimensions")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    vocab_size = logits.shape[-1]
    result: torch.Tensor | None = None
    for start in range(0, vocab_size, chunk_size):
        chunk = logits[..., start : start + chunk_size].float()
        chunk_lse = torch.logsumexp(chunk, dim=-1)
        result = chunk_lse if result is None else torch.logaddexp(result, chunk_lse)
    if result is None:
        raise ValueError("vocabulary dimension must not be empty")
    return result


@torch.no_grad()
def logits_logsumexp(
    logits: torch.Tensor,
    *,
    vocab_chunk_size: int = 8192,
) -> torch.Tensor:
    """Compute FP32 log-normalizers without materializing full FP32 logits."""

    return _chunked_logsumexp(logits, vocab_chunk_size)


@torch.no_grad()
def teacher_kl_sum(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    teacher_logsumexp: torch.Tensor | None = None,
    vocab_chunk_size: int = 8192,
) -> tuple[float, int]:
    """Return exact full-vocabulary ``KL(teacher || student)`` and token count.

    Teacher logits may live on CPU. Vocabulary chunking bounds FP32 temporary
    memory while retaining the exact softmax over the complete vocabulary.
    """

    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(
            "student and teacher logits must have identical shapes: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim != 3:
        raise ValueError("expected logits with shape [batch, tokens, vocabulary]")
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    device = student_logits.device
    student_lse = _chunked_logsumexp(student_logits, vocab_chunk_size)
    if teacher_logsumexp is None:
        if teacher_logits.device == device:
            teacher_lse = _chunked_logsumexp(teacher_logits, vocab_chunk_size)
        else:
            teacher_lse = _chunked_logsumexp(
                teacher_logits.to(device),
                vocab_chunk_size,
            )
    else:
        expected = student_logits.shape[:-1]
        if tuple(teacher_logsumexp.shape) != tuple(expected):
            raise ValueError(
                f"teacher_logsumexp must have shape {tuple(expected)}, "
                f"got {tuple(teacher_logsumexp.shape)}"
            )
        teacher_lse = teacher_logsumexp.to(device=device, dtype=torch.float32)

    token_kl = torch.zeros_like(student_lse, dtype=torch.float32)
    vocab_size = student_logits.shape[-1]
    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(vocab_size, start + vocab_chunk_size)
        teacher_chunk = teacher_logits[..., start:end].to(
            device=device,
            dtype=torch.float32,
        )
        student_chunk = student_logits[..., start:end].float()
        teacher_log_probs = teacher_chunk - teacher_lse.unsqueeze(-1)
        student_log_probs = student_chunk - student_lse.unsqueeze(-1)
        token_kl.add_(
            (
                teacher_log_probs.exp()
                * (teacher_log_probs - student_log_probs)
            ).sum(dim=-1)
        )

    # Exact dense replays are zero. Clamp only sub-zero roundoff at token level.
    token_kl.clamp_min_(0.0)
    return float(token_kl.double().sum().item()), int(token_kl.numel())


def differentiable_teacher_kl_mean(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    teacher_logsumexp: torch.Tensor | None = None,
    vocab_chunk_size: int = 8192,
) -> torch.Tensor:
    """Return exact full-vocabulary mean ``KL(teacher || student)`` with gradients."""

    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(
            "student and teacher logits must have identical shapes: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim != 3:
        raise ValueError("expected logits with shape [batch, tokens, vocabulary]")
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    device = student_logits.device
    student_lse = _chunked_logsumexp(student_logits, vocab_chunk_size)
    if teacher_logsumexp is None:
        teacher_lse = _chunked_logsumexp(
            teacher_logits.to(device),
            vocab_chunk_size,
        ).detach()
    else:
        if tuple(teacher_logsumexp.shape) != tuple(student_logits.shape[:-1]):
            raise ValueError("teacher_logsumexp shape does not match logits")
        teacher_lse = teacher_logsumexp.to(device=device, dtype=torch.float32)

    token_kl = torch.zeros_like(student_lse, dtype=torch.float32)
    for start in range(0, student_logits.shape[-1], vocab_chunk_size):
        end = min(student_logits.shape[-1], start + vocab_chunk_size)
        teacher_chunk = teacher_logits[..., start:end].to(
            device=device,
            dtype=torch.float32,
        )
        student_chunk = student_logits[..., start:end].float()
        teacher_log_probs = teacher_chunk - teacher_lse.unsqueeze(-1)
        student_log_probs = student_chunk - student_lse.unsqueeze(-1)
        token_kl = token_kl + (
            teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1)
    return token_kl.mean()


def adjacent_pairwise_rank_swaps(
    schedule: Sequence[int],
    *,
    candidate_ranks: Sequence[int],
    costs_by_layer: Sequence[Mapping[int, float]],
    rank_step: int = 16,
) -> tuple[PairwiseRankSwap, ...]:
    """Enumerate all ordered ``(+rank_step, -rank_step)`` schedule moves."""

    ranks = tuple(sorted(int(rank) for rank in candidate_ranks))
    current = tuple(int(rank) for rank in schedule)
    if not ranks or len(ranks) != len(set(ranks)):
        raise ValueError("candidate_ranks must be non-empty and unique")
    if rank_step <= 0:
        raise ValueError("rank_step must be positive")
    if len(costs_by_layer) != len(current):
        raise ValueError("costs_by_layer length must match the schedule")
    rank_set = set(ranks)
    if any(rank not in rank_set for rank in current):
        raise ValueError("schedule contains a rank absent from candidate_ranks")

    moves: list[PairwiseRankSwap] = []
    for receiver, receiver_rank in enumerate(current):
        receiver_after = receiver_rank + rank_step
        if receiver_after not in rank_set:
            continue
        receiver_costs = costs_by_layer[receiver]
        for donor, donor_rank in enumerate(current):
            if donor == receiver:
                continue
            donor_after = donor_rank - rank_step
            if donor_after not in rank_set:
                continue
            donor_costs = costs_by_layer[donor]
            required = (
                (receiver_costs, receiver_rank),
                (receiver_costs, receiver_after),
                (donor_costs, donor_rank),
                (donor_costs, donor_after),
            )
            if any(rank not in costs for costs, rank in required):
                raise ValueError("a layer cost curve is missing a required rank")
            predicted_delta = (
                float(receiver_costs[receiver_after])
                - float(receiver_costs[receiver_rank])
                + float(donor_costs[donor_after])
                - float(donor_costs[donor_rank])
            )
            if not math.isfinite(predicted_delta):
                raise ValueError("pairwise predicted delta must be finite")
            moves.append(
                PairwiseRankSwap(
                    receiver_layer=receiver,
                    donor_layer=donor,
                    receiver_rank_before=receiver_rank,
                    receiver_rank_after=receiver_after,
                    donor_rank_before=donor_rank,
                    donor_rank_after=donor_after,
                    predicted_delta=predicted_delta,
                )
            )
    return tuple(
        sorted(
            moves,
            key=lambda move: (
                move.predicted_delta,
                move.receiver_layer,
                move.donor_layer,
            ),
        )
    )
