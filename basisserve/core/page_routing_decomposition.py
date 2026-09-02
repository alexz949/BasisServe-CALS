"""Diagnostics for page-logit linearization and Top-page boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PageLSETerms:
    teacher_logits: torch.Tensor
    proxy_logits: torch.Tensor
    teacher_mass: torch.Tensor
    linearized_error: torch.Tensor
    true_error: torch.Tensor
    remainder: torch.Tensor


@dataclass(frozen=True)
class FisherPartition:
    total: torch.Tensor
    inside_inside: torch.Tensor
    outside_outside: torch.Tensor
    cross: torch.Tensor


def page_lse_terms(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    *,
    page_size: int,
) -> PageLSETerms:
    """Return true and teacher-linearized page-logit errors."""

    tokens = int(exact_scores.shape[-1])
    pages = (tokens + int(page_size) - 1) // int(page_size)
    padded_tokens = pages * int(page_size)
    padding = padded_tokens - tokens
    exact = torch.nn.functional.pad(exact_scores, (0, padding), value=-torch.inf)
    proxy = torch.nn.functional.pad(proxy_scores, (0, padding), value=-torch.inf)
    exact = exact.reshape(*exact_scores.shape[:-1], pages, int(page_size))
    proxy = proxy.reshape(*proxy_scores.shape[:-1], pages, int(page_size))
    teacher_logits = torch.logsumexp(exact, dim=-1)
    proxy_logits = torch.logsumexp(proxy, dim=-1)
    conditional = torch.softmax(exact, dim=-1)
    valid = torch.isfinite(exact)
    token_error = torch.where(valid, proxy - exact, torch.zeros_like(exact))
    linearized_error = torch.sum(conditional * token_error, dim=-1)
    true_error = proxy_logits - teacher_logits
    teacher_mass = torch.softmax(teacher_logits, dim=-1)
    return PageLSETerms(
        teacher_logits=teacher_logits,
        proxy_logits=proxy_logits,
        teacher_mass=teacher_mass,
        linearized_error=linearized_error,
        true_error=true_error,
        remainder=true_error - linearized_error,
    )


def _weighted_pair_loss(
    first_weight: torch.Tensor,
    first_value: torch.Tensor,
    second_weight: torch.Tensor,
    second_value: torch.Tensor,
    *,
    coefficient: float,
) -> torch.Tensor:
    first_mass = first_weight.sum(dim=-1)
    second_mass = second_weight.sum(dim=-1)
    first_mean = torch.sum(first_weight * first_value, dim=-1)
    second_mean = torch.sum(second_weight * second_value, dim=-1)
    first_second = torch.sum(first_weight * first_value.square(), dim=-1)
    second_second = torch.sum(second_weight * second_value.square(), dim=-1)
    return float(coefficient) * (
        second_mass * first_second
        + first_mass * second_second
        - 2.0 * first_mean * second_mean
    )


def fisher_partition(
    page_error: torch.Tensor,
    teacher_mass: torch.Tensor,
    inside: torch.Tensor,
) -> FisherPartition:
    """Partition ``0.5 delta.T J_rho delta`` around a page set."""

    inside_weight = teacher_mass * inside
    outside_weight = teacher_mass * ~inside
    inside_inside = _weighted_pair_loss(
        inside_weight,
        page_error,
        inside_weight,
        page_error,
        coefficient=0.25,
    )
    outside_outside = _weighted_pair_loss(
        outside_weight,
        page_error,
        outside_weight,
        page_error,
        coefficient=0.25,
    )
    cross = _weighted_pair_loss(
        inside_weight,
        page_error,
        outside_weight,
        page_error,
        coefficient=0.5,
    )
    mean = torch.sum(teacher_mass * page_error, dim=-1)
    total = 0.5 * (
        torch.sum(teacher_mass * page_error.square(), dim=-1) - mean.square()
    )
    return FisherPartition(
        total=total,
        inside_inside=inside_inside,
        outside_outside=outside_outside,
        cross=cross,
    )


def fisher_cross_loss(
    page_error: torch.Tensor,
    teacher_mass: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Return the Fisher pair loss between two disjoint page sets."""

    return _weighted_pair_loss(
        teacher_mass * first,
        page_error,
        teacher_mass * second,
        page_error,
        coefficient=0.5,
    )


def top_page_masks(
    page_logits: torch.Tensor,
    *,
    pages: int,
) -> torch.Tensor:
    """Return a boolean Top-page mask for each leading observation."""

    selected = min(int(pages), int(page_logits.shape[-1]))
    indices = page_logits.topk(selected, dim=-1).indices
    mask = torch.zeros_like(page_logits, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


def budget_recovery(
    teacher_logits: torch.Tensor,
    proxy_logits: torch.Tensor,
    teacher_mass: torch.Tensor,
    *,
    pages: int,
) -> dict[str, torch.Tensor]:
    """Measure mass and teacher-TopM recovery when M grows to 2M."""

    teacher = top_page_masks(teacher_logits, pages=pages)
    selected = top_page_masks(proxy_logits, pages=pages)
    expanded = top_page_masks(proxy_logits, pages=2 * int(pages))
    added = expanded & ~selected
    false_negative = teacher & ~selected
    recovered = false_negative & added
    false_negative_count = false_negative.sum(dim=-1)
    return {
        "selected_mass_m": torch.sum(teacher_mass * selected, dim=-1),
        "selected_mass_2m": torch.sum(teacher_mass * expanded, dim=-1),
        "added_mass": torch.sum(teacher_mass * added, dim=-1),
        "false_negative_count": false_negative_count,
        "false_negative_recovery": recovered.sum(dim=-1)
        / false_negative_count.clamp_min(1),
        "teacher_mask": teacher,
        "selected_mask_m": selected,
        "selected_mask_2m": expanded,
        "added_mask": added,
    }


__all__ = [
    "FisherPartition",
    "PageLSETerms",
    "budget_recovery",
    "fisher_cross_loss",
    "fisher_partition",
    "page_lse_terms",
    "top_page_masks",
]
