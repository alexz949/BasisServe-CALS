"""Utilities for layer-ahead exact-Key page prefetch oracles."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HeadwiseLinearFactors:
    """Independent linear maps for physical GQA head groups."""

    weight: Tensor

    def validate(self, *, heads: int, input_dim: int, output_dim: int) -> None:
        expected = (heads, input_dim, output_dim)
        if tuple(self.weight.shape) != expected:
            raise ValueError(
                f"headwise weight must have shape {expected}, got "
                f"{tuple(self.weight.shape)}"
            )
        if not self.weight.is_floating_point():
            raise TypeError("headwise weight must be floating point")


def fit_headwise_linear(source: Tensor, target: Tensor) -> HeadwiseLinearFactors:
    """Fit a parameter-free minimum-norm least square independently per head."""

    if source.ndim != 4 or target.ndim != 4:
        raise ValueError("source and target must be [batch, heads, sequence, dim]")
    if tuple(source.shape[:3]) != tuple(target.shape[:3]):
        raise ValueError("source and target must share batch/head/sequence geometry")
    if not source.is_floating_point() or not target.is_floating_point():
        raise TypeError("source and target must be floating point")
    heads = int(source.shape[1])
    input_dim = int(source.shape[-1])
    output_dim = int(target.shape[-1])
    design = source.detach().permute(1, 0, 2, 3).reshape(
        heads, -1, input_dim
    ).to(device="cpu", dtype=torch.float64)
    response = target.detach().permute(1, 0, 2, 3).reshape(
        heads, -1, output_dim
    ).to(device="cpu", dtype=torch.float64)
    solutions = [
        torch.linalg.lstsq(design[head], response[head], driver="gelsd").solution
        for head in range(heads)
    ]
    factors = HeadwiseLinearFactors(torch.stack(solutions).float())
    factors.validate(heads=heads, input_dim=input_dim, output_dim=output_dim)
    return factors


def apply_headwise_linear(source: Tensor, factors: HeadwiseLinearFactors) -> Tensor:
    """Apply independent physical-head linear maps."""

    if source.ndim != 4 or not source.is_floating_point():
        raise ValueError("source must be floating [batch, heads, sequence, dim]")
    heads, input_dim = int(source.shape[1]), int(source.shape[-1])
    output_dim = int(factors.weight.shape[-1])
    factors.validate(heads=heads, input_dim=input_dim, output_dim=output_dim)
    return torch.einsum(
        "bhsi,hio->bhso",
        source.float(),
        factors.weight.to(device=source.device, dtype=torch.float32),
    )


def group_query_heads(query: Tensor, *, kv_heads: int) -> Tensor:
    """Flatten each physical GQA group's query heads into one feature axis."""

    if query.ndim != 4:
        raise ValueError("query must be [batch, query heads, sequence, head dim]")
    batch, query_heads, sequence, head_dim = map(int, query.shape)
    if kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must divide evenly across physical KV heads")
    heads_per_group = query_heads // kv_heads
    return query.reshape(
        batch, kv_heads, heads_per_group, sequence, head_dim
    ).permute(0, 1, 3, 2, 4).reshape(
        batch, kv_heads, sequence, heads_per_group * head_dim
    )


def ungroup_query_heads(grouped: Tensor, *, query_heads: int) -> Tensor:
    """Undo :func:`group_query_heads`."""

    if grouped.ndim != 4:
        raise ValueError("grouped query must be [batch, KV heads, sequence, dim]")
    batch, kv_heads, sequence, grouped_dim = map(int, grouped.shape)
    if query_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must divide evenly across physical KV heads")
    heads_per_group = query_heads // kv_heads
    if grouped_dim % heads_per_group:
        raise ValueError("grouped query dimension must divide across query heads")
    head_dim = grouped_dim // heads_per_group
    return grouped.reshape(
        batch, kv_heads, sequence, heads_per_group, head_dim
    ).permute(0, 1, 3, 2, 4).reshape(
        batch, query_heads, sequence, head_dim
    )


def prefetch_set_statistics(
    actual_pages: Tensor,
    prefetched_pages: Tensor,
) -> dict[str, float]:
    """Compare predicted prefetch pages with the actual selector's page set."""

    if (
        actual_pages.dtype != torch.bool
        or prefetched_pages.dtype != torch.bool
        or tuple(actual_pages.shape) != tuple(prefetched_pages.shape)
        or actual_pages.ndim != 3
    ):
        raise ValueError("page masks must be boolean with shared [batch, heads, pages] geometry")
    intersection = actual_pages & prefetched_pages
    late = actual_pages & ~prefetched_pages
    waste = prefetched_pages & ~actual_pages
    actual_count = actual_pages.sum(dim=-1)
    prefetched_count = prefetched_pages.sum(dim=-1)
    intersection_count = intersection.sum(dim=-1)
    late_count = late.sum(dim=-1)
    waste_count = waste.sum(dim=-1)
    if torch.any(actual_count == 0) or torch.any(prefetched_count == 0):
        raise ValueError("each head must select at least one actual and prefetched page")
    return {
        "prefetch_page_recall": float(
            (intersection_count.float() / actual_count).mean()
        ),
        "prefetch_page_precision": float(
            (intersection_count.float() / prefetched_count).mean()
        ),
        "mean_actual_pages_per_kv_head": float(actual_count.float().mean()),
        "mean_prefetched_pages_per_kv_head": float(prefetched_count.float().mean()),
        "mean_late_pages_per_kv_head": float(late_count.float().mean()),
        "mean_wasted_pages_per_kv_head": float(waste_count.float().mean()),
        "fully_covered_kv_head_fraction": float((late_count == 0).float().mean()),
    }
