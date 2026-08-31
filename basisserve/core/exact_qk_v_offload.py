"""Exact-QK selection primitives for Value-offload quality ceilings."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class SparseValueAttention:
    """Sparse GQA output together with retained dense-attention mass."""

    output: Tensor
    selected_teacher_mass: Tensor


@dataclass(frozen=True)
class AdaptivePageMassSelection:
    """GQA-unioned pages and confidence statistics for adaptive routing."""

    token_mask: Tensor
    page_mask: Tensor
    eligible_query_heads: int
    refined_query_heads: int
    tail_mass_ratio_sum: float


def _validate_gqa_scores(scores: Tensor, *, num_kv_heads: int) -> int:
    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("scores must be floating [query heads, visible tokens]")
    query_heads, visible = map(int, scores.shape)
    if num_kv_heads <= 0 or query_heads % num_kv_heads:
        raise ValueError("query heads must divide evenly across positive KV heads")
    if visible <= 0:
        raise ValueError("at least one visible token is required")
    return query_heads // num_kv_heads


def gqa_union_token_topk_mask(
    scores: Tensor,
    *,
    num_kv_heads: int,
    top_k: int,
) -> Tensor:
    """Union each GQA group's exact per-query-head token Top-k selections."""

    heads_per_group = _validate_gqa_scores(scores, num_kv_heads=num_kv_heads)
    if top_k <= 0:
        raise ValueError("top-k must be positive")
    visible = int(scores.shape[1])
    selected = min(top_k, visible)
    indices = scores.reshape(
        num_kv_heads, heads_per_group, visible
    ).topk(selected, dim=-1).indices
    mask = torch.zeros(
        num_kv_heads,
        visible,
        dtype=torch.bool,
        device=scores.device,
    )
    mask.scatter_(1, indices.reshape(num_kv_heads, -1), True)
    return mask


def gqa_union_page_mass_mask(
    scores: Tensor,
    *,
    num_kv_heads: int,
    page_size: int,
    pages_per_query_head: int,
) -> tuple[Tensor, Tensor]:
    """Union exact per-query-head Top-page selections ranked by log-sum-exp."""

    heads_per_group = _validate_gqa_scores(scores, num_kv_heads=num_kv_heads)
    if page_size <= 0 or pages_per_query_head <= 0:
        raise ValueError("page size and page budget must be positive")
    query_heads, visible = map(int, scores.shape)
    pages = math.ceil(visible / page_size)
    padded_tokens = pages * page_size
    padded = F.pad(scores, (0, padded_tokens - visible), value=-torch.inf)
    page_log_mass = torch.logsumexp(
        padded.reshape(query_heads, pages, page_size), dim=-1
    )
    selected_pages = min(pages_per_query_head, pages)
    indices = page_log_mass.reshape(
        num_kv_heads, heads_per_group, pages
    ).topk(selected_pages, dim=-1).indices
    page_mask = torch.zeros(
        num_kv_heads,
        pages,
        dtype=torch.bool,
        device=scores.device,
    )
    page_mask.scatter_(1, indices.reshape(num_kv_heads, -1), True)
    token_mask = page_mask.repeat_interleave(page_size, dim=-1)[:, :visible]
    return token_mask, page_mask


def gqa_union_adaptive_page_mass_mask(
    scores: Tensor,
    *,
    num_kv_heads: int,
    page_size: int,
    base_pages_per_query_head: int,
    max_pages_per_query_head: int,
    tail_mass_ratio_threshold: float,
) -> AdaptivePageMassSelection:
    """Expand uncertain Query heads from a base to a maximum page budget.

    Confidence is measured from proxy page masses.  Let ``S_base`` be the
    summed mass of the base Top-pages and ``S_tail`` the summed mass of the
    additional pages up to the maximum budget.  A Query head is refined when
    ``S_tail / S_base >= tail_mass_ratio_threshold``.  Selected pages are then
    unioned across the Query heads belonging to each physical GQA Key head.
    """

    heads_per_group = _validate_gqa_scores(scores, num_kv_heads=num_kv_heads)
    if page_size <= 0:
        raise ValueError("page size must be positive")
    if base_pages_per_query_head <= 0:
        raise ValueError("base page budget must be positive")
    if max_pages_per_query_head < base_pages_per_query_head:
        raise ValueError("maximum page budget must not be smaller than base")
    if not 0.0 < tail_mass_ratio_threshold <= 1.0:
        raise ValueError("tail mass ratio threshold must lie in (0, 1]")

    query_heads, visible = map(int, scores.shape)
    pages = math.ceil(visible / page_size)
    padded_tokens = pages * page_size
    padded = F.pad(scores, (0, padded_tokens - visible), value=-torch.inf)
    page_log_mass = torch.logsumexp(
        padded.reshape(query_heads, pages, page_size), dim=-1
    )
    maximum = min(max_pages_per_query_head, pages)
    top_values, top_indices = torch.topk(
        page_log_mass,
        k=maximum,
        dim=-1,
        largest=True,
        sorted=True,
    )
    finite_counts = torch.isfinite(top_values).sum(dim=-1)
    if torch.any(finite_counts == 0):
        raise ValueError("every Query head must have at least one valid page")
    base_counts = finite_counts.clamp(max=min(base_pages_per_query_head, maximum))
    maximum_counts = finite_counts.clamp(max=maximum)
    positions = torch.arange(maximum, device=scores.device).view(1, -1)
    base_mask = positions < base_counts[:, None]
    tail_mask = (positions >= base_counts[:, None]) & (
        positions < maximum_counts[:, None]
    )
    base_log_mass = torch.logsumexp(
        top_values.masked_fill(~base_mask, -torch.inf), dim=-1
    )
    tail_log_mass = torch.logsumexp(
        top_values.masked_fill(~tail_mask, -torch.inf), dim=-1
    )
    eligible = maximum_counts > base_counts
    tail_mass_ratio = torch.where(
        eligible,
        torch.exp(tail_log_mass - base_log_mass),
        torch.zeros_like(base_log_mass),
    )
    refined = eligible & (tail_mass_ratio >= tail_mass_ratio_threshold)
    selected_counts = torch.where(refined, maximum_counts, base_counts)
    selected_query_pages = positions < selected_counts[:, None]
    query_page_mask = torch.zeros(
        query_heads,
        pages,
        dtype=torch.bool,
        device=scores.device,
    )
    query_page_mask.scatter_(1, top_indices, selected_query_pages)
    page_mask = query_page_mask.reshape(
        num_kv_heads, heads_per_group, pages
    ).any(dim=1)
    token_mask = page_mask.repeat_interleave(page_size, dim=-1)[:, :visible]
    return AdaptivePageMassSelection(
        token_mask=token_mask,
        page_mask=page_mask,
        eligible_query_heads=int(eligible.sum().item()),
        refined_query_heads=int(refined.sum().item()),
        tail_mass_ratio_sum=float(tail_mass_ratio[eligible].sum().item()),
    )


def full_gqa_value_attention(
    scores: Tensor,
    value: Tensor,
    *,
    heads_per_group: int,
) -> Tensor:
    """Compute full exact-score GQA attention over a Value payload."""

    if value.ndim != 3 or not value.is_floating_point():
        raise ValueError("Value must be floating [KV heads, visible tokens, dim]")
    query_heads, visible = map(int, scores.shape)
    kv_heads = int(value.shape[0])
    if (
        heads_per_group <= 0
        or query_heads != kv_heads * heads_per_group
        or int(value.shape[1]) != visible
    ):
        raise ValueError("scores and Value have incompatible GQA geometry")
    kv_index = torch.arange(query_heads, device=scores.device) // heads_per_group
    expanded_value = value.index_select(0, kv_index)
    probability = torch.softmax(scores.float(), dim=-1)
    return torch.einsum("hl,hld->hd", probability, expanded_value.float())


def sparse_gqa_value_attention(
    scores: Tensor,
    value: Tensor,
    union_token_mask: Tensor,
    *,
    heads_per_group: int,
) -> SparseValueAttention:
    """Attend over a GQA-unioned selected Value set with exact-score softmax."""

    query_heads, visible = map(int, scores.shape)
    kv_heads = int(value.shape[0])
    if tuple(union_token_mask.shape) != (kv_heads, visible):
        raise ValueError("union token mask must be [KV heads, visible tokens]")
    if not union_token_mask.dtype == torch.bool:
        raise TypeError("union token mask must be boolean")
    if not bool(union_token_mask.any(dim=-1).all()):
        raise ValueError("every KV head must select at least one token")
    if (
        value.ndim != 3
        or int(value.shape[1]) != visible
        or query_heads != kv_heads * heads_per_group
    ):
        raise ValueError("scores, mask, and Value have incompatible GQA geometry")
    kv_index = torch.arange(query_heads, device=scores.device) // heads_per_group
    query_mask = union_token_mask.index_select(0, kv_index)
    sparse_probability = torch.softmax(
        scores.float().masked_fill(~query_mask, -torch.inf), dim=-1
    )
    teacher_probability = torch.softmax(scores.float(), dim=-1)
    expanded_value = value.index_select(0, kv_index).float()
    return SparseValueAttention(
        output=torch.einsum("hl,hld->hd", sparse_probability, expanded_value),
        selected_teacher_mass=(teacher_probability * query_mask).sum(dim=-1),
    )
