"""QUEST page sparsity over completed pairwise/independent latent K history."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


_LANDMARK_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class PairwiseQuestConfig:
    """Fixed page geometry and historical-token budget for latent QUEST."""

    page_size: int
    historical_token_budget: int
    landmark_dtype: str = "bfloat16"

    def validate(self, latent_dim: int) -> None:
        if self.page_size <= 0:
            raise ValueError("QUEST page size must be positive")
        if self.historical_token_budget <= 0:
            raise ValueError("QUEST historical-token budget must be positive")
        if self.landmark_dtype not in _LANDMARK_DTYPES:
            raise ValueError(f"unsupported QUEST landmark dtype {self.landmark_dtype!r}")
        if latent_dim <= 0:
            raise ValueError("QUEST latent dimension must be positive")

    @property
    def page_budget(self) -> int:
        return math.ceil(self.historical_token_budget / self.page_size)

    @property
    def torch_landmark_dtype(self) -> torch.dtype:
        return _LANDMARK_DTYPES[self.landmark_dtype]


@dataclass(frozen=True)
class PairwiseQuestSelection:
    """One shared physical-KV-page selection for every GQA query group."""

    page_scores: Tensor
    selected_page_indices: Tensor
    selected_token_indices: Tensor
    selected_token_valid: Tensor
    page_mins: Tensor
    page_maxes: Tensor


@dataclass(frozen=True)
class PairwiseQuestAttentionResult:
    """Packed sparse attention output and logical work accounting."""

    output: Tensor
    selection: PairwiseQuestSelection
    statistics: dict[str, float]


def _page_bounds(
    history_code: Tensor,
    config: PairwiseQuestConfig,
) -> tuple[Tensor, Tensor]:
    batch, kv_heads, history_length, latent_dim = map(int, history_code.shape)
    config.validate(latent_dim)
    page_count = math.ceil(history_length / config.page_size)
    padded_length = page_count * config.page_size
    if padded_length != history_length:
        padding = history_code.new_zeros(
            batch,
            kv_heads,
            padded_length - history_length,
            latent_dim,
        )
        padded = torch.cat((history_code, padding), dim=2)
    else:
        padded = history_code
    pages = padded.float().reshape(
        batch,
        kv_heads,
        page_count,
        config.page_size,
        latent_dim,
    )
    if padded_length == history_length:
        page_mins = pages.amin(dim=3)
        page_maxes = pages.amax(dim=3)
    else:
        positions = torch.arange(padded_length, device=history_code.device)
        valid = (positions < history_length).reshape(
            1,
            1,
            page_count,
            config.page_size,
            1,
        )
        page_mins = pages.masked_fill(~valid, torch.inf).amin(dim=3)
        page_maxes = pages.masked_fill(~valid, -torch.inf).amax(dim=3)
    return (
        page_mins.to(config.torch_landmark_dtype),
        page_maxes.to(config.torch_landmark_dtype),
    )


def pairwise_quest_select_pages(
    projected_query: Tensor,
    history_code: Tensor,
    config: PairwiseQuestConfig,
    *,
    scaling: float,
) -> PairwiseQuestSelection:
    """Select one page set per physical KV group and query position.

    The four Qwen3 GQA Query heads first obtain independent QUEST bounds.  A
    maximum over those heads produces one traffic-controlled shared page set.
    """

    if projected_query.ndim != 4 or history_code.ndim != 4:
        raise ValueError("latent Query and historical K code must be rank four")
    batch, query_heads, query_length, latent_dim = map(int, projected_query.shape)
    history_batch, kv_heads, history_length, history_dim = map(int, history_code.shape)
    if batch != history_batch or latent_dim != history_dim:
        raise ValueError("latent Query and historical K geometry differs")
    if history_length <= 0:
        raise ValueError("QUEST selection requires nonempty historical K")
    if query_heads % kv_heads:
        raise ValueError("Query heads must be divisible by physical KV heads")
    config.validate(latent_dim)
    page_mins, page_maxes = _page_bounds(history_code, config)
    page_count = int(page_mins.shape[2])
    heads_per_group = query_heads // kv_heads
    grouped_query = projected_query.reshape(
        batch,
        kv_heads,
        heads_per_group,
        query_length,
        latent_dim,
    ).float()
    positive = grouped_query.clamp_min(0.0)
    negative = grouped_query.clamp_max(0.0)
    query_head_scores = torch.einsum(
        "bghqr,bgpr->bghqp",
        positive,
        page_maxes.float(),
    )
    query_head_scores.add_(
        torch.einsum(
            "bghqr,bgpr->bghqp",
            negative,
            page_mins.float(),
        )
    )
    query_head_scores.mul_(float(scaling))
    page_scores = query_head_scores.amax(dim=2)
    selected_pages = min(config.page_budget, page_count)
    page_indices = torch.topk(
        page_scores,
        k=selected_pages,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    page_indices = torch.sort(page_indices, dim=-1).values
    offsets = torch.arange(config.page_size, device=history_code.device)
    token_indices = (
        page_indices[..., None] * config.page_size + offsets
    ).flatten(start_dim=-2)
    token_valid = token_indices < history_length
    token_indices = token_indices.clamp_max(history_length - 1)
    return PairwiseQuestSelection(
        page_scores=page_scores,
        selected_page_indices=page_indices,
        selected_token_indices=token_indices,
        selected_token_valid=token_valid,
        page_mins=page_mins,
        page_maxes=page_maxes,
    )


def _gather_query_tokens(states: Tensor, token_indices: Tensor) -> Tensor:
    batch, kv_heads, tokens, width = map(int, states.shape)
    expected_prefix = (batch, kv_heads)
    if token_indices.ndim != 4 or tuple(token_indices.shape[:2]) != expected_prefix:
        raise ValueError("query-specific token indices differ from KV geometry")
    query_length, selected_tokens = map(int, token_indices.shape[-2:])
    expanded = states[:, :, None].expand(
        batch,
        kv_heads,
        query_length,
        tokens,
        width,
    )
    gather_indices = token_indices[..., None].expand(
        batch,
        kv_heads,
        query_length,
        selected_tokens,
        width,
    )
    return torch.gather(expanded, dim=3, index=gather_indices)


def _attention_bias(
    attention_mask: Tensor | None,
    *,
    batch: int,
    query_heads: int,
    query_length: int,
    total_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    if attention_mask is None:
        return None
    bias = attention_mask.to(device=device)
    if bias.ndim == 2:
        if tuple(bias.shape) != (batch, total_length):
            raise ValueError("rank-two attention mask has incompatible geometry")
        if bias.dtype == torch.bool:
            zero = torch.zeros((), dtype=dtype, device=device)
            negative = torch.full((), -torch.inf, dtype=dtype, device=device)
            bias = torch.where(bias, zero, negative)
        bias = bias[:, None, None].expand(batch, query_heads, query_length, total_length)
    elif bias.ndim == 4:
        if int(bias.shape[0]) not in (1, batch):
            raise ValueError("attention-mask batch dimension is incompatible")
        if int(bias.shape[1]) not in (1, query_heads):
            raise ValueError("attention-mask head dimension is incompatible")
        if int(bias.shape[2]) not in (1, query_length):
            raise ValueError("attention-mask Query dimension is incompatible")
        if int(bias.shape[3]) < total_length:
            raise ValueError("attention mask is shorter than the K/V cache")
        bias = bias[..., :total_length].expand(
            batch,
            query_heads,
            query_length,
            total_length,
        )
    else:
        raise ValueError("attention mask must be rank two or rank four")
    return bias.to(dtype=dtype)


def pairwise_quest_sparse_attention(
    query: Tensor,
    projected_query: Tensor,
    history_code: Tensor,
    current_key: Tensor,
    value_cache: Tensor,
    config: PairwiseQuestConfig,
    *,
    scaling: float,
    attention_mask: Tensor | None = None,
) -> PairwiseQuestAttentionResult:
    """Attend to QUEST-selected latent history plus the complete exact block."""

    if any(tensor.ndim != 4 for tensor in (
        query,
        projected_query,
        history_code,
        current_key,
        value_cache,
    )):
        raise ValueError("pairwise sparse Q/K/V tensors must be rank four")
    batch, query_heads, query_length, head_dim = map(int, query.shape)
    if tuple(projected_query.shape[:3]) != (batch, query_heads, query_length):
        raise ValueError("projected Query geometry differs from exact Query")
    kv_heads = int(history_code.shape[1])
    history_length = int(history_code.shape[2])
    latent_dim = int(history_code.shape[3])
    if query_heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    if tuple(current_key.shape[:3]) != (batch, kv_heads, query_length):
        raise ValueError("current exact K geometry differs from Query")
    if int(current_key.shape[3]) != head_dim:
        raise ValueError("current exact K dimension differs from Query")
    total_length = history_length + query_length
    if tuple(value_cache.shape[:3]) != (batch, kv_heads, total_length):
        raise ValueError("C1 Value cache length differs from historical/current K")
    if int(projected_query.shape[3]) != latent_dim:
        raise ValueError("projected Query rank differs from historical K rank")

    heads_per_group = query_heads // kv_heads
    selection = pairwise_quest_select_pages(
        projected_query,
        history_code,
        config,
        scaling=scaling,
    )
    selected_history = _gather_query_tokens(
        history_code,
        selection.selected_token_indices,
    ).repeat_interleave(heads_per_group, dim=1)
    selected_values = _gather_query_tokens(
        value_cache[:, :, :history_length],
        selection.selected_token_indices,
    ).repeat_interleave(heads_per_group, dim=1)
    expanded_token_indices = selection.selected_token_indices.repeat_interleave(
        heads_per_group,
        dim=1,
    )
    expanded_token_valid = selection.selected_token_valid.repeat_interleave(
        heads_per_group,
        dim=1,
    )
    historical_scores = torch.einsum(
        "bhqr,bhqkr->bhqk",
        projected_query,
        selected_history,
    ).mul_(float(scaling))
    historical_scores.masked_fill_(~expanded_token_valid, -torch.inf)

    repeated_current_key = current_key.repeat_interleave(heads_per_group, dim=1)
    current_scores = torch.matmul(
        query,
        repeated_current_key.transpose(2, 3),
    ).mul_(float(scaling))
    bias = _attention_bias(
        attention_mask,
        batch=batch,
        query_heads=query_heads,
        query_length=query_length,
        total_length=total_length,
        dtype=historical_scores.dtype,
        device=query.device,
    )
    if bias is not None:
        historical_scores.add_(torch.gather(bias, dim=-1, index=expanded_token_indices))
        current_scores.add_(bias[..., history_length:total_length])
    if query_length > 1:
        causal = torch.ones(
            query_length,
            query_length,
            dtype=torch.bool,
            device=query.device,
        ).tril()
        current_scores.masked_fill_(~causal.view(1, 1, query_length, query_length), -torch.inf)

    combined_scores = torch.cat((historical_scores, current_scores), dim=-1)
    probabilities = F.softmax(combined_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    historical_width = int(historical_scores.shape[-1])
    historical_output = torch.einsum(
        "bhqk,bhqkv->bhqv",
        probabilities[..., :historical_width],
        selected_values,
    )
    repeated_current_value = value_cache[:, :, history_length:].repeat_interleave(
        heads_per_group,
        dim=1,
    )
    current_output = torch.matmul(
        probabilities[..., historical_width:],
        repeated_current_value,
    )
    output = historical_output + current_output

    selected_physical_tokens = float(selection.selected_token_valid.sum())
    available_physical_tokens = float(
        batch * kv_heads * query_length * history_length
    )
    selected_page_instances = int(selection.selected_page_indices.numel())
    page_count = int(selection.page_scores.shape[-1])
    landmark_bytes = (
        selection.page_mins.numel() * selection.page_mins.element_size()
        + selection.page_maxes.numel() * selection.page_maxes.element_size()
    )
    selected_query_head_tokens = selected_physical_tokens * heads_per_group
    statistics = {
        "query_positions": float(batch * query_length),
        "query_heads": float(batch * query_heads * query_length),
        "history_length": float(history_length),
        "page_count": float(page_count),
        "selected_page_instances": float(selected_page_instances),
        "available_page_instances": float(batch * kv_heads * query_length * page_count),
        "selected_physical_tokens": selected_physical_tokens,
        "available_physical_tokens": available_physical_tokens,
        "selected_token_fraction": (
            selected_physical_tokens / available_physical_tokens
            if available_physical_tokens
            else 0.0
        ),
        "resident_selector_metadata_bytes": float(landmark_bytes),
        "selection_bound_flops": float(
            4 * batch * query_heads * query_length * page_count * latent_dim
        ),
        "sparse_historical_qk_flops": float(
            2 * selected_query_head_tokens * latent_dim
        ),
        "exact_current_qk_flops": float(
            2 * batch * query_heads * query_length * query_length * head_dim
        ),
        "sparse_historical_pv_flops": float(
            2 * selected_query_head_tokens * int(value_cache.shape[-1])
        ),
        "exact_current_pv_flops": float(
            2
            * batch
            * query_heads
            * query_length
            * query_length
            * int(value_cache.shape[-1])
        ),
    }
    return PairwiseQuestAttentionResult(
        output=output,
        selection=selection,
        statistics=statistics,
    )


__all__ = [
    "PairwiseQuestAttentionResult",
    "PairwiseQuestConfig",
    "PairwiseQuestSelection",
    "pairwise_quest_select_pages",
    "pairwise_quest_sparse_attention",
]
