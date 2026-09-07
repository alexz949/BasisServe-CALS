"""Sparse one-token attention over a vLLM Page32 KV cache."""

from __future__ import annotations

import math

import torch
from torch import Tensor

import triton
import triton.language as tl


PAGE_SIZE = 32
QK_DIM = 128
VALUE_DIM = 80
ROUTING_AUX_DIM = 32
CACHE_VALUE_DIM = 128
BASE_RANK = 16
RESIDUAL_RANK = 8


@triton.jit
def _conditional_page_lse_kernel(
    query_ptr,
    cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    base_right_ptr,
    base_bias_ptr,
    residual_query_ptr,
    rope_ptr,
    output_ptr,
    query_stride_batch,
    query_stride_head,
    query_stride_feature,
    cache_stride_block,
    cache_stride_token,
    cache_stride_head,
    cache_stride_feature,
    table_stride_batch,
    table_stride_block,
    right_stride_head,
    right_stride_rank,
    right_stride_feature,
    bias_stride_head,
    bias_stride_feature,
    residual_query_stride_head,
    residual_query_stride_feature,
    residual_query_stride_rank,
    rope_stride_token,
    rope_stride_feature,
    output_stride_batch,
    output_stride_head,
    output_stride_page,
    SCALE: tl.constexpr,
    PAGE: tl.constexpr,
    QK_WIDTH: tl.constexpr,
    VALUE_WIDTH: tl.constexpr,
    BASE_WIDTH: tl.constexpr,
    RESIDUAL_WIDTH: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
):
    page = tl.program_id(0)
    row = tl.program_id(1)
    batch_index = row // QUERY_HEADS
    query_head = row % QUERY_HEADS
    kv_head = query_head // HEADS_PER_KV
    sequence_length = tl.load(seq_lens_ptr + batch_index)
    page_valid = page * PAGE < sequence_length
    physical_block = tl.load(
        block_table_ptr + batch_index * table_stride_batch + page * table_stride_block,
        mask=page_valid,
        other=0,
    )

    token_offsets = tl.arange(0, PAGE)
    logical_tokens = page * PAGE + token_offsets
    token_valid = page_valid & (logical_tokens < sequence_length)
    base_offsets = tl.arange(0, BASE_WIDTH)
    half_offsets = tl.arange(0, QK_WIDTH // 2)
    residual_offsets = tl.arange(0, 16)
    base_code = tl.load(
        cache_ptr
        + physical_block * cache_stride_block
        + token_offsets[:, None] * cache_stride_token
        + kv_head * cache_stride_head
        + (QK_WIDTH + VALUE_WIDTH + base_offsets[None, :]) * cache_stride_feature,
        mask=token_valid[:, None],
        other=0.0,
    )
    right_first = tl.load(
        base_right_ptr
        + kv_head * right_stride_head
        + base_offsets[:, None] * right_stride_rank
        + half_offsets[None, :] * right_stride_feature,
    )
    right_second = tl.load(
        base_right_ptr
        + kv_head * right_stride_head
        + base_offsets[:, None] * right_stride_rank
        + (half_offsets[None, :] + QK_WIDTH // 2) * right_stride_feature,
    )
    predicted_first = tl.dot(base_code, right_first)
    predicted_second = tl.dot(base_code, right_second)
    predicted_first += tl.load(
        base_bias_ptr
        + kv_head * bias_stride_head
        + half_offsets[None, :] * bias_stride_feature,
    )
    predicted_second += tl.load(
        base_bias_ptr
        + kv_head * bias_stride_head
        + (half_offsets[None, :] + QK_WIDTH // 2) * bias_stride_feature,
    )
    predicted_first = predicted_first.to(tl.bfloat16).to(tl.float32)
    predicted_second = predicted_second.to(tl.bfloat16).to(tl.float32)

    cosine = tl.load(
        rope_ptr
        + logical_tokens[:, None] * rope_stride_token
        + half_offsets[None, :] * rope_stride_feature,
        mask=token_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    sine = tl.load(
        rope_ptr
        + logical_tokens[:, None] * rope_stride_token
        + (half_offsets[None, :] + QK_WIDTH // 2) * rope_stride_feature,
        mask=token_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    post_first = predicted_first * cosine - predicted_second * sine
    post_second = predicted_second * cosine + predicted_first * sine
    query_first = tl.load(
        query_ptr
        + batch_index * query_stride_batch
        + query_head * query_stride_head
        + half_offsets * query_stride_feature,
    ).to(tl.float32)
    query_second = tl.load(
        query_ptr
        + batch_index * query_stride_batch
        + query_head * query_stride_head
        + (half_offsets + QK_WIDTH // 2) * query_stride_feature,
    ).to(tl.float32)
    base_score = tl.sum(
        post_first * query_first[None, :] + post_second * query_second[None, :],
        axis=1,
    )

    residual_code = tl.load(
        cache_ptr
        + physical_block * cache_stride_block
        + token_offsets[:, None] * cache_stride_token
        + kv_head * cache_stride_head
        + (QK_WIDTH + VALUE_WIDTH + BASE_WIDTH + residual_offsets[None, :])
        * cache_stride_feature,
        mask=token_valid[:, None] & (residual_offsets[None, :] < RESIDUAL_WIDTH),
        other=0.0,
    ).to(tl.float32)
    residual_query = tl.load(
        residual_query_ptr
        + query_head * residual_query_stride_head
        + half_offsets[:, None] * residual_query_stride_feature
        + residual_offsets[None, :] * residual_query_stride_rank,
        mask=residual_offsets[None, :] < RESIDUAL_WIDTH,
        other=0.0,
    ).to(tl.float32)
    residual_query_second = tl.load(
        residual_query_ptr
        + query_head * residual_query_stride_head
        + (half_offsets[:, None] + QK_WIDTH // 2) * residual_query_stride_feature
        + residual_offsets[None, :] * residual_query_stride_rank,
        mask=residual_offsets[None, :] < RESIDUAL_WIDTH,
        other=0.0,
    ).to(tl.float32)
    query_code = tl.sum(
        query_first[:, None] * residual_query
        + query_second[:, None] * residual_query_second,
        axis=0,
    )
    residual_score = tl.sum(residual_code * query_code[None, :], axis=1)
    scores = (
        (
            (base_score.to(tl.bfloat16) + residual_score.to(tl.bfloat16))
            .to(tl.bfloat16)
            .to(tl.float32)
            * SCALE
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    scores = tl.where(token_valid, scores, -float("inf"))
    maximum = tl.max(scores, axis=0)
    maximum = tl.where(page_valid, maximum, 0.0)
    normalizer = tl.sum(
        tl.where(token_valid, tl.exp(scores - maximum), 0.0),
        axis=0,
    )
    log_mass = tl.where(
        page_valid,
        maximum + tl.log(normalizer),
        -float("inf"),
    )
    tl.store(
        output_ptr
        + batch_index * output_stride_batch
        + query_head * output_stride_head
        + page * output_stride_page,
        log_mass,
    )


@triton.jit
def _loki_scores_kernel(
    query_code_ptr,
    cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    output_ptr,
    query_stride_batch,
    query_stride_head,
    query_stride_rank,
    cache_stride_block,
    cache_stride_token,
    cache_stride_head,
    cache_stride_feature,
    table_stride_batch,
    table_stride_block,
    output_stride_batch,
    output_stride_head,
    output_stride_token,
    MAX_SEQUENCE: tl.constexpr,
    SCALE: tl.constexpr,
    PAGE: tl.constexpr,
    QK_WIDTH: tl.constexpr,
    VALUE_WIDTH: tl.constexpr,
    ROUTING_WIDTH: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    BLOCK_TOKEN: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_TOKEN + tl.arange(0, BLOCK_TOKEN)
    row = tl.program_id(1)
    batch_index = row // QUERY_HEADS
    query_head = row % QUERY_HEADS
    kv_head = query_head // HEADS_PER_KV
    sequence_length = tl.load(seq_lens_ptr + batch_index)
    valid = (token_offsets < MAX_SEQUENCE) & (token_offsets < sequence_length)
    logical_blocks = token_offsets // PAGE
    block_offsets = token_offsets % PAGE
    physical_blocks = tl.load(
        block_table_ptr
        + batch_index * table_stride_batch
        + logical_blocks * table_stride_block,
        mask=valid,
        other=0,
    )
    rank_offsets = tl.arange(0, ROUTING_WIDTH)
    query_code = tl.load(
        query_code_ptr
        + batch_index * query_stride_batch
        + query_head * query_stride_head
        + rank_offsets * query_stride_rank,
    ).to(tl.float32)
    key_code = tl.load(
        cache_ptr
        + physical_blocks[:, None] * cache_stride_block
        + block_offsets[:, None] * cache_stride_token
        + kv_head * cache_stride_head
        + (QK_WIDTH + VALUE_WIDTH + rank_offsets[None, :]) * cache_stride_feature,
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(key_code * query_code[None, :], axis=1) * SCALE
    tl.store(
        output_ptr
        + batch_index * output_stride_batch
        + query_head * output_stride_head
        + token_offsets * output_stride_token,
        tl.where(valid, scores, -float("inf")),
        mask=token_offsets < MAX_SEQUENCE,
    )


@triton.jit
def _paged_sparse_attention_kernel(
    query_ptr,
    cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    selected_ptr,
    output_ptr,
    query_stride_batch,
    query_stride_head,
    query_stride_feature,
    cache_stride_block,
    cache_stride_token,
    cache_stride_head,
    cache_stride_feature,
    table_stride_batch,
    table_stride_block,
    selected_stride_batch,
    selected_stride_head,
    selected_stride_token,
    output_stride_batch,
    output_stride_head,
    output_stride_feature,
    SCALE: tl.constexpr,
    PAGE: tl.constexpr,
    QK_WIDTH: tl.constexpr,
    VALUE_WIDTH: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    SELECTED_COUNT: tl.constexpr,
    BLOCK_SELECTED: tl.constexpr,
):
    row = tl.program_id(0)
    batch_index = row // QUERY_HEADS
    query_head = row % QUERY_HEADS
    kv_head = query_head // HEADS_PER_KV
    sequence_length = tl.load(seq_lens_ptr + batch_index)
    qk_offsets = tl.arange(0, QK_WIDTH)
    value_offsets = tl.arange(0, 128)
    query = tl.load(
        query_ptr
        + batch_index * query_stride_batch
        + query_head * query_stride_head
        + qk_offsets * query_stride_feature,
    ).to(tl.float32)
    running_maximum = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((128,), dtype=tl.float32)

    for selected_start in range(0, SELECTED_COUNT, BLOCK_SELECTED):
        selected_offsets = selected_start + tl.arange(0, BLOCK_SELECTED)
        token_ids = tl.load(
            selected_ptr
            + batch_index * selected_stride_batch
            + query_head * selected_stride_head
            + selected_offsets * selected_stride_token,
            mask=selected_offsets < SELECTED_COUNT,
            other=-1,
        )
        valid = (
            (selected_offsets < SELECTED_COUNT)
            & (token_ids >= 0)
            & (token_ids < sequence_length)
        )
        logical_blocks = token_ids // PAGE
        block_offsets = token_ids % PAGE
        physical_blocks = tl.load(
            block_table_ptr
            + batch_index * table_stride_batch
            + logical_blocks * table_stride_block,
            mask=valid,
            other=0,
        )
        keys = tl.load(
            cache_ptr
            + physical_blocks[:, None] * cache_stride_block
            + block_offsets[:, None] * cache_stride_token
            + kv_head * cache_stride_head
            + qk_offsets[None, :] * cache_stride_feature,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        block_maximum = tl.max(scores, axis=0)
        has_valid = tl.sum(valid.to(tl.int32), axis=0) > 0
        next_maximum = tl.where(
            has_valid,
            tl.maximum(running_maximum, block_maximum),
            running_maximum,
        )
        previous_scale = tl.where(
            has_valid,
            tl.exp(running_maximum - next_maximum),
            1.0,
        )
        probabilities = tl.where(valid, tl.exp(scores - next_maximum), 0.0)
        values = tl.load(
            cache_ptr
            + physical_blocks[:, None] * cache_stride_block
            + block_offsets[:, None] * cache_stride_token
            + kv_head * cache_stride_head
            + (QK_WIDTH + value_offsets[None, :]) * cache_stride_feature,
            mask=valid[:, None] & (value_offsets[None, :] < VALUE_WIDTH),
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * values,
            axis=0,
        )
        running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=0)
        running_maximum = next_maximum

    result = tl.where(running_sum > 0.0, accumulator / running_sum, 0.0)
    tl.store(
        output_ptr
        + batch_index * output_stride_batch
        + query_head * output_stride_head
        + value_offsets * output_stride_feature,
        result,
        mask=value_offsets < VALUE_WIDTH,
    )


def conditional_page32_log_mass(
    query: Tensor,
    kv_cache: Tensor,
    block_table: Tensor,
    seq_lens: Tensor,
    base_right: Tensor,
    base_bias: Tensor,
    residual_query: Tensor,
    rope_cache: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Return Base16+R8 Page32 log mass in physical GQA layout."""

    batch, query_heads, head_dim = map(int, query.shape)
    kv_heads = int(base_right.shape[0])
    pages = math.ceil(int(block_table.shape[1]) * PAGE_SIZE / PAGE_SIZE)
    assert head_dim == QK_DIM and query_heads == 4 * kv_heads
    assert int(kv_cache.shape[1]) == PAGE_SIZE
    assert int(kv_cache.shape[-1]) == QK_DIM + CACHE_VALUE_DIM
    assert tuple(base_right.shape) == (kv_heads, BASE_RANK, QK_DIM)
    assert tuple(base_bias.shape) == (kv_heads, QK_DIM)
    assert tuple(residual_query.shape) == (
        query_heads,
        QK_DIM,
        RESIDUAL_RANK,
    )
    assert query.dtype == kv_cache.dtype == torch.bfloat16
    assert all(
        tensor.is_cuda
        for tensor in (
            query,
            kv_cache,
            block_table,
            seq_lens,
            base_right,
            base_bias,
            residual_query,
            rope_cache,
        )
    )
    output = torch.empty(
        batch,
        query_heads,
        pages,
        dtype=torch.float32,
        device=query.device,
    )
    _conditional_page_lse_kernel[(pages, batch * query_heads)](
        query,
        kv_cache,
        block_table,
        seq_lens,
        base_right,
        base_bias,
        residual_query,
        rope_cache,
        output,
        *query.stride(),
        *kv_cache.stride(),
        *block_table.stride(),
        *base_right.stride(),
        *base_bias.stride(),
        *residual_query.stride(),
        *rope_cache.stride(),
        *output.stride(),
        SCALE=float(scale),
        PAGE=PAGE_SIZE,
        QK_WIDTH=QK_DIM,
        VALUE_WIDTH=VALUE_DIM,
        BASE_WIDTH=BASE_RANK,
        RESIDUAL_WIDTH=RESIDUAL_RANK,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        HEADS_PER_KV=query_heads // kv_heads,
        num_warps=4,
        num_stages=3,
    )
    return output.view(batch, kv_heads, query_heads // kv_heads, pages)


def select_group_shared_pages(
    page_log_mass: Tensor,
    seq_lens: Tensor,
    *,
    pages_per_kv_head: int,
    pinned_prefix_pages: int,
) -> Tensor:
    """Match the Page-Fisher evaluator's normalized group-max selection."""

    batch, kv_heads, _, pages = map(int, page_log_mass.shape)
    assert pages_per_kv_head <= pages
    indices = torch.arange(pages, device=page_log_mass.device)
    valid = indices[None, :] < torch.div(
        seq_lens[:, None] + PAGE_SIZE - 1,
        PAGE_SIZE,
        rounding_mode="floor",
    )
    routed_valid = valid[:, None, None] & (
        indices[None, None, None] >= pinned_prefix_pages
    )
    routed = page_log_mass.masked_fill(~routed_valid, -torch.inf)
    normalized = torch.softmax(routed, dim=-1).masked_fill(~routed_valid, 0.0)
    group_scores = normalized.amax(dim=2)
    group_valid = valid[:, None].expand(batch, kv_heads, pages)
    group_scores.masked_fill_(
        ~group_valid | (indices[None, None] < pinned_prefix_pages),
        -torch.inf,
    )
    pinned = min(pinned_prefix_pages, pages_per_kv_head)
    selected = []
    if pinned:
        selected.append(
            torch.arange(pinned, device=page_log_mass.device)
            .view(1, 1, pinned)
            .expand(batch, kv_heads, pinned)
        )
    routed_count = pages_per_kv_head - pinned
    if routed_count:
        selected.append(
            torch.topk(
                group_scores,
                routed_count,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
        )
    return torch.cat(selected, dim=-1).contiguous()


def loki_scores(
    query: Tensor,
    query_projector: Tensor,
    kv_cache: Tensor,
    block_table: Tensor,
    seq_lens: Tensor,
    *,
    scale: float,
    maximum_sequence: int,
) -> Tensor:
    """Return per-Query-head Loki token scores from the paged R32 sidecar."""

    batch, query_heads, head_dim = map(int, query.shape)
    kv_heads = int(kv_cache.shape[2])
    assert head_dim == QK_DIM and query_heads == 4 * kv_heads
    assert tuple(query_projector.shape) == (
        query_heads,
        QK_DIM,
        ROUTING_AUX_DIM,
    )
    query_code = torch.einsum(
        "bhd,hdr->bhr",
        query,
        query_projector.to(device=query.device, dtype=query.dtype),
    ).contiguous()
    output = torch.empty(
        batch,
        query_heads,
        maximum_sequence,
        dtype=torch.float32,
        device=query.device,
    )
    block_token = 64
    _loki_scores_kernel[
        (triton.cdiv(maximum_sequence, block_token), batch * query_heads)
    ](
        query_code,
        kv_cache,
        block_table,
        seq_lens,
        output,
        *query_code.stride(),
        *kv_cache.stride(),
        *block_table.stride(),
        *output.stride(),
        MAX_SEQUENCE=maximum_sequence,
        SCALE=float(scale),
        PAGE=PAGE_SIZE,
        QK_WIDTH=QK_DIM,
        VALUE_WIDTH=VALUE_DIM,
        ROUTING_WIDTH=ROUTING_AUX_DIM,
        QUERY_HEADS=query_heads,
        HEADS_PER_KV=query_heads // kv_heads,
        BLOCK_TOKEN=block_token,
        num_warps=4,
        num_stages=2,
    )
    return output


def selected_page_tokens(selected_pages: Tensor, *, query_heads: int) -> Tensor:
    """Expand one physical page list per KV head to Query-head token IDs."""

    batch, kv_heads, pages = map(int, selected_pages.shape)
    assert query_heads % kv_heads == 0
    offsets = torch.arange(PAGE_SIZE, device=selected_pages.device)
    tokens = (
        selected_pages[..., None] * PAGE_SIZE + offsets.view(1, 1, 1, PAGE_SIZE)
    ).reshape(batch, kv_heads, pages * PAGE_SIZE)
    return tokens.repeat_interleave(query_heads // kv_heads, dim=1).contiguous()


def paged_sparse_attention(
    query: Tensor,
    kv_cache: Tensor,
    block_table: Tensor,
    seq_lens: Tensor,
    selected_token_ids: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Evaluate exact QK and V80 only on selected logical token IDs."""

    batch, query_heads, head_dim = map(int, query.shape)
    selected_count = int(selected_token_ids.shape[-1])
    kv_heads = int(kv_cache.shape[2])
    assert head_dim == QK_DIM and query_heads == 4 * kv_heads
    assert tuple(selected_token_ids.shape[:2]) == (batch, query_heads)
    assert selected_count > 0
    output = torch.empty(
        batch,
        query_heads,
        VALUE_DIM,
        dtype=query.dtype,
        device=query.device,
    )
    _paged_sparse_attention_kernel[(batch * query_heads,)](
        query,
        kv_cache,
        block_table,
        seq_lens,
        selected_token_ids,
        output,
        *query.stride(),
        *kv_cache.stride(),
        *block_table.stride(),
        *selected_token_ids.stride(),
        *output.stride(),
        SCALE=float(scale),
        PAGE=PAGE_SIZE,
        QK_WIDTH=QK_DIM,
        VALUE_WIDTH=VALUE_DIM,
        QUERY_HEADS=query_heads,
        HEADS_PER_KV=query_heads // kv_heads,
        SELECTED_COUNT=selected_count,
        BLOCK_SELECTED=32,
        num_warps=4,
        num_stages=2,
    )
    return output


def physical_union_count(
    selected_token_ids: Tensor,
    *,
    kv_heads: int,
    maximum_sequence: int,
) -> Tensor:
    """Count distinct physical token reads after the Loki GQA union."""

    batch, query_heads, _ = map(int, selected_token_ids.shape)
    heads_per_kv = query_heads // kv_heads
    grouped = selected_token_ids.view(batch, kv_heads, heads_per_kv, -1)
    selected = torch.zeros(
        batch,
        kv_heads,
        maximum_sequence,
        dtype=torch.bool,
        device=selected_token_ids.device,
    )
    for head in range(heads_per_kv):
        selected.scatter_(2, grouped[:, :, head], True)
    return selected.sum(dtype=torch.int64)


__all__ = [
    "BASE_RANK",
    "CACHE_VALUE_DIM",
    "PAGE_SIZE",
    "QK_DIM",
    "RESIDUAL_RANK",
    "ROUTING_AUX_DIM",
    "VALUE_DIM",
    "conditional_page32_log_mass",
    "loki_scores",
    "paged_sparse_attention",
    "physical_union_count",
    "select_group_shared_pages",
    "selected_page_tokens",
]
