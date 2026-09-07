"""GQA-aware sparse routing and attention kernels for one-token decode.

The quality evaluators keep exact Keys and compressed Values resident on the
GPU.  Their former PyTorch path expanded every KV head into four Query heads,
then materialized gathered K/V tensors.  These Triton kernels preserve the
physical GQA layout: proxy scoring maps Query heads to their KV head inside the
kernel, and exact sparse attention streams selected token IDs directly.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

import triton
import triton.language as tl


_QUERY_TILE = 16
_RANK_TILE = 32


@triton.jit
def _gqa_proxy_scores_kernel(
    query_code_ptr,
    sidecar_ptr,
    output_ptr,
    tokens,
    query_stride_batch,
    query_stride_head,
    query_stride_rank,
    sidecar_stride_batch,
    sidecar_stride_head,
    sidecar_stride_token,
    sidecar_stride_rank,
    output_stride_batch,
    output_stride_head,
    output_stride_token,
    SCALE: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    ROUTING_RANK: tl.constexpr,
    BLOCK_QUERY: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
    BLOCK_TOKEN: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_TOKEN + tl.arange(0, BLOCK_TOKEN)
    kv_row = tl.program_id(1)
    batch_index = kv_row // KV_HEADS
    kv_head = kv_row % KV_HEADS
    query_in_group = tl.arange(0, BLOCK_QUERY)
    query_heads = kv_head * HEADS_PER_KV + query_in_group
    accumulator = tl.zeros((BLOCK_QUERY, BLOCK_TOKEN), dtype=tl.float32)

    for rank_start in range(0, ROUTING_RANK, BLOCK_RANK):
        rank_offsets = rank_start + tl.arange(0, BLOCK_RANK)
        query_code = tl.load(
            query_code_ptr
            + batch_index * query_stride_batch
            + query_heads[:, None] * query_stride_head
            + rank_offsets[None, :] * query_stride_rank,
            mask=(query_in_group[:, None] < HEADS_PER_KV)
            & (rank_offsets[None, :] < ROUTING_RANK),
            other=0.0,
        )
        sidecar = tl.load(
            sidecar_ptr
            + batch_index * sidecar_stride_batch
            + kv_head * sidecar_stride_head
            + rank_offsets[:, None] * sidecar_stride_rank
            + token_offsets[None, :] * sidecar_stride_token,
            mask=(rank_offsets[:, None] < ROUTING_RANK)
            & (token_offsets[None, :] < tokens),
            other=0.0,
        )
        accumulator = tl.dot(query_code, sidecar, accumulator)

    scores = accumulator * SCALE
    if OUTPUT_BF16:
        scores = scores.to(tl.bfloat16)
    else:
        scores = scores.to(tl.float16)
    tl.store(
        output_ptr
        + batch_index * output_stride_batch
        + query_heads[:, None] * output_stride_head
        + token_offsets[None, :] * output_stride_token,
        scores,
        mask=(query_in_group[:, None] < HEADS_PER_KV)
        & (token_offsets[None, :] < tokens),
    )


@triton.jit
def _gqa_page32_log_mass_kernel(
    query_code_ptr,
    sidecar_ptr,
    output_ptr,
    tokens,
    query_stride_batch,
    query_stride_head,
    query_stride_rank,
    sidecar_stride_batch,
    sidecar_stride_head,
    sidecar_stride_token,
    sidecar_stride_rank,
    output_stride_batch,
    output_stride_head,
    output_stride_query,
    output_stride_page,
    SCALE: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    ROUTING_RANK: tl.constexpr,
    BLOCK_QUERY: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    page = tl.program_id(0)
    kv_row = tl.program_id(1)
    batch_index = kv_row // KV_HEADS
    kv_head = kv_row % KV_HEADS
    query_in_group = tl.arange(0, BLOCK_QUERY)
    query_heads = kv_head * HEADS_PER_KV + query_in_group
    token_offsets = page * 32 + tl.arange(0, 32)
    token_valid = token_offsets < tokens
    accumulator = tl.zeros((BLOCK_QUERY, 32), dtype=tl.float32)

    for rank_start in range(0, ROUTING_RANK, BLOCK_RANK):
        rank_offsets = rank_start + tl.arange(0, BLOCK_RANK)
        query_code = tl.load(
            query_code_ptr
            + batch_index * query_stride_batch
            + query_heads[:, None] * query_stride_head
            + rank_offsets[None, :] * query_stride_rank,
            mask=(query_in_group[:, None] < HEADS_PER_KV)
            & (rank_offsets[None, :] < ROUTING_RANK),
            other=0.0,
        )
        sidecar = tl.load(
            sidecar_ptr
            + batch_index * sidecar_stride_batch
            + kv_head * sidecar_stride_head
            + rank_offsets[:, None] * sidecar_stride_rank
            + token_offsets[None, :] * sidecar_stride_token,
            mask=(rank_offsets[:, None] < ROUTING_RANK)
            & token_valid[None, :],
            other=0.0,
        )
        accumulator = tl.dot(query_code, sidecar, accumulator)

    scores = accumulator * SCALE
    if OUTPUT_BF16:
        scores = scores.to(tl.bfloat16).to(tl.float32)
    else:
        scores = scores.to(tl.float16).to(tl.float32)
    scores = tl.where(token_valid[None, :], scores, -float("inf"))
    maximum = tl.max(scores, axis=1)
    log_mass = maximum + tl.log(tl.sum(tl.exp(scores - maximum[:, None]), axis=1))
    tl.store(
        output_ptr
        + batch_index * output_stride_batch
        + kv_head * output_stride_head
        + query_in_group * output_stride_query
        + page * output_stride_page,
        log_mass,
        mask=query_in_group < HEADS_PER_KV,
    )


@triton.jit
def _gqa_indexed_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    selected_ptr,
    output_ptr,
    sequence_length,
    query_stride_batch,
    query_stride_head,
    query_stride_token,
    query_stride_feature,
    key_stride_batch,
    key_stride_head,
    key_stride_token,
    key_stride_feature,
    value_stride_batch,
    value_stride_head,
    value_stride_token,
    value_stride_feature,
    selected_stride_batch,
    selected_stride_head,
    selected_stride_token,
    output_stride_batch,
    output_stride_head,
    output_stride_token,
    output_stride_feature,
    SCALE: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    SELECTED_COUNT: tl.constexpr,
    BLOCK_QK: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
    BLOCK_SELECTED: tl.constexpr,
):
    row = tl.program_id(0)
    batch_index = row // QUERY_HEADS
    query_head = row % QUERY_HEADS
    kv_head = query_head // HEADS_PER_KV
    qk_offsets = tl.arange(0, BLOCK_QK)
    value_offsets = tl.arange(0, BLOCK_VALUE)
    query = tl.load(
        query_ptr
        + batch_index * query_stride_batch
        + query_head * query_stride_head
        + qk_offsets * query_stride_feature,
        mask=qk_offsets < QK_DIM,
        other=0.0,
    ).to(tl.float32)
    running_maximum = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_VALUE,), dtype=tl.float32)

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
        keys = tl.load(
            key_ptr
            + batch_index * key_stride_batch
            + kv_head * key_stride_head
            + token_ids[:, None] * key_stride_token
            + qk_offsets[None, :] * key_stride_feature,
            mask=valid[:, None] & (qk_offsets[None, :] < QK_DIM),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        block_maximum = tl.max(scores, axis=0)
        has_valid = tl.sum(valid.to(tl.int32), axis=0) > 0
        next_maximum = tl.maximum(running_maximum, block_maximum)
        next_maximum = tl.where(has_valid, next_maximum, running_maximum)
        previous_scale = tl.where(
            has_valid,
            tl.exp(running_maximum - next_maximum),
            1.0,
        )
        probabilities = tl.where(
            valid,
            tl.exp(scores - next_maximum),
            0.0,
        )
        values = tl.load(
            value_ptr
            + batch_index * value_stride_batch
            + kv_head * value_stride_head
            + token_ids[:, None] * value_stride_token
            + value_offsets[None, :] * value_stride_feature,
            mask=valid[:, None] & (value_offsets[None, :] < VALUE_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * values,
            axis=0,
        )
        running_sum = running_sum * previous_scale + tl.sum(
            probabilities,
            axis=0,
        )
        running_maximum = next_maximum

    result = tl.where(running_sum > 0.0, accumulator / running_sum, 0.0)
    tl.store(
        output_ptr
        + batch_index * output_stride_batch
        + query_head * output_stride_head
        + value_offsets * output_stride_feature,
        result,
        mask=value_offsets < VALUE_DIM,
    )


def _proxy_geometry(
    query_code: Tensor,
    routing_sidecar: Tensor,
) -> tuple[int, int, int, int, int]:
    assert query_code.ndim == 3 and routing_sidecar.ndim == 4
    batch, query_heads, routing_rank = map(int, query_code.shape)
    sidecar_batch, kv_heads, tokens, sidecar_rank = map(
        int, routing_sidecar.shape
    )
    assert batch == sidecar_batch and routing_rank == sidecar_rank
    assert kv_heads > 0 and query_heads % kv_heads == 0 and tokens > 0
    assert query_code.is_cuda and routing_sidecar.is_cuda
    assert query_code.device == routing_sidecar.device
    assert query_code.dtype == routing_sidecar.dtype
    assert query_code.dtype in (torch.float16, torch.bfloat16)
    assert query_code.stride(-1) == routing_sidecar.stride(-1) == 1
    assert 0 < routing_rank <= 256
    heads_per_kv = query_heads // kv_heads
    assert 0 < heads_per_kv <= _QUERY_TILE
    return batch, query_heads, kv_heads, tokens, routing_rank


def gqa_proxy_scores_triton(
    query_code: Tensor,
    routing_sidecar: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Return one-token proxy scores without physically expanding GQA Keys."""

    batch, query_heads, kv_heads, tokens, routing_rank = _proxy_geometry(
        query_code,
        routing_sidecar,
    )
    selected_scale = float(scale)
    assert math.isfinite(selected_scale) and selected_scale > 0.0
    output = torch.empty(
        batch,
        query_heads,
        tokens,
        dtype=query_code.dtype,
        device=query_code.device,
    )
    block_token = 64
    with torch.cuda.device(query_code.device):
        _gqa_proxy_scores_kernel[
            (triton.cdiv(tokens, block_token), batch * kv_heads)
        ](
            query_code,
            routing_sidecar,
            output,
            tokens,
            *query_code.stride(),
            *routing_sidecar.stride(),
            *output.stride(),
            SCALE=selected_scale,
            KV_HEADS=kv_heads,
            HEADS_PER_KV=query_heads // kv_heads,
            ROUTING_RANK=routing_rank,
            BLOCK_QUERY=_QUERY_TILE,
            BLOCK_RANK=_RANK_TILE,
            BLOCK_TOKEN=block_token,
            OUTPUT_BF16=query_code.dtype == torch.bfloat16,
            num_warps=4,
            num_stages=3,
        )
    return output.unsqueeze(2)


def gqa_page32_log_mass_triton(
    query_code: Tensor,
    routing_sidecar: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Fuse proxy scoring and Page32 log-sum-exp for one decode Query."""

    batch, query_heads, kv_heads, tokens, routing_rank = _proxy_geometry(
        query_code,
        routing_sidecar,
    )
    selected_scale = float(scale)
    assert math.isfinite(selected_scale) and selected_scale > 0.0
    pages = math.ceil(tokens / 32)
    heads_per_kv = query_heads // kv_heads
    output = torch.empty(
        batch,
        kv_heads,
        heads_per_kv,
        pages,
        dtype=torch.float32,
        device=query_code.device,
    )
    with torch.cuda.device(query_code.device):
        _gqa_page32_log_mass_kernel[(pages, batch * kv_heads)](
            query_code,
            routing_sidecar,
            output,
            tokens,
            *query_code.stride(),
            *routing_sidecar.stride(),
            *output.stride(),
            SCALE=selected_scale,
            KV_HEADS=kv_heads,
            HEADS_PER_KV=heads_per_kv,
            ROUTING_RANK=routing_rank,
            BLOCK_QUERY=_QUERY_TILE,
            BLOCK_RANK=_RANK_TILE,
            OUTPUT_BF16=query_code.dtype == torch.bfloat16,
            num_warps=4,
            num_stages=3,
        )
    return output


def gqa_indexed_sparse_decode_attention_triton(
    query: Tensor,
    exact_key: Tensor,
    value: Tensor,
    selected_token_ids: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Attend directly to per-Query selected token IDs without K/V gathers."""

    assert query.ndim == exact_key.ndim == value.ndim == 4
    batch, query_heads, query_tokens, qk_dim = map(int, query.shape)
    key_batch, kv_heads, sequence_length, key_dim = map(int, exact_key.shape)
    value_batch, value_heads, value_length, value_dim = map(int, value.shape)
    if selected_token_ids.ndim == 4:
        assert int(selected_token_ids.shape[2]) == 1
        selected_token_ids = selected_token_ids[:, :, 0]
    assert selected_token_ids.ndim == 3
    selected_count = int(selected_token_ids.shape[-1])
    assert query_tokens == 1 and qk_dim == key_dim
    assert batch == key_batch == value_batch == int(selected_token_ids.shape[0])
    assert kv_heads == value_heads and sequence_length == value_length
    assert query_heads % kv_heads == 0
    assert tuple(selected_token_ids.shape[:2]) == (batch, query_heads)
    assert sequence_length > 0 and selected_count > 0
    tensors = (query, exact_key, value)
    assert all(tensor.is_cuda for tensor in tensors)
    assert selected_token_ids.is_cuda and selected_token_ids.dtype == torch.int64
    assert all(tensor.device == query.device for tensor in tensors)
    assert selected_token_ids.device == query.device
    assert all(tensor.dtype == query.dtype for tensor in tensors)
    assert query.dtype in (torch.float16, torch.bfloat16)
    assert all(tensor.stride(-1) == 1 for tensor in tensors)
    selected_token_ids = selected_token_ids.contiguous()
    selected_scale = float(scale)
    assert math.isfinite(selected_scale) and selected_scale > 0.0
    output = torch.empty(
        batch,
        query_heads,
        1,
        value_dim,
        dtype=query.dtype,
        device=query.device,
    )
    block_qk = triton.next_power_of_2(qk_dim)
    block_value = triton.next_power_of_2(value_dim)
    block_selected = 32 if value_dim > 64 else 64
    with torch.cuda.device(query.device):
        _gqa_indexed_attention_kernel[(batch * query_heads,)](
            query,
            exact_key,
            value,
            selected_token_ids,
            output,
            sequence_length,
            *query.stride(),
            *exact_key.stride(),
            *value.stride(),
            *selected_token_ids.stride(),
            *output.stride(),
            SCALE=selected_scale,
            QUERY_HEADS=query_heads,
            HEADS_PER_KV=query_heads // kv_heads,
            QK_DIM=qk_dim,
            VALUE_DIM=value_dim,
            SELECTED_COUNT=selected_count,
            BLOCK_QK=block_qk,
            BLOCK_VALUE=block_value,
            BLOCK_SELECTED=block_selected,
            num_warps=4,
            num_stages=2,
        )
    return output


__all__ = [
    "gqa_indexed_sparse_decode_attention_triton",
    "gqa_page32_log_mass_triton",
    "gqa_proxy_scores_triton",
]
