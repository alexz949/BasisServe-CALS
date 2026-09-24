# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's triton_unified_attention_diffkv.py (vLLM contributors).
"""SM89 decode attention for QK128 GQA with four or eight local heads."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.v1.attention.ops.triton_attention_helpers import softmax_step


@triton.jit
def kernel_diffkv_decode_sm89(
    Q,
    K,
    V,
    O,
    PARTIAL,
    PARTIAL_MAX,
    PARTIAL_SUM,
    CU_Q,
    SEQ_K,
    TABLE,
    scale,
    q_stride0: tl.int64,
    q_stride1: tl.int64,
    o_stride0: tl.int64,
    o_stride1: tl.int64,
    o_stride2: tl.int64,
    k_stride0: tl.int64,
    k_stride1: tl.int64,
    v_stride0: tl.int64,
    v_stride1: tl.int64,
    partial_stride0: tl.int64,
    partial_stride1: tl.int64,
    partial_stride2: tl.int64,
    partial_max_stride0: tl.int64,
    partial_max_stride1: tl.int64,
    table_stride: tl.int64,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
    SEGMENTS: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SPLIT_KV: tl.constexpr,
):
    sequence = tl.program_id(0)
    segment = tl.program_id(1) if SPLIT_KV else 0
    query_start = tl.load(CU_Q + sequence)
    query_end = tl.load(CU_Q + sequence + 1)
    if query_start >= query_end:
        return

    sequence_length = tl.load(SEQ_K + sequence)
    tiles_per_segment = tl.cdiv(sequence_length, SEGMENTS * TILE)
    first_tile = segment * tiles_per_segment
    last_tile = tl.minimum(first_tile + tiles_per_segment, tl.cdiv(sequence_length, TILE))
    if first_tile >= last_tile:
        return

    rows = tl.arange(0, 16)
    heads = rows
    active_heads = rows < NUM_HEADS
    q_dimensions = tl.arange(0, 128)
    v_dimensions = tl.arange(0, BLOCK_V)
    valid_v = v_dimensions < VALUE_DIM
    query = tl.load(
        Q
        + query_start * q_stride0
        + heads[:, None] * q_stride1
        + q_dimensions[None, :],
        mask=active_heads[:, None],
        other=0.0,
    )

    maximum = tl.full((16,), float("-inf"), tl.float32)
    denominator = tl.full((16,), 1.0, tl.float32)
    accumulator = tl.zeros((16, BLOCK_V), tl.float32)

    for tile_index in range(first_tile, last_tile):
        positions = tile_index * TILE + tl.arange(0, TILE)
        valid_positions = positions < sequence_length
        physical_blocks = tl.load(
            TABLE + sequence * table_stride + positions // PAGE,
            mask=valid_positions,
            other=0,
        ).to(tl.int64)
        keys = tl.load(
            K
            + physical_blocks[None, :] * k_stride0
            + (positions % PAGE)[None, :] * k_stride1
            + q_dimensions[:, None],
            mask=valid_positions[None, :],
            other=0.0,
        )
        values = tl.load(
            V
            + physical_blocks[:, None] * v_stride0
            + (positions % PAGE)[:, None] * v_stride1
            + v_dimensions[None, :],
            mask=valid_positions[:, None] & valid_v[None, :],
            other=0.0,
        )
        scores = scale * tl.dot(query, keys)
        scores = tl.where(
            active_heads[:, None] & valid_positions[None, :],
            scores,
            float("-inf"),
        )
        maximum, denominator, probabilities, alpha = softmax_step(
            scores, maximum, denominator
        )
        accumulator = accumulator * alpha[:, None]
        accumulator += tl.dot(probabilities.to(values.dtype), values)

    if SPLIT_KV:
        partial_offsets = (
            query_start.to(tl.int64) * partial_stride0
            + heads[:, None] * partial_stride1
            + segment * partial_stride2
            + v_dimensions[None, :]
        )
        tl.store(
            PARTIAL + partial_offsets,
            accumulator,
            mask=active_heads[:, None] & valid_v[None, :],
        )
        scalar_offsets = (
            query_start.to(tl.int64) * partial_max_stride0
            + heads * partial_max_stride1
            + segment
        )
        tl.store(PARTIAL_MAX + scalar_offsets, maximum, mask=active_heads)
        tl.store(PARTIAL_SUM + scalar_offsets, denominator, mask=active_heads)
    else:
        answer = accumulator / denominator[:, None]
        output_offsets = (
            query_start * o_stride0
            + heads[:, None] * o_stride1
            + v_dimensions[None, :] * o_stride2
        )
        tl.store(
            O + output_offsets,
            answer,
            mask=active_heads[:, None] & valid_v[None, :],
        )


@triton.jit
def kernel_diffkv_decode_reduce_sm89(
    O,
    PARTIAL,
    PARTIAL_MAX,
    PARTIAL_SUM,
    CU_Q,
    SEQ_K,
    o_stride0: tl.int64,
    o_stride1: tl.int64,
    o_stride2: tl.int64,
    partial_stride0: tl.int64,
    partial_stride1: tl.int64,
    partial_stride2: tl.int64,
    partial_max_stride0: tl.int64,
    partial_max_stride1: tl.int64,
    TILE: tl.constexpr,
    SEGMENTS: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    sequence = tl.program_id(0)
    head = tl.program_id(1)
    query_start = tl.load(CU_Q + sequence)
    query_end = tl.load(CU_Q + sequence + 1)
    if query_start >= query_end:
        return

    sequence_length = tl.load(SEQ_K + sequence)
    tiles_per_segment = tl.cdiv(sequence_length, SEGMENTS * TILE)
    active_segments = tl.cdiv(sequence_length, tiles_per_segment * TILE)
    segments = tl.arange(0, SEGMENTS)
    segment_mask = segments < active_segments
    scalar_offsets = (
        query_start.to(tl.int64) * partial_max_stride0
        + head * partial_max_stride1
        + segments
    )
    maxima = tl.load(
        PARTIAL_MAX + scalar_offsets,
        mask=segment_mask,
        other=float("-inf"),
    )
    overall_maximum = tl.max(maxima)
    denominators = tl.load(
        PARTIAL_SUM + scalar_offsets,
        mask=segment_mask,
        other=0.0,
    )
    weights = tl.exp(maxima - overall_maximum)
    overall_denominator = tl.sum(denominators * weights)

    dimensions = tl.arange(0, BLOCK_V)
    valid_dimensions = dimensions < VALUE_DIM
    partial_offsets = (
        query_start.to(tl.int64) * partial_stride0
        + head * partial_stride1
        + segments[:, None] * partial_stride2
        + dimensions[None, :]
    )
    partials = tl.load(
        PARTIAL + partial_offsets,
        mask=segment_mask[:, None] & valid_dimensions[None, :],
        other=0.0,
    )
    numerator = tl.sum(partials * weights[:, None], axis=0)
    answer = tl.where(overall_denominator == 0.0, 0.0, numerator / overall_denominator)
    output_offsets = (
        query_start * o_stride0 + head * o_stride1 + dimensions * o_stride2
    )
    tl.store(O + output_offsets, answer, mask=valid_dimensions)


def _launch_config(
    num_sequences: int,
    max_sequence_length: int,
    split_threshold: int,
    num_heads: int,
    value_dim: int,
) -> tuple[int, int, int, int]:
    assert num_heads in (4, 8) and value_dim in (64, 96)
    if num_sequences > split_threshold:
        if num_heads == 8 and value_dim == 64:
            return 1, 64, 4, 3
        return 1, 32, 4, 3
    if num_sequences == split_threshold:
        return 1, 64, 4, 3
    if num_sequences == 1 and max_sequence_length <= 1024:
        if num_heads == 8 and value_dim == 64:
            return 16, 64, 4, 2
        return 16, 64, 8, 2
    if num_sequences <= 8:
        return 16, 128, 4, 2
    if num_sequences <= 32:
        if value_dim == 96 and num_heads == 4:
            return 8, 64, 4, 3
        if value_dim == 96:
            return 16, 32, 4, 2
        return 16, 64, 4, 3
    if num_sequences <= 64 and num_heads == 8 and value_dim == 64:
        return 8, 32, 4, 2
    return 1, 64, 4, 3


def diffkv_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    softmax_segm_output: torch.Tensor,
    softmax_segm_max: torch.Tensor,
    softmax_segm_expsum: torch.Tensor,
    *,
    split_threshold: int,
    max_sequence_length: int,
    segments: int | None = None,
    tile: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> None:
    """Run fixed-geometry paged decode without the generic metadata branches."""

    value_dim = int(v.shape[-1])
    num_heads = int(q.shape[1])
    block_v = triton.next_power_of_2(value_dim)
    assert q.shape[1:] == (num_heads, 128) and num_heads in (4, 8)
    assert k.shape[2:] == (1, 128) and v.shape[2:] == (1, value_dim)
    assert out.shape[1:] == (num_heads, value_dim)
    assert value_dim in (64, 96)
    assert k.shape[1] == v.shape[1]
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    assert softmax_segm_output.is_contiguous()
    assert softmax_segm_max.is_contiguous() and softmax_segm_expsum.is_contiguous()

    num_sequences = len(seqused_k)
    if segments is None:
        segments, tile, num_warps, num_stages = _launch_config(
            num_sequences, max_sequence_length, split_threshold,
            num_heads, value_dim,
        )
    assert segments in (1, 2, 4, 8, 16)
    assert tile in (16, 32, 64, 128)
    assert num_warps in (4, 8) and num_stages in (2, 3, 4)
    assert segments <= softmax_segm_output.shape[2]
    assert softmax_segm_output.shape[-1] >= block_v
    split_kv = segments > 1

    kernel_diffkv_decode_sm89[(num_sequences, segments)](
        q,
        k,
        v,
        out,
        softmax_segm_output,
        softmax_segm_max,
        softmax_segm_expsum,
        cu_seqlens_q,
        seqused_k,
        block_table,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        softmax_segm_output.stride(0),
        softmax_segm_output.stride(1),
        softmax_segm_output.stride(2),
        softmax_segm_max.stride(0),
        softmax_segm_max.stride(1),
        block_table.stride(0),
        PAGE=k.shape[1],
        TILE=tile,
        SEGMENTS=segments,
        VALUE_DIM=value_dim,
        BLOCK_V=block_v,
        NUM_HEADS=num_heads,
        SPLIT_KV=split_kv,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if split_kv:
        kernel_diffkv_decode_reduce_sm89[(num_sequences, num_heads)](
            out,
            softmax_segm_output,
            softmax_segm_max,
            softmax_segm_expsum,
            cu_seqlens_q,
            seqused_k,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            softmax_segm_output.stride(0),
            softmax_segm_output.stride(1),
            softmax_segm_output.stride(2),
            softmax_segm_max.stride(0),
            softmax_segm_max.stride(1),
            TILE=tile,
            SEGMENTS=segments,
            VALUE_DIM=value_dim,
            BLOCK_V=block_v,
            num_warps=4,
            num_stages=1,
        )


__all__ = ["diffkv_decode"]
