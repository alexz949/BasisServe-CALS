"""Fused GQA for dense keys and compressed Value caches.

PyTorch 2.6 Flash-SDPA requires equal Q/K/V head dimensions.  C1 deliberately
keeps the Q/K dimension dense while reducing the Value dimension, so the
generic SDPA call falls back to its materializing math implementation.  This
kernel implements the serving operation directly: it streams over the cache,
maintains an online softmax in float32, and accumulates the compact Value
coordinates without expanding grouped-query K/V heads or storing scores.

Decode exposes two deliberately separate implementations: a purpose-built
one-query Triton kernel and an architecture-specific handwritten CUDA kernel
family. Neither implementation falls back to the other. Causal prefill uses a
separately tiled Triton kernel that supports arbitrary compact Value widths.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch
from torch import Tensor
from torch.nn import functional as F

import triton
import triton.language as tl


_CUDA_DECODE_CONTEXT_LIMITS = (128, 512, 1024, 4096, 8192)
_CUDA_DECODE_TUNING: dict[
    tuple[int, int],
    dict[int, tuple[int, ...]],
] = {
    (8, 9): {
        1: (16, 16, 32, 32, 32),
        2: (4, 16, 32, 32, 32),
        4: (4, 16, 16, 16, 16),
        8: (8, 8, 8, 32, 32),
        16: (4, 4, 16, 32, 16),
        32: (1, 8, 8, 16, 16),
        64: (1, 4, 8, 8, 8),
        128: (1, 1, 4, 4, 16),
    },
}


def select_compressed_v_decode_cuda_splits(
    *,
    capability: tuple[int, int],
    batch: int,
    sequence_length: int,
) -> int:
    """Return the offline-tuned split count for an explicitly CUDA request."""

    if batch <= 0 or sequence_length <= 0:
        raise ValueError("decode batch and sequence length must be positive")
    if sequence_length < _CUDA_DECODE_CONTEXT_LIMITS[0]:
        return 1
    architecture_table = _CUDA_DECODE_TUNING.get(tuple(map(int, capability)))
    if architecture_table is None:
        major, minor = capability
        raise RuntimeError(f"no pure-CUDA tuning table is installed for SM{major}{minor}")
    batch_table = architecture_table.get(int(batch))
    if batch_table is None:
        raise ValueError(f"no pure-CUDA tuning is installed for batch {batch}")
    for context_index, context_limit in enumerate(_CUDA_DECODE_CONTEXT_LIMITS):
        if sequence_length <= context_limit:
            return batch_table[context_index]
    raise ValueError(
        f"pure-CUDA decode is tuned through context {_CUDA_DECODE_CONTEXT_LIMITS[-1]}"
    )


@dataclass(frozen=True)
class CompressedVDecodeWorkspace:
    """One flat split-K arena shared by sequential transformer layers."""

    batch: int
    query_heads: int
    max_splits: int
    max_value_dim: int
    workspace_storage: Tensor
    output_storage: Tensor

    @classmethod
    def allocate(
        cls,
        *,
        batch: int,
        query_heads: int,
        max_value_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        max_splits: int = 32,
    ) -> CompressedVDecodeWorkspace:
        rows = int(batch) * int(query_heads)
        if rows <= 0 or max_value_dim <= 0 or max_splits <= 0:
            raise ValueError("compressed-V workspace dimensions must be positive")
        return cls(
            batch=int(batch),
            query_heads=int(query_heads),
            max_splits=int(max_splits),
            max_value_dim=int(max_value_dim),
            workspace_storage=torch.empty(
                rows * int(max_splits) * (int(max_value_dim) + 2),
                dtype=torch.float32,
                device=device,
            ),
            output_storage=torch.empty(
                rows * int(max_value_dim),
                dtype=dtype,
                device=device,
            ),
        )

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.workspace_storage, self.output_storage)
        )

    def views(
        self,
        *,
        batch: int,
        query_heads: int,
        splits: int,
        value_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        if int(batch) != self.batch or int(query_heads) != self.query_heads:
            raise ValueError("decode inputs differ from the configured workspace")
        if not 0 < splits <= self.max_splits or not 0 < value_dim <= self.max_value_dim:
            raise ValueError("decode request exceeds the configured workspace")
        if self.output_storage.dtype != dtype or self.output_storage.device != device:
            raise ValueError("decode workspace dtype or device differs from its inputs")
        rows = self.batch * self.query_heads
        workspace_elements = rows * int(splits) * (int(value_dim) + 2)
        output_elements = rows * int(value_dim)
        workspace = self.workspace_storage[:workspace_elements].view(
            rows,
            int(splits),
            int(value_dim) + 2,
        )
        output = self.output_storage[:output_elements].view(
            self.batch,
            self.query_heads,
            1,
            int(value_dim),
        )
        return workspace, output


@lru_cache(maxsize=None)
def _device_capability(device_index: int) -> tuple[int, int]:
    return tuple(map(int, torch.cuda.get_device_capability(device_index)))


@lru_cache(maxsize=None)
def _cuda_architecture(device_index: int) -> int:
    capability = _device_capability(device_index)
    architectures = {(8, 0): 80, (8, 9): 89, (9, 0): 90}
    if capability not in architectures:
        major, minor = capability
        raise RuntimeError(
            "compressed-V CUDA split-K supports SM80, SM89, and SM90; "
            f"got SM{major}{minor}"
        )
    return architectures[capability]


@triton.jit
def _compressed_v_gqa_decode_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
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
    output_stride_batch,
    output_stride_head,
    output_stride_token,
    output_stride_feature,
    scale: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_QK: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
    BLOCK_SEQUENCE: tl.constexpr,
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
        + 0 * query_stride_token
        + qk_offsets * query_stride_feature,
        mask=qk_offsets < QK_DIM,
        other=0.0,
    ).to(tl.float32)

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_VALUE,), dtype=tl.float32)

    for sequence_start in range(0, sequence_length, BLOCK_SEQUENCE):
        sequence_offsets = sequence_start + tl.arange(0, BLOCK_SEQUENCE)
        sequence_mask = sequence_offsets < sequence_length
        keys = tl.load(
            key_ptr
            + batch_index * key_stride_batch
            + kv_head * key_stride_head
            + sequence_offsets[:, None] * key_stride_token
            + qk_offsets[None, :] * key_stride_feature,
            mask=sequence_mask[:, None] & (qk_offsets[None, :] < QK_DIM),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(sequence_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, block_max)
        previous_scale = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max)

        values = tl.load(
            value_ptr
            + batch_index * value_stride_batch
            + kv_head * value_stride_head
            + sequence_offsets[:, None] * value_stride_token
            + value_offsets[None, :] * value_stride_feature,
            mask=sequence_mask[:, None] & (value_offsets[None, :] < VALUE_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * values,
            axis=0,
        )
        running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=0)
        running_max = next_max

    output = accumulator / running_sum
    tl.store(
        output_ptr
        + batch_index * output_stride_batch
        + query_head * output_stride_head
        + 0 * output_stride_token
        + value_offsets * output_stride_feature,
        output,
        mask=value_offsets < VALUE_DIM,
    )


@triton.jit
def _compressed_v_gqa_prefill_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
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
    output_stride_batch,
    output_stride_head,
    output_stride_token,
    output_stride_feature,
    scale: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_QK: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_index = batch_head // QUERY_HEADS
    query_head = batch_head % QUERY_HEADS
    kv_head = query_head // HEADS_PER_KV

    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    qk_offsets = tl.arange(0, BLOCK_QK)
    value_offsets = tl.arange(0, BLOCK_VALUE)
    query_mask = query_offsets < sequence_length
    query = tl.load(
        query_ptr
        + batch_index * query_stride_batch
        + query_head * query_stride_head
        + query_offsets[:, None] * query_stride_token
        + qk_offsets[None, :] * query_stride_feature,
        mask=query_mask[:, None] & (qk_offsets[None, :] < QK_DIM),
        other=0.0,
    )

    running_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_VALUE), dtype=tl.float32)
    key_stop = tl.minimum(sequence_length, (query_block + 1) * BLOCK_M)

    for key_start in range(0, key_stop, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < sequence_length
        keys = tl.load(
            key_ptr
            + batch_index * key_stride_batch
            + kv_head * key_stride_head
            + key_offsets[:, None] * key_stride_token
            + qk_offsets[None, :] * key_stride_feature,
            mask=key_mask[:, None] & (qk_offsets[None, :] < QK_DIM),
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(keys)) * scale
        causal_mask = (
            query_mask[:, None]
            & key_mask[None, :]
            & (query_offsets[:, None] >= key_offsets[None, :])
        )
        scores = tl.where(causal_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        next_max = tl.where(query_mask, next_max, 0.0)
        previous_scale = tl.where(
            query_mask,
            tl.exp(running_max - next_max),
            0.0,
        )
        probabilities = tl.where(
            causal_mask,
            tl.exp(scores - next_max[:, None]),
            0.0,
        )

        values = tl.load(
            value_ptr
            + batch_index * value_stride_batch
            + kv_head * value_stride_head
            + key_offsets[:, None] * value_stride_token
            + value_offsets[None, :] * value_stride_feature,
            mask=key_mask[:, None] & (value_offsets[None, :] < VALUE_DIM),
            other=0.0,
        )
        accumulator = accumulator * previous_scale[:, None] + tl.dot(
            probabilities.to(values.dtype),
            values,
        )
        running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=1)
        running_max = next_max

    output = accumulator / running_sum[:, None]
    tl.store(
        output_ptr
        + batch_index * output_stride_batch
        + query_head * output_stride_head
        + query_offsets[:, None] * output_stride_token
        + value_offsets[None, :] * output_stride_feature,
        output,
        mask=query_mask[:, None] & (value_offsets[None, :] < VALUE_DIM),
    )


def _validate_inputs(query: Tensor, key: Tensor, value: Tensor) -> tuple[int, ...]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("compressed-V attention Q/K/V must be rank-4 tensors")
    batch, query_heads, query_tokens, qk_dim = map(int, query.shape)
    key_batch, kv_heads, sequence_length, key_dim = map(int, key.shape)
    value_batch, value_heads, value_sequence, value_dim = map(int, value.shape)
    if batch != key_batch or batch != value_batch:
        raise ValueError("compressed-V attention Q/K/V batch dimensions differ")
    if kv_heads != value_heads or sequence_length != value_sequence:
        raise ValueError("compressed-V attention K/V head or sequence dimensions differ")
    if qk_dim != key_dim:
        raise ValueError("compressed-V attention Q/K head dimensions differ")
    if kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if sequence_length <= 0 or not 0 < value_dim <= 128:
        raise ValueError("cache sequence and compressed Value width must be supported")
    if qk_dim <= 0 or qk_dim > 256:
        raise ValueError("Q/K head dimension must lie in [1, 256]")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError("compressed-V decode Q/K/V dtypes differ")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("compressed-V attention supports FP16 and BF16")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("compressed-V attention requires CUDA tensors")
    if query.device != key.device or query.device != value.device:
        raise ValueError("compressed-V attention Q/K/V devices differ")
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("compressed-V attention feature dimensions must be contiguous")
    return (
        batch,
        query_heads,
        kv_heads,
        query_tokens,
        sequence_length,
        qk_dim,
        value_dim,
    )


def compressed_v_decode_attention_triton(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Run online-softmax GQA with a one-token query and compact Values.

    Shapes are ``query=[B,Hq,1,D]``, ``key=[B,Hkv,S,D]``, and
    ``value=[B,Hkv,S,R]``.  The result is ``[B,Hq,1,R]``.
    """

    (
        batch,
        query_heads,
        kv_heads,
        query_tokens,
        sequence_length,
        qk_dim,
        value_dim,
    ) = _validate_inputs(query, key, value)
    if query_tokens != 1:
        raise ValueError("compressed-V Triton decode accepts one query token")
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(selected_scale) or selected_scale <= 0.0:
        raise ValueError("attention scale must be finite and positive")

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
    block_sequence = 64 if value_dim <= 64 else 32
    _compressed_v_gqa_decode_kernel[(batch * query_heads,)](
        query,
        key,
        value,
        output,
        sequence_length,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        scale=selected_scale,
        QUERY_HEADS=query_heads,
        HEADS_PER_KV=query_heads // kv_heads,
        QK_DIM=qk_dim,
        VALUE_DIM=value_dim,
        BLOCK_QK=block_qk,
        BLOCK_VALUE=block_value,
        BLOCK_SEQUENCE=block_sequence,
        num_warps=4,
        num_stages=2,
    )
    return output


def compressed_v_decode_attention_cuda(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
    splits: int = 1,
    workspace: Tensor | None = None,
    output: Tensor | None = None,
    feature_major_output: Tensor | None = None,
) -> Tensor:
    """Run the architecture-specialized exact-rank split-K CUDA decode.

    ``workspace`` and the selected output can be allocated once and reused so
    timed serving calls contain only the partial and reduction kernels. Passing
    ``feature_major_output=[query_heads * rank, batch]`` makes attention write
    directly into a packed collective slot.
    """

    (
        batch,
        query_heads,
        kv_heads,
        query_tokens,
        sequence_length,
        qk_dim,
        value_dim,
    ) = _validate_inputs(query, key, value)
    if query_tokens != 1:
        raise ValueError("compressed-V CUDA decode accepts one query token")
    if qk_dim != 128 or query_heads != 4 * kv_heads:
        raise ValueError("compressed-V CUDA requires QK dim 128 and GQA ratio 4")
    if value_dim not in (32, 48, 64, 80, 96, 112):
        raise ValueError(
            "compressed-V CUDA requires rank in {32,48,64,80,96,112}"
        )
    if splits not in (1, 2, 4, 8, 16, 32):
        raise ValueError("splits must be in {1,2,4,8,16,32}")
    if sequence_length < splits:
        raise ValueError("each CUDA split must contain at least one KV token")
    device_index = query.device.index
    architecture = _cuda_architecture(
        torch.cuda.current_device() if device_index is None else device_index
    )
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(selected_scale) or selected_scale <= 0.0:
        raise ValueError("attention scale must be finite and positive")

    expected_workspace = (batch * query_heads, splits, value_dim + 2)
    if workspace is None:
        workspace = torch.empty(
            expected_workspace,
            dtype=torch.float32,
            device=query.device,
        )
    elif (
        tuple(workspace.shape) != expected_workspace
        or workspace.dtype != torch.float32
        or workspace.device != query.device
        or not workspace.is_contiguous()
    ):
        raise ValueError("invalid CUDA split-K workspace")
    if output is not None and feature_major_output is not None:
        raise ValueError("select either token-major or feature-major CUDA output")
    uses_feature_major_output = feature_major_output is not None
    if uses_feature_major_output:
        output = feature_major_output
        expected_output = (query_heads * value_dim, batch)
    else:
        expected_output = (batch, query_heads, 1, value_dim)
        if output is None:
            output = torch.empty(
                expected_output,
                dtype=query.dtype,
                device=query.device,
            )
    assert output is not None
    if (
        tuple(output.shape) != expected_output
        or output.dtype != query.dtype
        or output.device != query.device
        or not output.is_contiguous()
    ):
        layout = "feature-major" if uses_feature_major_output else "token-major"
        raise ValueError(f"invalid {layout} CUDA split-K output")

    from basisserve.kernels.compressed_v_decode_cuda import (
        launch_compressed_v_decode_cuda,
    )

    return launch_compressed_v_decode_cuda(
        query,
        key,
        value,
        workspace,
        output,
        architecture=architecture,
        scale=selected_scale,
        splits=splits,
        feature_major_output=uses_feature_major_output,
    )


def compressed_v_decode_attention_cuda_tuned(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    workspace: CompressedVDecodeWorkspace,
    scale: float | None = None,
    feature_major_output: Tensor | None = None,
) -> Tensor:
    """Run only CUDA kernels using an offline-tuned split configuration."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("compressed-V attention Q/K/V must be rank-4 tensors")
    batch = int(query.shape[0])
    query_heads = int(query.shape[1])
    sequence_length = int(key.shape[2])
    value_dim = int(value.shape[3])
    device_index = query.device.index
    selected_device = (
        torch.cuda.current_device() if device_index is None else device_index
    )
    splits = select_compressed_v_decode_cuda_splits(
        capability=_device_capability(selected_device),
        batch=batch,
        sequence_length=sequence_length,
    )
    cuda_workspace, token_major_output = workspace.views(
        batch=batch,
        query_heads=query_heads,
        splits=splits,
        value_dim=value_dim,
        dtype=query.dtype,
        device=query.device,
    )
    return compressed_v_decode_attention_cuda(
        query,
        key,
        value,
        scale=scale,
        splits=splits,
        workspace=cuda_workspace,
        output=None if feature_major_output is not None else token_major_output,
        feature_major_output=feature_major_output,
    )


def compressed_v_prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Run fused causal GQA for a complete prompt with compact Values.

    Shapes are ``query=[B,Hq,S,D]``, ``key=[B,Hkv,S,D]``, and
    ``value=[B,Hkv,S,R]``. This deliberately accepts only the initial complete
    prefill. Chunked prefill would require a causal offset and is outside the
    benchmark protocol.
    """

    (
        batch,
        query_heads,
        kv_heads,
        query_tokens,
        sequence_length,
        qk_dim,
        value_dim,
    ) = _validate_inputs(query, key, value)
    if query_tokens <= 1 or query_tokens != sequence_length:
        raise ValueError(
            "compressed-V prefill requires matching Q/K lengths greater than one"
        )
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(selected_scale) or selected_scale <= 0.0:
        raise ValueError("attention scale must be finite and positive")
    output = torch.empty(
        batch,
        query_heads,
        sequence_length,
        value_dim,
        dtype=query.dtype,
        device=query.device,
    )
    block_m = 32
    block_n = 64
    block_qk = triton.next_power_of_2(qk_dim)
    block_value = triton.next_power_of_2(value_dim)
    _compressed_v_gqa_prefill_kernel[
        (triton.cdiv(sequence_length, block_m), batch * query_heads)
    ](
        query,
        key,
        value,
        output,
        sequence_length,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        scale=selected_scale,
        QUERY_HEADS=query_heads,
        HEADS_PER_KV=query_heads // kv_heads,
        QK_DIM=qk_dim,
        VALUE_DIM=value_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_QK=block_qk,
        BLOCK_VALUE=block_value,
        num_warps=4,
        num_stages=2,
    )
    return output


def reference_compressed_v_decode_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Materializing correctness reference that supports arbitrary float dtype."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("reference compressed-V decode Q/K/V must be rank-4")
    batch, query_heads, query_tokens, qk_dim = map(int, query.shape)
    if query_tokens != 1:
        raise ValueError("reference compressed-V decode accepts one query token")
    kv_heads = int(key.shape[1])
    if (
        int(key.shape[0]) != batch
        or int(value.shape[0]) != batch
        or int(value.shape[1]) != kv_heads
        or int(key.shape[2]) != int(value.shape[2])
        or int(key.shape[3]) != qk_dim
        or query_heads % kv_heads
    ):
        raise ValueError("reference compressed-V decode shapes are incompatible")
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    work_dtype = (
        torch.float64
        if query.dtype == torch.float64
        or key.dtype == torch.float64
        or value.dtype == torch.float64
        else torch.float32
    )
    head_map = torch.arange(query_heads, device=query.device) // (
        query_heads // kv_heads
    )
    expanded_key = key.index_select(1, head_map)
    expanded_value = value.index_select(1, head_map)
    scores = torch.matmul(
        query.to(work_dtype),
        expanded_key.to(work_dtype).transpose(-1, -2),
    ) * selected_scale
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, expanded_value.to(work_dtype)).to(query.dtype)


def reference_compressed_v_prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Materializing causal prefill reference for correctness tests."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("reference compressed-V prefill Q/K/V must be rank-4")
    query_tokens = int(query.shape[2])
    sequence_length = int(key.shape[2])
    if query_tokens <= 1 or query_tokens != sequence_length:
        raise ValueError("reference prefill requires equal Q/K lengths greater than one")
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
        enable_gqa=True,
    )


__all__ = [
    "CompressedVDecodeWorkspace",
    "compressed_v_decode_attention_triton",
    "compressed_v_decode_attention_cuda",
    "compressed_v_decode_attention_cuda_tuned",
    "compressed_v_prefill_attention",
    "reference_compressed_v_decode_attention",
    "reference_compressed_v_prefill_attention",
    "select_compressed_v_decode_cuda_splits",
]
