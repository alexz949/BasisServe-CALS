"""Decoder GEMM over tensor-parallel rank-major AllGather output.

NCCL ``all_gather_into_tensor`` produces ``[P, M, K_local]`` storage while
the C1 decoder logically consumes ``[M, P * K_local]``.  Materializing that
transpose costs one prompt-sized buffer and one kernel per transformer layer.
For medium matrices, a Triton GEMM applies the logical transpose in its A-tile
address calculation. At the largest prompt size, four cuBLAS GEMMs over the
already-contiguous rank blocks are faster on L40S. Both paths write the decoded
``[M, N]`` result directly and avoid a token-major prompt buffer.
"""

from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl


@triton.jit
def _rank_major_decoder_kernel(
    rank_major_ptr,
    decoder_ptr,
    output_ptr,
    rows,
    output_width,
    LOCAL_WIDTH: tl.constexpr,
    PROCESSES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    column_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # The decoder is token-major in its reduction dimension, but the input is
    # rank-major. The address mapping remains correct even if a reduction tile
    # crosses a process-block boundary.
    for reduction_start in range(0, PROCESSES * LOCAL_WIDTH, BLOCK_K):
        reduction_offsets = reduction_start + tl.arange(0, BLOCK_K)
        process_offsets = reduction_offsets // LOCAL_WIDTH
        local_offsets = reduction_offsets % LOCAL_WIDTH
        rank_major_offsets = (
            process_offsets[None, :] * rows * LOCAL_WIDTH
            + row_offsets[:, None] * LOCAL_WIDTH
            + local_offsets[None, :]
        )
        decoder_offsets = (
            reduction_offsets[:, None] * output_width + column_offsets[None, :]
        )
        lhs = tl.load(
            rank_major_ptr + rank_major_offsets,
            mask=row_offsets[:, None] < rows,
            other=0.0,
        )
        rhs = tl.load(
            decoder_ptr + decoder_offsets,
            mask=column_offsets[None, :] < output_width,
            other=0.0,
        )
        accumulator += tl.dot(lhs, rhs)

    output_offsets = row_offsets[:, None] * output_width + column_offsets[None, :]
    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=(row_offsets[:, None] < rows)
        & (column_offsets[None, :] < output_width),
    )


def _validate_rank_major_decoder(
    rank_major: Tensor,
    decoder: Tensor,
    processes: int,
) -> tuple[int, int, int]:
    if rank_major.ndim != 2 or decoder.ndim != 2:
        raise ValueError("rank-major coordinates and decoder must be matrices")
    selected_processes = int(processes)
    if selected_processes <= 1:
        raise ValueError("rank-major decoder requires at least two processes")
    gathered_rows, local_width = map(int, rank_major.shape)
    if gathered_rows <= 0 or gathered_rows % selected_processes:
        raise ValueError("rank-major row count must be divisible by processes")
    rows = gathered_rows // selected_processes
    if tuple(decoder.shape[:1]) != (selected_processes * local_width,):
        raise ValueError(
            "decoder reduction width differs from gathered coordinates: "
            f"{tuple(decoder.shape)} versus {selected_processes}*{local_width}"
        )
    output_width = int(decoder.shape[1])
    if local_width <= 0 or local_width % 32:
        raise ValueError("local coordinate width must be a positive multiple of 32")
    if output_width <= 0:
        raise ValueError("decoder output width must be positive")
    if rank_major.dtype != decoder.dtype:
        raise TypeError("rank-major coordinates and decoder dtypes differ")
    if rank_major.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("rank-major Triton decoder supports FP16 and BF16")
    if not rank_major.is_cuda or not decoder.is_cuda:
        raise ValueError("rank-major Triton decoder requires CUDA tensors")
    if rank_major.device != decoder.device:
        raise ValueError("rank-major coordinates and decoder devices differ")
    if not rank_major.is_contiguous() or not decoder.is_contiguous():
        raise ValueError("rank-major coordinates and decoder must be contiguous")
    return rows, local_width, output_width


def rank_major_decoder(
    rank_major: Tensor,
    decoder: Tensor,
    *,
    processes: int,
) -> Tensor:
    """Decode rank-major coordinates without a token-major materialization.

    ``rank_major`` has physical shape ``[processes * rows, local_width]`` and
    rank-contiguous storage. ``decoder`` has shape
    ``[processes * local_width, output_width]``. The logical result is
    equivalent to first transposing the gathered coordinates to
    ``[rows, processes * local_width]`` and then calling ``torch.mm``.
    """

    rows, local_width, output_width = _validate_rank_major_decoder(
        rank_major,
        decoder,
        processes,
    )
    if rows >= 32768:
        blocks = rank_major.view(int(processes), rows, local_width)
        output = torch.mm(blocks[0], decoder[:local_width])
        for process in range(1, int(processes)):
            start = process * local_width
            output.addmm_(
                blocks[process],
                decoder[start : start + local_width],
            )
        return output

    output = torch.empty(
        rows,
        output_width,
        device=rank_major.device,
        dtype=rank_major.dtype,
    )
    if rows <= 256:
        block_m, block_n, block_k, num_warps, num_stages = 32, 64, 32, 4, 3
    elif rows >= 16384 and local_width > 512:
        block_m, block_n, block_k, num_warps, num_stages = 64, 256, 32, 8, 4
    else:
        block_m, block_n, block_k, num_warps, num_stages = 128, 256, 64, 8, 3
    _rank_major_decoder_kernel[
        (triton.cdiv(rows, block_m), triton.cdiv(output_width, block_n))
    ](
        rank_major,
        decoder,
        output,
        rows,
        output_width,
        LOCAL_WIDTH=local_width,
        PROCESSES=int(processes),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def reference_rank_major_decoder(
    rank_major: Tensor,
    decoder: Tensor,
    *,
    processes: int,
) -> Tensor:
    """PyTorch layout-and-GEMM reference used by tests and benchmarks."""

    if rank_major.ndim != 2 or decoder.ndim != 2:
        raise ValueError("rank-major coordinates and decoder must be matrices")
    selected_processes = int(processes)
    gathered_rows, local_width = map(int, rank_major.shape)
    if selected_processes <= 1 or gathered_rows % selected_processes:
        raise ValueError("invalid process count for rank-major coordinates")
    rows = gathered_rows // selected_processes
    if int(decoder.shape[0]) != selected_processes * local_width:
        raise ValueError("decoder reduction width differs from gathered coordinates")
    token_major = (
        rank_major.reshape(selected_processes, rows, local_width)
        .permute(1, 0, 2)
        .reshape(rows, selected_processes * local_width)
        .contiguous()
    )
    return torch.mm(token_major, decoder)


__all__ = ["rank_major_decoder", "reference_rank_major_decoder"]
