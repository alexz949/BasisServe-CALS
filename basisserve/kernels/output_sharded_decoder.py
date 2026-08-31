"""Output-sharded tensor-parallel decoder collectives."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor


def pack_hidden_for_reduce_scatter(value: Tensor, *, processes: int) -> Tensor:
    """Pack token-major hidden columns into rank-major ReduceScatter chunks."""

    selected_processes = int(processes)
    if value.ndim != 2:
        raise ValueError("decoder output must be a matrix")
    if selected_processes <= 1:
        raise ValueError("output sharding requires at least two processes")
    rows, hidden = map(int, value.shape)
    if hidden % selected_processes:
        raise ValueError("hidden width must be divisible by the process count")
    local_hidden = hidden // selected_processes
    return (
        value.view(rows, selected_processes, local_hidden)
        .permute(1, 0, 2)
        .contiguous()
        .view(selected_processes * rows, local_hidden)
    )


def restore_hidden_from_rank_major_shards(
    value: Tensor,
    *,
    processes: int,
) -> Tensor:
    """Restore token-major hidden columns after rank-major AllGather."""

    selected_processes = int(processes)
    if value.ndim != 2:
        raise ValueError("gathered output shards must form a matrix")
    if selected_processes <= 1:
        raise ValueError("output restoration requires at least two processes")
    gathered_rows, local_hidden = map(int, value.shape)
    if gathered_rows % selected_processes:
        raise ValueError("gathered row count must be divisible by the process count")
    rows = gathered_rows // selected_processes
    return (
        value.view(selected_processes, rows, local_hidden)
        .permute(1, 0, 2)
        .contiguous()
        .view(rows, selected_processes * local_hidden)
    )


class OutputShardedDecoder:
    """Prepared local decoder GEMM followed by hidden ReduceScatter.

    Each TP rank owns one contiguous reduction-axis block of the decoder.  It
    computes a full-width partial output, then ReduceScatters hidden columns.
    Optionally, an AllGather restores the replicated hidden tensor expected by
    conventional one-dimensional tensor-parallel transformer blocks.
    """

    def __init__(
        self,
        local_decoder: Tensor,
        *,
        rows: int,
        reconstruct: bool,
        group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        if str(dist.get_backend(group)).lower() != "nccl":
            raise RuntimeError("output-sharded decoder requires an NCCL group")
        if local_decoder.ndim != 2:
            raise ValueError("local decoder must be a matrix")
        if not local_decoder.is_cuda or local_decoder.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise TypeError("local decoder must be a CUDA FP16 or BF16 matrix")
        if not local_decoder.is_contiguous():
            raise ValueError("local decoder must be contiguous")
        if torch.is_grad_enabled() and local_decoder.requires_grad:
            raise RuntimeError("output-sharded decoder is inference-only")

        self.group = group
        self.processes = dist.get_world_size(group)
        self.rows = int(rows)
        self.local_width = int(local_decoder.shape[0])
        self.hidden_size = int(local_decoder.shape[1])
        self.reconstruct = bool(reconstruct)
        if self.rows <= 0:
            raise ValueError("row count must be positive")
        if self.hidden_size % self.processes:
            raise ValueError("hidden width must be divisible by the process count")
        self.local_hidden = self.hidden_size // self.processes
        self.local_decoder = local_decoder
        self.device = local_decoder.device
        self.dtype = local_decoder.dtype

        self.partial_output = torch.empty(
            self.rows,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.reduce_input = torch.empty(
            self.processes * self.rows,
            self.local_hidden,
            device=self.device,
            dtype=self.dtype,
        )
        self.output_shard = torch.empty(
            self.rows,
            self.local_hidden,
            device=self.device,
            dtype=self.dtype,
        )
        if self.reconstruct:
            self.gathered_shards = torch.empty_like(self.reduce_input)
            self.replicated_output = torch.empty_like(self.partial_output)
        else:
            self.gathered_shards = None
            self.replicated_output = None

    def __call__(self, local_coordinates: Tensor) -> Tensor:
        expected = (self.rows, self.local_width)
        if tuple(local_coordinates.shape) != expected:
            raise ValueError(
                f"local coordinates must have shape {expected}, got "
                f"{tuple(local_coordinates.shape)}"
            )
        if (
            local_coordinates.device != self.device
            or local_coordinates.dtype != self.dtype
        ):
            raise ValueError("local coordinates must match the local decoder")
        if not local_coordinates.is_contiguous():
            raise ValueError("local coordinates must be contiguous")
        if torch.is_grad_enabled() and local_coordinates.requires_grad:
            raise RuntimeError("output-sharded decoder is inference-only")

        torch.mm(local_coordinates, self.local_decoder, out=self.partial_output)
        self.reduce_input.view(
            self.processes,
            self.rows,
            self.local_hidden,
        ).copy_(
            self.partial_output.view(
                self.rows,
                self.processes,
                self.local_hidden,
            ).permute(1, 0, 2)
        )
        work = dist.reduce_scatter_tensor(
            self.output_shard,
            self.reduce_input,
            op=dist.ReduceOp.SUM,
            group=self.group,
            async_op=True,
        )
        work.wait()
        if not self.reconstruct:
            return self.output_shard

        assert self.gathered_shards is not None
        assert self.replicated_output is not None
        work = dist.all_gather_into_tensor(
            self.gathered_shards,
            self.output_shard,
            group=self.group,
            async_op=True,
        )
        work.wait()
        self.replicated_output.view(
            self.rows,
            self.processes,
            self.local_hidden,
        ).copy_(
            self.gathered_shards.view(
                self.processes,
                self.rows,
                self.local_hidden,
            ).permute(1, 0, 2)
        )
        return self.replicated_output


__all__ = [
    "OutputShardedDecoder",
    "pack_hidden_for_reduce_scatter",
    "restore_hidden_from_rank_major_shards",
]
