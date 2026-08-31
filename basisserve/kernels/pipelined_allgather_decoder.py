"""Overlap uniform tensor-parallel AllGather waves with partial decoding."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from basisserve.kernels.rank_major_decoder import rank_major_decoder


def pack_uniform_decoder_waves(
    decoder: Tensor,
    *,
    processes: int,
    local_width: int,
    waves: int,
) -> tuple[Tensor, ...]:
    """Pack matching per-rank decoder slices for feature-wise AG waves."""

    selected_processes = int(processes)
    selected_local_width = int(local_width)
    selected_waves = int(waves)
    if decoder.ndim != 2:
        raise ValueError("decoder must be a matrix")
    if selected_processes <= 1:
        raise ValueError("pipelined AllGather requires at least two processes")
    if selected_local_width <= 0:
        raise ValueError("local width must be positive")
    if selected_waves not in (2, 4):
        raise ValueError("decoder pipeline supports exactly 2 or 4 waves")
    if selected_local_width % selected_waves:
        raise ValueError("local width must be divisible by the wave count")
    if int(decoder.shape[0]) != selected_processes * selected_local_width:
        raise ValueError(
            "decoder reduction width differs from process-local coordinates"
        )

    wave_width = selected_local_width // selected_waves
    process_blocks = decoder.view(
        selected_processes,
        selected_local_width,
        int(decoder.shape[1]),
    )
    return tuple(
        process_blocks[:, start : start + wave_width]
        .reshape(selected_processes * wave_width, int(decoder.shape[1]))
        .contiguous()
        for start in range(0, selected_local_width, wave_width)
    )


class PipelinedAllGatherDecoder:
    """Prepared fixed-shape NCCL AllGather plus partial decoder pipeline.

    The local token-major coordinate matrix is split along its feature axis.
    Every wave AllGathers the same feature slice from all TP ranks into a
    rank-major receive buffer.  The main stream decodes a completed wave while
    the side stream communicates the next one.
    """

    def __init__(
        self,
        decoder: Tensor,
        *,
        rows: int,
        local_width: int,
        waves: int,
        group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        if str(dist.get_backend(group)).lower() != "nccl":
            raise RuntimeError("pipelined AllGather requires an NCCL process group")
        if not decoder.is_cuda or decoder.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise TypeError("decoder must be a CUDA FP16 or BF16 matrix")
        if not decoder.is_contiguous():
            raise ValueError("decoder must be contiguous")
        if torch.is_grad_enabled() and decoder.requires_grad:
            raise RuntimeError("pipelined AllGather decode is inference-only")

        self.group = group
        self.processes = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.rows = int(rows)
        self.local_width = int(local_width)
        self.waves = int(waves)
        if self.rows <= 0:
            raise ValueError("row count must be positive")
        self.wave_width = self.local_width // self.waves
        self.decoder_waves = pack_uniform_decoder_waves(
            decoder,
            processes=self.processes,
            local_width=self.local_width,
            waves=self.waves,
        )
        self.device = decoder.device
        self.dtype = decoder.dtype
        self.communication_stream = torch.cuda.Stream(device=self.device)
        self.ready_events = tuple(
            torch.cuda.Event(enable_timing=False) for _ in range(self.waves)
        )
        self.local_waves = tuple(
            torch.empty(
                self.rows,
                self.wave_width,
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.waves)
        )
        self.gathered_waves = tuple(
            torch.empty(
                self.processes * self.rows,
                self.wave_width,
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.waves)
        )

    def __call__(self, local_coordinates: Tensor) -> Tensor:
        if tuple(local_coordinates.shape) != (self.rows, self.local_width):
            raise ValueError(
                "local coordinates must have prepared shape "
                f"[{self.rows}, {self.local_width}], got "
                f"{tuple(local_coordinates.shape)}"
            )
        if (
            local_coordinates.device != self.device
            or local_coordinates.dtype != self.dtype
        ):
            raise ValueError("local coordinates must match the prepared decoder")
        if not local_coordinates.is_contiguous():
            raise ValueError("local coordinates must be contiguous")
        if torch.is_grad_enabled() and local_coordinates.requires_grad:
            raise RuntimeError("pipelined AllGather decode is inference-only")

        compute_stream = torch.cuda.current_stream(self.device)
        for wave_index, local_wave in enumerate(self.local_waves):
            start = wave_index * self.wave_width
            local_wave.copy_(
                local_coordinates.narrow(1, start, self.wave_width),
                non_blocking=True,
            )

        self.communication_stream.wait_stream(compute_stream)
        with torch.cuda.stream(self.communication_stream):
            for local_wave, gathered_wave, ready_event in zip(
                self.local_waves,
                self.gathered_waves,
                self.ready_events,
                strict=True,
            ):
                work = dist.all_gather_into_tensor(
                    gathered_wave,
                    local_wave,
                    group=self.group,
                    async_op=True,
                )
                # NCCL Work.wait() inserts a dependency on the active CUDA
                # stream without serializing the CPU launch thread.
                work.wait()
                ready_event.record(self.communication_stream)

        output: Tensor | None = None
        for gathered_wave, decoder_wave, ready_event in zip(
            self.gathered_waves,
            self.decoder_waves,
            self.ready_events,
            strict=True,
        ):
            compute_stream.wait_event(ready_event)
            partial = rank_major_decoder(
                gathered_wave,
                decoder_wave,
                processes=self.processes,
            )
            if output is None:
                output = partial
            else:
                output.add_(partial)
        assert output is not None
        return output


__all__ = ["PipelinedAllGatherDecoder", "pack_uniform_decoder_waves"]
