"""Streaming statistics for routed GQA Value/output factorization.

The tensor captured at a dense attention module's ``o_proj`` input is ordered
as concatenated query heads.  This module accumulates its full uncentered
second moment while retaining an explicit ``[head, head, channel, channel]``
view for structured offline solvers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutedOVStatistics:
    covariance_blocks: torch.Tensor
    row_count: int
    dense_output_energy: float

    @property
    def num_query_heads(self) -> int:
        return int(self.covariance_blocks.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self.covariance_blocks.shape[2])

    def flat_covariance(self) -> torch.Tensor:
        heads = self.num_query_heads
        width = self.head_dim
        return (
            self.covariance_blocks.permute(0, 2, 1, 3)
            .reshape(heads * width, heads * width)
            .contiguous()
        )


class RoutedOVAccumulator:
    """Accumulate the exact dense ``o_proj`` input covariance.

    Accumulation normally happens on the layer device in FP32.  Final
    statistics may be converted to FP64 on CPU without transferring every
    activation row to the host.
    """

    def __init__(
        self,
        *,
        num_query_heads: int,
        head_dim: int,
        accumulation_dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        if num_query_heads <= 0 or head_dim <= 0:
            raise ValueError("head count and head dimension must be positive")
        if accumulation_dtype not in (torch.float32, torch.float64):
            raise ValueError("accumulation dtype must be float32 or float64")
        self.num_query_heads = int(num_query_heads)
        self.head_dim = int(head_dim)
        self.width = self.num_query_heads * self.head_dim
        self.accumulation_dtype = accumulation_dtype
        self.device = None if device is None else torch.device(device)
        self._gram: torch.Tensor | None = None
        self._output_energy: torch.Tensor | None = None
        self._rows = 0

    @torch.no_grad()
    def update(
        self,
        o_proj_input: torch.Tensor,
        o_proj_output: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        if o_proj_input.shape[:-1] != o_proj_output.shape[:-1]:
            raise ValueError("o_proj input/output leading dimensions must match")
        if o_proj_input.shape[-1] != self.width:
            raise ValueError(
                f"o_proj input width must be {self.width}, got {o_proj_input.shape[-1]}"
            )
        rows = o_proj_input.detach().reshape(-1, self.width)
        outputs = o_proj_output.detach().reshape(-1, o_proj_output.shape[-1])
        if valid_mask is not None:
            if tuple(valid_mask.shape) != tuple(o_proj_input.shape[:-1]):
                raise ValueError("valid mask must match o_proj input leading dimensions")
            selected = valid_mask.detach().reshape(-1).to(dtype=torch.bool)
            rows = rows[selected]
            outputs = outputs[selected]
        if rows.shape[0] == 0:
            return
        target_device = self.device or rows.device
        rows = rows.to(device=target_device, dtype=self.accumulation_dtype)
        outputs = outputs.to(device=target_device, dtype=self.accumulation_dtype)
        if not torch.isfinite(rows).all() or not torch.isfinite(outputs).all():
            raise ValueError("routed activations must contain only finite values")
        if self._gram is None:
            self._gram = torch.zeros(
                self.width,
                self.width,
                device=target_device,
                dtype=self.accumulation_dtype,
            )
            self._output_energy = torch.zeros(
                (),
                device=target_device,
                dtype=self.accumulation_dtype,
            )
        elif self._gram.device != target_device:
            raise RuntimeError("routed statistic device changed during accumulation")
        assert self._output_energy is not None
        self._gram.addmm_(rows.transpose(0, 1), rows)
        self._output_energy.add_(outputs.square().sum())
        self._rows += int(rows.shape[0])

    @property
    def rows(self) -> int:
        return self._rows

    def finalize(
        self,
        *,
        output_dtype: torch.dtype = torch.float64,
        output_device: torch.device | str = "cpu",
    ) -> RoutedOVStatistics:
        if self._gram is None or self._output_energy is None or self._rows <= 0:
            raise RuntimeError("no routed rows were accumulated")
        if output_dtype not in (torch.float32, torch.float64):
            raise ValueError("statistics output dtype must be float32 or float64")
        covariance = self._gram / self._rows
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        covariance = covariance.to(device=output_device, dtype=output_dtype)
        blocks = (
            covariance.reshape(
                self.num_query_heads,
                self.head_dim,
                self.num_query_heads,
                self.head_dim,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return RoutedOVStatistics(
            covariance_blocks=blocks,
            row_count=self._rows,
            dense_output_energy=float(self._output_energy.double() / self._rows),
        )


def flatten_covariance_blocks(blocks: torch.Tensor) -> torch.Tensor:
    if blocks.ndim != 4:
        raise ValueError("covariance blocks must have shape [H, H, d, d]")
    heads, heads_again, width, width_again = blocks.shape
    if heads != heads_again or width != width_again:
        raise ValueError("covariance block axes must be square")
    return (
        blocks.permute(0, 2, 1, 3)
        .reshape(heads * width, heads * width)
        .contiguous()
    )


def unflatten_covariance(
    covariance: torch.Tensor,
    *,
    num_query_heads: int,
    head_dim: int,
) -> torch.Tensor:
    expected = num_query_heads * head_dim
    if tuple(covariance.shape) != (expected, expected):
        raise ValueError(
            f"flat covariance must have shape {(expected, expected)}, "
            f"got {tuple(covariance.shape)}"
        )
    return (
        covariance.reshape(
            num_query_heads,
            head_dim,
            num_query_heads,
            head_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def covariance_block_energy_diagnostics(
    blocks: torch.Tensor,
    head_to_kv_group: torch.Tensor,
) -> dict[str, float]:
    """Summarize diagonal, within-group, and cross-group block energy."""

    if blocks.ndim != 4 or blocks.shape[0] != blocks.shape[1]:
        raise ValueError("blocks must have shape [H, H, d, d]")
    mapping = torch.as_tensor(head_to_kv_group, dtype=torch.long)
    if mapping.numel() != blocks.shape[0]:
        raise ValueError("head mapping length must match covariance")
    energy = blocks.double().square().sum(dim=(-2, -1))
    diagonal_mask = torch.eye(mapping.numel(), dtype=torch.bool)
    same_group = mapping[:, None] == mapping[None, :]
    within_mask = same_group & ~diagonal_mask
    cross_mask = ~same_group
    total = float(energy.sum())
    diagonal = float(energy[diagonal_mask].sum())
    within = float(energy[within_mask].sum())
    cross = float(energy[cross_mask].sum())
    denominator = max(total, torch.finfo(torch.float64).tiny)
    return {
        "total_energy": total,
        "diagonal_energy": diagonal,
        "within_group_offdiagonal_energy": within,
        "cross_group_energy": cross,
        "offdiagonal_energy_ratio": (within + cross) / denominator,
        "within_group_energy_ratio": within / denominator,
        "cross_group_energy_ratio": cross / denominator,
    }
