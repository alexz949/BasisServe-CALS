"""Pairwise tall-skinny QR reduction.

The reduction retains only one square ``R`` factor per row block.  Pairwise
merges therefore have bounded row dimension even when the original matrix has
many calibration rows.  The returned factor satisfies ``R.T @ R ~= X.T @ X``
without explicitly forming the normal equations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import torch


@dataclass(frozen=True)
class PairwiseQRDiagnostics:
    column_count: int
    input_row_count: int
    leaf_row_counts: tuple[int, ...]
    node_counts_by_level: tuple[int, ...]


@dataclass(frozen=True)
class SquareRootLeastSquaresDiagnostics:
    design_rows: int
    design_columns: int
    right_hand_side_count: int
    absolute_product_damping: float
    minimum_abs_r_diagonal: float
    maximum_abs_r_diagonal: float
    r_diagonal_ratio: float
    relative_residual_norm: float
    wall_time_seconds: float


@dataclass(frozen=True)
class StreamingTSQRDiagnostics:
    column_count: int
    block_rows: int
    total_rows: int
    completed_leaf_count: int
    merge_count: int
    buffered_rows: int


def _canonicalize_r_diagonal(r: torch.Tensor) -> torch.Tensor:
    """Choose the QR sign gauge with a non-negative diagonal."""

    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return signs.unsqueeze(1) * r


def _synchronize(value: torch.Tensor) -> None:
    if value.device.type == "cuda":
        torch.cuda.synchronize(value.device)


@torch.no_grad()
def pack_upper_triangular(matrix: torch.Tensor) -> torch.Tensor:
    """Pack a square upper-triangular matrix row by row without index tensors."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("upper-triangular packing requires a square matrix")
    dimension = int(matrix.shape[0])
    packed = torch.empty(
        dimension * (dimension + 1) // 2,
        device=matrix.device,
        dtype=matrix.dtype,
    )
    offset = 0
    for row in range(dimension):
        count = dimension - row
        packed[offset : offset + count].copy_(matrix[row, row:])
        offset += count
    return packed


@torch.no_grad()
def unpack_upper_triangular(
    packed: torch.Tensor,
    *,
    dimension: int,
) -> torch.Tensor:
    """Restore a row-wise packed upper-triangular matrix."""

    dimension = int(dimension)
    expected = dimension * (dimension + 1) // 2
    if packed.ndim != 1 or int(packed.numel()) != expected:
        raise ValueError(
            f"packed upper triangle must have {expected} values, "
            f"got shape {tuple(packed.shape)}"
        )
    matrix = torch.zeros(
        dimension,
        dimension,
        device=packed.device,
        dtype=packed.dtype,
    )
    offset = 0
    for row in range(dimension):
        count = dimension - row
        matrix[row, row:].copy_(packed[offset : offset + count])
        offset += count
    return matrix


class StreamingTSQR:
    """Online pairwise TSQR with one bounded row buffer and binary-tree R state."""

    def __init__(
        self,
        *,
        columns: int,
        block_rows: int | None = None,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.columns = int(columns)
        self.block_rows = self.columns if block_rows is None else int(block_rows)
        if self.columns <= 0 or self.block_rows < self.columns:
            raise ValueError("TSQR block rows must be at least the positive column count")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("streaming TSQR dtype must be float32 or float64")
        self.device = torch.device(device)
        self.dtype = dtype
        self._buffer = torch.empty(
            self.block_rows,
            self.columns,
            device=self.device,
            dtype=self.dtype,
        )
        self._buffered_rows = 0
        self._levels: list[torch.Tensor | None] = []
        self._total_rows = 0
        self._completed_leaf_count = 0
        self._merge_count = 0

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def buffered_rows(self) -> int:
        return self._buffered_rows

    @torch.no_grad()
    def append(self, rows: torch.Tensor) -> None:
        if rows.ndim != 2 or int(rows.shape[1]) != self.columns:
            raise ValueError(
                f"TSQR rows must have shape [rows, {self.columns}], "
                f"got {tuple(rows.shape)}"
            )
        if rows.device != self.device:
            raise ValueError("TSQR rows must already reside on the accumulator device")
        if not rows.is_floating_point() or not torch.isfinite(rows).all():
            raise ValueError("TSQR rows must be finite floating-point values")
        source_start = 0
        source_rows = int(rows.shape[0])
        while source_start < source_rows:
            count = min(
                self.block_rows - self._buffered_rows,
                source_rows - source_start,
            )
            self._buffer[
                self._buffered_rows : self._buffered_rows + count
            ].copy_(
                rows[source_start : source_start + count],
            )
            self._buffered_rows += count
            self._total_rows += count
            source_start += count
            if self._buffered_rows == self.block_rows:
                self._consume_full_leaf()

    @torch.no_grad()
    def _consume_full_leaf(self) -> None:
        if self._buffered_rows != self.block_rows:
            raise RuntimeError("cannot consume an incomplete TSQR leaf")
        _, factor = torch.linalg.qr(self._buffer, mode="r")
        factor = _canonicalize_r_diagonal(factor)
        self._completed_leaf_count += 1
        self._buffered_rows = 0

        level = 0
        while True:
            if level == len(self._levels):
                self._levels.append(factor)
                break
            previous = self._levels[level]
            if previous is None:
                self._levels[level] = factor
                break
            pair = torch.cat((previous, factor), dim=0)
            _, factor = torch.linalg.qr(pair, mode="r")
            factor = _canonicalize_r_diagonal(factor)
            self._levels[level] = None
            self._merge_count += 1
            level += 1

    @torch.no_grad()
    def snapshot_r(self) -> torch.Tensor:
        """Return a canonical R for all complete rows without mutating the tree."""

        if self._buffered_rows:
            raise RuntimeError("TSQR snapshots require an empty row buffer")
        factors = [factor for factor in reversed(self._levels) if factor is not None]
        if not factors:
            raise RuntimeError("TSQR has no completed row block")
        result = factors[0]
        for factor in factors[1:]:
            pair = torch.cat((result, factor), dim=0)
            _, result = torch.linalg.qr(pair, mode="r")
            result = _canonicalize_r_diagonal(result)
        return result.clone()

    def diagnostics(self) -> StreamingTSQRDiagnostics:
        return StreamingTSQRDiagnostics(
            column_count=self.columns,
            block_rows=self.block_rows,
            total_rows=self._total_rows,
            completed_leaf_count=self._completed_leaf_count,
            merge_count=self._merge_count,
            buffered_rows=self._buffered_rows,
        )


@torch.no_grad()
def pairwise_qr_r(
    blocks: Iterable[torch.Tensor],
) -> tuple[torch.Tensor, PairwiseQRDiagnostics]:
    """Reduce compatible tall row blocks to one canonical upper-triangular R.

    Every leaf must have at least as many rows as columns.  This keeps every
    leaf and merge factor square, which is the standard TSQR regime and makes
    memory use independent of the total number of rows.
    """

    factors: list[torch.Tensor] = []
    leaf_rows: list[int] = []
    columns: int | None = None
    reference_dtype: torch.dtype | None = None
    reference_device: torch.device | None = None

    for index, block in enumerate(blocks):
        if block.ndim != 2:
            raise ValueError(f"QR leaf {index} must be a matrix")
        rows, block_columns = map(int, block.shape)
        if columns is None:
            columns = block_columns
            reference_dtype = block.dtype
            reference_device = block.device
            if columns <= 0:
                raise ValueError("QR leaves must have a positive column count")
        elif block_columns != columns:
            raise ValueError("all QR leaves must have the same column count")
        if rows < block_columns:
            raise ValueError(
                f"QR leaf {index} has {rows} rows but {block_columns} columns"
            )
        if not block.is_floating_point():
            raise ValueError("QR leaves must use a floating-point dtype")
        if block.dtype != reference_dtype or block.device != reference_device:
            raise ValueError("all QR leaves must share one dtype and device")
        if not torch.isfinite(block).all():
            raise ValueError(f"QR leaf {index} contains a non-finite value")

        _, r = torch.linalg.qr(block, mode="r")
        factors.append(_canonicalize_r_diagonal(r))
        leaf_rows.append(rows)

    if not factors or columns is None:
        raise ValueError("pairwise QR requires at least one row block")

    node_counts = [len(factors)]
    while len(factors) > 1:
        merged: list[torch.Tensor] = []
        for index in range(0, len(factors), 2):
            if index + 1 == len(factors):
                merged.append(factors[index])
                continue
            pair = torch.cat((factors[index], factors[index + 1]), dim=0)
            _, r = torch.linalg.qr(pair, mode="r")
            merged.append(_canonicalize_r_diagonal(r))
        factors = merged
        node_counts.append(len(factors))

    return factors[0], PairwiseQRDiagnostics(
        column_count=columns,
        input_row_count=sum(leaf_rows),
        leaf_row_counts=tuple(leaf_rows),
        node_counts_by_level=tuple(node_counts),
    )


@torch.no_grad()
def solve_square_root_least_squares(
    *,
    left_factor: torch.Tensor,
    basis: torch.Tensor,
    target_weight: torch.Tensor,
    absolute_product_damping: float = 0.0,
    output_chunk_size: int = 256,
) -> tuple[torch.Tensor, SquareRootLeastSquaresDiagnostics]:
    """Solve ``min_D ||R(BD-W)||^2 + delta ||BD-W||^2`` by QR.

    ``left_factor`` is normally the R returned by :func:`pairwise_qr_r` for
    ``X / sqrt(N)``.  Appending ``sqrt(delta) * B`` and
    ``sqrt(delta) * W`` exactly represents isotropic damping of the original
    product residual; it is not a ridge penalty on decoder coordinates.
    """

    if left_factor.ndim != 2 or left_factor.shape[0] != left_factor.shape[1]:
        raise ValueError("left factor must be a square matrix")
    width = int(left_factor.shape[1])
    if basis.ndim != 2 or int(basis.shape[0]) != width:
        raise ValueError("basis must have shape [left width, reduced width]")
    if target_weight.ndim != 2 or int(target_weight.shape[0]) != width:
        raise ValueError("target weight must have shape [left width, outputs]")
    reduced_width = int(basis.shape[1])
    output_width = int(target_weight.shape[1])
    if reduced_width <= 0 or output_width <= 0:
        raise ValueError("basis and target must have positive widths")
    if reduced_width > width:
        raise ValueError("the reduced basis cannot be wider than the left factor")
    if output_chunk_size <= 0:
        raise ValueError("output chunk size must be positive")
    if absolute_product_damping < 0 or not math.isfinite(absolute_product_damping):
        raise ValueError("absolute product damping must be finite and non-negative")
    tensors = (left_factor, basis, target_weight)
    if any(value.dtype != left_factor.dtype for value in tensors[1:]) or any(
        value.device != left_factor.device for value in tensors[1:]
    ):
        raise ValueError("left factor, basis, and target must share dtype and device")
    if not left_factor.is_floating_point():
        raise ValueError("square-root least squares requires floating-point tensors")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("square-root least-squares inputs must be finite")

    _synchronize(left_factor)
    started = time.monotonic()
    top_design = left_factor @ basis
    if absolute_product_damping:
        square_root_damping = math.sqrt(absolute_product_damping)
        design = torch.cat((top_design, square_root_damping * basis), dim=0)
    else:
        square_root_damping = 0.0
        design = top_design

    q, r = torch.linalg.qr(design, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q.mul_(signs.unsqueeze(0))
    r.mul_(signs.unsqueeze(1))

    decoder = torch.empty(
        reduced_width,
        output_width,
        device=left_factor.device,
        dtype=left_factor.dtype,
    )
    residual_squared = 0.0
    target_squared = 0.0
    for start in range(0, output_width, output_chunk_size):
        stop = min(start + output_chunk_size, output_width)
        top_target = left_factor @ target_weight[:, start:stop]
        if absolute_product_damping:
            target = torch.cat(
                (
                    top_target,
                    square_root_damping * target_weight[:, start:stop],
                ),
                dim=0,
            )
        else:
            target = top_target
        projected = q.transpose(0, 1) @ target
        decoder[:, start:stop] = torch.linalg.solve_triangular(
            r,
            projected,
            upper=True,
        )
        target_norm_squared = float(target.square().sum(dtype=torch.float64))
        projected_norm_squared = float(projected.square().sum(dtype=torch.float64))
        target_squared += target_norm_squared
        residual_squared += max(target_norm_squared - projected_norm_squared, 0.0)

    _synchronize(left_factor)
    elapsed = time.monotonic() - started
    diagonal = torch.diagonal(r).abs()
    minimum = float(diagonal.min())
    maximum = float(diagonal.max())
    if not torch.isfinite(decoder).all():
        raise torch.linalg.LinAlgError(
            "square-root QR produced a non-finite decoder; design may be rank deficient"
        )
    return decoder, SquareRootLeastSquaresDiagnostics(
        design_rows=int(design.shape[0]),
        design_columns=reduced_width,
        right_hand_side_count=output_width,
        absolute_product_damping=absolute_product_damping,
        minimum_abs_r_diagonal=minimum,
        maximum_abs_r_diagonal=maximum,
        r_diagonal_ratio=maximum
        / max(minimum, torch.finfo(left_factor.dtype).tiny),
        relative_residual_norm=math.sqrt(
            residual_squared
            / max(target_squared, torch.finfo(torch.float64).tiny)
        ),
        wall_time_seconds=elapsed,
    )
