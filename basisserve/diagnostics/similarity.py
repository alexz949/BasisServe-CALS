"""Numerically explicit representation-similarity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


def _matrix(name: str, value: Tensor) -> Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got {tuple(value.shape)}")
    if value.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two aligned rows")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return value.to(dtype=torch.float64)


def centered_linear_cka(x: Tensor, y: Tensor) -> float:
    """Return centered linear CKA for token-aligned row representations."""

    x = _matrix("x", x)
    y = _matrix("y", y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must have identical token rows")
    xc = x - x.mean(dim=0, keepdim=True)
    yc = y - y.mean(dim=0, keepdim=True)
    numerator = torch.linalg.matrix_norm(xc.T @ yc, ord="fro") ** 2
    denominator = torch.linalg.matrix_norm(xc.T @ xc, ord="fro") * torch.linalg.matrix_norm(
        yc.T @ yc, ord="fro"
    )
    return float((numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny)).item())


@dataclass(frozen=True)
class PrincipalAngleSummary:
    rank_x: int
    rank_y: int
    compared_rank: int
    mean_angle_degrees: float
    max_angle_degrees: float
    projection_overlap: float


def _top_right_subspace(x: Tensor, retained_variance: float, fixed_rank: int | None) -> Tensor:
    x = _matrix("representation", x)
    xc = x - x.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(xc, full_matrices=False)
    if fixed_rank is not None:
        rank = min(int(fixed_rank), int(singular_values.numel()))
    else:
        if not 0.0 < retained_variance <= 1.0:
            raise ValueError("retained_variance must be in (0, 1]")
        energy = singular_values.square()
        rank = int(torch.searchsorted(torch.cumsum(energy, 0), retained_variance * energy.sum()).item()) + 1
    return vh[: max(1, rank)].T.contiguous()


def principal_angle_summary(
    x: Tensor,
    y: Tensor,
    *,
    retained_variance: float = 0.99,
    fixed_rank: int | None = None,
) -> PrincipalAngleSummary:
    """Compare activation feature subspaces without assuming equal widths."""

    qx = _top_right_subspace(x, retained_variance, fixed_rank)
    qy = _top_right_subspace(y, retained_variance, fixed_rank)
    singular_values = torch.linalg.svdvals(qx.T @ qy).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(singular_values))
    return PrincipalAngleSummary(
        rank_x=int(qx.shape[1]),
        rank_y=int(qy.shape[1]),
        compared_rank=int(singular_values.numel()),
        mean_angle_degrees=float(angles.mean().item()),
        max_angle_degrees=float(angles.max().item()),
        projection_overlap=float(singular_values.square().mean().item()),
    )


@dataclass(frozen=True)
class SVCCASummary:
    retained_rank_x: int
    retained_rank_y: int
    compared_rank: int
    mean_correlation: float
    median_correlation: float
    minimum_correlation: float


def svcca_summary(
    x: Tensor,
    y: Tensor,
    *,
    retained_variance: float = 0.99,
    fixed_rank: int | None = None,
) -> SVCCASummary:
    """SVCCA summary for aligned rows, fit for selected pairs rather than all pairs."""

    x = _matrix("x", x)
    y = _matrix("y", y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("SVCCA inputs must have identical token rows")
    qx = _top_right_subspace(x, retained_variance, fixed_rank)
    qy = _top_right_subspace(y, retained_variance, fixed_rank)
    xc = (x - x.mean(0, keepdim=True)) @ qx
    yc = (y - y.mean(0, keepdim=True)) @ qy
    covariance = xc.T @ yc
    wx, vx = torch.linalg.eigh(xc.T @ xc)
    wy, vy = torch.linalg.eigh(yc.T @ yc)
    keep_x = wx > torch.finfo(wx.dtype).eps * wx.max().clamp_min(1.0)
    keep_y = wy > torch.finfo(wy.dtype).eps * wy.max().clamp_min(1.0)
    inv_x = vx[:, keep_x] / torch.sqrt(wx[keep_x]).unsqueeze(0)
    inv_y = vy[:, keep_y] / torch.sqrt(wy[keep_y]).unsqueeze(0)
    correlations = torch.linalg.svdvals(inv_x.T @ covariance @ inv_y).clamp(0.0, 1.0)
    return SVCCASummary(
        retained_rank_x=int(qx.shape[1]),
        retained_rank_y=int(qy.shape[1]),
        compared_rank=int(correlations.numel()),
        mean_correlation=float(correlations.mean().item()),
        median_correlation=float(correlations.median().item()),
        minimum_correlation=float(correlations.min().item()),
    )


def context_bucket_indices(positions: Tensor, boundaries: Sequence[tuple[int, int | None]]) -> Tensor:
    """Assign every non-negative token position to exactly one configured bucket."""

    if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be a one-dimensional integer tensor")
    assignments = torch.full_like(positions, -1)
    for index, (start, stop) in enumerate(boundaries):
        mask = positions >= start
        if stop is not None:
            mask &= positions <= stop
        assignments[mask] = index
    if (assignments < 0).any() or (positions < 0).any():
        raise ValueError("every position must be covered by exactly one bucket")
    return assignments
