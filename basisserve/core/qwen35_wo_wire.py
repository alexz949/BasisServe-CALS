"""Spectrum utilities for Qwen3.5 GDN output-projection wires."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


class FullSecondMoment:
    """Streaming full-channel uncentered second moment ``E[x^T x]``."""

    def __init__(
        self,
        width: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if width <= 0:
            raise ValueError("moment width must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("moment dtype must be float32 or float64")
        self.width = int(width)
        self.device = torch.device(device)
        self.dtype = dtype
        self.rows = 0
        self.sum = torch.zeros(self.width, device=self.device, dtype=self.dtype)
        self.gram = torch.zeros(
            self.width,
            self.width,
            device=self.device,
            dtype=self.dtype,
        )

    @torch.no_grad()
    def update(self, rows: Tensor) -> None:
        if rows.ndim != 2 or rows.shape[1] != self.width:
            raise ValueError(f"moment rows must have shape [N,{self.width}]")
        if rows.shape[0] == 0:
            return
        work = rows.detach().to(device=self.device, dtype=self.dtype)
        if not torch.isfinite(work).all():
            raise ValueError("moment rows contain non-finite values")
        self.sum.add_(work.sum(dim=0))
        self.gram.addmm_(work.transpose(0, 1), work)
        self.rows += int(work.shape[0])

    def second_moment(self) -> Tensor:
        if self.rows <= 0:
            raise ValueError("cannot normalize empty moments")
        result = self.gram / float(self.rows)
        return 0.5 * (result + result.transpose(0, 1))

    def mean(self) -> Tensor:
        if self.rows <= 0:
            raise ValueError("cannot normalize empty moments")
        return self.sum / float(self.rows)

    def state_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "storage_dtype": str(self.dtype),
            "rows": self.rows,
            "sum": self.sum.detach().cpu().contiguous(),
            "gram": self.gram.detach().cpu().contiguous(),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "FullSecondMoment":
        result = cls(int(payload["width"]), dtype=torch.float64)
        result.rows = int(payload["rows"])
        result.sum.copy_(payload["sum"].to(dtype=torch.float64, device="cpu"))
        result.gram.copy_(payload["gram"].to(dtype=torch.float64, device="cpu"))
        return result


@dataclass(frozen=True)
class ActivationAwareWoFactors:
    """Nested activation-weighted factors for ``x @ W.T``."""

    input_factor: Tensor
    output_basis: Tensor
    singular_values: Tensor
    total_weighted_energy: float
    damping: float

    @property
    def maximum_rank(self) -> int:
        return int(self.input_factor.shape[1])

    def prefix(self, rank: int) -> tuple[Tensor, Tensor]:
        if not 0 < rank <= self.maximum_rank:
            raise ValueError(f"wire rank must lie in [1,{self.maximum_rank}]")
        return (
            self.input_factor[:, :rank].contiguous(),
            self.output_basis[:, :rank].contiguous(),
        )

    def retained_fraction(self, rank: int) -> float:
        if not 0 < rank <= self.maximum_rank:
            raise ValueError(f"wire rank must lie in [1,{self.maximum_rank}]")
        retained = self.singular_values[:rank].double().square().sum().item()
        return float(retained / max(self.total_weighted_energy, 1e-30))


@torch.no_grad()
def fit_activation_aware_wo_factors(
    weight: Tensor,
    second_moment: Tensor,
    maximum_rank: int,
    *,
    relative_damping: float = 1e-5,
    factor_dtype: torch.dtype = torch.bfloat16,
) -> ActivationAwareWoFactors:
    """Fit ``W.T ~= A B.T`` under the input second-moment metric.

    For ``C = E[x.T x]`` and ``C + lambda I = L L.T``, this computes the
    truncated SVD of ``L.T @ W.T`` and maps its left factor back through
    ``L.T``.  Prefixes of the returned factors are nested optimal solutions
    for the damped activation-weighted output-MSE objective.
    """

    if weight.ndim != 2 or second_moment.ndim != 2:
        raise ValueError("weight and second moment must be matrices")
    out_features, in_features = map(int, weight.shape)
    if tuple(second_moment.shape) != (in_features, in_features):
        raise ValueError("second moment does not match Wo input width")
    if not 0 < maximum_rank <= min(in_features, out_features):
        raise ValueError("invalid maximum Wo wire rank")
    if relative_damping < 0:
        raise ValueError("relative damping must be nonnegative")
    device = weight.device
    matrix = second_moment.detach().to(device=device, dtype=torch.float32)
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    scale = matrix.diagonal().mean().clamp_min(torch.finfo(matrix.dtype).tiny)
    requested_damping = scale * float(relative_damping)
    damping_tensor = requested_damping
    cholesky = None
    last_info = -1
    # Large FP32 streaming Grams can acquire tiny negative eigenvalues through
    # accumulation roundoff.  Use the smallest decade of diagonal jitter that
    # restores positive definiteness instead of silently selecting a large
    # fixed regularizer for every layer.
    for _ in range(6):
        regularized = matrix.clone()
        regularized.diagonal().add_(damping_tensor)
        candidate, info = torch.linalg.cholesky_ex(regularized)
        last_info = int(info.max().item())
        if last_info == 0:
            cholesky = candidate
            break
        damping_tensor = damping_tensor * 10.0
    if cholesky is None:
        raise RuntimeError(
            "activation second moment Cholesky failed after adaptive damping; "
            f"last_info={last_info}, requested={float(requested_damping.item()):.6g}, "
            f"last={float(damping_tensor.item()):.6g}"
        )
    logical = weight.detach().to(device=device, dtype=torch.float32).transpose(0, 1)
    weighted = cholesky.transpose(0, 1) @ logical
    left, singular_values, right_t = torch.linalg.svd(weighted, full_matrices=False)
    total_energy = float(singular_values.double().square().sum().item())
    left_scaled = left[:, :maximum_rank] * singular_values[:maximum_rank].unsqueeze(0)
    input_factor = torch.linalg.solve_triangular(
        cholesky.transpose(0, 1),
        left_scaled,
        upper=True,
    )
    output_basis = right_t[:maximum_rank].transpose(0, 1)
    return ActivationAwareWoFactors(
        input_factor=input_factor.to(dtype=factor_dtype).contiguous(),
        output_basis=output_basis.to(dtype=factor_dtype).contiguous(),
        singular_values=singular_values.float().cpu().contiguous(),
        total_weighted_energy=total_energy,
        damping=float(damping_tensor.item()),
    )


@dataclass(frozen=True)
class WoWireRankMetric:
    rank: int
    wire_fraction: float
    payload_reduction: float
    factor_parameter_ratio: float
    energy_retained: float
    relative_frobenius_error: float


@dataclass(frozen=True)
class WoWireSpectrum:
    singular_values: Tensor
    in_features: int
    out_features: int

    def metric(self, rank: int) -> WoWireRankMetric:
        maximum = min(self.in_features, self.out_features)
        if not 0 < rank <= maximum:
            raise ValueError(f"wire rank must lie in [1,{maximum}]")
        energy = self.singular_values.double().square()
        retained = float((energy[:rank].sum() / energy.sum().clamp_min(1e-30)).item())
        return WoWireRankMetric(
            rank=int(rank),
            wire_fraction=rank / self.out_features,
            payload_reduction=self.out_features / rank,
            factor_parameter_ratio=(
                rank * (self.in_features + self.out_features)
                / (self.in_features * self.out_features)
            ),
            energy_retained=retained,
            relative_frobenius_error=math.sqrt(max(0.0, 1.0 - retained)),
        )


@torch.no_grad()
def qwen35_wo_wire_spectrum(
    weight: Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> WoWireSpectrum:
    if weight.ndim != 2:
        raise ValueError("Wo weight must be two-dimensional")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Wo spectrum work dtype must be float32 or float64")
    work = weight.detach().to(
        device=weight.device if device is None else device,
        dtype=dtype,
    )
    singular_values = torch.linalg.svdvals(work).float().cpu().contiguous()
    return WoWireSpectrum(
        singular_values=singular_values,
        in_features=int(weight.shape[1]),
        out_features=int(weight.shape[0]),
    )


def parse_wire_ranks(raw: str, maximum: int) -> tuple[int, ...]:
    values = tuple(sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()}))
    if not values or values[0] <= 0 or values[-1] > maximum:
        raise ValueError(f"wire ranks must lie in [1,{maximum}]")
    return values


def aggregate_wire_metrics(
    layers: Sequence[dict[str, Any]],
    ranks: Sequence[int],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for rank in ranks:
        metrics = [layer["ranks"][str(rank)] for layer in layers]
        result[str(rank)] = {
            "mean_energy_retained": sum(item["energy_retained"] for item in metrics)
            / len(metrics),
            "minimum_energy_retained": min(item["energy_retained"] for item in metrics),
            "mean_relative_frobenius_error": sum(
                item["relative_frobenius_error"] for item in metrics
            )
            / len(metrics),
            "maximum_relative_frobenius_error": max(
                item["relative_frobenius_error"] for item in metrics
            ),
            "wire_fraction": metrics[0]["wire_fraction"],
            "payload_reduction": metrics[0]["payload_reduction"],
            "factor_parameter_ratio": metrics[0]["factor_parameter_ratio"],
        }
    return result


__all__ = [
    "ActivationAwareWoFactors",
    "FullSecondMoment",
    "WoWireRankMetric",
    "WoWireSpectrum",
    "aggregate_wire_metrics",
    "fit_activation_aware_wo_factors",
    "parse_wire_ranks",
    "qwen35_wo_wire_spectrum",
]
