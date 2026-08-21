"""Deterministic strong-RRQR bases for activation-aware low-rank maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class StrongRRQRDiagnostics:
    rank: int
    bound: float
    swaps: int
    converged: bool
    initial_max_rho: float
    final_max_rho: float
    max_interpolation_coefficient: float
    r11_condition: float
    relative_weighted_error: float
    optimal_svd_relative_error: float
    error_over_svd_optimal: float
    selected_columns: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["selected_columns"] = list(self.selected_columns)
        return payload


@dataclass(frozen=True)
class StrongRRQRResult:
    basis: torch.Tensor
    diagnostics: StrongRRQRDiagnostics


@dataclass(frozen=True)
class ActivationAwareStrongRRQRFactors:
    left: torch.Tensor
    right: torch.Tensor
    diagnostics: StrongRRQRDiagnostics


def _validate_inputs(matrix: torch.Tensor, rank: int, bound: float, max_swaps: int) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional, got shape {tuple(matrix.shape)}")
    if not 0 < rank <= min(matrix.shape):
        raise ValueError(f"rank must be in [1, {min(matrix.shape)}], got {rank}")
    if not math.isfinite(bound) or bound < 1.0:
        raise ValueError(f"bound must be finite and at least 1, got {bound}")
    if max_swaps < 0:
        raise ValueError(f"max_swaps must be nonnegative, got {max_swaps}")
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")


def _rrqr_state(
    matrix: np.ndarray,
    selected: np.ndarray,
    remaining: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, tuple[int, int] | None]:
    from scipy.linalg import qr, solve_triangular

    q, r11 = qr(matrix[:, selected], mode="economic", pivoting=False, check_finite=False)
    scale = max(float(np.linalg.norm(r11, ord=np.inf)), 1.0)
    diagonal = np.abs(np.diag(r11))
    tolerance = np.finfo(matrix.dtype).eps * max(matrix.shape) * scale
    if diagonal.size and float(diagonal.min()) <= tolerance:
        raise np.linalg.LinAlgError(
            "selected strong-RRQR columns are numerically rank deficient: "
            f"min_diag={float(diagonal.min()):.6e}, tolerance={tolerance:.6e}"
        )

    if remaining.size == 0:
        return q, r11, 0.0, 0.0, None

    trailing = matrix[:, remaining]
    r12 = q.T @ trailing
    interpolation = solve_triangular(r11, r12, lower=False, check_finite=False)
    r11_inverse = solve_triangular(
        r11,
        np.eye(r11.shape[0], dtype=matrix.dtype),
        lower=False,
        check_finite=False,
    )
    inverse_row_norms = np.linalg.norm(r11_inverse, axis=1)
    residual_squared = np.sum(trailing * trailing, axis=0) - np.sum(r12 * r12, axis=0)
    residual_norms = np.sqrt(np.maximum(residual_squared, 0.0))
    rho_squared = interpolation * interpolation
    rho_squared += (inverse_row_norms[:, None] * residual_norms[None, :]) ** 2
    flat_index = int(np.argmax(rho_squared))
    swap_index = tuple(int(value) for value in np.unravel_index(flat_index, rho_squared.shape))
    return (
        q,
        r11,
        float(np.sqrt(rho_squared[swap_index])),
        float(np.max(np.abs(interpolation))),
        swap_index,
    )


def strong_rrqr_basis(
    matrix: torch.Tensor,
    rank: int,
    *,
    bound: float = 2.0,
    max_swaps: int = 512,
) -> StrongRRQRResult:
    """Return a Gu-Eisenstat strong-RRQR column-space basis.

    The implementation starts from deterministic column-pivoted QR and applies
    the Gu-Eisenstat swap test. QR is recomputed after every swap; this is less
    clever than rank-one updates, but keeps the fixed-rank reference compact and
    auditable for the small 128-row value-head matrices used here.
    """

    _validate_inputs(matrix, rank, bound, max_swaps)
    from scipy.linalg import qr

    original_device = matrix.device
    work = np.asarray(matrix.detach().to(device="cpu", dtype=torch.float64).numpy(), order="F")
    _, _, pivots = qr(work, mode="economic", pivoting=True, check_finite=False)
    selected = np.array(pivots[:rank], dtype=np.int64, copy=True)
    remaining = np.array(pivots[rank:], dtype=np.int64, copy=True)

    q, r11, max_rho, max_coefficient, swap_index = _rrqr_state(work, selected, remaining)
    initial_max_rho = max_rho
    swaps = 0
    threshold = bound * (1.0 + 32.0 * np.finfo(work.dtype).eps)
    while swap_index is not None and max_rho > threshold and swaps < max_swaps:
        selected_index, remaining_index = swap_index
        selected[selected_index], remaining[remaining_index] = (
            remaining[remaining_index],
            selected[selected_index],
        )
        swaps += 1
        q, r11, max_rho, max_coefficient, swap_index = _rrqr_state(
            work,
            selected,
            remaining,
        )

    converged = bool(swap_index is None or max_rho <= threshold)
    if not converged:
        raise RuntimeError(
            "strong RRQR did not satisfy the requested bound before max_swaps: "
            f"rank={rank}, bound={bound}, max_swaps={max_swaps}, final_max_rho={max_rho:.6f}"
        )

    total_energy = float(np.sum(work * work))
    captured_energy = float(np.sum((q.T @ work) ** 2))
    relative_error = max(total_energy - captured_energy, 0.0) / max(total_energy, np.finfo(float).tiny)
    eigenvalues = np.linalg.eigvalsh(work @ work.T)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    optimal_residual = float(np.sum(eigenvalues[: max(work.shape[0] - rank, 0)]))
    optimal_relative_error = optimal_residual / max(total_energy, np.finfo(float).tiny)
    if optimal_relative_error <= np.finfo(float).eps:
        error_ratio = 1.0 if relative_error <= 128.0 * np.finfo(float).eps else math.inf
    else:
        error_ratio = relative_error / optimal_relative_error

    diagnostics = StrongRRQRDiagnostics(
        rank=rank,
        bound=float(bound),
        swaps=swaps,
        converged=converged,
        initial_max_rho=initial_max_rho,
        final_max_rho=max_rho,
        max_interpolation_coefficient=max_coefficient,
        r11_condition=float(np.linalg.cond(r11)),
        relative_weighted_error=relative_error,
        optimal_svd_relative_error=optimal_relative_error,
        error_over_svd_optimal=error_ratio,
        selected_columns=tuple(int(index) for index in selected),
    )
    basis = torch.from_numpy(np.array(q, copy=True)).to(device=original_device, dtype=torch.float32)
    return StrongRRQRResult(basis=basis, diagnostics=diagnostics)


def activation_aware_strong_rrqr_factors(
    weight: torch.Tensor,
    scaling_matrix: torch.Tensor,
    rank: int,
    *,
    bound: float = 2.0,
    max_swaps: int = 512,
) -> ActivationAwareStrongRRQRFactors:
    """Factor ``weight`` using a strong-RRQR basis of ``weight @ scale``."""

    if weight.ndim != 2:
        raise ValueError(f"weight must be two-dimensional, got shape {tuple(weight.shape)}")
    if scaling_matrix.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError(
            "scaling matrix shape mismatch: "
            f"expected {(weight.shape[1], weight.shape[1])}, got {tuple(scaling_matrix.shape)}"
        )

    original_dtype = weight.dtype
    work_weight = weight.detach().to(dtype=torch.float32)
    scale = scaling_matrix.detach().to(device=weight.device, dtype=torch.float32)
    weighted = work_weight @ scale
    rrqr = strong_rrqr_basis(weighted, rank, bound=bound, max_swaps=max_swaps)

    # Rotate and balance factors inside the selected RRQR subspace. This leaves
    # the approximation unchanged while matching PaLu's sqrt(sigma) SVD split.
    basis = rrqr.basis.to(device=weight.device, dtype=torch.float32)
    core = basis.T @ weighted
    core_left, singular_values, _ = torch.linalg.svd(core, full_matrices=False)
    balanced_basis = basis @ core_left
    singular_floor = torch.finfo(singular_values.dtype).eps * singular_values.max()
    if torch.any(singular_values <= singular_floor):
        raise RuntimeError(
            "strong-RRQR projected core is numerically rank deficient: "
            f"min_singular={float(singular_values.min()):.6e}, floor={float(singular_floor):.6e}"
        )
    sqrt_singular_values = torch.sqrt(singular_values)
    left = balanced_basis * sqrt_singular_values.unsqueeze(0)
    right = (balanced_basis.T @ work_weight) / sqrt_singular_values.unsqueeze(1)
    return ActivationAwareStrongRRQRFactors(
        left=left.to(dtype=original_dtype),
        right=right.to(dtype=original_dtype),
        diagnostics=rrqr.diagnostics,
    )
