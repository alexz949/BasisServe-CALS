"""Matrix-free least-squares solvers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch


@dataclass(frozen=True)
class LSQRDiagnostics:
    iterations: int
    converged: bool
    relative_residual: float
    relative_normal_residual: float
    absolute_damping: float
    operator_norm: float
    condition_estimate: float
    solution_norm: float


def _symmetric_rotation(left: float, right: float) -> tuple[float, float, float]:
    radius = math.hypot(left, right)
    if radius == 0.0:
        return 1.0, 0.0, 0.0
    return left / radius, right / radius, radius


@torch.no_grad()
def least_squares_matrix(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rmatvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    solution_shape: tuple[int, ...],
    relative_tolerance: float,
    max_iterations: int,
    absolute_damping: float = 0.0,
) -> tuple[torch.Tensor, LSQRDiagnostics]:
    """Solve ``min_x ||A(x) - rhs||^2 + damping * ||x||^2`` with LSQR.

    ``matvec`` and ``rmatvec`` operate on tensors rather than flattened
    vectors.  They must be exact adjoints; LSQR never forms ``A.T @ A``.
    """

    if relative_tolerance <= 0 or max_iterations <= 0 or absolute_damping < 0:
        raise ValueError("invalid LSQR tolerance, iteration count, or damping")
    if not solution_shape or any(int(size) <= 0 for size in solution_shape):
        raise ValueError("LSQR solution shape must contain positive dimensions")

    solution = rhs.new_zeros(solution_shape)
    rhs_norm = float(torch.linalg.vector_norm(rhs))
    if rhs_norm == 0.0:
        return solution, LSQRDiagnostics(
            iterations=0,
            converged=True,
            relative_residual=0.0,
            relative_normal_residual=0.0,
            absolute_damping=float(absolute_damping),
            operator_norm=0.0,
            condition_estimate=0.0,
            solution_norm=0.0,
        )

    u = rhs / rhs_norm
    v = rmatvec(u)
    if tuple(v.shape) != solution_shape:
        raise ValueError("LSQR rmatvec returned the wrong solution shape")
    alpha = float(torch.linalg.vector_norm(v))
    if alpha == 0.0:
        return solution, LSQRDiagnostics(
            iterations=0,
            converged=True,
            relative_residual=1.0,
            relative_normal_residual=0.0,
            absolute_damping=float(absolute_damping),
            operator_norm=0.0,
            condition_estimate=0.0,
            solution_norm=0.0,
        )
    v = v / alpha
    w = v.clone()

    beta = rhs_norm
    rho_bar = alpha
    phi_bar = beta
    damping = math.sqrt(absolute_damping)
    operator_norm = 0.0
    inverse_rho_norm_squared = 0.0
    residual_norm = beta
    normal_residual_norm = alpha * beta
    condition_estimate = 0.0
    solution_norm = 0.0
    converged = False

    for iteration in range(1, max_iterations + 1):
        next_u = matvec(v)
        if tuple(next_u.shape) != tuple(rhs.shape):
            raise ValueError("LSQR matvec returned the wrong residual shape")
        next_u = next_u - alpha * u
        beta = float(torch.linalg.vector_norm(next_u))
        if beta:
            u = next_u / beta
            next_v = rmatvec(u) - beta * v
            alpha = float(torch.linalg.vector_norm(next_v))
            v = next_v / alpha if alpha else torch.zeros_like(v)
        else:
            u = torch.zeros_like(u)
            alpha = 0.0
            v = torch.zeros_like(v)

        operator_norm = math.sqrt(
            operator_norm**2 + alpha**2 + beta**2 + absolute_damping
        )
        damped_rho = math.hypot(rho_bar, damping)
        damping_cosine = rho_bar / damped_rho if damped_rho else 1.0
        damping_sine = damping / damped_rho if damped_rho else 0.0
        damped_residual = damping_sine * phi_bar
        phi_bar *= damping_cosine

        cosine, sine, rho = _symmetric_rotation(damped_rho, beta)
        theta = sine * alpha
        rho_bar = -cosine * alpha
        phi = cosine * phi_bar
        phi_bar *= sine
        tau = sine * phi

        if rho == 0.0:
            break
        inverse_rho = 1.0 / rho
        inverse_rho_norm_squared += float(
            torch.sum((inverse_rho * w).square())
        )
        solution.add_((phi * inverse_rho) * w)
        w = v - (theta * inverse_rho) * w

        residual_norm = math.hypot(phi_bar, damped_residual)
        normal_residual_norm = alpha * abs(tau)
        solution_norm = float(torch.linalg.vector_norm(solution))
        condition_estimate = operator_norm * math.sqrt(inverse_rho_norm_squared)
        relative_residual = residual_norm / rhs_norm
        relative_normal_residual = normal_residual_norm / max(
            operator_norm * residual_norm,
            torch.finfo(rhs.dtype).tiny,
        )
        compatible_tolerance = relative_tolerance * (
            1.0 + operator_norm * solution_norm / rhs_norm
        )
        if (
            relative_residual <= compatible_tolerance
            or relative_normal_residual <= relative_tolerance
        ):
            converged = True
            break

    return solution, LSQRDiagnostics(
        iterations=iteration,
        converged=converged,
        relative_residual=residual_norm / rhs_norm,
        relative_normal_residual=normal_residual_norm
        / max(operator_norm * residual_norm, torch.finfo(rhs.dtype).tiny),
        absolute_damping=float(absolute_damping),
        operator_norm=operator_norm,
        condition_estimate=condition_estimate,
        solution_norm=solution_norm,
    )


__all__ = ["LSQRDiagnostics", "least_squares_matrix"]
