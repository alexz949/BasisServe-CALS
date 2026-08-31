"""Numerically stable design-matrix solvers for routed output decoders.

The routed decoder problem has many right-hand sides but a shared design:

    min_D ||X D - Y||_F^2.

Forming ``X.T @ X`` squares the condition number of ``X``.  This module keeps
the direct Householder-QR path explicit and provides a prior-centered spectral
ridge solve from the resulting triangular factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DirectQRSolve:
    decoder: torch.Tensor
    triangular_factor: torch.Tensor
    projected_target: torch.Tensor
    relative_stationarity: float
    relative_reconstruction_residual: float


@dataclass(frozen=True)
class SpectralRidgeWorkspace:
    left_vectors: torch.Tensor
    singular_values: torch.Tensor
    right_vectors_t: torch.Tensor
    projected_centered_residual: torch.Tensor
    center: torch.Tensor
    trace_scale: float


@dataclass(frozen=True)
class TraceNormalizedPriorRidgeWorkspace:
    """One full-layer decoder system with a fixed prior center.

    The workspace represents

        H D = B

    and caches the eigendecomposition needed to solve the one-parameter
    prior-centered family

        (H + alpha * trace(H) / dim(H) * I) (D - D0) = B - H D0.
    """

    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    projected_gradient: torch.Tensor
    center: torch.Tensor
    hessian: torch.Tensor
    rhs: torch.Tensor
    trace_scale: float
    numerical_negative_tolerance: float


@dataclass(frozen=True)
class TraceNormalizedPriorRidgeDiagnostics:
    alpha: float
    absolute_ridge: float
    trace_scale: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    hessian_condition: float
    regularized_condition: float
    minimum_filter: float
    median_filter: float
    maximum_filter: float
    relative_regularized_residual: float
    relative_movement_from_center: float


@torch.no_grad()
def full_layer_decoder_normal_equations(
    *,
    covariance: torch.Tensor,
    cross: torch.Tensor,
    encoders: torch.Tensor,
    head_to_kv_group: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack the fixed-encoder routed decoder system into ``H D = B``."""

    if covariance.ndim != 4:
        raise ValueError("covariance must have shape [H,H,d,d]")
    heads = covariance.shape[0]
    head_dim = covariance.shape[2]
    if tuple(covariance.shape) != (heads, heads, head_dim, head_dim):
        raise ValueError("covariance blocks must be square and head-aligned")
    if cross.ndim != 3 or tuple(cross.shape[:2]) != (heads, head_dim):
        raise ValueError("cross must have shape [H,d,D]")
    if encoders.ndim != 3 or encoders.shape[1] != head_dim:
        raise ValueError("encoders must have shape [G,d,r]")
    mapping = torch.as_tensor(
        head_to_kv_group,
        device=encoders.device,
        dtype=torch.long,
    )
    if tuple(mapping.shape) != (heads,):
        raise ValueError("head mapping must have shape [H]")
    if bool(torch.any(mapping < 0)) or bool(
        torch.any(mapping >= encoders.shape[0])
    ):
        raise ValueError("head mapping contains an invalid group")
    if (
        covariance.device != cross.device
        or covariance.device != encoders.device
        or covariance.dtype != cross.dtype
        or covariance.dtype != encoders.dtype
    ):
        raise ValueError("covariance, cross, and encoders must share device/dtype")
    by_head = encoders.index_select(0, mapping)
    blocks = torch.einsum(
        "hia,hkij,kjb->hkab",
        by_head,
        covariance,
        by_head,
    )
    rhs = torch.einsum("hia,hio->hao", by_head, cross)
    rank = encoders.shape[2]
    return (
        blocks.permute(0, 2, 1, 3).reshape(heads * rank, heads * rank),
        rhs.reshape(heads * rank, cross.shape[2]),
    )


def _validate_design(design: torch.Tensor, target: torch.Tensor) -> None:
    if design.ndim != 2 or target.ndim != 2:
        raise ValueError("design and target must be matrices")
    if design.shape[0] != target.shape[0]:
        raise ValueError("design and target row counts differ")
    if design.shape[0] < design.shape[1]:
        raise ValueError("direct QR requires at least as many rows as columns")
    if design.device != target.device or design.dtype != target.dtype:
        raise ValueError("design and target must share device and dtype")
    if not torch.isfinite(design).all() or not torch.isfinite(target).all():
        raise FloatingPointError("design and target must be finite")


@torch.no_grad()
def factor_design_qr(design: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the reduced Householder QR factorization of a tall design."""

    if design.ndim != 2 or design.shape[0] < design.shape[1]:
        raise ValueError("design must be a tall matrix")
    if not torch.isfinite(design).all():
        raise FloatingPointError("design must be finite")
    return torch.linalg.qr(design, mode="reduced")


@torch.no_grad()
def solve_reduced_qr(
    triangular_factor: torch.Tensor,
    projected_target: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Solve ``R D=Q.T Y`` after an externally managed direct QR."""

    if (
        triangular_factor.ndim != 2
        or triangular_factor.shape[0] != triangular_factor.shape[1]
    ):
        raise ValueError("triangular factor must be square")
    if (
        projected_target.ndim != 2
        or projected_target.shape[0] != triangular_factor.shape[0]
    ):
        raise ValueError("projected target does not match triangular factor")
    decoder = torch.linalg.solve_triangular(
        triangular_factor,
        projected_target,
        upper=True,
    )
    normal_residual = triangular_factor.transpose(0, 1) @ (
        triangular_factor @ decoder - projected_target
    )
    normal_scale = (
        torch.linalg.vector_norm(
            triangular_factor.transpose(0, 1) @ projected_target
        ).clamp_min(torch.finfo(triangular_factor.dtype).tiny)
    )
    return decoder, float(torch.linalg.vector_norm(normal_residual) / normal_scale)


@torch.no_grad()
def solve_direct_qr(
    design: torch.Tensor,
    target: torch.Tensor,
) -> DirectQRSolve:
    """Solve a full-column-rank least-squares problem without a Gram matrix."""

    _validate_design(design, target)
    q_factor, triangular = factor_design_qr(design)
    projected = q_factor.transpose(0, 1) @ target
    decoder, stationarity = solve_reduced_qr(triangular, projected)
    residual = design @ decoder - target
    target_scale = torch.linalg.vector_norm(target).clamp_min(
        torch.finfo(design.dtype).tiny
    )
    return DirectQRSolve(
        decoder=decoder,
        triangular_factor=triangular,
        projected_target=projected,
        relative_stationarity=stationarity,
        relative_reconstruction_residual=float(
            torch.linalg.vector_norm(residual) / target_scale
        ),
    )


@torch.no_grad()
def prepare_centered_spectral_ridge(
    *,
    triangular_factor: torch.Tensor,
    projected_target: torch.Tensor,
    center: torch.Tensor,
) -> SpectralRidgeWorkspace:
    """Prepare a stable centered-ridge workspace from a direct QR reduction."""

    if triangular_factor.ndim != 2:
        raise ValueError("triangular factor must be a matrix")
    width = triangular_factor.shape[0]
    if triangular_factor.shape[1] != width:
        raise ValueError("triangular factor must be square")
    if projected_target.shape[0] != width:
        raise ValueError("projected target width does not match QR factor")
    if tuple(center.shape) != tuple(projected_target.shape):
        raise ValueError("ridge center must match the decoder shape")
    if (
        triangular_factor.device != projected_target.device
        or triangular_factor.device != center.device
        or triangular_factor.dtype != projected_target.dtype
        or triangular_factor.dtype != center.dtype
    ):
        raise ValueError("ridge tensors must share device and dtype")
    left, singular_values, right_t = torch.linalg.svd(
        triangular_factor,
        full_matrices=False,
    )
    centered_residual = (
        projected_target - triangular_factor @ center
    )
    projected_residual = left.transpose(0, 1) @ centered_residual
    trace_scale = float(singular_values.square().mean())
    if not trace_scale > 0:
        raise FloatingPointError("QR design has zero trace scale")
    return SpectralRidgeWorkspace(
        left_vectors=left,
        singular_values=singular_values,
        right_vectors_t=right_t,
        projected_centered_residual=projected_residual,
        center=center,
        trace_scale=trace_scale,
    )


@torch.no_grad()
def solve_centered_spectral_ridge(
    workspace: SpectralRidgeWorkspace,
    *,
    relative_ridge: float,
) -> tuple[torch.Tensor, float]:
    """Solve ``||R D-C||^2 + lambda ||D-D0||^2`` spectrally."""

    if relative_ridge < 0:
        raise ValueError("relative ridge must be non-negative")
    absolute = float(relative_ridge) * workspace.trace_scale
    singular = workspace.singular_values
    denominator = singular.square() + absolute
    if relative_ridge == 0 and bool(torch.any(singular == 0)):
        raise torch.linalg.LinAlgError(
            "unregularized centered solve has a singular design"
        )
    filtered = (
        singular / denominator.clamp_min(torch.finfo(singular.dtype).tiny)
    ).unsqueeze(1) * workspace.projected_centered_residual
    delta = workspace.right_vectors_t.transpose(0, 1) @ filtered
    return workspace.center + delta, absolute


@torch.no_grad()
def prepare_trace_normalized_prior_ridge(
    *,
    hessian: torch.Tensor,
    rhs: torch.Tensor,
    center: torch.Tensor,
) -> TraceNormalizedPriorRidgeWorkspace:
    """Prepare the global-alpha decoder closure from ``H``, ``B``, and ``D0``.

    Tiny negative eigenvalues consistent with floating-point symmetrization
    are clipped to zero.  Material negative curvature is rejected.  This
    numerical PSD projection is not an algorithmic condition floor: positive
    eigenvalues are left unchanged, and an unregularized solve still rejects
    an exactly singular system.
    """

    if (
        hessian.ndim != 2
        or hessian.shape[0] != hessian.shape[1]
        or rhs.ndim != 2
        or center.ndim != 2
    ):
        raise ValueError("H, B, and D0 must be matrices with square H")
    width = hessian.shape[0]
    if rhs.shape[0] != width or tuple(center.shape) != tuple(rhs.shape):
        raise ValueError("decoder RHS and center must match the Hessian width")
    if (
        hessian.device != rhs.device
        or hessian.device != center.device
        or hessian.dtype != rhs.dtype
        or hessian.dtype != center.dtype
    ):
        raise ValueError("H, B, and D0 must share device and dtype")
    if not hessian.dtype.is_floating_point:
        raise ValueError("decoder system must use floating-point tensors")
    if (
        not torch.isfinite(hessian).all()
        or not torch.isfinite(rhs).all()
        or not torch.isfinite(center).all()
    ):
        raise FloatingPointError("decoder system must be finite")

    symmetric = 0.5 * (hessian + hessian.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    scale = float(eigenvalues.abs().max())
    if not scale > 0:
        raise FloatingPointError("decoder Hessian has zero spectral scale")
    tolerance = (
        1024.0 * torch.finfo(hessian.dtype).eps * max(scale, 1.0)
    )
    minimum = float(eigenvalues[0])
    if minimum < -tolerance:
        raise torch.linalg.LinAlgError(
            "decoder Hessian has material negative curvature: "
            f"minimum={minimum:.6e}, tolerance={tolerance:.6e}"
        )
    eigenvalues = eigenvalues.clamp_min(0)
    trace_scale = float(eigenvalues.mean())
    if not trace_scale > 0:
        raise FloatingPointError("decoder Hessian has zero trace scale")
    gradient = rhs - symmetric @ center
    projected_gradient = eigenvectors.transpose(0, 1) @ gradient
    return TraceNormalizedPriorRidgeWorkspace(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        projected_gradient=projected_gradient,
        center=center,
        hessian=symmetric,
        rhs=rhs,
        trace_scale=trace_scale,
        numerical_negative_tolerance=tolerance,
    )


@torch.no_grad()
def solve_trace_normalized_prior_ridge(
    workspace: TraceNormalizedPriorRidgeWorkspace,
    *,
    alpha: float,
) -> tuple[torch.Tensor, TraceNormalizedPriorRidgeDiagnostics]:
    """Solve the one-global-``alpha`` prior-centered decoder closure."""

    if not isinstance(alpha, (float, int)) or not torch.isfinite(
        torch.tensor(float(alpha))
    ):
        raise ValueError("alpha must be finite")
    alpha = float(alpha)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    eigenvalues = workspace.eigenvalues
    absolute = alpha * workspace.trace_scale
    denominator = eigenvalues + absolute
    if alpha == 0 and bool(torch.any(denominator == 0)):
        raise torch.linalg.LinAlgError(
            "unregularized decoder closure has a singular Hessian"
        )
    inverse = denominator.clamp_min(torch.finfo(eigenvalues.dtype).tiny).reciprocal()
    delta = workspace.eigenvectors @ (
        inverse.unsqueeze(1) * workspace.projected_gradient
    )
    decoder = workspace.center + delta

    filter_values = eigenvalues / denominator.clamp_min(
        torch.finfo(eigenvalues.dtype).tiny
    )
    if alpha == 0:
        filter_values = torch.ones_like(filter_values)
    residual = (
        workspace.hessian @ delta
        + absolute * delta
        - (workspace.rhs - workspace.hessian @ workspace.center)
    )
    residual_scale = torch.linalg.vector_norm(
        workspace.rhs - workspace.hessian @ workspace.center
    ).clamp_min(torch.finfo(residual.dtype).tiny)
    positive = eigenvalues[eigenvalues > 0]
    if positive.numel() == 0:
        hessian_condition = float("inf")
    else:
        hessian_condition = float(eigenvalues.max() / positive.min())
    regularized_minimum = float(eigenvalues.min()) + absolute
    regularized_condition = (
        float((eigenvalues.max() + absolute) / regularized_minimum)
        if regularized_minimum > 0
        else float("inf")
    )
    center_scale = torch.linalg.vector_norm(workspace.center).clamp_min(
        torch.finfo(workspace.center.dtype).tiny
    )
    diagnostics = TraceNormalizedPriorRidgeDiagnostics(
        alpha=alpha,
        absolute_ridge=absolute,
        trace_scale=workspace.trace_scale,
        minimum_eigenvalue=float(eigenvalues.min()),
        maximum_eigenvalue=float(eigenvalues.max()),
        hessian_condition=hessian_condition,
        regularized_condition=regularized_condition,
        minimum_filter=float(filter_values.min()),
        median_filter=float(filter_values.median()),
        maximum_filter=float(filter_values.max()),
        relative_regularized_residual=float(
            torch.linalg.vector_norm(residual) / residual_scale
        ),
        relative_movement_from_center=float(
            torch.linalg.vector_norm(delta) / center_scale
        ),
    )
    return decoder, diagnostics


@torch.no_grad()
def solve_fp32_gram_cholesky(
    design: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Reproduce the fragile FP32-Gram then FP64-Cholesky reference path."""

    _validate_design(design, target)
    design32 = design.to(torch.float32)
    target32 = target.to(torch.float32)
    gram = (design32.transpose(0, 1) @ design32).to(torch.float64)
    cross = (design32.transpose(0, 1) @ target32).to(torch.float64)
    gram = 0.5 * (gram + gram.transpose(0, 1))
    cholesky, info = torch.linalg.cholesky_ex(gram, check_errors=False)
    if int(info.max()) != 0:
        raise torch.linalg.LinAlgError("FP32 Gram matrix is not Cholesky positive")
    decoder = torch.cholesky_solve(cross, cholesky)
    diagonal = torch.diagonal(cholesky).abs()
    condition_estimate = float(
        (
            diagonal.max()
            / diagonal.min().clamp_min(torch.finfo(diagonal.dtype).tiny)
        ).square()
    )
    relative_residual = float(
        torch.linalg.vector_norm(gram @ decoder - cross)
        / torch.linalg.vector_norm(cross).clamp_min(
            torch.finfo(cross.dtype).tiny
        )
    )
    return decoder, condition_estimate, relative_residual


__all__ = [
    "DirectQRSolve",
    "SpectralRidgeWorkspace",
    "TraceNormalizedPriorRidgeDiagnostics",
    "TraceNormalizedPriorRidgeWorkspace",
    "factor_design_qr",
    "full_layer_decoder_normal_equations",
    "prepare_centered_spectral_ridge",
    "prepare_trace_normalized_prior_ridge",
    "solve_centered_spectral_ridge",
    "solve_direct_qr",
    "solve_fp32_gram_cholesky",
    "solve_reduced_qr",
    "solve_trace_normalized_prior_ridge",
]
