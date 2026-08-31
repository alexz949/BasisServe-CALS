"""Gate-family-aware polynomial primitives for activated output projections.

The supported Qwen3.5 paths share the form

    output = (other_factor * activation(gate)) @ Wo.T

where full attention uses a sigmoid gate and GDN uses a SiLU gate after an
exact headwise RMSNorm.  This module fits only the scalar gate activation in
the canonical communication wire.  It does not approximate attention, the
GDN recurrent state, or RMSNorm.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from basisserve.analysis.mlp_topk_chebyshev import chebyshev_sequence


GateFamily = Literal["sigmoid", "silu"]
GatePolynomialVariant = Literal[
    "unconstrained",
    "sigmoid_parity",
    "silu_parity",
]


def _matrix(name: str, value: Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def variants_for_family(family: GateFamily) -> tuple[GatePolynomialVariant, ...]:
    if family == "sigmoid":
        return ("unconstrained", "sigmoid_parity")
    if family == "silu":
        return ("unconstrained", "silu_parity")
    raise ValueError(f"unsupported gate family: {family}")


def constrained_feature_orders(
    degree: int,
    variant: GatePolynomialVariant,
) -> tuple[int, ...]:
    if degree < 0:
        raise ValueError("polynomial degree must be nonnegative")
    if variant == "unconstrained":
        return tuple(range(degree + 1))
    if variant == "sigmoid_parity":
        if degree < 1:
            raise ValueError("sigmoid parity requires degree at least one")
        return tuple(range(1, degree + 1, 2))
    if variant == "silu_parity":
        if degree < 2:
            raise ValueError("SiLU parity requires degree at least two")
        return tuple(range(2, degree + 1, 2))
    raise ValueError(f"unsupported polynomial variant: {variant}")


def _validate_family_variant(
    family: GateFamily,
    variant: GatePolynomialVariant,
) -> None:
    if variant == "unconstrained":
        return
    expected = "sigmoid_parity" if family == "sigmoid" else "silu_parity"
    if variant != expected:
        raise ValueError(f"variant {variant!r} is incompatible with {family!r}")


def exact_gate_activation(values: Tensor, family: GateFamily) -> Tensor:
    if family == "sigmoid":
        # Full-attention snapshots are BF16 and the checkpoint applies sigmoid
        # before promoting any value to FP32.  Preserve that rounding here.
        return torch.sigmoid(values).float()
    if family == "silu":
        return F.silu(values.float())
    raise ValueError(f"unsupported gate family: {family}")


@torch.no_grad()
def gate_chebyshev_polynomial(
    values: Tensor,
    coefficients: Tensor,
    *,
    radius: float,
    degree: int,
    family: GateFamily,
    variant: GatePolynomialVariant,
    clipped: bool,
) -> Tensor:
    """Evaluate a stable Chebyshev gate polynomial without monomial conversion."""

    if coefficients.ndim != 1:
        raise ValueError("Chebyshev coefficients must be a vector")
    _validate_family_variant(family, variant)
    sequence = chebyshev_sequence(
        values,
        radius=radius,
        degree=degree,
        clipped=clipped,
    )
    coefficient = coefficients.to(device=values.device, dtype=torch.float32)
    orders = constrained_feature_orders(degree, variant)
    if int(coefficient.numel()) != len(orders):
        raise ValueError("coefficient count differs from polynomial design")
    if variant == "unconstrained":
        result = torch.zeros_like(sequence[0])
        for theta, order in zip(coefficient, orders, strict=True):
            result.add_(sequence[order], alpha=float(theta))
        return result
    if family == "sigmoid":
        result = torch.full_like(sequence[0], 0.5)
        for beta, order in zip(coefficient, orders, strict=True):
            result.add_(sequence[order], alpha=float(beta))
        return result
    base = values.float()
    if clipped:
        base = base.clamp(min=-float(radius), max=float(radius))
    result = 0.5 * base
    for beta, order in zip(coefficient, orders, strict=True):
        feature_index = order // 2
        at_zero = -1.0 if feature_index % 2 else 1.0
        result.add_(sequence[order] - at_zero, alpha=float(beta))
    return result


@dataclass(frozen=True)
class GateWireComponents:
    """Reusable maximum-degree canonical-wire terms for one gate design."""

    target: Tensor
    constrained_base: Tensor
    terms: tuple[Tensor, ...]
    radius: float
    maximum_degree: int
    clipped: bool
    family: GateFamily


@dataclass(frozen=True)
class GateNormalEquations:
    gram: Tensor
    rhs: Tensor
    target_energy: float
    rows: int
    wire_rank: int
    degree: int
    family: GateFamily
    variant: GatePolynomialVariant
    radius: float
    clipped: bool


@torch.no_grad()
def build_gate_wire_components(
    gate_preactivation: Tensor,
    other_factor: Tensor,
    exact_post_gate: Tensor,
    encoder: Tensor,
    *,
    radius: float,
    maximum_degree: int,
    family: GateFamily,
    clipped: bool,
    chunk_size: int,
    device: torch.device,
) -> GateWireComponents:
    """Build all large wire terms once for reuse across degrees and ridges."""

    for name, value in (
        ("gate_preactivation", gate_preactivation),
        ("other_factor", other_factor),
        ("exact_post_gate", exact_post_gate),
        ("encoder", encoder),
    ):
        _matrix(name, value)
    if (
        gate_preactivation.shape != other_factor.shape
        or gate_preactivation.shape != exact_post_gate.shape
    ):
        raise ValueError("activated-Wo snapshot shapes differ")
    if int(encoder.shape[0]) != int(gate_preactivation.shape[1]):
        raise ValueError("encoder input width differs from activated-Wo width")
    if maximum_degree < 0 or chunk_size <= 0:
        raise ValueError("maximum degree and chunk size are invalid")
    if family not in ("sigmoid", "silu"):
        raise ValueError(f"unsupported gate family: {family}")
    rows = int(gate_preactivation.shape[0])
    rank = int(encoder.shape[1])
    encoder_device = encoder.to(device=device, dtype=torch.float32)
    target = torch.empty(rows, rank, device=device, dtype=torch.float32)
    base = torch.empty_like(target)
    terms = tuple(torch.empty_like(target) for _ in range(maximum_degree + 1))
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        gate = gate_preactivation[start:stop].to(device=device, dtype=torch.float32)
        other = other_factor[start:stop].to(device=device, dtype=torch.float32)
        exact = exact_post_gate[start:stop].to(device=device, dtype=torch.float32)
        sequence = chebyshev_sequence(
            gate,
            radius=radius,
            degree=maximum_degree,
            clipped=clipped,
        )
        target[start:stop] = exact @ encoder_device
        if family == "sigmoid":
            base[start:stop] = (0.5 * other) @ encoder_device
        else:
            base_gate = gate.clamp(-radius, radius) if clipped else gate
            base[start:stop] = (0.5 * base_gate * other) @ encoder_device
        for order, term in enumerate(sequence):
            terms[order][start:stop] = (term * other) @ encoder_device
    return GateWireComponents(
        target=target,
        constrained_base=base,
        terms=terms,
        radius=float(radius),
        maximum_degree=int(maximum_degree),
        clipped=bool(clipped),
        family=family,
    )


def _component_design(
    components: GateWireComponents,
    *,
    degree: int,
    variant: GatePolynomialVariant,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    if not 0 <= degree <= components.maximum_degree:
        raise ValueError("degree is outside the cached component bank")
    _validate_family_variant(components.family, variant)
    orders = constrained_feature_orders(degree, variant)
    if variant == "unconstrained":
        return torch.zeros_like(components.target), tuple(
            components.terms[order] for order in orders
        )
    if components.family == "sigmoid":
        return components.constrained_base, tuple(
            components.terms[order] for order in orders
        )
    features = []
    for order in orders:
        feature_index = order // 2
        at_zero = -1.0 if feature_index % 2 else 1.0
        features.append(components.terms[order] - at_zero * components.terms[0])
    return components.constrained_base, tuple(features)


@torch.no_grad()
def gate_normal_equations_from_components(
    components: GateWireComponents,
    *,
    degree: int,
    variant: GatePolynomialVariant,
    chunk_size: int = 256,
) -> GateNormalEquations:
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    base, features = _component_design(components, degree=degree, variant=variant)
    count = len(features)
    gram = torch.zeros(
        count,
        count,
        dtype=torch.float64,
        device=components.target.device,
    )
    rhs = torch.zeros(count, dtype=torch.float64, device=components.target.device)
    target_energy = torch.zeros(
        (), dtype=torch.float64, device=components.target.device
    )
    rows = int(components.target.shape[0])
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        target = (components.target[start:stop] - base[start:stop]).double()
        feature = torch.stack([value[start:stop] for value in features], dim=0).double()
        gram += torch.einsum("pnr,qnr->pq", feature, feature)
        rhs += torch.einsum("pnr,nr->p", feature, target)
        target_energy += target.square().sum()
    return GateNormalEquations(
        gram=gram.cpu(),
        rhs=rhs.cpu(),
        target_energy=float(target_energy),
        rows=rows,
        wire_rank=int(components.target.shape[1]),
        degree=int(degree),
        family=components.family,
        variant=variant,
        radius=components.radius,
        clipped=components.clipped,
    )


@torch.no_grad()
def solve_gate_ridge_coefficients(
    equations: GateNormalEquations,
    *,
    relative_lambda: float,
) -> tuple[Tensor, dict[str, Any]]:
    if not math.isfinite(float(relative_lambda)) or relative_lambda < 0.0:
        raise ValueError("relative ridge lambda must be finite and nonnegative")
    gram = 0.5 * (equations.gram.double() + equations.gram.double().transpose(0, 1))
    rhs = equations.rhs.double()
    scale = float(torch.diagonal(gram).mean())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Chebyshev Gram matrix has nonpositive scale")
    absolute_lambda = float(relative_lambda) * scale
    regularized = gram + absolute_lambda * torch.eye(
        int(gram.shape[0]), dtype=torch.float64
    )
    chol, info = torch.linalg.cholesky_ex(regularized)
    if int(info.max()) == 0:
        coefficients = torch.cholesky_solve(rhs[:, None], chol).squeeze(1)
        solver = "cholesky"
    else:
        coefficients = torch.linalg.lstsq(regularized, rhs[:, None]).solution.squeeze(1)
        solver = "lstsq_fallback"
    if not bool(torch.isfinite(coefficients).all()):
        raise RuntimeError("Chebyshev ridge solve returned non-finite coefficients")
    singular_values = torch.linalg.svdvals(regularized)
    return coefficients, {
        "relative_lambda": float(relative_lambda),
        "absolute_lambda": absolute_lambda,
        "gram_average_diagonal": scale,
        "regularized_condition": float(singular_values.max() / singular_values.min()),
        "solver": solver,
    }


@torch.no_grad()
def gate_latent_from_components(
    components: GateWireComponents,
    coefficients: Tensor,
    *,
    degree: int,
    variant: GatePolynomialVariant,
) -> Tensor:
    base, features = _component_design(components, degree=degree, variant=variant)
    if coefficients.ndim != 1 or int(coefficients.numel()) != len(features):
        raise ValueError("coefficient count differs from cached feature count")
    result = base.clone()
    coefficient = coefficients.to(device=result.device, dtype=torch.float32)
    for theta, feature in zip(coefficient, features, strict=True):
        result.add_(feature, alpha=float(theta))
    return result


@torch.no_grad()
def gate_component_wire_relative_mse(
    components: GateWireComponents,
    coefficients: Tensor,
    *,
    degree: int,
    variant: GatePolynomialVariant,
) -> float:
    prediction = gate_latent_from_components(
        components,
        coefficients,
        degree=degree,
        variant=variant,
    )
    residual = components.target.double() - prediction.double()
    return float(
        residual.square().sum()
        / components.target.double().square().sum().clamp_min(1.0e-300)
    )


@torch.no_grad()
def gate_polynomial_candidate_activation(
    gate_preactivation: Tensor,
    other_factor: Tensor,
    coefficients: Tensor,
    *,
    radius: float,
    degree: int,
    family: GateFamily,
    variant: GatePolynomialVariant,
    clipped: bool,
) -> Tensor:
    if gate_preactivation.shape != other_factor.shape:
        raise ValueError("gate and other-factor shapes differ")
    approximation = gate_chebyshev_polynomial(
        gate_preactivation,
        coefficients,
        radius=radius,
        degree=degree,
        family=family,
        variant=variant,
        clipped=clipped,
    )
    candidate = approximation * other_factor.float()
    if not bool(torch.isfinite(candidate).all()):
        raise FloatingPointError("Chebyshev candidate activation is non-finite")
    return candidate


@torch.no_grad()
def scalar_gate_diagnostics(
    gate_preactivation: Tensor,
    approximation: Tensor,
    *,
    radius: float,
    family: GateFamily,
    quantile_max_elements: int = 4_000_000,
) -> dict[str, Any]:
    if gate_preactivation.shape != approximation.shape:
        raise ValueError("scalar gate diagnostic tensors differ")
    if quantile_max_elements <= 0:
        raise ValueError("scalar diagnostic sample size must be positive")
    exact = exact_gate_activation(gate_preactivation, family)
    absolute_error = (exact - approximation.float()).abs()
    squared_error = absolute_error.square()
    inside = gate_preactivation.abs() <= float(radius)
    outside = ~inside
    flattened = absolute_error.flatten()
    if int(flattened.numel()) > quantile_max_elements:
        sample_indices = (
            torch.linspace(
                0,
                int(flattened.numel()) - 1,
                quantile_max_elements,
                device=flattened.device,
                dtype=torch.float64,
            )
            .round()
            .to(torch.long)
        )
        quantile_values = flattened.index_select(0, sample_indices)
        quantile_method = "deterministic_even_subsample"
    else:
        quantile_values = flattened
        quantile_method = "exact_all_elements"
    inside_count = int(inside.sum())
    outside_count = int(outside.sum())
    return {
        "gate_family": family,
        "scalar_gate_mse": float(
            torch.sum(squared_error, dtype=torch.float64) / squared_error.numel()
        ),
        "maximum_scalar_error": float(absolute_error.max()),
        "p99_scalar_error": float(torch.quantile(quantile_values.float(), 0.99)),
        "scalar_quantile_method": quantile_method,
        "scalar_quantile_elements": int(quantile_values.numel()),
        "inside_interval_mse": (
            float(torch.sum(squared_error[inside], dtype=torch.float64) / inside_count)
            if inside_count
            else None
        ),
        "outside_interval_mse": (
            float(
                torch.sum(squared_error[outside], dtype=torch.float64) / outside_count
            )
            if outside_count
            else None
        ),
        "out_of_range_fraction": outside_count / squared_error.numel(),
        "maximum_abs_gate_preactivation": float(gate_preactivation.abs().max()),
    }


__all__ = [
    "GateFamily",
    "GateNormalEquations",
    "GatePolynomialVariant",
    "GateWireComponents",
    "build_gate_wire_components",
    "constrained_feature_orders",
    "exact_gate_activation",
    "gate_chebyshev_polynomial",
    "gate_component_wire_relative_mse",
    "gate_latent_from_components",
    "gate_normal_equations_from_components",
    "gate_polynomial_candidate_activation",
    "scalar_gate_diagnostics",
    "solve_gate_ridge_coefficients",
    "variants_for_family",
]
