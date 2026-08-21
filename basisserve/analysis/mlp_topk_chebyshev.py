"""Training-free MLP top-k and Chebyshev diagnostic primitives.

The functions in this module deliberately keep the communication wire in a
canonical gauge.  A decoder has orthonormal columns, so Euclidean latent error
equals decoded incremental error.  Final metrics are nevertheless measured
after decoding against the exact dense MLP output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F


ChebyshevVariant = Literal["unconstrained", "silu_parity"]
TopKScope = Literal["global_oracle", "tp_local"]
TopKScore = Literal["magnitude", "decoder_weighted"]


def _matrix(name: str, value: Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _quantiles(values: Tensor) -> dict[str, float | None]:
    if values.numel() == 0:
        return {name: None for name in ("mean", "median", "p90", "p95", "p99")}
    work = values.detach().to(device="cpu", dtype=torch.float64)
    qs = torch.quantile(
        work,
        torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64),
    )
    return {
        "mean": float(work.mean()),
        "median": float(qs[0]),
        "p90": float(qs[1]),
        "p95": float(qs[2]),
        "p99": float(qs[3]),
    }


@dataclass(frozen=True)
class CanonicalWire:
    """A row-major encoder and an orthonormal-column output decoder."""

    encoder: Tensor  # [intermediate, rank]
    decoder: Tensor  # [hidden, rank]
    rank: int
    source: str
    orthogonality_max_abs: float


@torch.no_grad()
def polar_retract_columns(
    basis: Tensor,
    *,
    tolerance: float = 2.0e-5,
) -> tuple[Tensor, dict[str, float]]:
    """Return the FP32 column-polar factor using an FP64 rank-side solve."""

    _matrix("basis", basis)
    rows, rank = map(int, basis.shape)
    if not 0 < rank <= rows:
        raise ValueError("basis must be tall with at least one column")
    work = basis.double()
    identity = torch.eye(rank, device=work.device, dtype=work.dtype)
    gram = work.transpose(0, 1) @ work
    gram = 0.5 * (gram + gram.transpose(0, 1))
    before = float((gram - identity).abs().max())
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    minimum_eigenvalue = float(eigenvalues[0])
    maximum_eigenvalue = float(eigenvalues[-1])
    rank_threshold = (
        torch.finfo(torch.float64).eps * max(rows, rank) * max(maximum_eigenvalue, 1.0)
    )
    if minimum_eigenvalue <= rank_threshold:
        raise ValueError(
            "basis Gram matrix is numerically rank deficient: "
            f"minimum eigenvalue {minimum_eigenvalue:.6g}"
        )
    inverse_sqrt = (
        eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
    ) @ eigenvectors.transpose(0, 1)
    polar_double = work @ inverse_sqrt
    retracted = polar_double.float().contiguous()
    retracted_double = retracted.double()
    retracted_gram = retracted_double.transpose(0, 1) @ retracted_double
    after = float((retracted_gram - identity).abs().max())
    if after > tolerance:
        raise RuntimeError(
            "FP64 polar retraction did not meet the FP32 storage tolerance: "
            f"max discrepancy {after:.6g}"
        )
    relative_change = float(
        torch.linalg.vector_norm(polar_double - work)
        / torch.linalg.vector_norm(work).clamp_min(1e-300)
    )
    return retracted, {
        "pre_retraction_orthogonality_max_abs_fp64": before,
        "post_retraction_orthogonality_max_abs_fp64": after,
        "gram_minimum_eigenvalue_fp64": minimum_eigenvalue,
        "gram_maximum_eigenvalue_fp64": maximum_eigenvalue,
        "relative_basis_change_fp64": relative_change,
    }


@torch.no_grad()
def canonical_wire_from_output_basis(
    down_weight: Tensor,
    output_basis: Tensor,
    *,
    source: str,
    tolerance: float = 2.0e-5,
) -> CanonicalWire:
    """Compile ``W_down`` through a fixed orthonormal output basis.

    ``down_weight`` follows ``torch.nn.Linear`` convention ``[hidden, m]``.
    The resulting approximation is

        C @ encoder @ decoder.T.
    """

    _matrix("down_weight", down_weight)
    _matrix("output_basis", output_basis)
    hidden, intermediate = map(int, down_weight.shape)
    if int(output_basis.shape[0]) != hidden:
        raise ValueError("output basis and down projection widths differ")
    rank = int(output_basis.shape[1])
    if not 0 < rank <= hidden:
        raise ValueError("output basis rank is invalid")
    decoder = output_basis.float().contiguous()
    gram = decoder.transpose(0, 1) @ decoder
    identity = torch.eye(rank, device=decoder.device, dtype=decoder.dtype)
    orthogonality = float((gram - identity).abs().max())
    if orthogonality > tolerance:
        raise ValueError(
            f"output basis is not orthonormal: max discrepancy {orthogonality:.6g}"
        )
    encoder = down_weight.float().transpose(0, 1) @ decoder
    if tuple(encoder.shape) != (intermediate, rank):
        raise AssertionError("canonical wire compiler returned an invalid shape")
    return CanonicalWire(
        encoder=encoder.contiguous(),
        decoder=decoder,
        rank=rank,
        source=str(source),
        orthogonality_max_abs=orthogonality,
    )


@torch.no_grad()
def exact_output_wire(down_weight: Tensor) -> CanonicalWire:
    """Represent the exact down projection in an identity output gauge."""

    _matrix("down_weight", down_weight)
    hidden = int(down_weight.shape[0])
    identity = torch.eye(hidden, device=down_weight.device, dtype=torch.float32)
    return canonical_wire_from_output_basis(
        down_weight,
        identity,
        source="exact_output_identity",
        tolerance=0.0,
    )


@torch.no_grad()
def decoder_aware_channel_weights(wire: CanonicalWire) -> Tensor:
    """Return gauge-invariant decoded row norms for cheap dynamic scoring."""

    # With an orthonormal decoder this equals
    # ||encoder[i, :] @ decoder.T||_2 without materializing the dense product.
    return torch.linalg.vector_norm(wire.encoder.float(), dim=1)


@dataclass(frozen=True)
class TopKDiagnostics:
    requested_keep_ratio: float
    realized_keep_ratio: float
    scope: TopKScope
    score: TopKScore
    tp_size: int
    block_size: int
    selected_blocks_per_partition: int
    blocks_per_partition: int
    retained_activation_energy: float


@torch.no_grad()
def dynamic_topk(
    activation: Tensor,
    keep_ratio: float,
    *,
    scope: TopKScope,
    tp_size: int,
    block_size: int = 1,
    score: TopKScore = "magnitude",
    channel_weights: Tensor | None = None,
) -> tuple[Tensor, TopKDiagnostics]:
    """Apply token-wise coordinate or contiguous-block top-k.

    ``global_oracle`` is explicitly non-deployable.  ``tp_local`` independently
    selects an equal number of blocks inside every contiguous TP shard.
    """

    _matrix("activation", activation)
    if not math.isfinite(float(keep_ratio)) or not 0.0 <= keep_ratio <= 1.0:
        raise ValueError("keep ratio must lie in [0, 1]")
    if tp_size <= 0 or block_size <= 0:
        raise ValueError("TP size and block size must be positive")
    if scope not in ("global_oracle", "tp_local"):
        raise ValueError(f"unsupported top-k scope: {scope}")
    if score not in ("magnitude", "decoder_weighted"):
        raise ValueError(f"unsupported top-k score: {score}")
    rows, width = map(int, activation.shape)
    partitions = 1 if scope == "global_oracle" else int(tp_size)
    if width % partitions:
        raise ValueError(
            "activation width is not divisible by the selection partitions"
        )
    partition_width = width // partitions
    if partition_width % block_size:
        raise ValueError("block size must divide every selection partition")
    blocks = partition_width // block_size
    if keep_ratio == 0.0:
        selected_blocks = 0
    elif keep_ratio == 1.0:
        selected_blocks = blocks
    else:
        selected_blocks = max(1, min(blocks, int(round(keep_ratio * blocks))))

    if score == "decoder_weighted":
        if channel_weights is None:
            raise ValueError("decoder-weighted top-k requires channel weights")
        if channel_weights.ndim != 1 or int(channel_weights.numel()) != width:
            raise ValueError("channel weights do not match activation width")
        if not bool(torch.isfinite(channel_weights).all()) or bool(
            (channel_weights < 0).any()
        ):
            raise ValueError("channel weights must be finite and nonnegative")
        weighted = activation.float() * channel_weights.to(
            device=activation.device,
            dtype=torch.float32,
        )
    else:
        weighted = activation.float()

    grouped = weighted.reshape(rows, partitions, blocks, block_size)
    block_scores = grouped.square().sum(dim=-1)
    block_mask = torch.zeros_like(block_scores, dtype=torch.bool)
    if selected_blocks == blocks:
        block_mask.fill_(True)
    elif selected_blocks:
        indices = torch.topk(
            block_scores,
            selected_blocks,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        block_mask.scatter_(-1, indices, True)
    coordinate_mask = (
        block_mask.unsqueeze(-1)
        .expand(rows, partitions, blocks, block_size)
        .reshape(rows, width)
    )
    sparse = activation * coordinate_mask.to(dtype=activation.dtype)
    dense_energy = float(activation.double().square().sum())
    sparse_energy = float(sparse.double().square().sum())
    return sparse.contiguous(), TopKDiagnostics(
        requested_keep_ratio=float(keep_ratio),
        realized_keep_ratio=(selected_blocks * block_size * partitions) / width,
        scope=scope,
        score=score,
        tp_size=int(tp_size),
        block_size=int(block_size),
        selected_blocks_per_partition=int(selected_blocks),
        blocks_per_partition=int(blocks),
        retained_activation_energy=sparse_energy / max(dense_energy, 1.0e-300),
    )


def chebyshev_sequence(
    values: Tensor,
    *,
    radius: float,
    degree: int,
    clipped: bool,
) -> tuple[Tensor, ...]:
    """Evaluate a Chebyshev sequence in FP32 without monomial conversion."""

    if not math.isfinite(float(radius)) or radius <= 0.0:
        raise ValueError("Chebyshev radius must be finite and positive")
    if degree < 0:
        raise ValueError("Chebyshev degree must be nonnegative")
    work = values.float()
    if clipped:
        work = work.clamp(min=-float(radius), max=float(radius))
    scaled = work / float(radius)
    sequence: list[Tensor] = [torch.ones_like(scaled)]
    if degree:
        sequence.append(scaled)
    for _ in range(2, degree + 1):
        sequence.append(2.0 * scaled * sequence[-1] - sequence[-2])
    return tuple(sequence)


def _parity_feature_count(degree: int) -> int:
    if degree < 2:
        raise ValueError("SiLU-parity Chebyshev requires degree at least 2")
    return degree // 2


def chebyshev_polynomial(
    values: Tensor,
    coefficients: Tensor,
    *,
    radius: float,
    degree: int,
    variant: ChebyshevVariant,
    clipped: bool,
) -> Tensor:
    """Evaluate one fitted Chebyshev SiLU approximation in FP32."""

    if coefficients.ndim != 1:
        raise ValueError("Chebyshev coefficients must be a vector")
    sequence = chebyshev_sequence(
        values,
        radius=radius,
        degree=degree,
        clipped=clipped,
    )
    coefficient = coefficients.to(device=values.device, dtype=torch.float32)
    if variant == "unconstrained":
        if int(coefficient.numel()) != degree + 1:
            raise ValueError("unconstrained coefficient count differs from degree")
        result = torch.zeros_like(sequence[0])
        for order, theta in enumerate(coefficient):
            result = result + theta * sequence[order]
        return result
    if variant != "silu_parity":
        raise ValueError(f"unsupported Chebyshev variant: {variant}")
    count = _parity_feature_count(degree)
    if int(coefficient.numel()) != count:
        raise ValueError("parity coefficient count differs from degree")
    base = values.float()
    if clipped:
        base = base.clamp(min=-float(radius), max=float(radius))
    result = 0.5 * base
    for feature_index, beta in enumerate(coefficient, start=1):
        order = 2 * feature_index
        at_zero = -1.0 if feature_index % 2 else 1.0
        result = result + beta * (sequence[order] - at_zero)
    return result


@dataclass(frozen=True)
class ChebyshevNormalEquations:
    gram: Tensor
    rhs: Tensor
    target_energy: float
    rows: int
    wire_rank: int
    degree: int
    variant: ChebyshevVariant
    radius: float
    clipped: bool


@dataclass(frozen=True)
class ChebyshevWireComponents:
    """Reusable maximum-degree wire terms for one split/radius/clip mode."""

    target: Tensor  # exact post-SwiGLU wire [N, r]
    linear_base: Tensor  # [0.5 * gate * up] wire [N, r]
    terms: tuple[Tensor, ...]  # [T_k(gate / radius) * up] wire
    radius: float
    maximum_degree: int
    clipped: bool


@torch.no_grad()
def build_chebyshev_wire_components(
    gate_preactivation: Tensor,
    up_activation: Tensor,
    exact_post_swiglu: Tensor,
    encoder: Tensor,
    *,
    radius: float,
    maximum_degree: int,
    clipped: bool,
    chunk_size: int,
    device: torch.device,
) -> ChebyshevWireComponents:
    """Build every Chebyshev wire term once for reuse across degrees/ridges."""

    for name, value in (
        ("gate_preactivation", gate_preactivation),
        ("up_activation", up_activation),
        ("exact_post_swiglu", exact_post_swiglu),
        ("encoder", encoder),
    ):
        _matrix(name, value)
    if (
        gate_preactivation.shape != up_activation.shape
        or gate_preactivation.shape != exact_post_swiglu.shape
    ):
        raise ValueError("MLP activation snapshot shapes differ")
    if int(encoder.shape[0]) != int(gate_preactivation.shape[1]):
        raise ValueError("encoder input width differs from MLP activation width")
    if maximum_degree < 0 or chunk_size <= 0:
        raise ValueError("maximum degree and chunk size are invalid")
    rows = int(gate_preactivation.shape[0])
    rank = int(encoder.shape[1])
    encoder_device = encoder.to(device=device, dtype=torch.float32)
    target = torch.empty(rows, rank, device=device, dtype=torch.float32)
    linear = torch.empty_like(target)
    terms = tuple(torch.empty_like(target) for _ in range(maximum_degree + 1))
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        gate = gate_preactivation[start:stop].to(device=device, dtype=torch.float32)
        up = up_activation[start:stop].to(device=device, dtype=torch.float32)
        exact = exact_post_swiglu[start:stop].to(device=device, dtype=torch.float32)
        sequence = chebyshev_sequence(
            gate,
            radius=radius,
            degree=maximum_degree,
            clipped=clipped,
        )
        base_gate = gate.clamp(-radius, radius) if clipped else gate
        target[start:stop] = exact @ encoder_device
        linear[start:stop] = (0.5 * base_gate * up) @ encoder_device
        for order, term in enumerate(sequence):
            terms[order][start:stop] = (term * up) @ encoder_device
    return ChebyshevWireComponents(
        target=target,
        linear_base=linear,
        terms=terms,
        radius=float(radius),
        maximum_degree=int(maximum_degree),
        clipped=bool(clipped),
    )


def _component_design(
    components: ChebyshevWireComponents,
    *,
    degree: int,
    variant: ChebyshevVariant,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    if not 0 <= degree <= components.maximum_degree:
        raise ValueError("degree is outside the cached Chebyshev component bank")
    if variant == "unconstrained":
        return torch.zeros_like(components.target), components.terms[: degree + 1]
    if variant != "silu_parity":
        raise ValueError(f"unsupported Chebyshev variant: {variant}")
    count = _parity_feature_count(degree)
    features = []
    for feature_index in range(1, count + 1):
        at_zero = -1.0 if feature_index % 2 else 1.0
        features.append(
            components.terms[2 * feature_index] - at_zero * components.terms[0]
        )
    return components.linear_base, tuple(features)


@torch.no_grad()
def normal_equations_from_components(
    components: ChebyshevWireComponents,
    *,
    degree: int,
    variant: ChebyshevVariant,
    chunk_size: int = 256,
) -> ChebyshevNormalEquations:
    """Accumulate FP64 normal equations from a reusable wire-term bank."""

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
    return ChebyshevNormalEquations(
        gram=gram.cpu(),
        rhs=rhs.cpu(),
        target_energy=float(target_energy),
        rows=rows,
        wire_rank=int(components.target.shape[1]),
        degree=int(degree),
        variant=variant,
        radius=components.radius,
        clipped=components.clipped,
    )


@torch.no_grad()
def latent_from_components(
    components: ChebyshevWireComponents,
    coefficients: Tensor,
    *,
    degree: int,
    variant: ChebyshevVariant,
) -> Tensor:
    """Evaluate fitted coefficients without repeating the large MLP encoder GEMMs."""

    base, features = _component_design(components, degree=degree, variant=variant)
    if coefficients.ndim != 1 or int(coefficients.numel()) != len(features):
        raise ValueError("coefficient count differs from cached feature count")
    result = base.clone()
    coefficient = coefficients.to(device=result.device, dtype=torch.float32)
    for theta, feature in zip(coefficient, features, strict=True):
        result.add_(feature, alpha=float(theta))
    return result


@torch.no_grad()
def component_wire_relative_mse(
    components: ChebyshevWireComponents,
    coefficients: Tensor,
    *,
    degree: int,
    variant: ChebyshevVariant,
) -> float:
    prediction = latent_from_components(
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


def _chebyshev_feature_activations(
    gate_preactivation: Tensor,
    up_activation: Tensor,
    *,
    radius: float,
    degree: int,
    variant: ChebyshevVariant,
    clipped: bool,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    if gate_preactivation.shape != up_activation.shape:
        raise ValueError("gate and up activation shapes differ")
    sequence = chebyshev_sequence(
        gate_preactivation,
        radius=radius,
        degree=degree,
        clipped=clipped,
    )
    up = up_activation.float()
    if variant == "unconstrained":
        return torch.zeros_like(up), tuple(term * up for term in sequence)
    if variant != "silu_parity":
        raise ValueError(f"unsupported Chebyshev variant: {variant}")
    count = _parity_feature_count(degree)
    base_gate = gate_preactivation.float()
    if clipped:
        base_gate = base_gate.clamp(min=-float(radius), max=float(radius))
    base = 0.5 * base_gate * up
    features = []
    for feature_index in range(1, count + 1):
        at_zero = -1.0 if feature_index % 2 else 1.0
        features.append((sequence[2 * feature_index] - at_zero) * up)
    return base, tuple(features)


@torch.no_grad()
def accumulate_chebyshev_normal_equations(
    gate_preactivation: Tensor,
    up_activation: Tensor,
    exact_post_swiglu: Tensor,
    encoder: Tensor,
    *,
    radius: float,
    degree: int,
    variant: ChebyshevVariant,
    clipped: bool,
    chunk_size: int,
    device: torch.device,
) -> ChebyshevNormalEquations:
    """Accumulate a small FP64 wire-space regression system by streaming."""

    for name, value in (
        ("gate_preactivation", gate_preactivation),
        ("up_activation", up_activation),
        ("exact_post_swiglu", exact_post_swiglu),
        ("encoder", encoder),
    ):
        _matrix(name, value)
    if (
        gate_preactivation.shape != up_activation.shape
        or gate_preactivation.shape != exact_post_swiglu.shape
    ):
        raise ValueError("MLP activation snapshot shapes differ")
    if int(encoder.shape[0]) != int(gate_preactivation.shape[1]):
        raise ValueError("encoder input width differs from MLP activation width")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    feature_count = (
        degree + 1 if variant == "unconstrained" else _parity_feature_count(degree)
    )
    gram = torch.zeros(feature_count, feature_count, dtype=torch.float64, device=device)
    rhs = torch.zeros(feature_count, dtype=torch.float64, device=device)
    target_energy = torch.zeros((), dtype=torch.float64, device=device)
    encoder_device = encoder.to(device=device, dtype=torch.float32)
    rows = int(gate_preactivation.shape[0])
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        gate = gate_preactivation[start:stop].to(device=device, dtype=torch.float32)
        up = up_activation[start:stop].to(device=device, dtype=torch.float32)
        exact = exact_post_swiglu[start:stop].to(device=device, dtype=torch.float32)
        base, features = _chebyshev_feature_activations(
            gate,
            up,
            radius=radius,
            degree=degree,
            variant=variant,
            clipped=clipped,
        )
        target = exact @ encoder_device - base @ encoder_device
        feature_wire = torch.stack(
            [feature @ encoder_device for feature in features],
            dim=0,
        ).double()
        target64 = target.double()
        gram += torch.einsum("pnr,qnr->pq", feature_wire, feature_wire)
        rhs += torch.einsum("pnr,nr->p", feature_wire, target64)
        target_energy += target64.square().sum()
    return ChebyshevNormalEquations(
        gram=gram.cpu(),
        rhs=rhs.cpu(),
        target_energy=float(target_energy),
        rows=rows,
        wire_rank=int(encoder.shape[1]),
        degree=int(degree),
        variant=variant,
        radius=float(radius),
        clipped=bool(clipped),
    )


@torch.no_grad()
def solve_ridge_coefficients(
    equations: ChebyshevNormalEquations,
    *,
    relative_lambda: float,
) -> tuple[Tensor, dict[str, Any]]:
    """Solve one tiny normalized ridge system without an explicit inverse."""

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
def polynomial_candidate_activation(
    gate_preactivation: Tensor,
    up_activation: Tensor,
    coefficients: Tensor,
    *,
    radius: float,
    degree: int,
    variant: ChebyshevVariant,
    clipped: bool,
) -> Tensor:
    if gate_preactivation.shape != up_activation.shape:
        raise ValueError("gate and up activation shapes differ")
    approximation = chebyshev_polynomial(
        gate_preactivation,
        coefficients,
        radius=radius,
        degree=degree,
        variant=variant,
        clipped=clipped,
    )
    candidate = approximation * up_activation.float()
    if not bool(torch.isfinite(candidate).all()):
        raise FloatingPointError("Chebyshev candidate activation is non-finite")
    return candidate


@torch.no_grad()
def canonical_output_metrics(
    teacher_output: Tensor,
    dense_latent: Tensor,
    candidate_latent: Tensor,
    decoder: Tensor,
    *,
    teacher_projection: Tensor | None = None,
    dense_activation: Tensor | None = None,
    candidate_activation: Tensor | None = None,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Measure canonical-wire incremental error and exact-output total error."""

    for name, value in (
        ("teacher_output", teacher_output),
        ("dense_latent", dense_latent),
        ("candidate_latent", candidate_latent),
        ("decoder", decoder),
    ):
        _matrix(name, value)
    if dense_latent.shape != candidate_latent.shape:
        raise ValueError("dense and candidate latent shapes differ")
    if int(teacher_output.shape[0]) != int(dense_latent.shape[0]):
        raise ValueError("teacher and latent row counts differ")
    if tuple(decoder.shape) != (
        int(teacher_output.shape[1]),
        int(dense_latent.shape[1]),
    ):
        raise ValueError("decoder shape is incompatible with outputs and latents")
    if teacher_projection is not None:
        _matrix("teacher_projection", teacher_projection)
        if teacher_projection.shape != dense_latent.shape:
            raise ValueError("teacher projection and latent shapes differ")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if (dense_activation is None) != (candidate_activation is None):
        raise ValueError("dense and candidate activations must be provided together")
    if dense_activation is not None and (
        dense_activation.shape != candidate_activation.shape  # type: ignore[union-attr]
        or int(dense_activation.shape[0]) != int(teacher_output.shape[0])
    ):
        raise ValueError("activation metric tensors have incompatible shapes")

    rank = int(decoder.shape[1])
    decoder32 = decoder.float()
    gram = decoder32.transpose(0, 1) @ decoder32
    identity = torch.eye(rank, device=gram.device, dtype=gram.dtype)
    decoder_orthogonality = float((gram - identity).abs().max())
    totals = {
        name: 0.0
        for name in (
            "teacher",
            "dense_output",
            "dense_wire",
            "baseline",
            "incremental_output",
            "incremental_wire",
            "total",
            "cross",
            "prediction",
            "prediction_cross",
            "dense_activation",
            "candidate_activation",
        )
    }
    per_token_total: list[Tensor] = []
    per_token_incremental: list[Tensor] = []
    near_zero_teacher = 0
    rows = int(teacher_output.shape[0])
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        teacher = teacher_output[start:stop].float()
        dense_z = dense_latent[start:stop].float()
        candidate_z = candidate_latent[start:stop].float()
        projected_teacher = (
            teacher @ decoder32
            if teacher_projection is None
            else teacher_projection[start:stop].float()
        )
        teacher64 = teacher.double()
        projected64 = projected_teacher.double()
        dense64 = dense_z.double()
        candidate64 = candidate_z.double()
        incremental64 = dense64 - candidate64
        teacher_per = teacher64.square().sum(dim=1)
        dense_per = dense64.square().sum(dim=1)
        incremental_per = incremental64.square().sum(dim=1)
        baseline_per = (
            teacher_per + dense_per - 2.0 * (projected64 * dense64).sum(dim=1)
        ).clamp_min(0.0)
        prediction_per = candidate64.square().sum(dim=1)
        prediction_cross_per = (projected64 * candidate64).sum(dim=1)
        total_per = (
            teacher_per + prediction_per - 2.0 * prediction_cross_per
        ).clamp_min(0.0)
        cross_per = ((projected64 - dense64) * incremental64).sum(dim=1)
        totals["teacher"] += float(teacher_per.sum())
        totals["dense_output"] += float(dense_per.sum())
        totals["dense_wire"] += float(dense_per.sum())
        totals["baseline"] += float(baseline_per.sum())
        totals["incremental_output"] += float(incremental_per.sum())
        totals["incremental_wire"] += float(incremental_per.sum())
        totals["total"] += float(total_per.sum())
        totals["cross"] += float(cross_per.sum())
        totals["prediction"] += float(prediction_per.sum())
        totals["prediction_cross"] += float(prediction_cross_per.sum())
        valid_teacher = teacher_per > 1.0e-24
        near_zero_teacher += int((~valid_teacher).sum())
        per_token_total.append(
            (total_per[valid_teacher] / teacher_per[valid_teacher]).cpu()
        )
        valid_dense = dense_per > 1.0e-24
        per_token_incremental.append(
            (incremental_per[valid_dense] / dense_per[valid_dense]).cpu()
        )
        if dense_activation is not None:
            totals["dense_activation"] += float(
                dense_activation[start:stop].double().square().sum()
            )
            totals["candidate_activation"] += float(
                candidate_activation[start:stop].double().square().sum()  # type: ignore[index]
            )

    teacher_energy = totals["teacher"]
    dense_energy = totals["dense_output"]
    dense_wire_energy = totals["dense_wire"]
    if min(teacher_energy, dense_energy, dense_wire_energy) <= 0.0:
        raise ValueError("teacher or dense canonical-wire energy is zero")
    baseline = totals["baseline"] / teacher_energy
    incremental = totals["incremental_output"] / teacher_energy
    total = totals["total"] / teacher_energy
    cross = 2.0 * totals["cross"] / teacher_energy
    prediction_energy = totals["prediction"]
    cosine_denominator = math.sqrt(teacher_energy * prediction_energy)
    result: dict[str, Any] = {
        "baseline_output_relative_mse": baseline,
        "candidate_incremental_output_relative_mse": incremental,
        "candidate_incremental_relative_to_dense_output": totals["incremental_output"]
        / dense_energy,
        "wire_relative_mse": totals["incremental_wire"] / dense_wire_energy,
        "total_output_relative_mse": total,
        "baseline_candidate_cross_term": cross,
        "error_identity_sum": baseline + incremental + cross,
        "error_identity_absolute_discrepancy": abs(
            total - (baseline + incremental + cross)
        ),
        "decoder_isometry_relative_discrepancy": abs(
            totals["incremental_output"] - totals["incremental_wire"]
        )
        / max(totals["incremental_wire"], 1.0e-30),
        "decoder_orthogonality_max_abs": decoder_orthogonality,
        "normalized_cross": (
            totals["prediction_cross"] / cosine_denominator
            if cosine_denominator > 0.0
            else 0.0
        ),
        "prediction_teacher_energy": prediction_energy / teacher_energy,
        "per_token_total_relative_squared_error": _quantiles(
            torch.cat(per_token_total) if per_token_total else torch.empty(0)
        ),
        "per_token_incremental_relative_squared_error": _quantiles(
            torch.cat(per_token_incremental)
            if per_token_incremental
            else torch.empty(0)
        ),
        "near_zero_teacher_tokens": near_zero_teacher,
        "teacher_energy": teacher_energy,
        "dense_output_energy": dense_energy,
        "dense_wire_energy": dense_wire_energy,
    }
    if dense_activation is not None:
        result["retained_activation_energy"] = totals["candidate_activation"] / max(
            totals["dense_activation"], 1.0e-300
        )
    return result


@torch.no_grad()
def scalar_silu_diagnostics(
    gate_preactivation: Tensor,
    approximation: Tensor,
    *,
    radius: float,
    quantile_max_elements: int = 4_000_000,
) -> dict[str, Any]:
    if gate_preactivation.shape != approximation.shape:
        raise ValueError("scalar SiLU diagnostic tensors differ")
    if quantile_max_elements <= 0:
        raise ValueError("scalar diagnostic sample size must be positive")
    exact = F.silu(gate_preactivation.float())
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
        "scalar_silu_mse": float(
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
    "CanonicalWire",
    "ChebyshevNormalEquations",
    "ChebyshevWireComponents",
    "ChebyshevVariant",
    "TopKDiagnostics",
    "TopKScope",
    "TopKScore",
    "accumulate_chebyshev_normal_equations",
    "build_chebyshev_wire_components",
    "canonical_output_metrics",
    "canonical_wire_from_output_basis",
    "chebyshev_polynomial",
    "chebyshev_sequence",
    "decoder_aware_channel_weights",
    "dynamic_topk",
    "exact_output_wire",
    "component_wire_relative_mse",
    "latent_from_components",
    "normal_equations_from_components",
    "polynomial_candidate_activation",
    "scalar_silu_diagnostics",
    "solve_ridge_coefficients",
]
