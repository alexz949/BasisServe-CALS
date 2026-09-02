"""Joint Store80/Route32/Payload80 optimization for GQA attention.

The first 32 latent coordinates are shared by routing and payload.  The
remaining 48 coordinates are payload-only.  This front-loaded coordinate
system is used during fitting and deployment, so the optimizer has no latent
selector or selector gauge.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Sequence

import torch

from basisserve.calibration.gqa_joint_routing_payload_s80_stats import (
    S80DirectResidualData,
    S80PayloadStatistics,
    S80RoutingStatistics,
)
from basisserve.core.iterative_least_squares import (
    LSQRDiagnostics,
    least_squares_matrix,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    S80CompactSoftmaxFisherRouting,
    compact_softmax_fisher_adapter_system,
    compact_softmax_fisher_encoder_diagonal,
    compact_softmax_fisher_loss,
    compact_softmax_fisher_roots,
)
from basisserve.core.gqa_routed_ov_joint import (
    LinearSolveDiagnostics,
    RoutedOVQuadratic,
    conjugate_gradient_matrix,
    encoder_hessian_vector_product,
    evaluate_quadratic,
    quadratic_from_target,
    solve_free_decoder,
)


@dataclass(frozen=True)
class S80Layout:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    value_dim: int
    key_dim: int
    joint_rank: int
    routing_rank: int

    @property
    def joint_dim(self) -> int:
        return self.value_dim + self.key_dim

    @property
    def query_heads_per_kv_group(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def payload_only_rank(self) -> int:
        return self.joint_rank - self.routing_rank

    def head_to_kv_group(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        return (
            torch.arange(
                self.num_attention_heads,
                device=device,
                dtype=torch.long,
            )
            // self.query_heads_per_kv_group
        )


@dataclass(frozen=True)
class S80Factors:
    routing_payload_encoders: torch.Tensor
    payload_only_encoders: torch.Tensor
    payload_decoders: torch.Tensor
    routing_query_factors: torch.Tensor

    @property
    def joint_encoders(self) -> torch.Tensor:
        return torch.cat(
            (self.routing_payload_encoders, self.payload_only_encoders),
            dim=-1,
        )

    def validate(self, layout: S80Layout) -> None:
        return None


@dataclass(frozen=True)
class S80Objective:
    payload: RoutedOVQuadratic
    payload_target: torch.Tensor
    routing: S80RoutingStatistics
    payload_normalizer: float
    routing_normalizer: float
    routing_weight: float
    normalization_epsilon: float

    def validate(self, layout: S80Layout) -> None:
        return None


@dataclass(frozen=True)
class S80Loss:
    payload: float
    routing: float
    normalized_payload: float
    normalized_routing: float
    total: float


@dataclass(frozen=True)
class S80LSQRStep:
    index: int
    loss_before: float
    loss_after: float
    lsqr: LSQRDiagnostics


@dataclass(frozen=True)
class S80SweepDiagnostics:
    sweep: int
    fit_before: S80Loss
    fit_after_decoder: S80Loss
    fit_after_encoders: S80Loss
    validation_after_encoders: S80Loss | None
    decoder: LinearSolveDiagnostics
    encoder_steps: tuple[S80LSQRStep, ...]
    maximum_gauge_payload_error: float
    maximum_gauge_routing_error: float
    wall_time_seconds: float


@dataclass(frozen=True)
class S80FitDiagnostics:
    sweeps: tuple[S80SweepDiagnostics, ...]
    loss_after_final_decoder: S80Loss
    validation_after_final_decoder: S80Loss | None
    loss_after_routing_queries: S80Loss
    validation_after_routing_queries: S80Loss | None
    final_decoder: LinearSolveDiagnostics
    routing_query_steps: tuple[S80LSQRStep, ...]
    wall_time_seconds: float


@dataclass(frozen=True)
class S80FitResult:
    factors: S80Factors
    initial_loss: S80Loss
    final_loss: S80Loss
    diagnostics: S80FitDiagnostics
    final_routing_query_steps: tuple[S80LSQRStep, ...]
    routing_metric: str
    routing_normalizer: float


@dataclass(frozen=True)
class FoldedS80Factors:
    v_joint_proj_weight: torch.Tensor
    v_joint_proj_bias: torch.Tensor | None
    k_joint_encoder: torch.Tensor
    routing_query_factor: torch.Tensor
    o_decoder_weight: torch.Tensor
    o_decoder_bias: torch.Tensor | None
    head_to_kv_group: torch.Tensor
    joint_encoder: torch.Tensor | None = None


@dataclass(frozen=True)
class _StackedRoutingMoments:
    query_grams: torch.Tensor
    joint_grams: torch.Tensor
    target_score_energy: torch.Tensor


def joint_payload_target(
    dense_head_output_blocks: torch.Tensor,
    *,
    key_dim: int,
) -> torch.Tensor:
    """Embed dense ``O_h`` blocks as ``[O_h; 0_K]``."""

    zeros = dense_head_output_blocks.new_zeros(
        dense_head_output_blocks.shape[0],
        key_dim,
        dense_head_output_blocks.shape[2],
    )
    return torch.cat((dense_head_output_blocks, zeros), dim=1)


def dense_o_weight_to_head_blocks(
    dense_o_proj_weight: torch.Tensor,
    layout: S80Layout,
) -> torch.Tensor:
    """Convert PyTorch ``o_proj`` weight to ``[Hq, dV, hidden]`` blocks."""

    return dense_o_proj_weight.mT.reshape(
        layout.num_attention_heads,
        layout.value_dim,
        layout.hidden_size,
    ).contiguous()


def s80_objective_from_statistics(
    *,
    layout: S80Layout,
    payload_statistics: S80PayloadStatistics,
    routing_statistics: S80RoutingStatistics,
    dense_o_proj_weight: torch.Tensor,
    routing_weight: float,
    normalization_epsilon: float = 1e-30,
) -> S80Objective:
    payload_statistics.validate()
    routing_statistics.validate()
    target = joint_payload_target(
        dense_o_weight_to_head_blocks(dense_o_proj_weight, layout),
        key_dim=layout.key_dim,
    ).to(payload_statistics.covariance_blocks)
    payload = quadratic_from_target(
        covariance=payload_statistics.covariance_blocks,
        target=target,
        name="s80_payload",
        trace_normalize=False,
    )
    result = S80Objective(
        payload=payload,
        payload_target=target,
        routing=routing_statistics,
        payload_normalizer=max(
            float(payload_statistics.dense_output_energy), normalization_epsilon
        ),
        routing_normalizer=max(
            float(routing_statistics.target_score_energy), normalization_epsilon
        ),
        routing_weight=float(routing_weight),
        normalization_epsilon=float(normalization_epsilon),
    )
    result.validate(layout)
    return result


def _selector_transpose(layout: S80Layout, reference: torch.Tensor) -> torch.Tensor:
    selector = reference.new_zeros(layout.key_dim, layout.joint_dim)
    selector[:, layout.value_dim :] = torch.eye(
        layout.key_dim,
        device=reference.device,
        dtype=reference.dtype,
    )
    return selector


def _stack_routing_moments(
    statistics: S80RoutingStatistics,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> _StackedRoutingMoments:
    return _StackedRoutingMoments(
        query_grams=torch.stack(
            [
                shard.query_grams.to(device=device, dtype=dtype)
                for shard in statistics.shards
            ]
        ),
        joint_grams=torch.stack(
            [
                shard.joint_grams.to(device=device, dtype=dtype)
                for shard in statistics.shards
            ]
        ),
        target_score_energy=torch.as_tensor(
            statistics.target_score_energy,
            device=device,
            dtype=dtype,
        ),
    )


def compose_routing_query_maps(
    *,
    layout: S80Layout,
    routing_query_factors: torch.Tensor,
) -> torch.Tensor:
    """Return the front-loaded per-head Route32 query maps."""

    return routing_query_factors


def _batched_routing_scores(
    left: torch.Tensor,
    joint_rows: torch.Tensor,
) -> torch.Tensor:
    return torch.bmm(left.unsqueeze(1), joint_rows.mT).squeeze(1)


def _batched_weighted_joint(
    scores: torch.Tensor,
    joint_rows: torch.Tensor,
) -> torch.Tensor:
    return torch.bmm(scores.unsqueeze(1), joint_rows).squeeze(1)


def routing_loss_tensor(
    *,
    layout: S80Layout,
    routing_payload_encoders: torch.Tensor,
    routing_query_factors: torch.Tensor,
    routing_moments: _StackedRoutingMoments,
) -> torch.Tensor:
    selector = _selector_transpose(layout, routing_payload_encoders)
    mapping = layout.head_to_kv_group(device=routing_payload_encoders.device)
    loss = routing_payload_encoders.new_zeros(())
    for head, group in enumerate(mapping.tolist()):
        loss.add_(
            _routing_head_loss_tensor(
                selector=selector,
                group_encoder=routing_payload_encoders[group],
                routing_query_map=routing_query_factors[head],
                query_grams=routing_moments.query_grams[:, head],
                joint_grams=routing_moments.joint_grams[:, group],
            )
        )
    return loss


def _routing_head_loss_tensor(
    *,
    selector: torch.Tensor,
    group_encoder: torch.Tensor,
    routing_query_map: torch.Tensor,
    query_grams: torch.Tensor,
    joint_grams: torch.Tensor,
) -> torch.Tensor:
    delta = routing_query_map @ group_encoder.mT - selector
    expanded_delta = delta.unsqueeze(0).expand(query_grams.shape[0], -1, -1)
    transformed = torch.bmm(
        torch.bmm(query_grams, expanded_delta),
        joint_grams,
    )
    return torch.sum(expanded_delta * transformed)


def routing_loss(
    *,
    layout: S80Layout,
    routing_payload_encoders: torch.Tensor,
    routing_query_factors: torch.Tensor,
    routing_statistics: S80RoutingStatistics,
) -> float:
    moments = _stack_routing_moments(
        routing_statistics,
        device=routing_payload_encoders.device,
        dtype=routing_payload_encoders.dtype,
    )
    return float(
        routing_loss_tensor(
            layout=layout,
            routing_payload_encoders=routing_payload_encoders,
            routing_query_factors=routing_query_factors,
            routing_moments=moments,
        )
    )


def evaluate_s80_objective(
    *,
    layout: S80Layout,
    objective: S80Objective,
    factors: S80Factors,
    routing_moments: _StackedRoutingMoments | None = None,
) -> S80Loss:
    factors.validate(layout)
    moments = routing_moments or _stack_routing_moments(
        objective.routing,
        device=factors.joint_encoders.device,
        dtype=factors.joint_encoders.dtype,
    )
    mapping = layout.head_to_kv_group(device=factors.joint_encoders.device)
    payload = evaluate_quadratic(
        objective.payload,
        factors.joint_encoders,
        factors.payload_decoders,
        mapping,
    )
    route = float(
        routing_loss_tensor(
            layout=layout,
            routing_payload_encoders=factors.routing_payload_encoders,
            routing_query_factors=factors.routing_query_factors,
            routing_moments=moments,
        )
    )
    normalized_payload = payload / objective.payload_normalizer
    normalized_routing = route / objective.routing_normalizer
    return S80Loss(
        payload=payload,
        routing=route,
        normalized_payload=normalized_payload,
        normalized_routing=normalized_routing,
        total=normalized_payload + objective.routing_weight * normalized_routing,
    )


def routing_map_half_gradient(
    *,
    layout: S80Layout,
    group_encoder: torch.Tensor,
    routing_map: torch.Tensor,
    query_grams: torch.Tensor,
    joint_grams: torch.Tensor,
) -> torch.Tensor:
    selector = _selector_transpose(layout, group_encoder)
    delta = routing_map @ group_encoder.mT - selector
    shards = query_grams.shape[0]
    joint_encoder = torch.bmm(
        joint_grams,
        group_encoder.unsqueeze(0).expand(shards, -1, -1),
    )
    transformed = torch.bmm(
        query_grams,
        torch.bmm(
            delta.unsqueeze(0).expand(shards, -1, -1),
            joint_encoder,
        ),
    )
    return transformed.sum(dim=0)


def _project_joint_grams(
    group_encoder: torch.Tensor,
    joint_grams: torch.Tensor,
) -> torch.Tensor:
    shards = joint_grams.shape[0]
    expanded_encoder = group_encoder.unsqueeze(0).expand(shards, -1, -1)
    return torch.bmm(
        group_encoder.mT.unsqueeze(0).expand(shards, -1, -1),
        torch.bmm(joint_grams, expanded_encoder),
    )


def _routing_map_hvp_from_projected(
    *,
    direction: torch.Tensor,
    query_grams: torch.Tensor,
    projected_joint_grams: torch.Tensor,
) -> torch.Tensor:
    shards = query_grams.shape[0]
    transformed = torch.bmm(
        torch.bmm(
            query_grams,
            direction.unsqueeze(0).expand(shards, -1, -1),
        ),
        projected_joint_grams,
    )
    return transformed.sum(dim=0)


def routing_map_hessian_vector_product(
    *,
    group_encoder: torch.Tensor,
    direction: torch.Tensor,
    query_grams: torch.Tensor,
    joint_grams: torch.Tensor,
) -> torch.Tensor:
    return _routing_map_hvp_from_projected(
        direction=direction,
        query_grams=query_grams,
        projected_joint_grams=_project_joint_grams(group_encoder, joint_grams),
    )


def routing_encoder_half_gradient(
    *,
    layout: S80Layout,
    group_encoder: torch.Tensor,
    routing_query_maps: torch.Tensor,
    head_indices: Sequence[int],
    query_grams: torch.Tensor,
    joint_grams: torch.Tensor,
) -> torch.Tensor:
    selector = _selector_transpose(layout, group_encoder)
    result = torch.zeros_like(group_encoder)
    shards = joint_grams.shape[0]
    for head in head_indices:
        routing_map = routing_query_maps[head]
        delta = routing_map @ group_encoder.mT - selector
        query_map = torch.bmm(
            query_grams[:, head],
            routing_map.unsqueeze(0).expand(shards, -1, -1),
        )
        delta_query_map = torch.bmm(
            delta.mT.unsqueeze(0).expand(shards, -1, -1),
            query_map,
        )
        result.add_(torch.bmm(joint_grams, delta_query_map).sum(dim=0))
    return result


def _project_routing_query_grams(
    *,
    routing_query_maps: torch.Tensor,
    head_indices: Sequence[int],
    query_grams: torch.Tensor,
) -> torch.Tensor:
    shards = query_grams.shape[0]
    rank = routing_query_maps.shape[-1]
    result = query_grams.new_zeros(shards, rank, rank)
    for head in head_indices:
        routing_map = routing_query_maps[head]
        expanded_map = routing_map.unsqueeze(0).expand(shards, -1, -1)
        result.add_(
            torch.bmm(
                routing_map.mT.unsqueeze(0).expand(shards, -1, -1),
                torch.bmm(query_grams[:, head], expanded_map),
            )
        )
    return result


def _routing_encoder_hvp_from_projected(
    *,
    direction: torch.Tensor,
    joint_grams: torch.Tensor,
    projected_query_grams: torch.Tensor,
) -> torch.Tensor:
    shards = joint_grams.shape[0]
    transformed = torch.bmm(
        torch.bmm(
            joint_grams,
            direction.unsqueeze(0).expand(shards, -1, -1),
        ),
        projected_query_grams,
    )
    return transformed.sum(dim=0)


def _sum_of_kronecker_products_diagonal(
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
) -> torch.Tensor:
    """Return the exact matrix-shaped diagonal of ``sum L_s X R_s``."""

    return torch.einsum(
        "si,sj->ij",
        torch.diagonal(left_factors, dim1=-2, dim2=-1),
        torch.diagonal(right_factors, dim1=-2, dim2=-1),
    )


def _mean_hessian_diagonal(diagonal: torch.Tensor) -> float:
    return max(
        abs(float(diagonal.sum())) / diagonal.numel(),
        torch.finfo(diagonal.dtype).tiny,
    )


def _sum_of_kronecker_products_cholesky(
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
    *,
    absolute_damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a balanced partial-trace preconditioner for ``sum L_s X R_s``."""

    left_dimension = left_factors.shape[-1]
    right_dimension = right_factors.shape[-1]
    left = torch.einsum(
        "sij,s->ij",
        left_factors,
        torch.diagonal(right_factors, dim1=-2, dim2=-1).sum(dim=1) / right_dimension,
    )
    right = torch.einsum(
        "sij,s->ij",
        right_factors,
        torch.diagonal(left_factors, dim1=-2, dim2=-1).sum(dim=1) / left_dimension,
    )
    left.diagonal().add_(absolute_damping)
    right.diagonal().add_(absolute_damping)
    mean_eigenvalue = max(
        float(torch.trace(left)) / left_dimension,
        torch.finfo(left.dtype).tiny,
    )
    scale = mean_eigenvalue**0.5
    left = 0.5 * (left + left.mT) / scale
    right = 0.5 * (right + right.mT) / scale
    return torch.linalg.cholesky(left), torch.linalg.cholesky(right)


def _two_sided_kronecker_cholesky(
    *,
    payload_covariance: torch.Tensor,
    decoder_grams: torch.Tensor,
    head_indices: Sequence[int],
    routing_joint_grams: torch.Tensor,
    projected_query_grams: torch.Tensor,
    payload_scale: float,
    routing_scale: float,
    absolute_damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    joint_dim = payload_covariance.shape[-1]
    joint_rank = decoder_grams.shape[-1]
    left = payload_covariance.new_zeros(joint_dim, joint_dim)
    right = decoder_grams.new_zeros(joint_rank, joint_rank)
    for head in head_indices:
        for other in head_indices:
            left_factor = payload_covariance[head, other]
            right_factor = decoder_grams[other, head]
            left.add_(
                left_factor,
                alpha=(payload_scale * float(torch.trace(right_factor)) / joint_rank),
            )
            right.add_(
                right_factor,
                alpha=(payload_scale * float(torch.trace(left_factor)) / joint_dim),
            )
    for left_factor, right_factor in zip(
        routing_joint_grams,
        projected_query_grams,
        strict=True,
    ):
        left.add_(
            left_factor,
            alpha=(routing_scale * float(torch.trace(right_factor)) / joint_rank),
        )
        right.add_(
            right_factor,
            alpha=(routing_scale * float(torch.trace(left_factor)) / joint_dim),
        )
    left.diagonal().add_(absolute_damping)
    right.diagonal().add_(absolute_damping)
    mean_eigenvalue = max(
        float(torch.trace(left)) / joint_dim,
        torch.finfo(left.dtype).tiny,
    )
    scale = mean_eigenvalue**0.5
    left = 0.5 * (left + left.mT) / scale
    right = 0.5 * (right + right.mT) / scale
    return torch.linalg.cholesky(left), torch.linalg.cholesky(right)


def _two_sided_inverse(
    value: torch.Tensor,
    left_cholesky: torch.Tensor,
    right_cholesky: torch.Tensor,
) -> torch.Tensor:
    left_solved = torch.linalg.solve_triangular(
        left_cholesky.mT,
        value,
        upper=True,
    )
    return torch.linalg.solve_triangular(
        right_cholesky.mT,
        left_solved.mT,
        upper=True,
    ).mT


def _two_sided_inverse_adjoint(
    value: torch.Tensor,
    left_cholesky: torch.Tensor,
    right_cholesky: torch.Tensor,
) -> torch.Tensor:
    left_solved = torch.linalg.solve_triangular(
        left_cholesky,
        value,
        upper=False,
    )
    return torch.linalg.solve_triangular(
        right_cholesky,
        left_solved.mT,
        upper=False,
    ).mT


def _two_sided_preconditioned_least_squares(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rmatvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    solution_shape: tuple[int, ...],
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
    relative_tolerance: float,
    max_iterations: int,
    absolute_damping: float,
) -> tuple[torch.Tensor, LSQRDiagnostics]:
    """Solve a damped matrix least-squares problem in two-sided coordinates."""

    left_cholesky, right_cholesky = _sum_of_kronecker_products_cholesky(
        left_factors,
        right_factors,
        absolute_damping=absolute_damping,
    )
    residual_count = rhs.numel()
    solution_count = math.prod(solution_shape)
    damping_root = absolute_damping**0.5

    def preconditioned_matvec(value: torch.Tensor) -> torch.Tensor:
        direction = _two_sided_inverse(
            value,
            left_cholesky,
            right_cholesky,
        )
        return torch.cat(
            (
                matvec(direction),
                damping_root * direction.reshape(-1),
            )
        )

    def preconditioned_rmatvec(value: torch.Tensor) -> torch.Tensor:
        gradient = rmatvec(value[:residual_count])
        gradient.add_(damping_root * value[residual_count:].reshape(solution_shape))
        return _two_sided_inverse_adjoint(
            gradient,
            left_cholesky,
            right_cholesky,
        )

    preconditioned_solution, diagnostics = least_squares_matrix(
        preconditioned_matvec,
        preconditioned_rmatvec,
        torch.cat((rhs, rhs.new_zeros(solution_count))),
        solution_shape=solution_shape,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    solution = _two_sided_inverse(
        preconditioned_solution,
        left_cholesky,
        right_cholesky,
    )
    return solution, LSQRDiagnostics(
        iterations=diagnostics.iterations,
        converged=diagnostics.converged,
        relative_residual=diagnostics.relative_residual,
        relative_normal_residual=diagnostics.relative_normal_residual,
        absolute_damping=absolute_damping,
        operator_norm=diagnostics.operator_norm,
        condition_estimate=diagnostics.condition_estimate,
        solution_norm=float(torch.linalg.vector_norm(solution)),
    )


def _batched_kronecker_pcg(
    *,
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
    rhs: torch.Tensor,
    absolute_damping: torch.Tensor,
    relative_tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, tuple[LSQRDiagnostics, ...]]:
    """Solve independent ``sum_s L X R + mu X = rhs`` systems together."""

    systems, left_dim, right_dim = rhs.shape
    left_cholesky = []
    right_cholesky = []
    for system in range(systems):
        left, right = _sum_of_kronecker_products_cholesky(
            left_factors[:, system],
            right_factors[:, system],
            absolute_damping=float(absolute_damping[system]),
        )
        left_cholesky.append(left)
        right_cholesky.append(right)
    left_cholesky_tensor = torch.stack(left_cholesky)
    right_cholesky_tensor = torch.stack(right_cholesky)

    def operator(value: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(value)
        for shard in range(left_factors.shape[0]):
            result.add_(
                torch.bmm(
                    torch.bmm(left_factors[shard], value),
                    right_factors[shard],
                )
            )
        return result + absolute_damping[:, None, None] * value

    def precondition(value: torch.Tensor) -> torch.Tensor:
        left_solved = torch.cholesky_solve(value, left_cholesky_tensor)
        return torch.cholesky_solve(
            left_solved.mT,
            right_cholesky_tensor,
        ).mT

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    preconditioned = precondition(residual)
    direction = preconditioned.clone()
    inner = torch.sum(residual * preconditioned, dim=(1, 2))
    rhs_norm = torch.linalg.vector_norm(rhs, dim=(1, 2)).clamp_min(
        torch.finfo(rhs.dtype).tiny
    )
    initial_norm = torch.linalg.vector_norm(residual, dim=(1, 2))
    active = initial_norm > 0
    iterations = torch.zeros(systems, device=rhs.device, dtype=torch.int64)
    for iteration in range(1, max_iterations + 1):
        product = operator(direction)
        curvature = torch.sum(direction * product, dim=(1, 2))
        usable = active & torch.isfinite(curvature) & (curvature > 0)
        alpha = torch.where(usable, inner / curvature, torch.zeros_like(inner))
        solution.add_(alpha[:, None, None] * direction)
        residual.add_(-alpha[:, None, None] * product)
        relative = torch.linalg.vector_norm(residual, dim=(1, 2)) / rhs_norm
        iterations = torch.where(usable, iteration, iterations)
        active = usable & (relative > relative_tolerance)
        if not bool(active.any()):
            break
        new_preconditioned = precondition(residual)
        new_inner = torch.sum(residual * new_preconditioned, dim=(1, 2))
        beta = torch.where(active, new_inner / inner, torch.zeros_like(inner))
        direction = new_preconditioned + beta[:, None, None] * direction
        direction = torch.where(active[:, None, None], direction, torch.zeros_like(direction))
        preconditioned = new_preconditioned
        inner = new_inner
    final_norm = torch.linalg.vector_norm(residual, dim=(1, 2))
    relative = final_norm / rhs_norm
    diagnostics = tuple(
        LSQRDiagnostics(
            iterations=int(iterations[index]),
            converged=bool(relative[index] <= relative_tolerance),
            relative_residual=float(relative[index]),
            relative_normal_residual=float(relative[index]),
            absolute_damping=float(absolute_damping[index]),
            operator_norm=float("nan"),
            condition_estimate=float("nan"),
            solution_norm=float(torch.linalg.vector_norm(solution[index])),
        )
        for index in range(systems)
    )
    return solution, diagnostics


def _payload_covariance_root_system(
    *,
    covariance: torch.Tensor,
    current_errors: torch.Tensor,
    head_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress the exact conditional payload LS into covariance-root rows."""

    group_covariance = covariance.index_select(0, head_indices).index_select(
        1,
        head_indices,
    )
    rows = group_covariance.permute(0, 2, 1, 3).reshape(
        head_indices.numel() * covariance.shape[-1],
        head_indices.numel() * covariance.shape[-1],
    )
    rows = 0.5 * (rows + rows.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(rows)
    threshold = (
        torch.finfo(rows.dtype).eps
        * rows.shape[0]
        * eigenvalues.abs().amax().clamp_min(torch.finfo(rows.dtype).tiny)
    )
    keep = eigenvalues > threshold
    roots = eigenvalues[keep].sqrt()
    basis = eigenvectors[:, keep]
    design = roots[:, None] * basis.mT

    cross_blocks = []
    for head in head_indices.tolist():
        cross_blocks.append(
            -torch.einsum("hij,hjo->io", covariance[head], current_errors)
        )
    cross = torch.cat(cross_blocks, dim=0)
    target = (basis.mT @ cross) / roots[:, None]
    return design, target


def _payload_encoder_hessian_diagonal(
    *,
    covariance: torch.Tensor,
    decoder_grams: torch.Tensor,
    head_indices: Sequence[int],
) -> torch.Tensor:
    diagonal = covariance.new_zeros(
        covariance.shape[-1],
        decoder_grams.shape[-1],
    )
    for head in head_indices:
        for other in head_indices:
            diagonal.add_(
                torch.diagonal(covariance[head, other]).unsqueeze(1)
                * torch.diagonal(decoder_grams[other, head]).unsqueeze(0)
            )
    return diagonal


def _jacobi_preconditioner(
    hessian_diagonal: torch.Tensor,
    *,
    absolute_damping: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    shifted = hessian_diagonal.clamp_min(0).add(absolute_damping)
    shifted.clamp_min_(torch.finfo(shifted.dtype).tiny)

    def apply(value: torch.Tensor) -> torch.Tensor:
        return value / shifted

    return apply


def routing_encoder_hessian_vector_product(
    *,
    direction: torch.Tensor,
    routing_query_maps: torch.Tensor,
    head_indices: Sequence[int],
    query_grams: torch.Tensor,
    joint_grams: torch.Tensor,
) -> torch.Tensor:
    return _routing_encoder_hvp_from_projected(
        direction=direction,
        joint_grams=joint_grams,
        projected_query_grams=_project_routing_query_grams(
            routing_query_maps=routing_query_maps,
            head_indices=head_indices,
            query_grams=query_grams,
        ),
    )


def _decoder_cross_grams(payload_decoders: torch.Tensor) -> torch.Tensor:
    return torch.einsum("kao,hbo->khab", payload_decoders, payload_decoders)


def payload_encoder_half_gradient(
    *,
    objective: RoutedOVQuadratic,
    joint_encoders: torch.Tensor,
    payload_decoders: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    group_index: int,
    decoder_grams: torch.Tensor | None = None,
) -> torch.Tensor:
    mapping = head_to_kv_group.to(device=joint_encoders.device, dtype=torch.long)
    by_head = joint_encoders.index_select(0, mapping)
    grams = (
        _decoder_cross_grams(payload_decoders)
        if decoder_grams is None
        else decoder_grams
    )
    quadratic = torch.einsum(
        "hkij,kja,khab->hib",
        objective.covariance,
        by_head,
        grams,
    )
    linear = torch.einsum("hio,hbo->hib", objective.cross, payload_decoders)
    heads = torch.nonzero(mapping == group_index, as_tuple=False).flatten()
    return (quadratic - linear).index_select(0, heads).sum(dim=0)


def combined_encoder_hessian_vector_product(
    *,
    direction: torch.Tensor,
    payload_covariance: torch.Tensor,
    payload_decoders: torch.Tensor,
    routing_query_maps: torch.Tensor,
    head_indices: Sequence[int],
    routing_query_grams: torch.Tensor,
    routing_joint_grams: torch.Tensor,
    payload_scale: float,
    routing_scale: float,
) -> torch.Tensor:
    payload = encoder_hessian_vector_product(
        covariance=payload_covariance,
        D_heads=payload_decoders,
        head_indices=head_indices,
        delta=direction,
    )
    route = routing_encoder_hessian_vector_product(
        direction=direction,
        routing_query_maps=routing_query_maps,
        head_indices=head_indices,
        query_grams=routing_query_grams,
        joint_grams=routing_joint_grams,
    )
    return payload_scale * payload + routing_scale * route


def gauge_canonicalize_s80(
    *,
    layout: S80Layout,
    factors: S80Factors,
) -> tuple[S80Factors, float, float]:
    """Orthonormalize the Route32 block, then its Payload48 complement."""

    factors.validate(layout)
    mapping = layout.head_to_kv_group(device=factors.joint_encoders.device)
    routing_encoders = factors.routing_payload_encoders.clone()
    payload_encoders = factors.payload_only_encoders.clone()
    decoders = factors.payload_decoders.clone()
    query_factors = factors.routing_query_factors.clone()
    max_payload_error = 0.0
    max_routing_error = 0.0
    tiny = torch.finfo(routing_encoders.dtype).tiny
    route = layout.routing_rank
    for group in range(layout.num_key_value_heads):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        old_b = routing_encoders[group]
        old_f = payload_encoders[group]
        old_d = decoders.index_select(0, heads)
        old_u = query_factors.index_select(0, heads)
        old_payload = torch.cat((old_b, old_f), dim=1) @ old_d
        old_routing = old_u @ old_b.mT

        new_b, transform_b = torch.linalg.qr(old_b, mode="reduced")
        signs = torch.sign(torch.diagonal(transform_b))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        new_b = new_b * signs.unsqueeze(0)
        transform_b = signs.unsqueeze(1) * transform_b
        decoder_b = torch.einsum("ab,hbo->hao", transform_b, old_d[:, :route])
        decoder_f = old_d[:, route:]
        new_u = torch.einsum("hka,ba->hkb", old_u, transform_b)

        coupling = new_b.mT @ old_f
        orthogonal_f = old_f - new_b @ coupling
        if layout.payload_only_rank:
            new_f, transform_f = torch.linalg.qr(orthogonal_f, mode="reduced")
            signs = torch.sign(torch.diagonal(transform_f))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            new_f = new_f * signs.unsqueeze(0)
            transform_f = signs.unsqueeze(1) * transform_f
            decoder_b = decoder_b + torch.einsum(
                "ab,hbo->hao",
                coupling,
                decoder_f,
            )
            decoder_f = torch.einsum("ab,hbo->hao", transform_f, decoder_f)
        else:
            new_f = orthogonal_f
        new_d = torch.cat((decoder_b, decoder_f), dim=1)
        new_payload = torch.cat((new_b, new_f), dim=1) @ new_d
        new_routing = new_u @ new_b.mT
        max_payload_error = max(
            max_payload_error,
            float(
                torch.linalg.vector_norm(new_payload - old_payload)
                / torch.linalg.vector_norm(old_payload).clamp_min(tiny)
            ),
        )
        max_routing_error = max(
            max_routing_error,
            float(
                torch.linalg.vector_norm(new_routing - old_routing)
                / torch.linalg.vector_norm(old_routing).clamp_min(tiny)
            ),
        )
        routing_encoders[group] = new_b
        payload_encoders[group] = new_f
        decoders.index_copy_(0, heads, new_d)
        query_factors.index_copy_(0, heads, new_u)
    return (
        S80Factors(routing_encoders, payload_encoders, decoders, query_factors),
        max_payload_error,
        max_routing_error,
    )


def initialize_s80_from_c1_and_kq(
    *,
    layout: S80Layout,
    value_encoders: torch.Tensor,
    value_decoders: torch.Tensor,
    key_encoders: torch.Tensor,
    query_encoders: torch.Tensor,
    routing_statistics: S80RoutingStatistics,
    cg_relative_tolerance: float = 1e-5,
    cg_max_iterations: int = 100,
    cg_relative_damping: float = 1e-5,
) -> S80Factors:
    """Preserve C1-V80 and front-load its regressed KQ-SVD32 route block."""

    routing_rank = int(key_encoders.shape[-1])
    if tuple(query_encoders.shape) == (
        layout.num_key_value_heads,
        layout.key_dim,
        routing_rank,
    ):
        query_encoders = query_encoders.repeat_interleave(
            layout.query_heads_per_kv_group,
            dim=0,
        )
    dtype = value_encoders.dtype
    device = value_encoders.device
    routing_statistics.validate()
    A = torch.zeros(
        layout.num_key_value_heads,
        layout.joint_dim,
        layout.joint_rank,
        device=device,
        dtype=dtype,
    )
    D = torch.zeros(
        layout.num_attention_heads,
        layout.joint_rank,
        layout.hidden_size,
        device=device,
        dtype=dtype,
    )
    U = torch.zeros(
        layout.num_attention_heads,
        layout.key_dim,
        layout.routing_rank,
        device=device,
        dtype=dtype,
    )
    R = torch.zeros(
        layout.num_key_value_heads,
        layout.joint_rank,
        layout.routing_rank,
        device=device,
        dtype=dtype,
    )
    A[:, : layout.value_dim] = value_encoders
    D.copy_(value_decoders)
    moments = _stack_routing_moments(
        routing_statistics,
        device=device,
        dtype=dtype,
    )
    mapping = layout.head_to_kv_group(device=device)
    for group in range(layout.num_key_value_heads):
        joint_gram = moments.joint_grams[:, group].sum(dim=0)
        value_gram = joint_gram[: layout.value_dim, : layout.value_dim]
        value_key_cross = joint_gram[
            : layout.value_dim,
            layout.value_dim :,
        ]
        latent_gram = value_encoders[group].mT @ value_gram @ value_encoders[group]
        latent_gram = 0.5 * (latent_gram + latent_gram.mT)
        rhs = value_encoders[group].mT @ value_key_cross @ key_encoders[group]

        def operator(value: torch.Tensor) -> torch.Tensor:
            return latent_gram @ value

        hessian_diagonal = torch.diagonal(latent_gram).unsqueeze(1).expand_as(rhs)
        damping = cg_relative_damping * _mean_hessian_diagonal(hessian_diagonal)
        raw_selector, _ = conjugate_gradient_matrix(
            operator,
            rhs,
            relative_tolerance=cg_relative_tolerance,
            max_iterations=cg_max_iterations,
            absolute_damping=damping,
            preconditioner=_jacobi_preconditioner(
                hessian_diagonal,
                absolute_damping=damping,
            ),
        )
        selector, transform = torch.linalg.qr(raw_selector, mode="reduced")
        signs = torch.sign(torch.diagonal(transform))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        selector = selector * signs.unsqueeze(0)
        transform = signs.unsqueeze(1) * transform
        R[group] = selector
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        grouped_query = query_encoders.index_select(0, heads)
        U.index_copy_(
            0,
            heads,
            torch.einsum("hka,ba->hkb", grouped_query, transform),
        )
    routing_encoders = torch.empty(
        layout.num_key_value_heads,
        layout.joint_dim,
        layout.routing_rank,
        device=device,
        dtype=dtype,
    )
    payload_encoders = torch.empty(
        layout.num_key_value_heads,
        layout.joint_dim,
        layout.payload_only_rank,
        device=device,
        dtype=dtype,
    )
    mapping = layout.head_to_kv_group(device=device)
    for group in range(layout.num_key_value_heads):
        complete = torch.linalg.qr(R[group], mode="complete").Q
        rotation = torch.cat((R[group], complete[:, layout.routing_rank :]), dim=1)
        rotated = A[group] @ rotation
        routing_encoders[group] = rotated[:, : layout.routing_rank]
        payload_encoders[group] = rotated[:, layout.routing_rank :]
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        rotated_decoders = torch.einsum(
            "ra,nad->nrd",
            rotation.mT,
            D.index_select(0, heads),
        )
        D.index_copy_(0, heads, rotated_decoders)
    result = S80Factors(routing_encoders, payload_encoders, D, U)
    result, _, _ = gauge_canonicalize_s80(layout=layout, factors=result)
    result.validate(layout)
    return result


def frontload_s80_routing_coordinates(
    *,
    layout: S80Layout,
    factors: S80Factors,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the already front-loaded Store80 encoder and payload decoder."""

    factors.validate(layout)
    return factors.joint_encoders, factors.payload_decoders


def fold_s80_factors(
    *,
    layout: S80Layout,
    factors: S80Factors,
    dense_v_proj_weight: torch.Tensor,
    dense_v_proj_bias: torch.Tensor | None = None,
    dense_o_proj_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> FoldedS80Factors:
    """Fold only the V branch; the post-RoPE K branch remains explicit."""

    factors.validate(layout)
    frontloaded_encoders, frontloaded_decoders = frontload_s80_routing_coordinates(
        layout=layout,
        factors=factors,
    )
    A_v = frontloaded_encoders[:, : layout.value_dim]
    A_k = frontloaded_encoders[:, layout.value_dim :]
    dense_v = dense_v_proj_weight.to(A_v).reshape(
        layout.num_key_value_heads,
        layout.value_dim,
        layout.hidden_size,
    )
    folded_v = torch.bmm(A_v.mT, dense_v).reshape(
        layout.num_key_value_heads * layout.joint_rank,
        layout.hidden_size,
    )
    folded_bias = None
    if dense_v_proj_bias is not None:
        grouped_bias = dense_v_proj_bias.to(A_v).reshape(
            layout.num_key_value_heads,
            layout.value_dim,
            1,
        )
        folded_bias = torch.bmm(A_v.mT, grouped_bias).reshape(-1)
    o_weight = frontloaded_decoders.permute(2, 0, 1).reshape(
        layout.hidden_size,
        layout.num_attention_heads * layout.joint_rank,
    )
    dtype = output_dtype or dense_v_proj_weight.dtype
    return FoldedS80Factors(
        v_joint_proj_weight=folded_v.detach().cpu().to(dtype),
        v_joint_proj_bias=(
            None if folded_bias is None else folded_bias.detach().cpu().to(dtype)
        ),
        k_joint_encoder=A_k.detach().cpu().to(dtype),
        routing_query_factor=(factors.routing_query_factors.detach().cpu().to(dtype)),
        o_decoder_weight=o_weight.detach().cpu().to(dtype),
        o_decoder_bias=(
            None
            if dense_o_proj_bias is None
            else dense_o_proj_bias.detach().cpu().to(dtype)
        ),
        head_to_kv_group=layout.head_to_kv_group(),
        joint_encoder=frontloaded_encoders.detach().cpu().to(dtype),
    )


def _with_factors(
    factors: S80Factors,
    *,
    A: torch.Tensor | None = None,
    B: torch.Tensor | None = None,
    F: torch.Tensor | None = None,
    D: torch.Tensor | None = None,
    U: torch.Tensor | None = None,
) -> S80Factors:
    joint = factors.joint_encoders if A is None else A
    route = factors.routing_payload_encoders.shape[-1]
    return S80Factors(
        joint[..., :route] if B is None else B,
        joint[..., route:] if F is None else F,
        factors.payload_decoders if D is None else D,
        factors.routing_query_factors if U is None else U,
    )


def fit_s80_joint(
    *,
    layout: S80Layout,
    objective: S80Objective,
    validation_objective: S80Objective | None = None,
    direct_residuals: S80DirectResidualData | None = None,
    fisher_statistics: S80CompactSoftmaxFisherRouting | None = None,
    validation_fisher_statistics: S80CompactSoftmaxFisherRouting | None = None,
    initial_factors: S80Factors,
    routing_metric: str = "raw_qk",
    outer_sweeps: int = 1,
    u_mode: str = "full_u_final",
    lsqr_relative_tolerance: float = 1e-5,
    lsqr_max_iterations: int = 100,
    lsqr_relative_damping: float = 1e-5,
    decoder_relative_jitter: float = 0.0,
    work_dtype: torch.dtype = torch.float64,
    work_device: torch.device | str | None = None,
    progress_callback: Callable[[S80SweepDiagnostics], None] | None = None,
) -> S80FitResult:
    """Fit one front-loaded BF update, close D, then solve U once."""
    assert routing_metric in {"raw_qk", "page_fisher"}
    assert routing_metric == "raw_qk" or u_mode in {"frozen_u", "adapter_u"}
    objective.validate(layout)
    if direct_residuals is not None:
        direct_residuals.validate(
            num_query_heads=layout.num_attention_heads,
            num_kv_heads=layout.num_key_value_heads,
            key_dim=layout.key_dim,
            joint_dim=layout.joint_dim,
        )
    initial_factors.validate(layout)
    device = (
        torch.device(work_device)
        if work_device is not None
        else initial_factors.joint_encoders.device
    )
    mapping = layout.head_to_kv_group(device=device)
    assert routing_metric != "raw_qk" or direct_residuals is not None
    assert routing_metric != "page_fisher" or fisher_statistics is not None
    assert routing_metric != "page_fisher" or validation_fisher_statistics is not None
    fisher_fit = fisher_statistics
    fisher_validation = validation_fisher_statistics
    routing_normalizer = (
        objective.routing_normalizer
        if fisher_fit is None
        else max(
            fisher_fit.teacher_fisher_energy,
            objective.normalization_epsilon,
        )
    )
    payload = RoutedOVQuadratic(
        covariance=objective.payload.covariance.to(device=device, dtype=work_dtype),
        cross=objective.payload.cross.to(device=device, dtype=work_dtype),
        constant=objective.payload.constant.to(device=device, dtype=work_dtype),
        name=objective.payload.name,
    )
    working_objective = S80Objective(
        payload=payload,
        payload_target=objective.payload_target.to(
            device=device,
            dtype=work_dtype,
        ),
        routing=objective.routing,
        payload_normalizer=objective.payload_normalizer,
        routing_normalizer=routing_normalizer,
        routing_weight=objective.routing_weight,
        normalization_epsilon=objective.normalization_epsilon,
    )
    working_validation = None
    validation_moments = None
    if validation_objective is not None:
        validation_routing_normalizer = (
            validation_objective.routing_normalizer
            if fisher_validation is None
            else max(
                fisher_validation.teacher_fisher_energy,
                validation_objective.normalization_epsilon,
            )
        )
        working_validation = S80Objective(
            payload=RoutedOVQuadratic(
                covariance=validation_objective.payload.covariance.to(
                    device=device,
                    dtype=work_dtype,
                ),
                cross=validation_objective.payload.cross.to(
                    device=device,
                    dtype=work_dtype,
                ),
                constant=validation_objective.payload.constant.to(
                    device=device,
                    dtype=work_dtype,
                ),
                name=validation_objective.payload.name,
            ),
            payload_target=validation_objective.payload_target.to(
                device=device,
                dtype=work_dtype,
            ),
            routing=validation_objective.routing,
            payload_normalizer=validation_objective.payload_normalizer,
            routing_normalizer=validation_routing_normalizer,
            routing_weight=validation_objective.routing_weight,
            normalization_epsilon=validation_objective.normalization_epsilon,
        )
        validation_moments = _stack_routing_moments(
            validation_objective.routing,
            device=device,
            dtype=work_dtype,
        )
    factors = S80Factors(
        initial_factors.routing_payload_encoders.to(
            device=device,
            dtype=work_dtype,
        ).clone(),
        initial_factors.payload_only_encoders.to(
            device=device,
            dtype=work_dtype,
        ).clone(),
        initial_factors.payload_decoders.to(device=device, dtype=work_dtype).clone(),
        initial_factors.routing_query_factors.to(
            device=device,
            dtype=work_dtype,
        ).clone(),
    )
    factors, _, _ = gauge_canonicalize_s80(layout=layout, factors=factors)
    moments = _stack_routing_moments(
        objective.routing,
        device=device,
        dtype=work_dtype,
    )
    routing_queries_by_head = (
        fisher_fit.queries_by_head.to(device=device, dtype=work_dtype)
        if fisher_fit is not None
        else direct_residuals.routing_queries.permute(1, 0, 2)
        .contiguous()
        .to(device=device, dtype=work_dtype)
    )
    routing_rows_by_group = (
        None
        if fisher_fit is not None
        else direct_residuals.routing_joint_rows.permute(2, 0, 1, 3)
        .contiguous()
        .to(device=device, dtype=work_dtype)
    )
    fisher_roots = (
        None if fisher_fit is None else compact_softmax_fisher_roots(fisher_fit)
    )
    routing_document_count = int(routing_queries_by_head.shape[1])
    routing_residual_width = (
        int(routing_rows_by_group.shape[2])
        if routing_rows_by_group is not None
        else layout.joint_dim
    )

    def metric_loss(
        current: S80Factors,
        selected_objective: S80Objective,
        selected_moments: _StackedRoutingMoments,
        fisher: S80CompactSoftmaxFisherRouting | None,
    ) -> S80Loss:
        if fisher is None:
            return evaluate_s80_objective(
                layout=layout,
                objective=selected_objective,
                factors=current,
                routing_moments=selected_moments,
            )
        payload_value = evaluate_quadratic(
            selected_objective.payload,
            current.joint_encoders,
            current.payload_decoders,
            mapping,
        )
        routing_value = compact_softmax_fisher_loss(
            fisher,
            routing_payload_encoders=current.routing_payload_encoders,
            routing_query_factors=current.routing_query_factors,
        )
        normalized_payload = payload_value / selected_objective.payload_normalizer
        normalized_routing = routing_value / selected_objective.routing_normalizer
        return S80Loss(
            payload=payload_value,
            routing=routing_value,
            normalized_payload=normalized_payload,
            normalized_routing=normalized_routing,
            total=(
                normalized_payload
                + selected_objective.routing_weight * normalized_routing
            ),
        )

    def loss(current: S80Factors) -> S80Loss:
        return metric_loss(current, working_objective, moments, fisher_fit)

    def heldout_loss(current: S80Factors) -> S80Loss | None:
        if working_validation is None or validation_moments is None:
            return None
        return metric_loss(
            current,
            working_validation,
            validation_moments,
            fisher_validation,
        )

    def solve_decoder(current: S80Factors) -> tuple[S80Factors, LinearSolveDiagnostics]:
        proposed, diagnostics = solve_free_decoder(
            objective=payload,
            A_unique=current.joint_encoders,
            head_to_kv_group=mapping,
            coupling_mode="full_layer",
            relative_jitter=decoder_relative_jitter,
        )
        return _with_factors(current, D=proposed), diagnostics

    def solve_encoders(
        current: S80Factors,
    ) -> tuple[S80Factors, tuple[S80LSQRStep, ...]]:
        encoders = current.joint_encoders.clone()
        steps: list[S80LSQRStep] = []
        decoder_grams = _decoder_cross_grams(current.payload_decoders)
        payload_scale = 1.0 / working_objective.payload_normalizer
        routing_scale = (
            working_objective.routing_weight / working_objective.routing_normalizer
        )
        selector = _selector_transpose(layout, encoders)
        for group in range(layout.num_key_value_heads):
            heads_tensor = torch.nonzero(mapping == group, as_tuple=False).flatten()
            heads = tuple(int(item) for item in heads_tensor.tolist())
            state = _with_factors(current, A=encoders)
            before = loss(state).total

            decoder_stack = current.payload_decoders.index_select(
                0,
                heads_tensor,
            ).reshape(len(heads) * layout.joint_rank, layout.hidden_size)
            output_basis = torch.linalg.qr(decoder_stack.mT, mode="reduced").Q
            projected_decoders = current.payload_decoders @ output_basis
            projected_targets = working_objective.payload_target @ output_basis
            encoders_by_head = encoders.index_select(0, mapping)
            current_errors = torch.bmm(encoders_by_head, projected_decoders)
            current_errors.sub_(projected_targets)
            payload_design, payload_target = _payload_covariance_root_system(
                covariance=payload.covariance,
                current_errors=current_errors,
                head_indices=heads_tensor,
            )
            payload_design = payload_design.reshape(
                payload_design.shape[0],
                len(heads),
                layout.joint_dim,
            )
            group_decoders = projected_decoders.index_select(0, heads_tensor)
            payload_scale_root = payload_scale**0.5

            group_queries = routing_queries_by_head.index_select(
                0,
                heads_tensor,
            ).permute(1, 0, 2)
            query_factors = current.routing_query_factors.index_select(
                0,
                heads_tensor,
            )
            projected_queries = torch.einsum(
                "dhk,hkr->dhr",
                group_queries,
                query_factors,
            )
            joint_rows = (
                None
                if routing_rows_by_group is None
                else routing_rows_by_group[group]
            )
            group_fisher_roots = (
                None
                if fisher_roots is None
                else fisher_roots.index_select(0, heads_tensor).permute(1, 0, 2, 3)
            )
            routing_scale_root = routing_scale**0.5
            payload_count = payload_target.numel()
            routing_count = (
                routing_document_count * len(heads) * routing_residual_width
                if routing_scale
                else 0
            )
            residual_count = payload_count + routing_count

            projected_query = _project_routing_query_grams(
                routing_query_maps=current.routing_query_factors,
                head_indices=heads,
                query_grams=moments.query_grams,
            )
            padded_projected_query = projected_query.new_zeros(
                projected_query.shape[0],
                layout.joint_rank,
                layout.joint_rank,
            )
            padded_projected_query[
                :, : layout.routing_rank, : layout.routing_rank
            ] = projected_query
            payload_diagonal = _payload_encoder_hessian_diagonal(
                covariance=payload.covariance,
                decoder_grams=decoder_grams,
                head_indices=heads,
            )
            raw_routing_diagonal = _sum_of_kronecker_products_diagonal(
                moments.joint_grams[:, group],
                padded_projected_query,
            )
            if fisher_fit is None:
                routing_diagonal = raw_routing_diagonal
                preconditioner_routing_scale = routing_scale
            else:
                fisher_route_diagonal = compact_softmax_fisher_encoder_diagonal(
                    fisher_fit,
                    head_indices=heads_tensor,
                    projected_queries=projected_queries,
                )
                routing_diagonal = raw_routing_diagonal.new_zeros(
                    layout.joint_dim,
                    layout.joint_rank,
                )
                routing_diagonal[:, : layout.routing_rank] = fisher_route_diagonal
                raw_mean = raw_routing_diagonal.abs().mean().clamp_min(
                    torch.finfo(raw_routing_diagonal.dtype).tiny
                )
                fisher_mean = routing_diagonal.abs().mean()
                preconditioner_routing_scale = routing_scale * float(
                    fisher_mean / raw_mean
                )
            hessian_diagonal = (
                payload_scale * payload_diagonal + routing_scale * routing_diagonal
            )
            damping = lsqr_relative_damping * _mean_hessian_diagonal(
                hessian_diagonal
            )
            damping_root = damping**0.5
            left_cholesky, right_cholesky = _two_sided_kronecker_cholesky(
                payload_covariance=payload.covariance,
                decoder_grams=decoder_grams,
                head_indices=heads,
                routing_joint_grams=moments.joint_grams[:, group],
                projected_query_grams=padded_projected_query,
                payload_scale=payload_scale,
                routing_scale=preconditioner_routing_scale,
                absolute_damping=damping,
            )

            def matvec(value: torch.Tensor) -> torch.Tensor:
                result = value.new_empty(residual_count + value.numel())
                encoded = torch.einsum("nhi,ir->nhr", payload_design, value)
                result[:payload_count].copy_(
                    (
                        payload_scale_root
                        * torch.einsum("nhr,hrk->nk", encoded, group_decoders)
                    ).reshape(-1)
                )
                if routing_scale:
                    left = torch.einsum(
                        "dhr,ir->dhi",
                        projected_queries,
                        value[:, : layout.routing_rank],
                    )
                    if group_fisher_roots is None:
                        routing_residual = torch.stack(
                            [
                                _batched_routing_scores(left[:, local], joint_rows)
                                for local in range(len(heads))
                            ],
                            dim=1,
                        )
                    else:
                        routing_residual = (
                            (0.5**0.5)
                            * fisher_fit.scaling
                            * torch.einsum(
                                "dhi,dhij->dhj",
                                left,
                                group_fisher_roots,
                            )
                        )
                    result[payload_count:residual_count].copy_(
                        (routing_scale_root * routing_residual).reshape(-1)
                    )
                result[residual_count:].copy_(damping_root * value.reshape(-1))
                return result

            def rmatvec(value: torch.Tensor) -> torch.Tensor:
                payload_value = value[:payload_count].reshape_as(payload_target)
                decoded = torch.einsum(
                    "nk,hrk->nhr",
                    payload_value,
                    group_decoders,
                )
                result = payload_scale_root * torch.einsum(
                    "nhi,nhr->ir",
                    payload_design,
                    decoded,
                )
                if routing_scale:
                    scores = value[payload_count:residual_count].reshape(
                        routing_document_count,
                        len(heads),
                        routing_residual_width,
                    )
                    if group_fisher_roots is not None:
                        scores = (
                            (0.5**0.5)
                            * fisher_fit.scaling
                            * torch.einsum(
                                "dhj,dhij->dhi",
                                scores,
                                group_fisher_roots,
                            )
                        )
                    route_gradient = result.new_zeros(
                        layout.joint_dim,
                        layout.routing_rank,
                    )
                    if group_fisher_roots is None:
                        for local in range(len(heads)):
                            weighted_joint = _batched_weighted_joint(
                                scores[:, local],
                                joint_rows,
                            )
                            route_gradient.add_(
                                torch.einsum(
                                    "di,dr->ir",
                                    weighted_joint,
                                    projected_queries[:, local],
                                )
                            )
                    else:
                        route_gradient.copy_(
                            torch.einsum(
                                "dhi,dhr->ir",
                                scores,
                                projected_queries,
                            )
                        )
                    result[:, : layout.routing_rank].add_(
                        routing_scale_root * route_gradient
                    )
                result.add_(
                    damping_root * value[residual_count:].reshape_as(result)
                )
                return result

            routing_rhs = []
            if routing_scale:
                current_b = encoders[group, :, : layout.routing_rank]
                if fisher_fit is None:
                    for local in range(len(heads)):
                        residual_left = (
                            projected_queries[:, local] @ current_b.mT
                            - group_queries[:, local] @ selector
                        )
                        routing_rhs.append(
                            -routing_scale_root
                            * _batched_routing_scores(residual_left, joint_rows)
                        )
                else:
                    current_error = torch.einsum(
                        "dhr,ir->dhi",
                        projected_queries,
                        current_b,
                    )
                    current_error.sub_(
                        torch.einsum("dhk,ki->dhi", group_queries, selector)
                    )
                    transformed = (
                        (0.5**0.5)
                        * fisher_fit.scaling
                        * torch.einsum(
                            "dhi,dhij->dhj",
                            current_error,
                            group_fisher_roots,
                        )
                    )
                    routing_rhs.extend(
                        -routing_scale_root * transformed[:, local]
                        for local in range(len(heads))
                    )
            rhs_blocks = [(payload_scale_root * payload_target).reshape(-1)]
            if routing_rhs:
                rhs_blocks.append(torch.stack(routing_rhs, dim=1).reshape(-1))
            rhs_blocks.append(encoders.new_zeros(encoders[group].numel()))
            rhs = torch.cat(rhs_blocks)

            def preconditioned_matvec(value: torch.Tensor) -> torch.Tensor:
                return matvec(
                    _two_sided_inverse(value, left_cholesky, right_cholesky)
                )

            def preconditioned_rmatvec(value: torch.Tensor) -> torch.Tensor:
                return _two_sided_inverse_adjoint(
                    rmatvec(value),
                    left_cholesky,
                    right_cholesky,
                )

            preconditioned_direction, lsqr = least_squares_matrix(
                preconditioned_matvec,
                preconditioned_rmatvec,
                rhs,
                solution_shape=tuple(encoders[group].shape),
                relative_tolerance=lsqr_relative_tolerance,
                max_iterations=lsqr_max_iterations,
            )
            direction = _two_sided_inverse(
                preconditioned_direction,
                left_cholesky,
                right_cholesky,
            )
            encoders[group].add_(direction)
            after = loss(_with_factors(current, A=encoders)).total
            steps.append(
                S80LSQRStep(
                    group,
                    before,
                    after,
                    LSQRDiagnostics(
                        iterations=lsqr.iterations,
                        converged=lsqr.converged,
                        relative_residual=lsqr.relative_residual,
                        relative_normal_residual=lsqr.relative_normal_residual,
                        absolute_damping=damping,
                        operator_norm=lsqr.operator_norm,
                        condition_estimate=lsqr.condition_estimate,
                        solution_norm=float(torch.linalg.vector_norm(direction)),
                    ),
                )
            )
        return _with_factors(current, A=encoders), tuple(steps)

    def solve_routing_queries(
        current: S80Factors,
    ) -> tuple[S80Factors, tuple[S80LSQRStep, ...]]:
        if u_mode == "frozen_u":
            return current, ()
        if fisher_fit is not None:
            left_factors, right_factors, rhs = compact_softmax_fisher_adapter_system(
                fisher_fit,
                routing_payload_encoders=current.routing_payload_encoders,
                routing_query_factors=current.routing_query_factors,
            )
            diagonals = torch.stack(
                [
                    _sum_of_kronecker_products_diagonal(
                        left_factors[:, head],
                        right_factors[:, head],
                    )
                    for head in range(layout.num_attention_heads)
                ]
            )
            damping = (
                lsqr_relative_damping
                * diagonals.mean(dim=(1, 2))
                .abs()
                .clamp_min(torch.finfo(diagonals.dtype).tiny)
            )
            direction, diagnostics = _batched_kronecker_pcg(
                left_factors=left_factors,
                right_factors=right_factors,
                rhs=rhs,
                absolute_damping=damping,
                relative_tolerance=lsqr_relative_tolerance,
                max_iterations=lsqr_max_iterations,
            )
            identity = torch.eye(
                layout.routing_rank,
                device=device,
                dtype=work_dtype,
            ).expand(layout.num_attention_heads, -1, -1)
            proposed = torch.bmm(
                current.routing_query_factors,
                identity + direction,
            )
            before = loss(current).total
            updated = _with_factors(current, U=proposed)
            after = loss(updated).total
            steps = tuple(
                S80LSQRStep(head, before, after, diagnostics[head])
                for head in range(layout.num_attention_heads)
            )
            return updated, steps
        query_factors = current.routing_query_factors
        routing_encoders = current.routing_payload_encoders
        encoders_by_head = routing_encoders.index_select(0, mapping)
        joint_by_head = moments.joint_grams.index_select(1, mapping)
        joint_encoder = torch.matmul(joint_by_head, encoders_by_head.unsqueeze(0))
        projected_joint = torch.matmul(
            encoders_by_head.mT.unsqueeze(0),
            joint_encoder,
        )
        selector = _selector_transpose(layout, routing_encoders)
        delta = torch.matmul(query_factors, encoders_by_head.mT) - selector
        gradient = torch.matmul(delta.unsqueeze(0), joint_encoder)
        gradient = torch.matmul(moments.query_grams, gradient).sum(dim=0)

        if u_mode == "adapter_u":
            left_factors = torch.matmul(
                query_factors.mT.unsqueeze(0),
                torch.matmul(moments.query_grams, query_factors.unsqueeze(0)),
            )
            rhs = -torch.matmul(query_factors.mT, gradient)
        else:
            left_factors = moments.query_grams
            rhs = -gradient
        diagonals = torch.stack(
            [
                _sum_of_kronecker_products_diagonal(
                    left_factors[:, head],
                    projected_joint[:, head],
                )
                for head in range(layout.num_attention_heads)
            ]
        )
        damping = lsqr_relative_damping * diagonals.mean(dim=(1, 2)).abs().clamp_min(
            torch.finfo(diagonals.dtype).tiny
        )
        direction, diagnostics = _batched_kronecker_pcg(
            left_factors=left_factors,
            right_factors=projected_joint,
            rhs=rhs,
            absolute_damping=damping,
            relative_tolerance=lsqr_relative_tolerance,
            max_iterations=lsqr_max_iterations,
        )
        if u_mode == "adapter_u":
            identity = torch.eye(
                layout.routing_rank,
                device=device,
                dtype=work_dtype,
            ).expand(layout.num_attention_heads, -1, -1)
            proposed = torch.bmm(query_factors, identity + direction)
        else:
            proposed = query_factors + direction
        before = loss(current).total
        updated = _with_factors(current, U=proposed)
        after = loss(updated).total
        steps = tuple(
            S80LSQRStep(head, before, after, diagnostics[head])
            for head in range(layout.num_attention_heads)
        )
        return updated, steps

    started = time.monotonic()
    initial_loss = loss(factors)
    sweeps = []
    for sweep in range(1, outer_sweeps + 1):
        sweep_started = time.monotonic()
        fit_before = loss(factors)
        factors, decoder = solve_decoder(factors)
        fit_after_decoder = loss(factors)
        factors, encoder_steps = solve_encoders(factors)
        factors, payload_gauge_error, routing_gauge_error = gauge_canonicalize_s80(
            layout=layout,
            factors=factors,
        )
        fit_after_encoders = loss(factors)
        sweep_diagnostics = S80SweepDiagnostics(
            sweep=sweep,
            fit_before=fit_before,
            fit_after_decoder=fit_after_decoder,
            fit_after_encoders=fit_after_encoders,
            validation_after_encoders=heldout_loss(factors),
            decoder=decoder,
            encoder_steps=encoder_steps,
            maximum_gauge_payload_error=payload_gauge_error,
            maximum_gauge_routing_error=routing_gauge_error,
            wall_time_seconds=time.monotonic() - sweep_started,
        )
        sweeps.append(sweep_diagnostics)
        if progress_callback is not None:
            progress_callback(sweep_diagnostics)
    factors, final_decoder = solve_decoder(factors)
    after_final_decoder = loss(factors)
    validation_after_final_decoder = heldout_loss(factors)
    factors, final_routing_query_steps = solve_routing_queries(factors)
    final_loss = loss(factors)
    diagnostics = S80FitDiagnostics(
        sweeps=tuple(sweeps),
        loss_after_final_decoder=after_final_decoder,
        validation_after_final_decoder=validation_after_final_decoder,
        loss_after_routing_queries=final_loss,
        validation_after_routing_queries=heldout_loss(factors),
        final_decoder=final_decoder,
        routing_query_steps=final_routing_query_steps,
        wall_time_seconds=time.monotonic() - started,
    )
    return S80FitResult(
        factors=S80Factors(
            factors.routing_payload_encoders.detach().cpu(),
            factors.payload_only_encoders.detach().cpu(),
            factors.payload_decoders.detach().cpu(),
            factors.routing_query_factors.detach().cpu(),
        ),
        initial_loss=initial_loss,
        final_loss=final_loss,
        diagnostics=diagnostics,
        final_routing_query_steps=final_routing_query_steps,
        routing_metric=routing_metric,
        routing_normalizer=working_objective.routing_normalizer,
    )


__all__ = [
    "FoldedS80Factors",
    "S80Factors",
    "S80FitResult",
    "S80Layout",
    "S80Loss",
    "S80Objective",
    "S80LSQRStep",
    "S80FitDiagnostics",
    "S80SweepDiagnostics",
    "combined_encoder_hessian_vector_product",
    "compose_routing_query_maps",
    "dense_o_weight_to_head_blocks",
    "evaluate_s80_objective",
    "fit_s80_joint",
    "fold_s80_factors",
    "frontload_s80_routing_coordinates",
    "gauge_canonicalize_s80",
    "initialize_s80_from_c1_and_kq",
    "joint_payload_target",
    "payload_encoder_half_gradient",
    "routing_encoder_half_gradient",
    "routing_encoder_hessian_vector_product",
    "routing_loss",
    "routing_loss_tensor",
    "routing_map_half_gradient",
    "routing_map_hessian_vector_product",
    "s80_objective_from_statistics",
]
