"""Anchor-agnostic routed GQA Value/output joint factorization.

Row-vector notation is used throughout.  A query head ``h`` uses the encoder
``A[g(h)]`` shared by its physical KV group and a free decoder ``D[h]``:

    M_h = A[g(h)] @ D[h].

The offline objective is a general PSD quadratic

    const - 2 <M, cross> + <M, C M>,

which supports both dense routed-output fitting and product-centered anchors.
All learned interaction is folded into ordinary compressed ``Wv`` and ``Wo``.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from basisserve.calibration.gqa_routed_ov_stats import flatten_covariance_blocks
from basisserve.core.gqa_vo_svdllm import GQAVOLayout


@dataclass(frozen=True)
class RoutedOVQuadratic:
    covariance: torch.Tensor
    cross: torch.Tensor
    constant: torch.Tensor
    name: str

    @property
    def num_query_heads(self) -> int:
        return int(self.covariance.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self.covariance.shape[2])

    @property
    def hidden_size(self) -> int:
        return int(self.cross.shape[2])


@dataclass(frozen=True)
class LinearSolveDiagnostics:
    partition_sizes: tuple[int, ...]
    absolute_jitters: tuple[float, ...]
    condition_estimates: tuple[float, ...]
    matrix_dimensions: tuple[int, ...] = ()
    relative_residuals: tuple[float, ...] = ()
    wall_times_seconds: tuple[float, ...] = ()


@dataclass(frozen=True)
class CGDiagnostics:
    iterations: int
    converged: bool
    relative_residual: float
    absolute_damping: float
    negative_curvature: bool
    initial_residual_norm: float
    final_residual_norm: float
    fixed_iterations: bool


@dataclass(frozen=True)
class EncoderDirectionResult:
    direction: torch.Tensor
    solver_name: str
    diagnostics: dict[str, Any]
    complexity: dict[str, int | float]


class EncoderDirectionSolver(Protocol):
    name: str

    def propose(
        self,
        *,
        half_gradient: torch.Tensor,
        covariance: torch.Tensor,
        decoder_grams: torch.Tensor,
        head_indices: Sequence[int],
    ) -> EncoderDirectionResult:
        """Return one unscaled descent direction for an encoder coordinate."""


@dataclass(frozen=True)
class EncoderDirectionDiagnostics:
    solver_name: str
    half_gradient_norm: float
    direction_norm: float
    gradient_direction_inner_product: float
    exact_hessian_curvature: float
    eta_exact: float
    eta_used: float
    requested_step_scale: float
    exact_hessian_relative_residual: float
    optimal_line_reduction: float
    relative_encoder_movement: float
    maximum_anchor_principal_angle_radians: float
    mean_anchor_principal_angle_radians: float
    solver_diagnostics: dict[str, Any]
    complexity: dict[str, int | float]


@dataclass(frozen=True)
class EncoderStepDiagnostics:
    group_index: int
    old_loss: float
    new_loss: float
    predicted_change: float
    realized_change: float
    accepted_scale: float
    retries: int
    cg: CGDiagnostics
    direction: EncoderDirectionDiagnostics | None = None
    relative_encoder_movement: float = 0.0
    maximum_anchor_principal_angle_radians: float = 0.0
    mean_anchor_principal_angle_radians: float = 0.0


@dataclass(frozen=True)
class RoutedOVSweep:
    sweep: int
    loss_before_decoder: float
    loss_after_decoder: float
    loss_after_encoders: float
    relative_improvement: float
    decoder: LinearSolveDiagnostics
    encoder_steps: tuple[EncoderStepDiagnostics, ...]
    maximum_qr_product_error: float


@dataclass(frozen=True)
class RoutedOVJointResult:
    A_unique: torch.Tensor
    D_heads: torch.Tensor
    initial_loss: float
    decoder_only_loss: float
    final_loss: float
    sweeps: tuple[RoutedOVSweep, ...]
    initial_decoder: LinearSolveDiagnostics
    legacy_final_loss: float
    final_decoder: LinearSolveDiagnostics | None
    checkpoints: tuple["RoutedOVCheckpointDiagnostics", ...]
    attribution: "RoutedOVAttribution"


@dataclass(frozen=True)
class RoutedOVCheckpointDiagnostics:
    boundary: str
    sweep: int
    loss: float
    component_losses: tuple[tuple[str, float], ...]
    decoder_relative_stationarity: float


@dataclass(frozen=True)
class RoutedOVAttributionStep:
    sweep: int
    loss_before_encoder: float
    loss_after_encoder: float
    loss_after_redecoder: float | None
    encoder_reduction: float
    redecoder_reduction: float


@dataclass(frozen=True)
class RoutedOVAttribution:
    anchor_loss: float
    decoder_only_loss: float
    endpoint_loss: float
    initial_decoder_reduction: float
    steps: tuple[RoutedOVAttributionStep, ...]
    decoder_reduction: float
    encoder_reduction: float
    total_reduction: float
    decoder_fraction: float
    encoder_fraction: float
    identity_error: float


RoutedOVCheckpointCallback = Callable[
    [RoutedOVCheckpointDiagnostics, torch.Tensor, torch.Tensor],
    None,
]


@dataclass(frozen=True)
class FoldedRoutedOVFactors:
    v_proj_compressed_weight: torch.Tensor
    o_decoder_weight: torch.Tensor
    maximum_qr_product_error: float
    maximum_dense_fold_error: float


@dataclass(frozen=True)
class RaggedHeadLayout:
    """Active latent coordinates for one fixed per-group rank vector."""

    group_ranks: tuple[int, ...]
    head_to_kv_group: tuple[int, ...]
    head_ranks: tuple[int, ...]
    head_offsets: tuple[int, ...]
    maximum_rank: int
    total_width: int

    @classmethod
    def from_group_ranks(
        cls,
        group_ranks: Sequence[int],
        head_to_kv_group: Sequence[int] | torch.Tensor,
        *,
        maximum_rank: int | None = None,
    ) -> "RaggedHeadLayout":
        ranks = tuple(int(item) for item in group_ranks)
        mapping = tuple(
            int(item)
            for item in torch.as_tensor(
                head_to_kv_group,
                dtype=torch.long,
            )
            .cpu()
            .tolist()
        )
        if not ranks or any(rank <= 0 for rank in ranks):
            raise ValueError("group ranks must be positive")
        if not mapping or min(mapping) < 0 or max(mapping) >= len(ranks):
            raise ValueError("head mapping does not match group ranks")
        largest = max(ranks) if maximum_rank is None else int(maximum_rank)
        if largest <= 0 or any(rank > largest for rank in ranks):
            raise ValueError("group rank exceeds the padded maximum rank")
        head_ranks = tuple(ranks[group] for group in mapping)
        offsets: list[int] = []
        running = 0
        for rank in head_ranks:
            offsets.append(running)
            running += rank
        return cls(
            group_ranks=ranks,
            head_to_kv_group=mapping,
            head_ranks=head_ranks,
            head_offsets=tuple(offsets),
            maximum_rank=largest,
            total_width=running,
        )

    def head_slice(self, head_index: int) -> slice:
        start = self.head_offsets[int(head_index)]
        return slice(start, start + self.head_ranks[int(head_index)])

    def active_flat_indices(self, *, device: torch.device) -> torch.Tensor:
        pieces = [
            head * self.maximum_rank
            + torch.arange(rank, device=device, dtype=torch.long)
            for head, rank in enumerate(self.head_ranks)
        ]
        return torch.cat(pieces)


@dataclass(frozen=True)
class FoldedRaggedRoutedOVFactors:
    v_group_weights: tuple[torch.Tensor, ...]
    o_group_weights: tuple[torch.Tensor, ...]
    maximum_qr_product_error: float
    maximum_dense_fold_error: float


@dataclass(frozen=True)
class GroupPooledRoutedSVDGroupDiagnostics:
    group_index: int
    head_indices: tuple[int, ...]
    rank: int
    covariance_trace_scale: float
    requested_relative_ridge: float
    requested_absolute_ridge: float
    effective_relative_ridge: float
    effective_absolute_ridge: float
    covariance_condition_estimate: float
    leading_singular_values: tuple[float, ...]
    boundary_singular_value: float
    next_singular_value: float | None
    relative_boundary_gap: float | None
    weighted_tail_energy_fraction: float
    metric_orthogonality_error: float
    euclidean_orthogonality_error: float


@dataclass(frozen=True)
class GroupPooledRoutedSVDInitialization:
    A_unique: torch.Tensor
    group_ranks: tuple[int, ...]
    maximum_rank: int
    groups: tuple[GroupPooledRoutedSVDGroupDiagnostics, ...]


def _validate_covariance_blocks(blocks: torch.Tensor) -> None:
    if blocks.ndim != 4:
        raise ValueError("covariance must have shape [H, H, d, d]")
    heads, heads_again, width, width_again = blocks.shape
    if heads <= 0 or width <= 0 or heads != heads_again or width != width_again:
        raise ValueError("covariance block axes must be positive and square")
    if not torch.isfinite(blocks).all():
        raise ValueError("covariance must contain only finite values")


def _symmetric_blocks(blocks: torch.Tensor) -> torch.Tensor:
    return 0.5 * (blocks + blocks.permute(1, 0, 3, 2))


def trace_normalize_covariance(
    covariance: torch.Tensor,
    *,
    epsilon: float | None = None,
) -> tuple[torch.Tensor, float]:
    _validate_covariance_blocks(covariance)
    covariance = _symmetric_blocks(covariance)
    trace = sum(torch.trace(covariance[index, index]) for index in range(covariance.shape[0]))
    tiny = (
        float(epsilon)
        if epsilon is not None
        else torch.finfo(covariance.dtype).eps * covariance.shape[0] * covariance.shape[2]
    )
    if not torch.isfinite(trace) or float(trace) <= tiny:
        raise ValueError("covariance must have positive finite trace")
    dimension = covariance.shape[0] * covariance.shape[2]
    scale = float(dimension / trace)
    return covariance * scale, scale


def covariance_with_trace_damping(
    covariance: torch.Tensor,
    *,
    relative_damping: float,
) -> tuple[torch.Tensor, float]:
    if relative_damping < 0:
        raise ValueError("relative damping must be non-negative")
    _validate_covariance_blocks(covariance)
    work = _symmetric_blocks(covariance).clone()
    trace = sum(torch.trace(work[index, index]) for index in range(work.shape[0]))
    dimension = work.shape[0] * work.shape[2]
    absolute = relative_damping * float(trace) / dimension
    if absolute:
        identity = torch.eye(work.shape[2], device=work.device, dtype=work.dtype)
        for index in range(work.shape[0]):
            work[index, index].add_(absolute * identity)
    return work, absolute


def _canonicalize_column_signs(value: torch.Tensor) -> torch.Tensor:
    """Make each column's largest-magnitude entry non-negative."""

    if value.ndim != 2:
        raise ValueError("column sign canonicalization requires a matrix")
    if value.shape[1] == 0:
        return value
    pivots = value.abs().argmax(dim=0)
    columns = torch.arange(value.shape[1], device=value.device)
    signs = torch.sign(value[pivots, columns])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return value * signs.unsqueeze(0)


def _deterministic_thin_qr(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q, r = torch.linalg.qr(value, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0), signs.unsqueeze(1) * r


def _trace_ridge_cholesky(
    covariance: torch.Tensor,
    *,
    relative_ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Return a trace-ridge-stabilized covariance and its Cholesky factor."""

    if relative_ridge < 0:
        raise ValueError("relative covariance ridge must be non-negative")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("pooled covariance must be square")
    symmetric = 0.5 * (covariance + covariance.T)
    if not torch.isfinite(symmetric).all():
        raise FloatingPointError("pooled covariance contains non-finite values")
    trace_scale = float(torch.trace(symmetric)) / symmetric.shape[0]
    if not torch.isfinite(torch.as_tensor(trace_scale)) or trace_scale <= 0:
        raise ValueError("pooled covariance must have positive finite trace")
    identity = torch.eye(
        symmetric.shape[0],
        device=symmetric.device,
        dtype=symmetric.dtype,
    )
    candidates = [float(relative_ridge)]
    candidates.extend(10.0**exponent for exponent in range(-12, -1))
    for candidate in sorted(
        {
            value
            for value in candidates
            if value >= relative_ridge and value <= 1e-2
        }
    ):
        absolute = candidate * trace_scale
        damped = symmetric + absolute * identity
        cholesky, info = torch.linalg.cholesky_ex(
            damped,
            check_errors=False,
        )
        if int(info.max()) != 0:
            continue
        diagonal = torch.diagonal(cholesky).abs()
        condition = float(
            (
                diagonal.max()
                / diagonal.min().clamp_min(torch.finfo(diagonal.dtype).tiny)
            ).square()
        )
        return damped, cholesky, trace_scale, candidate, condition
    raise torch.linalg.LinAlgError(
        "group-pooled covariance Cholesky failed through relative ridge 1e-2"
    )


@torch.no_grad()
def initialize_group_pooled_routed_svd(
    *,
    covariance: torch.Tensor,
    target: torch.Tensor,
    head_to_kv_group: torch.Tensor | Sequence[int],
    group_ranks: Sequence[int],
    covariance_ridge: float = 1e-7,
) -> GroupPooledRoutedSVDInitialization:
    """Initialize foldable group encoders from a pooled routed metric.

    For group ``g``, the diagonal routed covariance blocks assigned to the
    group are averaged into ``C_pool``.  If

        C_pool + ridge * trace(C_pool) / d * I = L L^T,

    the leading left singular subspace of ``L^T O_cat`` is mapped back through
    ``L^{-T}`` and Euclidean-QR canonicalized.  The returned encoder therefore
    spans the exact optimum of the damped group-pooled weighted low-rank
    surrogate.  Cross-head covariance is intentionally left to the subsequent
    full-layer decoder solve and encoder update.
    """

    _validate_covariance_blocks(covariance)
    heads, _, head_dim, _ = covariance.shape
    if target.ndim != 3 or tuple(target.shape[:2]) != (heads, head_dim):
        raise ValueError(
            "target must have shape [num_query_heads, head_dim, hidden_size]"
        )
    if not torch.isfinite(target).all():
        raise FloatingPointError("routed output target contains non-finite values")
    mapping = torch.as_tensor(
        head_to_kv_group,
        dtype=torch.long,
        device=covariance.device,
    )
    if tuple(mapping.shape) != (heads,):
        raise ValueError("head-to-KV mapping must have one entry per query head")
    ranks = tuple(int(rank) for rank in group_ranks)
    if not ranks or any(rank <= 0 or rank > head_dim for rank in ranks):
        raise ValueError("group ranks must lie within [1, head_dim]")
    if int(mapping.min()) < 0 or int(mapping.max()) >= len(ranks):
        raise ValueError("head-to-KV mapping contains an invalid group")
    if covariance.device != target.device or covariance.dtype != target.dtype:
        raise ValueError("covariance and target must share device and dtype")

    maximum_rank = max(ranks)
    encoders = covariance.new_zeros(len(ranks), head_dim, maximum_rank)
    diagnostics: list[GroupPooledRoutedSVDGroupDiagnostics] = []
    tiny = torch.finfo(covariance.dtype).tiny
    for group, rank in enumerate(ranks):
        head_index_tensor = torch.nonzero(
            mapping == group,
            as_tuple=False,
        ).flatten()
        if head_index_tensor.numel() == 0:
            raise ValueError(f"physical KV group {group} has no query heads")
        head_indices = tuple(int(item) for item in head_index_tensor.tolist())
        pooled = covariance[
            head_index_tensor,
            head_index_tensor,
        ].mean(dim=0)
        (
            damped,
            cholesky,
            trace_scale,
            effective_relative_ridge,
            condition,
        ) = _trace_ridge_cholesky(
            pooled,
            relative_ridge=covariance_ridge,
        )
        output_blocks = target.index_select(0, head_index_tensor)
        output_concatenated = output_blocks.permute(1, 0, 2).reshape(
            head_dim,
            -1,
        )
        whitened = cholesky.T @ output_concatenated
        left_gram = whitened @ whitened.T
        left_gram = 0.5 * (left_gram + left_gram.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(left_gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues.index_select(0, order).clamp_min(0)
        eigenvectors = _canonicalize_column_signs(
            eigenvectors.index_select(1, order)
        )
        singular_values = torch.sqrt(eigenvalues)
        leading = eigenvectors[:, :rank]
        raw_encoder = torch.linalg.solve_triangular(
            cholesky.T,
            leading,
            upper=True,
        )
        metric_identity = torch.eye(
            rank,
            device=covariance.device,
            dtype=covariance.dtype,
        )
        metric_error = float(
            torch.linalg.vector_norm(
                raw_encoder.T @ damped @ raw_encoder - metric_identity
            )
        )
        encoder, _ = _deterministic_thin_qr(raw_encoder)
        euclidean_error = float(
            torch.linalg.vector_norm(encoder.T @ encoder - metric_identity)
        )
        encoders[group, :, :rank] = encoder
        total_energy = float(eigenvalues.sum())
        tail_energy = float(eigenvalues[rank:].sum())
        boundary = float(singular_values[rank - 1])
        if rank < head_dim:
            next_value: float | None = float(singular_values[rank])
            relative_gap: float | None = (
                boundary - next_value
            ) / max(boundary, float(tiny))
        else:
            next_value = None
            relative_gap = None
        diagnostics.append(
            GroupPooledRoutedSVDGroupDiagnostics(
                group_index=group,
                head_indices=head_indices,
                rank=rank,
                covariance_trace_scale=trace_scale,
                requested_relative_ridge=float(covariance_ridge),
                requested_absolute_ridge=float(covariance_ridge) * trace_scale,
                effective_relative_ridge=effective_relative_ridge,
                effective_absolute_ridge=effective_relative_ridge * trace_scale,
                covariance_condition_estimate=condition,
                leading_singular_values=tuple(
                    float(item)
                    for item in singular_values[
                        : min(head_dim, max(rank + 1, 16))
                    ].tolist()
                ),
                boundary_singular_value=boundary,
                next_singular_value=next_value,
                relative_boundary_gap=relative_gap,
                weighted_tail_energy_fraction=(
                    tail_energy / total_energy if total_energy > 0 else 0.0
                ),
                metric_orthogonality_error=metric_error,
                euclidean_orthogonality_error=euclidean_error,
            )
        )
    return GroupPooledRoutedSVDInitialization(
        A_unique=encoders,
        group_ranks=ranks,
        maximum_rank=maximum_rank,
        groups=tuple(diagnostics),
    )


def mask_routed_covariance(
    covariance: torch.Tensor,
    *,
    head_to_kv_group: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    _validate_covariance_blocks(covariance)
    mapping = torch.as_tensor(
        head_to_kv_group,
        dtype=torch.long,
        device=covariance.device,
    )
    if tuple(mapping.shape) != (covariance.shape[0],):
        raise ValueError("head mapping must have shape [H]")
    mode = mode.lower()
    if mode not in {"diagonal", "within_group", "full_layer"}:
        raise ValueError(f"unsupported head coupling mode: {mode}")
    if mode == "full_layer":
        return _symmetric_blocks(covariance)
    heads = mapping.numel()
    keep = torch.eye(heads, dtype=torch.bool, device=covariance.device)
    if mode == "within_group":
        keep = mapping[:, None] == mapping[None, :]
    masked = covariance * keep[:, :, None, None]
    return _symmetric_blocks(masked)


def function_prior_covariance(
    value_metrics: torch.Tensor,
    *,
    head_to_kv_group: torch.Tensor,
) -> torch.Tensor:
    """Repeat one physical-Value metric on each assigned query-head block."""

    if value_metrics.ndim != 3 or value_metrics.shape[1] != value_metrics.shape[2]:
        raise ValueError("value metrics must have shape [G, d, d]")
    mapping = torch.as_tensor(
        head_to_kv_group,
        dtype=torch.long,
        device=value_metrics.device,
    )
    if torch.any(mapping < 0) or torch.any(mapping >= value_metrics.shape[0]):
        raise ValueError("head mapping contains an invalid group")
    heads = mapping.numel()
    width = value_metrics.shape[1]
    result = value_metrics.new_zeros(heads, heads, width, width)
    for head_index, group_index in enumerate(mapping.tolist()):
        result[head_index, head_index] = value_metrics[group_index]
    return _symmetric_blocks(result)


def block_matrix_product(
    covariance: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    _validate_covariance_blocks(covariance)
    expected = (
        covariance.shape[0],
        covariance.shape[2],
    )
    if target.ndim != 3 or tuple(target.shape[:2]) != expected:
        raise ValueError(f"target must have shape [H, d, D], got {tuple(target.shape)}")
    return torch.einsum("hkij,kjo->hio", covariance, target)


def quadratic_from_target(
    *,
    covariance: torch.Tensor,
    target: torch.Tensor,
    name: str,
    trace_normalize: bool = True,
    precomputed_cross: torch.Tensor | None = None,
    precomputed_constant: float | torch.Tensor | None = None,
) -> RoutedOVQuadratic:
    _validate_covariance_blocks(covariance)
    work = _symmetric_blocks(covariance)
    scale = 1.0
    if trace_normalize:
        work, scale = trace_normalize_covariance(work)
    if precomputed_cross is None:
        cross = block_matrix_product(work, target)
    else:
        if tuple(precomputed_cross.shape) != tuple(target.shape):
            raise ValueError("precomputed cross must match target shape")
        cross = precomputed_cross.to(device=work.device, dtype=work.dtype) * scale
    if precomputed_constant is None:
        constant = torch.sum(target.to(work) * cross)
    else:
        constant = torch.as_tensor(
            precomputed_constant,
            device=work.device,
            dtype=work.dtype,
        ) * scale
    return RoutedOVQuadratic(
        covariance=work,
        cross=cross,
        constant=constant,
        name=name,
    )


def combine_quadratics(
    left: RoutedOVQuadratic,
    right: RoutedOVQuadratic,
    *,
    right_weight: float,
    name: str | None = None,
) -> RoutedOVQuadratic:
    if not 0.0 <= right_weight <= 1.0:
        raise ValueError("quadratic mixture weight must be in [0, 1]")
    if (
        left.covariance.shape != right.covariance.shape
        or left.cross.shape != right.cross.shape
    ):
        raise ValueError("quadratics must have matching shapes")
    left_weight = 1.0 - right_weight
    return RoutedOVQuadratic(
        covariance=left_weight * left.covariance + right_weight * right.covariance,
        cross=left_weight * left.cross + right_weight * right.cross,
        constant=left_weight * left.constant + right_weight * right.constant,
        name=name or f"{left.name}_{left_weight:g}+{right.name}_{right_weight:g}",
    )


def add_isotropic_product_prior(
    objective: RoutedOVQuadratic,
    *,
    center_product: torch.Tensor,
    absolute_weight: float,
    name: str | None = None,
) -> RoutedOVQuadratic:
    """Add ``weight * ||M - M0||_F^2`` to a routed product objective.

    The prior is expressed in dense head-product coordinates rather than in a
    particular encoder/decoder gauge.  Consequently it remains well-defined
    while alternating updates change the shared Value encoders.
    """

    weight = float(absolute_weight)
    if not torch.isfinite(torch.as_tensor(weight)) or weight < 0:
        raise ValueError("product-prior weight must be finite and non-negative")
    expected = (
        objective.num_query_heads,
        objective.head_dim,
        objective.hidden_size,
    )
    if tuple(center_product.shape) != expected:
        raise ValueError(
            f"center product must have shape {expected}, "
            f"got {tuple(center_product.shape)}"
        )
    if (
        center_product.device != objective.covariance.device
        or center_product.dtype != objective.covariance.dtype
    ):
        raise ValueError("product prior and objective must share device/dtype")
    if not torch.isfinite(center_product).all():
        raise FloatingPointError("product-prior center contains non-finite values")
    if weight == 0:
        return RoutedOVQuadratic(
            covariance=objective.covariance.clone(),
            cross=objective.cross.clone(),
            constant=objective.constant.clone(),
            name=name or objective.name,
        )
    covariance = objective.covariance.clone()
    identity = torch.eye(
        objective.head_dim,
        device=covariance.device,
        dtype=covariance.dtype,
    )
    for head in range(objective.num_query_heads):
        covariance[head, head].add_(weight * identity)
    return RoutedOVQuadratic(
        covariance=covariance,
        cross=objective.cross + weight * center_product,
        constant=(
            objective.constant
            + weight * torch.sum(center_product.square())
        ),
        name=name or f"{objective.name}+isotropic_product_prior_{weight:g}",
    )


def head_products(
    A_unique: torch.Tensor,
    D_heads: torch.Tensor,
    head_to_kv_group: torch.Tensor,
) -> torch.Tensor:
    if A_unique.ndim != 3 or D_heads.ndim != 3:
        raise ValueError("A and D must have shapes [G,d,r] and [H,r,D]")
    mapping = torch.as_tensor(
        head_to_kv_group,
        dtype=torch.long,
        device=A_unique.device,
    )
    if mapping.numel() != D_heads.shape[0]:
        raise ValueError("head mapping and decoder head count differ")
    if A_unique.shape[2] != D_heads.shape[1]:
        raise ValueError("encoder and decoder ranks differ")
    return torch.einsum("hia,hao->hio", A_unique.index_select(0, mapping), D_heads)


def _reduced_normal_equations(
    objective: RoutedOVQuadratic,
    A_unique: torch.Tensor,
    head_to_kv_group: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mapping = torch.as_tensor(
        head_to_kv_group,
        dtype=torch.long,
        device=A_unique.device,
    )
    by_head = A_unique.index_select(0, mapping)
    hessian = torch.einsum(
        "hia,hkij,kjb->hkab",
        by_head,
        objective.covariance,
        by_head,
    )
    rhs = torch.einsum("hia,hio->hao", by_head, objective.cross)
    return hessian, rhs


def _partitions(
    head_to_kv_group: torch.Tensor,
    *,
    mode: str,
) -> tuple[tuple[int, ...], ...]:
    mapping = torch.as_tensor(head_to_kv_group, dtype=torch.long).cpu()
    if mode == "diagonal":
        return tuple((index,) for index in range(mapping.numel()))
    if mode == "within_group":
        return tuple(
            tuple(torch.nonzero(mapping == group, as_tuple=False).flatten().tolist())
            for group in torch.unique(mapping, sorted=True).tolist()
        )
    if mode == "full_layer":
        return (tuple(range(mapping.numel())),)
    raise ValueError(f"unsupported coupling mode: {mode}")


def _cholesky_with_jitter(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    relative_jitter: float,
) -> tuple[torch.Tensor, float, float]:
    if relative_jitter < 0:
        raise ValueError("relative jitter must be non-negative")
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    scale = float(torch.diagonal(matrix).abs().mean().clamp_min(torch.finfo(matrix.dtype).tiny))
    relatives = [relative_jitter]
    if relative_jitter == 0:
        relatives = [0.0]
    relatives.extend(10.0**exponent for exponent in range(-12, -1))
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    seen: set[float] = set()
    for relative in relatives:
        if relative in seen or relative > 1e-2:
            continue
        seen.add(relative)
        absolute = relative * scale
        chol, info = torch.linalg.cholesky_ex(
            matrix + absolute * identity,
            check_errors=False,
        )
        if int(info.max()) != 0:
            continue
        solution = torch.cholesky_solve(rhs, chol)
        diagonal = torch.diagonal(chol).abs()
        condition = float(
            (
                diagonal.max()
                / diagonal.min().clamp_min(torch.finfo(matrix.dtype).tiny)
            ).square()
        )
        return solution, absolute, condition
    raise torch.linalg.LinAlgError("reduced decoder solve failed through relative jitter 1e-2")


def solve_free_decoder(
    *,
    objective: RoutedOVQuadratic,
    A_unique: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    coupling_mode: str,
    relative_jitter: float = 0.0,
    linear_solve_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, LinearSolveDiagnostics]:
    hessian, rhs = _reduced_normal_equations(
        objective,
        A_unique,
        head_to_kv_group,
    )
    heads, rank, hidden = rhs.shape
    result = rhs.new_empty(heads, rank, hidden)
    jitters = []
    conditions = []
    dimensions = []
    residuals = []
    wall_times = []
    partitions = _partitions(head_to_kv_group, mode=coupling_mode)
    for partition in partitions:
        index = torch.tensor(partition, device=rhs.device, dtype=torch.long)
        block_hessian = (
            hessian.index_select(0, index)
            .index_select(1, index)
            .permute(0, 2, 1, 3)
            .reshape(len(partition) * rank, len(partition) * rank)
        )
        block_rhs = rhs.index_select(0, index).reshape(
            len(partition) * rank,
            hidden,
        )
        solve_hessian = (
            block_hessian
            if linear_solve_dtype is None
            else block_hessian.to(dtype=linear_solve_dtype)
        )
        solve_rhs = (
            block_rhs
            if linear_solve_dtype is None
            else block_rhs.to(dtype=linear_solve_dtype)
        )
        started = time.monotonic()
        solved_work, absolute, condition = _cholesky_with_jitter(
            solve_hessian,
            solve_rhs,
            relative_jitter=relative_jitter,
        )
        wall_times.append(time.monotonic() - started)
        residuals.append(
            float(
                torch.linalg.vector_norm(
                    solve_hessian @ solved_work - solve_rhs
                )
                / torch.linalg.vector_norm(solve_rhs).clamp_min(
                    torch.finfo(solve_rhs.dtype).tiny
                )
            )
        )
        solved = solved_work.to(dtype=rhs.dtype)
        result.index_copy_(0, index, solved.reshape(len(partition), rank, hidden))
        jitters.append(absolute)
        conditions.append(condition)
        dimensions.append(int(block_hessian.shape[0]))
    return result, LinearSolveDiagnostics(
        partition_sizes=tuple(len(item) for item in partitions),
        absolute_jitters=tuple(jitters),
        condition_estimates=tuple(conditions),
        matrix_dimensions=tuple(dimensions),
        relative_residuals=tuple(residuals),
        wall_times_seconds=tuple(wall_times),
    )


def _validate_ragged_padded_factors(
    A_unique: torch.Tensor,
    D_heads: torch.Tensor,
    layout: RaggedHeadLayout,
) -> None:
    if A_unique.ndim != 3 or D_heads.ndim != 3:
        raise ValueError("ragged factors must be padded rank-3 tensors")
    if A_unique.shape[0] != len(layout.group_ranks):
        raise ValueError("ragged encoder group count does not match layout")
    if D_heads.shape[0] != len(layout.head_ranks):
        raise ValueError("ragged decoder head count does not match layout")
    if (
        A_unique.shape[2] != layout.maximum_rank
        or D_heads.shape[1] != layout.maximum_rank
    ):
        raise ValueError("ragged factors do not use the declared padded rank")
    tolerance = 128.0 * torch.finfo(A_unique.dtype).eps
    for group, rank in enumerate(layout.group_ranks):
        if rank < layout.maximum_rank and bool(
            torch.any(A_unique[group, :, rank:].abs() > tolerance)
        ):
            raise ValueError("ragged encoder has nonzero inactive columns")
    for head, rank in enumerate(layout.head_ranks):
        if rank < layout.maximum_rank and bool(
            torch.any(D_heads[head, rank:].abs() > tolerance)
        ):
            raise ValueError("ragged decoder has nonzero inactive rows")


def solve_free_decoder_ragged(
    *,
    objective: RoutedOVQuadratic,
    A_unique: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    group_ranks: Sequence[int],
    relative_jitter: float = 0.0,
    linear_solve_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, LinearSolveDiagnostics]:
    """Solve the full-layer decoder over only active ragged coordinates."""

    layout = RaggedHeadLayout.from_group_ranks(
        group_ranks,
        head_to_kv_group,
        maximum_rank=A_unique.shape[2],
    )
    dummy = A_unique.new_zeros(
        len(layout.head_ranks),
        layout.maximum_rank,
        objective.hidden_size,
    )
    _validate_ragged_padded_factors(A_unique, dummy, layout)
    hessian, rhs = _reduced_normal_equations(
        objective,
        A_unique,
        head_to_kv_group,
    )
    heads, padded_rank, hidden = rhs.shape
    flat_hessian = hessian.permute(0, 2, 1, 3).reshape(
        heads * padded_rank,
        heads * padded_rank,
    )
    flat_rhs = rhs.reshape(heads * padded_rank, hidden)
    active = layout.active_flat_indices(device=rhs.device)
    packed_hessian = flat_hessian.index_select(0, active).index_select(1, active)
    packed_rhs = flat_rhs.index_select(0, active)
    solve_hessian = (
        packed_hessian
        if linear_solve_dtype is None
        else packed_hessian.to(dtype=linear_solve_dtype)
    )
    solve_rhs = (
        packed_rhs
        if linear_solve_dtype is None
        else packed_rhs.to(dtype=linear_solve_dtype)
    )
    started = time.monotonic()
    solved_work, absolute, condition = _cholesky_with_jitter(
        solve_hessian,
        solve_rhs,
        relative_jitter=relative_jitter,
    )
    wall_time = time.monotonic() - started
    residual = float(
        torch.linalg.vector_norm(solve_hessian @ solved_work - solve_rhs)
        / torch.linalg.vector_norm(solve_rhs).clamp_min(
            torch.finfo(solve_rhs.dtype).tiny
        )
    )
    solved = solved_work.to(dtype=rhs.dtype)
    flat_result = rhs.new_zeros(heads * padded_rank, hidden)
    flat_result.index_copy_(0, active, solved)
    return flat_result.reshape(heads, padded_rank, hidden), LinearSolveDiagnostics(
        partition_sizes=(heads,),
        absolute_jitters=(absolute,),
        condition_estimates=(condition,),
        matrix_dimensions=(layout.total_width,),
        relative_residuals=(residual,),
        wall_times_seconds=(wall_time,),
    )


def _evaluate_quadratic_with_scale(
    objective: RoutedOVQuadratic,
    A_unique: torch.Tensor,
    D_heads: torch.Tensor,
    head_to_kv_group: torch.Tensor,
) -> tuple[float, float]:
    hessian, rhs = _reduced_normal_equations(
        objective,
        A_unique,
        head_to_kv_group,
    )
    quadratic = torch.einsum("hao,hkab,kbo->", D_heads, hessian, D_heads)
    linear = torch.sum(D_heads * rhs)
    loss = objective.constant - 2.0 * linear + quadratic
    expanded_scale = torch.stack(
        (
            objective.constant.abs(),
            (2.0 * linear).abs(),
            quadratic.abs(),
            loss.new_ones(()),
        )
    ).max()
    tolerance = (
        max(1024.0 * torch.finfo(loss.dtype).eps, 1e-10)
        * expanded_scale
    )
    if float(loss) < 0:
        if bool(loss.abs() <= tolerance):
            loss = loss.new_zeros(())
        else:
            raise FloatingPointError(
                "PSD routed OV objective is materially negative: "
                f"loss={float(loss):.6e}, tolerance={float(tolerance):.6e}"
            )
    if not torch.isfinite(loss):
        raise FloatingPointError("routed OV objective became non-finite")
    return float(loss), float(expanded_scale)


def evaluate_quadratic(
    objective: RoutedOVQuadratic,
    A_unique: torch.Tensor,
    D_heads: torch.Tensor,
    head_to_kv_group: torch.Tensor,
) -> float:
    return _evaluate_quadratic_with_scale(
        objective,
        A_unique,
        D_heads,
        head_to_kv_group,
    )[0]


def decoder_is_stationary(
    *,
    objective: RoutedOVQuadratic,
    A_unique: torch.Tensor,
    D_heads: torch.Tensor,
    head_to_kv_group: torch.Tensor,
) -> float:
    hessian, rhs = _reduced_normal_equations(
        objective,
        A_unique,
        head_to_kv_group,
    )
    gradient = torch.einsum("hkab,kbo->hao", hessian, D_heads) - rhs
    denominator = torch.linalg.vector_norm(rhs).clamp_min(
        torch.finfo(rhs.dtype).tiny
    )
    return float(torch.linalg.vector_norm(gradient) / denominator)


def decoder_is_stationary_ragged(
    *,
    objective: RoutedOVQuadratic,
    A_unique: torch.Tensor,
    D_heads: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    group_ranks: Sequence[int],
) -> float:
    layout = RaggedHeadLayout.from_group_ranks(
        group_ranks,
        head_to_kv_group,
        maximum_rank=A_unique.shape[2],
    )
    _validate_ragged_padded_factors(A_unique, D_heads, layout)
    hessian, rhs = _reduced_normal_equations(
        objective,
        A_unique,
        head_to_kv_group,
    )
    gradient = torch.einsum("hkab,kbo->hao", hessian, D_heads) - rhs
    active_gradient = []
    active_rhs = []
    for head, rank in enumerate(layout.head_ranks):
        active_gradient.append(gradient[head, :rank])
        active_rhs.append(rhs[head, :rank])
    numerator = torch.linalg.vector_norm(torch.cat(active_gradient, dim=0))
    denominator = torch.linalg.vector_norm(torch.cat(active_rhs, dim=0)).clamp_min(
        torch.finfo(rhs.dtype).tiny
    )
    return float(numerator / denominator)


def _decoder_cross_grams(D_heads: torch.Tensor) -> torch.Tensor:
    return torch.einsum("kao,hbo->khab", D_heads, D_heads)


def _per_head_encoder_half_gradient(
    *,
    objective: RoutedOVQuadratic,
    A_unique: torch.Tensor,
    D_heads: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    decoder_grams: torch.Tensor,
) -> torch.Tensor:
    mapping = torch.as_tensor(
        head_to_kv_group,
        dtype=torch.long,
        device=A_unique.device,
    )
    by_head = A_unique.index_select(0, mapping)
    quadratic = torch.einsum(
        "hkij,kja,khab->hib",
        objective.covariance,
        by_head,
        decoder_grams,
    )
    linear = torch.einsum("hio,hbo->hib", objective.cross, D_heads)
    return quadratic - linear


def encoder_hessian_vector_product(
    *,
    covariance: torch.Tensor,
    D_heads: torch.Tensor,
    head_indices: Sequence[int],
    delta: torch.Tensor,
    decoder_grams: torch.Tensor | None = None,
) -> torch.Tensor:
    indices = tuple(int(item) for item in head_indices)
    if not indices:
        raise ValueError("encoder group must contain at least one head")
    grams = (
        _decoder_cross_grams(D_heads)
        if decoder_grams is None
        else decoder_grams
    )
    result = torch.zeros_like(delta)
    for h in indices:
        for k in indices:
            result.add_(covariance[h, k] @ delta @ grams[k, h])
    return result


def encoder_group_half_gradient(
    per_head_half_gradient: torch.Tensor,
    head_indices: Sequence[int],
) -> torch.Tensor:
    index = torch.tensor(
        tuple(int(item) for item in head_indices),
        device=per_head_half_gradient.device,
        dtype=torch.long,
    )
    return per_head_half_gradient.index_select(0, index).sum(dim=0)


def _encoder_operator_scale(
    *,
    covariance: torch.Tensor,
    decoder_grams: torch.Tensor,
    head_indices: Sequence[int],
    head_dim: int,
    rank: int,
) -> float:
    trace = covariance.new_zeros(())
    for h in head_indices:
        for k in head_indices:
            trace.add_(
                torch.trace(covariance[h, k])
                * torch.trace(decoder_grams[k, h])
            )
    return max(
        float(trace.abs()) / (head_dim * rank),
        torch.finfo(covariance.dtype).tiny,
    )


def exact_encoder_hessian_diagonal(
    *,
    covariance: torch.Tensor,
    decoder_grams: torch.Tensor,
    head_indices: Sequence[int],
) -> torch.Tensor:
    """Return the exact diagonal of the repository's half-Hessian operator."""

    indices = tuple(int(item) for item in head_indices)
    if not indices:
        raise ValueError("encoder group must contain at least one head")
    head_dim = covariance.shape[-1]
    rank = decoder_grams.shape[-1]
    result = covariance.new_zeros(head_dim, rank)
    for h in indices:
        for k in indices:
            result.add_(
                torch.diagonal(covariance[h, k]).unsqueeze(1)
                * torch.diagonal(decoder_grams[k, h]).unsqueeze(0)
            )
    return result


def separable_encoder_hessian_factors(
    *,
    covariance: torch.Tensor,
    decoder_grams: torch.Tensor,
    head_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the diagonal-head two-sided approximation Cbar ΔA Rbar."""

    indices = tuple(int(item) for item in head_indices)
    if not indices:
        raise ValueError("encoder group must contain at least one head")
    left = sum(
        (covariance[h, h] for h in indices),
        start=torch.zeros_like(covariance[indices[0], indices[0]]),
    )
    right = sum(
        (decoder_grams[h, h] for h in indices),
        start=torch.zeros_like(decoder_grams[indices[0], indices[0]]),
    )
    return 0.5 * (left + left.T), 0.5 * (right + right.T)


def _trace_relative_damping(
    value: torch.Tensor,
    relative_damping: float,
) -> tuple[torch.Tensor, float]:
    if relative_damping < 0:
        raise ValueError("relative damping must be non-negative")
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("damped matrix must be square")
    symmetric = 0.5 * (value + value.T)
    scale = float(torch.trace(symmetric)) / symmetric.shape[0]
    if not torch.isfinite(symmetric).all() or not torch.isfinite(
        torch.as_tensor(scale)
    ):
        raise FloatingPointError("two-sided Hessian factor is non-finite")
    absolute = relative_damping * max(
        scale,
        torch.finfo(symmetric.dtype).tiny,
    )
    if absolute:
        symmetric = symmetric + absolute * torch.eye(
            symmetric.shape[0],
            device=symmetric.device,
            dtype=symmetric.dtype,
        )
    return symmetric, absolute


def _spd_cholesky(
    value: torch.Tensor,
    *,
    name: str,
) -> tuple[torch.Tensor, list[float], float]:
    symmetric = 0.5 * (value + value.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    maximum = float(eigenvalues[-1])
    minimum = float(eigenvalues[0])
    tolerance = (
        1024.0
        * torch.finfo(value.dtype).eps
        * max(abs(maximum), 1.0)
    )
    if minimum <= tolerance:
        raise torch.linalg.LinAlgError(
            f"{name} is not numerically positive definite: "
            f"minimum={minimum:.6e}, tolerance={tolerance:.6e}"
        )
    cholesky, info = torch.linalg.cholesky_ex(symmetric)
    if int(info.max()) != 0:
        raise torch.linalg.LinAlgError(f"{name} Cholesky factorization failed")
    condition = maximum / minimum
    return cholesky, [float(item) for item in eigenvalues], condition


def _two_sided_solve(
    *,
    left: torch.Tensor,
    right: torch.Tensor,
    rhs: torch.Tensor,
    left_name: str,
    right_name: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    left_cholesky, left_eigenvalues, left_condition = _spd_cholesky(
        left,
        name=left_name,
    )
    right_cholesky, right_eigenvalues, right_condition = _spd_cholesky(
        right,
        name=right_name,
    )
    temporary = torch.cholesky_solve(rhs, left_cholesky)
    result = torch.cholesky_solve(
        temporary.T,
        right_cholesky,
    ).T
    relative_residual = float(
        torch.linalg.vector_norm(left @ result @ right - rhs)
        / torch.linalg.vector_norm(rhs).clamp_min(
            torch.finfo(rhs.dtype).tiny
        )
    )
    return result, {
        "left_eigenvalues": left_eigenvalues,
        "right_eigenvalues": right_eigenvalues,
        "left_condition_estimate": left_condition,
        "right_condition_estimate": right_condition,
        "two_sided_solve_relative_residual": relative_residual,
    }


def _frobenius_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (
        torch.linalg.vector_norm(left)
        * torch.linalg.vector_norm(right)
    ).clamp_min(torch.finfo(left.dtype).tiny)
    return float(torch.sum(left * right) / denominator)


def _normalize_frobenius(value: torch.Tensor, *, name: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value)
    if not torch.isfinite(norm) or float(norm) <= torch.finfo(value.dtype).tiny:
        raise FloatingPointError(f"cannot normalize zero/non-finite {name}")
    return value / norm


def _project_numerical_psd(
    value: torch.Tensor,
    *,
    relative_negative_tolerance: float,
    name: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    symmetric = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    scale = max(float(eigenvalues.abs().max()), torch.finfo(value.dtype).tiny)
    tolerance = relative_negative_tolerance * scale
    minimum = float(eigenvalues[0])
    if minimum < -tolerance:
        raise torch.linalg.LinAlgError(
            f"{name} has material negative curvature: "
            f"minimum={minimum:.6e}, tolerance={tolerance:.6e}"
        )
    clipped = eigenvalues.clamp_min(0)
    clipping_mass = float(torch.sum((clipped - eigenvalues).abs()))
    denominator = float(torch.sum(eigenvalues.abs()).clamp_min(
        torch.finfo(value.dtype).tiny
    ))
    projected = (eigenvectors * clipped.unsqueeze(0)) @ eigenvectors.T
    return projected, {
        "raw_eigenvalues": [float(item) for item in eigenvalues],
        "negative_clipping_mass": clipping_mass,
        "relative_negative_clipping_mass": clipping_mass / denominator,
        "negative_tolerance": tolerance,
    }


def hessian_best_kronecker_factors(
    *,
    covariance: torch.Tensor,
    decoder_grams: torch.Tensor,
    head_indices: Sequence[int],
    iterations: int,
    convergence_tolerance: float,
    eigenvalue_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fit one matrix-free Kronecker term to an encoder half-Hessian."""

    if iterations <= 0 or convergence_tolerance < 0 or eigenvalue_floor < 0:
        raise ValueError("invalid Hessian Best-Kronecker controls")
    indices = tuple(int(item) for item in head_indices)
    if not indices:
        raise ValueError("encoder group must contain at least one head")
    left_terms = torch.stack(
        [covariance[h, k] for h in indices for k in indices]
    )
    right_terms = torch.stack(
        [decoder_grams[k, h] for h in indices for k in indices]
    )

    def left_action(right_probe: torch.Tensor) -> torch.Tensor:
        coefficients = torch.einsum("tij,ij->t", right_terms, right_probe)
        return torch.einsum("t,tij->ij", coefficients, left_terms)

    def right_action(left_probe: torch.Tensor) -> torch.Tensor:
        coefficients = torch.einsum("tij,ij->t", left_terms, left_probe)
        return torch.einsum("t,tij->ij", coefficients, right_terms)

    separable_left, separable_right = separable_encoder_hessian_factors(
        covariance=covariance,
        decoder_grams=decoder_grams,
        head_indices=indices,
    )
    initializations = {
        "separable": (
            _normalize_frobenius(separable_left, name="separable left"),
            _normalize_frobenius(separable_right, name="separable right"),
        ),
        "identity": (
            torch.eye(
                covariance.shape[-1],
                device=covariance.device,
                dtype=covariance.dtype,
            )
            / covariance.shape[-1] ** 0.5,
            torch.eye(
                decoder_grams.shape[-1],
                device=decoder_grams.device,
                dtype=decoder_grams.dtype,
            )
            / decoder_grams.shape[-1] ** 0.5,
        ),
    }
    runs: dict[str, dict[str, Any]] = {}
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for initialization, (left, right) in initializations.items():
        history = []
        converged = False
        for iteration in range(1, iterations + 1):
            new_left = _normalize_frobenius(
                left_action(right),
                name="Hessian Best-Kronecker left factor",
            )
            new_right = _normalize_frobenius(
                right_action(new_left),
                name="Hessian Best-Kronecker right factor",
            )
            left_change = min(
                float(torch.linalg.vector_norm(new_left - left)),
                float(torch.linalg.vector_norm(new_left + left)),
            )
            right_change = min(
                float(torch.linalg.vector_norm(new_right - right)),
                float(torch.linalg.vector_norm(new_right + right)),
            )
            left, right = new_left, new_right
            sigma = float(torch.sum(left * left_action(right)))
            history.append(
                {
                    "iteration": iteration,
                    "sigma": sigma,
                    "left_change": left_change,
                    "right_change": right_change,
                }
            )
            if max(left_change, right_change) <= convergence_tolerance:
                converged = True
                break
        sigma = float(torch.sum(left * left_action(right)))
        if sigma < 0:
            left = -left
            sigma = -sigma
        runs[initialization] = {
            "sigma": sigma,
            "iterations": len(history),
            "converged": converged,
            "history": history,
        }
        factors[initialization] = (left, right)
    selected = max(runs, key=lambda name: runs[name]["sigma"])
    unit_left, unit_right = factors[selected]
    sigma = float(runs[selected]["sigma"])
    if sigma <= torch.finfo(covariance.dtype).tiny:
        raise FloatingPointError("Hessian Best-Kronecker singular value is zero")
    unit_left, left_psd = _project_numerical_psd(
        unit_left,
        relative_negative_tolerance=eigenvalue_floor,
        name="Hessian Best-Kronecker left factor",
    )
    unit_right, right_psd = _project_numerical_psd(
        unit_right,
        relative_negative_tolerance=eigenvalue_floor,
        name="Hessian Best-Kronecker right factor",
    )
    left_trace = torch.trace(unit_left)
    right_trace = torch.trace(unit_right)
    tiny = torch.finfo(covariance.dtype).tiny
    if float(left_trace) <= tiny or float(right_trace) <= tiny:
        raise FloatingPointError("Hessian Best-Kronecker factor has zero trace")
    canonical_left = unit_left * (unit_left.shape[0] / left_trace)
    canonical_right = unit_right * (unit_right.shape[0] / right_trace)

    exact_norm_square = covariance.new_zeros(())
    for term_index in range(left_terms.shape[0]):
        for other_index in range(left_terms.shape[0]):
            exact_norm_square.add_(
                torch.sum(left_terms[term_index] * left_terms[other_index])
                * torch.sum(right_terms[term_index] * right_terms[other_index])
            )
    explained_fraction = sigma**2 / max(
        float(exact_norm_square),
        torch.finfo(covariance.dtype).tiny,
    )
    diagnostics = {
        "selected_initialization": selected,
        "top_rearranged_singular_value": sigma,
        "exact_hessian_frobenius_norm_square": float(exact_norm_square),
        "explained_hessian_fraction": explained_fraction,
        "power_runs": runs,
        "left_psd": left_psd,
        "right_psd": right_psd,
        "left_overlap_with_separable": _frobenius_cosine(
            canonical_left,
            separable_left,
        ),
        "right_overlap_with_separable": _frobenius_cosine(
            canonical_right,
            separable_right,
        ),
    }
    return canonical_left, canonical_right, diagnostics


@dataclass(frozen=True)
class JacobiEncoderSolver:
    relative_damping: float = 1e-8
    name: str = "jacobi"

    def propose(
        self,
        *,
        half_gradient: torch.Tensor,
        covariance: torch.Tensor,
        decoder_grams: torch.Tensor,
        head_indices: Sequence[int],
    ) -> EncoderDirectionResult:
        diagonal = exact_encoder_hessian_diagonal(
            covariance=covariance,
            decoder_grams=decoder_grams,
            head_indices=head_indices,
        )
        scale = float(diagonal.mean())
        negative_tolerance = (
            1024.0
            * torch.finfo(diagonal.dtype).eps
            * max(abs(float(diagonal.abs().max())), 1.0)
        )
        if float(diagonal.min()) < -negative_tolerance:
            raise FloatingPointError("exact encoder Hessian diagonal is negative")
        damping = self.relative_damping * max(
            scale,
            torch.finfo(diagonal.dtype).tiny,
        )
        damped = diagonal + damping
        if float(damped.min()) <= 0:
            raise torch.linalg.LinAlgError(
                "damped Jacobi encoder Hessian diagonal is singular"
            )
        direction = -half_gradient / damped
        return EncoderDirectionResult(
            direction=direction,
            solver_name=self.name,
            diagnostics={
                "minimum_hessian_diagonal": float(diagonal.min()),
                "maximum_hessian_diagonal": float(diagonal.max()),
                "mean_hessian_diagonal": scale,
                "damped_minimum_hessian_diagonal": float(damped.min()),
                "damped_maximum_hessian_diagonal": float(damped.max()),
                "diagonal_condition_ratio": float(damped.max() / damped.min()),
                "relative_damping": self.relative_damping,
                "absolute_damping": damping,
            },
            complexity={
                "cg_iterations": 0,
                "cholesky_solves": 0,
                "left_matrix_solves": 0,
                "right_matrix_solves": 0,
                "bestkron_power_iterations": 0,
                "head_pair_contractions": len(tuple(head_indices)) ** 2,
            },
        )


@dataclass(frozen=True)
class SeparableEncoderSolver:
    left_relative_damping: float = 1e-8
    right_relative_damping: float = 1e-8
    name: str = "separable"

    def propose(
        self,
        *,
        half_gradient: torch.Tensor,
        covariance: torch.Tensor,
        decoder_grams: torch.Tensor,
        head_indices: Sequence[int],
    ) -> EncoderDirectionResult:
        left, right = separable_encoder_hessian_factors(
            covariance=covariance,
            decoder_grams=decoder_grams,
            head_indices=head_indices,
        )
        damped_left, left_damping = _trace_relative_damping(
            left,
            self.left_relative_damping,
        )
        damped_right, right_damping = _trace_relative_damping(
            right,
            self.right_relative_damping,
        )
        direction, diagnostics = _two_sided_solve(
            left=damped_left,
            right=damped_right,
            rhs=-half_gradient,
            left_name="separable left factor",
            right_name="separable right factor",
        )
        diagnostics.update(
            {
                "left_relative_damping": self.left_relative_damping,
                "right_relative_damping": self.right_relative_damping,
                "left_absolute_damping": left_damping,
                "right_absolute_damping": right_damping,
            }
        )
        return EncoderDirectionResult(
            direction=direction,
            solver_name=self.name,
            diagnostics=diagnostics,
            complexity={
                "cg_iterations": 0,
                "cholesky_solves": 2,
                "left_matrix_solves": 1,
                "right_matrix_solves": 1,
                "bestkron_power_iterations": 0,
                "head_pair_contractions": 2 * len(tuple(head_indices)),
            },
        )


@dataclass(frozen=True)
class HessianBestKronEncoderSolver:
    left_relative_damping: float = 1e-8
    right_relative_damping: float = 1e-8
    iterations: int = 8
    convergence_tolerance: float = 1e-8
    eigenvalue_floor: float = 1e-10
    name: str = "hessian_bestkron"

    def propose(
        self,
        *,
        half_gradient: torch.Tensor,
        covariance: torch.Tensor,
        decoder_grams: torch.Tensor,
        head_indices: Sequence[int],
    ) -> EncoderDirectionResult:
        left, right, diagnostics = hessian_best_kronecker_factors(
            covariance=covariance,
            decoder_grams=decoder_grams,
            head_indices=head_indices,
            iterations=self.iterations,
            convergence_tolerance=self.convergence_tolerance,
            eigenvalue_floor=self.eigenvalue_floor,
        )
        damped_left, left_damping = _trace_relative_damping(
            left,
            self.left_relative_damping,
        )
        damped_right, right_damping = _trace_relative_damping(
            right,
            self.right_relative_damping,
        )
        direction, solve_diagnostics = _two_sided_solve(
            left=damped_left,
            right=damped_right,
            rhs=-half_gradient,
            left_name="Hessian Best-Kronecker left factor",
            right_name="Hessian Best-Kronecker right factor",
        )
        diagnostics.update(solve_diagnostics)
        diagnostics.update(
            {
                "left_relative_damping": self.left_relative_damping,
                "right_relative_damping": self.right_relative_damping,
                "left_absolute_damping": left_damping,
                "right_absolute_damping": right_damping,
            }
        )
        pairs = len(tuple(head_indices)) ** 2
        return EncoderDirectionResult(
            direction=direction,
            solver_name=self.name,
            diagnostics=diagnostics,
            complexity={
                "cg_iterations": 0,
                "cholesky_solves": 2,
                "left_matrix_solves": 1,
                "right_matrix_solves": 1,
                "bestkron_power_iterations": self.iterations,
                "head_pair_contractions": 2 * pairs * self.iterations,
            },
        )


def conjugate_gradient_matrix(
    operator: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    relative_tolerance: float,
    max_iterations: int,
    absolute_damping: float = 0.0,
    fixed_iterations: bool = False,
) -> tuple[torch.Tensor, CGDiagnostics]:
    if relative_tolerance <= 0 or max_iterations <= 0 or absolute_damping < 0:
        raise ValueError("invalid CG tolerance, iteration count, or damping")
    solution = torch.zeros_like(rhs)

    def damped(value: torch.Tensor) -> torch.Tensor:
        result = operator(value)
        if absolute_damping:
            result = result + absolute_damping * value
        return result

    residual = rhs.clone()
    direction = residual.clone()
    residual_square = torch.sum(residual * residual)
    rhs_norm = torch.sqrt(residual_square).clamp_min(torch.finfo(rhs.dtype).tiny)
    initial_residual_norm = float(torch.sqrt(residual_square))
    negative_curvature = False
    converged = initial_residual_norm == 0.0
    iterations = 0
    if converged:
        return solution, CGDiagnostics(
            iterations=0,
            converged=True,
            relative_residual=0.0,
            absolute_damping=float(absolute_damping),
            negative_curvature=False,
            initial_residual_norm=0.0,
            final_residual_norm=0.0,
            fixed_iterations=bool(fixed_iterations),
        )
    for iterations in range(1, max_iterations + 1):
        product = damped(direction)
        curvature = torch.sum(direction * product)
        tolerance = (
            torch.finfo(rhs.dtype).eps
            * torch.linalg.vector_norm(direction).square()
        )
        if float(curvature) <= -float(tolerance):
            negative_curvature = True
            break
        if float(curvature) <= float(tolerance):
            break
        step = residual_square / curvature
        solution.add_(step * direction)
        residual.add_(-step * product)
        new_square = torch.sum(residual * residual)
        relative = float(torch.sqrt(new_square) / rhs_norm)
        if relative <= relative_tolerance:
            residual_square = new_square
            converged = True
            if not fixed_iterations or float(new_square) == 0.0:
                break
        direction = residual + (new_square / residual_square) * direction
        residual_square = new_square
    relative = float(torch.sqrt(residual_square) / rhs_norm)
    converged = relative <= relative_tolerance
    return solution, CGDiagnostics(
        iterations=iterations,
        converged=converged,
        relative_residual=relative,
        absolute_damping=float(absolute_damping),
        negative_curvature=negative_curvature,
        initial_residual_norm=initial_residual_norm,
        final_residual_norm=float(torch.sqrt(residual_square)),
        fixed_iterations=bool(fixed_iterations),
    )


def _increment_encoder_gradients(
    *,
    per_head_gradient: torch.Tensor,
    covariance: torch.Tensor,
    delta: torch.Tensor,
    changed_heads: Sequence[int],
    decoder_grams: torch.Tensor,
) -> None:
    for h in range(per_head_gradient.shape[0]):
        for k in changed_heads:
            per_head_gradient[h].add_(
                covariance[h, k] @ delta @ decoder_grams[k, h]
            )


def gauge_canonicalize(
    A_unique: torch.Tensor,
    D_heads: torch.Tensor,
    head_to_kv_group: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    mapping = torch.as_tensor(
        head_to_kv_group,
        dtype=torch.long,
        device=A_unique.device,
    )
    new_A = A_unique.clone()
    new_D = D_heads.clone()
    maximum_error = 0.0
    for group in range(A_unique.shape[0]):
        old_A = A_unique[group]
        q, r = torch.linalg.qr(old_A, mode="reduced")
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q = q * signs.unsqueeze(0)
        r = signs.unsqueeze(1) * r
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        old_products = old_A @ D_heads.index_select(0, heads)
        adjusted = torch.einsum("ab,hbo->hao", r, D_heads.index_select(0, heads))
        new_products = q @ adjusted
        denominator = torch.linalg.vector_norm(old_products).clamp_min(
            torch.finfo(old_products.dtype).tiny
        )
        maximum_error = max(
            maximum_error,
            float(torch.linalg.vector_norm(new_products - old_products) / denominator),
        )
        new_A[group] = q
        new_D.index_copy_(0, heads, adjusted)
    return new_A, new_D, maximum_error


def gauge_canonicalize_ragged(
    A_unique: torch.Tensor,
    D_heads: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    group_ranks: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Apply deterministic Thin QR independently at every active group rank."""

    layout = RaggedHeadLayout.from_group_ranks(
        group_ranks,
        head_to_kv_group,
        maximum_rank=A_unique.shape[2],
    )
    _validate_ragged_padded_factors(A_unique, D_heads, layout)
    mapping = torch.as_tensor(
        head_to_kv_group,
        dtype=torch.long,
        device=A_unique.device,
    )
    new_A = torch.zeros_like(A_unique)
    new_D = torch.zeros_like(D_heads)
    maximum_error = 0.0
    for group, rank in enumerate(layout.group_ranks):
        old_A = A_unique[group, :, :rank]
        q, r = torch.linalg.qr(old_A, mode="reduced")
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q = q * signs.unsqueeze(0)
        r = signs.unsqueeze(1) * r
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        old_D = D_heads.index_select(0, heads)[:, :rank]
        old_products = old_A @ old_D
        adjusted = torch.einsum("ab,hbo->hao", r, old_D)
        new_products = q @ adjusted
        denominator = torch.linalg.vector_norm(old_products).clamp_min(
            torch.finfo(old_products.dtype).tiny
        )
        maximum_error = max(
            maximum_error,
            float(torch.linalg.vector_norm(new_products - old_products) / denominator),
        )
        new_A[group, :, :rank] = q
        for local, head in enumerate(heads.tolist()):
            new_D[head, :rank] = adjusted[local]
    return new_A, new_D, maximum_error


def _subspace_movement(
    reference: torch.Tensor,
    current: torch.Tensor,
) -> tuple[float, float]:
    reference_q = torch.linalg.qr(reference, mode="reduced").Q
    current_q = torch.linalg.qr(current, mode="reduced").Q
    singular_values = torch.linalg.svdvals(reference_q.T @ current_q).clamp(
        min=0.0,
        max=1.0,
    )
    angles = torch.arccos(singular_values)
    return float(angles.max()), float(angles.mean())


def _zero_cg_diagnostics() -> CGDiagnostics:
    return CGDiagnostics(
        iterations=0,
        converged=True,
        relative_residual=0.0,
        absolute_damping=0.0,
        negative_curvature=False,
        initial_residual_norm=0.0,
        final_residual_norm=0.0,
        fixed_iterations=False,
    )


def fit_routed_ov_joint(
    *,
    objective: RoutedOVQuadratic,
    initial_A: torch.Tensor,
    initial_D: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    coupling_mode: str,
    maximum_sweeps: int = 5,
    minimum_sweeps: int = 1,
    relative_objective_tolerance: float = 1e-7,
    patience: int = 2,
    decoder_relative_jitter: float = 0.0,
    encoder_relative_damping: float = 1e-8,
    cg_relative_tolerance: float = 1e-8,
    cg_max_iterations: int = 200,
    cg_fixed_iterations: bool = False,
    maximum_backtracks: int = 10,
    encoder_group_indices: Sequence[int] | None = None,
    component_objectives: Mapping[str, RoutedOVQuadratic] | None = None,
    checkpoint_callback: RoutedOVCheckpointCallback | None = None,
    final_decoder_solve: bool = False,
    verify_encoder_step_objective: bool = False,
    encoder_direction_solver: EncoderDirectionSolver | None = None,
    encoder_step_scale: float = 1.0,
    group_ranks: Sequence[int] | None = None,
    decoder_solver_override: Callable[
        [
            RoutedOVQuadratic,
            torch.Tensor,
            torch.Tensor,
            RaggedHeadLayout | None,
        ],
        tuple[torch.Tensor, LinearSolveDiagnostics],
    ]
    | None = None,
    decoder_stationarity_override: Callable[
        [
            RoutedOVQuadratic,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            RaggedHeadLayout | None,
        ],
        float,
    ]
    | None = None,
    work_dtype: torch.dtype = torch.float64,
    work_device: torch.device | str | None = None,
) -> RoutedOVJointResult:
    """Run deterministic decoder/group-encoder alternating least squares."""

    if maximum_sweeps < 0 or minimum_sweeps < 0 or minimum_sweeps > maximum_sweeps:
        raise ValueError("invalid sweep limits")
    if patience <= 0 or relative_objective_tolerance < 0:
        raise ValueError("invalid patience or objective tolerance")
    if not 0.0 < encoder_step_scale <= 1.0:
        raise ValueError("encoder step scale must be in (0, 1]")
    if encoder_direction_solver is not None and (
        maximum_sweeps != 1 or minimum_sweeps != 1
    ):
        raise ValueError("one-shot encoder solvers require exactly one sweep")
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("work dtype must be float32 or float64")
    device = (
        torch.device(work_device)
        if work_device is not None
        else objective.covariance.device
    )
    objective = RoutedOVQuadratic(
        covariance=objective.covariance.detach().to(device=device, dtype=work_dtype),
        cross=objective.cross.detach().to(device=device, dtype=work_dtype),
        constant=objective.constant.detach().to(device=device, dtype=work_dtype),
        name=objective.name,
    )
    components = {
        str(name): RoutedOVQuadratic(
            covariance=value.covariance.detach().to(
                device=device,
                dtype=work_dtype,
            ),
            cross=value.cross.detach().to(device=device, dtype=work_dtype),
            constant=value.constant.detach().to(device=device, dtype=work_dtype),
            name=value.name,
        )
        for name, value in (component_objectives or {}).items()
    }
    A = initial_A.detach().to(device=device, dtype=work_dtype).clone()
    D = initial_D.detach().to(device=device, dtype=work_dtype).clone()
    def monotonicity_tolerance(
        loss: float,
        *,
        expanded_scale: float | None = None,
    ) -> float:
        relative = max(
            1.0e-9,
            1024.0 * torch.finfo(work_dtype).eps,
        )
        return relative * max(
            abs(float(objective.constant)),
            abs(loss),
            abs(expanded_scale or 0.0),
            1.0,
        )
    mapping = torch.as_tensor(
        head_to_kv_group,
        device=device,
        dtype=torch.long,
    )
    if A.ndim != 3 or D.ndim != 3:
        raise ValueError("initial factors must have shapes [G,d,r] and [H,r,D]")
    if (
        A.shape[0] <= int(mapping.max())
        or D.shape[0] != mapping.numel()
        or A.shape[1] != objective.head_dim
        or A.shape[2] != D.shape[1]
        or D.shape[2] != objective.hidden_size
    ):
        raise ValueError("initial factors do not match objective geometry")
    ragged_layout = (
        None
        if group_ranks is None
        else RaggedHeadLayout.from_group_ranks(
            group_ranks,
            mapping,
            maximum_rank=A.shape[2],
        )
    )
    if ragged_layout is not None:
        _validate_ragged_padded_factors(A, D, ragged_layout)

    def canonicalize_factors(
        current_A: torch.Tensor,
        current_D: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        if ragged_layout is None:
            return gauge_canonicalize(current_A, current_D, mapping)
        return gauge_canonicalize_ragged(
            current_A,
            current_D,
            mapping,
            ragged_layout.group_ranks,
        )

    def solve_decoder(
        current_A: torch.Tensor,
        *,
        relative_jitter_override: float | None = None,
        linear_solve_dtype_override: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, LinearSolveDiagnostics]:
        if decoder_solver_override is not None:
            if (
                relative_jitter_override is not None
                or linear_solve_dtype_override is not None
            ):
                raise ValueError("decoder solver override cannot use numeric retries")
            solved, diagnostics = decoder_solver_override(
                objective,
                current_A,
                mapping,
                ragged_layout,
            )
            if tuple(solved.shape) != tuple(D.shape):
                raise ValueError(
                    "decoder solver override returned an incompatible tensor"
                )
            if not torch.isfinite(solved).all():
                raise FloatingPointError(
                    "decoder solver override returned non-finite values"
                )
            return solved, diagnostics
        if ragged_layout is None:
            return solve_free_decoder(
                objective=objective,
                A_unique=current_A,
                head_to_kv_group=mapping,
                coupling_mode=coupling_mode,
                relative_jitter=(
                    decoder_relative_jitter
                    if relative_jitter_override is None
                    else relative_jitter_override
                ),
                linear_solve_dtype=linear_solve_dtype_override,
            )
        if coupling_mode != "full_layer":
            raise ValueError("ragged routed fitting currently requires full-layer coupling")
        return solve_free_decoder_ragged(
            objective=objective,
            A_unique=current_A,
            head_to_kv_group=mapping,
            group_ranks=ragged_layout.group_ranks,
            relative_jitter=(
                decoder_relative_jitter
                if relative_jitter_override is None
                else relative_jitter_override
            ),
            linear_solve_dtype=linear_solve_dtype_override,
        )

    def decoder_stationarity(
        current_A: torch.Tensor,
        current_D: torch.Tensor,
    ) -> float:
        if decoder_stationarity_override is not None:
            return float(
                decoder_stationarity_override(
                    objective,
                    current_A,
                    current_D,
                    mapping,
                    ragged_layout,
                )
            )
        if ragged_layout is None:
            return decoder_is_stationary(
                objective=objective,
                A_unique=current_A,
                D_heads=current_D,
                head_to_kv_group=mapping,
            )
        return decoder_is_stationary_ragged(
            objective=objective,
            A_unique=current_A,
            D_heads=current_D,
            head_to_kv_group=mapping,
            group_ranks=ragged_layout.group_ranks,
        )
    selected_groups = (
        tuple(range(A.shape[0]))
        if encoder_group_indices is None
        else tuple(int(item) for item in encoder_group_indices)
    )
    if (
        len(selected_groups) != len(set(selected_groups))
        or any(item < 0 or item >= A.shape[0] for item in selected_groups)
    ):
        raise ValueError("encoder group indices must be unique valid group indices")
    checkpoints: list[RoutedOVCheckpointDiagnostics] = []

    def emit_checkpoint(
        boundary: str,
        sweep: int,
        loss: float,
    ) -> None:
        if checkpoint_callback is None and not components:
            return
        checkpoint = RoutedOVCheckpointDiagnostics(
            boundary=boundary,
            sweep=int(sweep),
            loss=float(loss),
            component_losses=tuple(
                (
                    name,
                    evaluate_quadratic(component, A, D, mapping),
                )
                for name, component in components.items()
            ),
            decoder_relative_stationarity=decoder_stationarity(A, D),
        )
        checkpoints.append(checkpoint)
        if checkpoint_callback is not None:
            checkpoint_callback(checkpoint, A, D)

    A, D, _ = canonicalize_factors(A, D)
    anchor_A = A.clone()

    def monotone_decoder_solve(
        current_A: torch.Tensor,
        current_D: torch.Tensor,
        current_loss: float,
        *,
        boundary: str,
    ) -> tuple[torch.Tensor, LinearSolveDiagnostics, float]:
        candidate_D, candidate_diagnostics = solve_decoder(current_A)
        candidate_loss, candidate_scale = _evaluate_quadratic_with_scale(
            objective, current_A, candidate_D, mapping
        )
        tolerance = monotonicity_tolerance(
            current_loss,
            expanded_scale=candidate_scale,
        )
        if candidate_loss <= current_loss + tolerance:
            if candidate_loss > current_loss:
                print(
                    f"[RoutedOV] {boundary} apparent decoder increase is within "
                    f"FP{torch.finfo(work_dtype).bits} cancellation tolerance: "
                    f"before={current_loss:.9g} after={candidate_loss:.9g} "
                    f"tolerance={tolerance:.9g}",
                    flush=True,
                )
            return candidate_D, candidate_diagnostics, candidate_loss
        if decoder_solver_override is not None or work_dtype != torch.float32:
            raise RuntimeError(
                f"{boundary} free-decoder solve increased the fitted objective: "
                f"before={current_loss:.9g} after={candidate_loss:.9g} "
                f"tolerance={tolerance:.9g}"
            )
        best_D = candidate_D
        best_diagnostics = candidate_diagnostics
        best_loss = candidate_loss
        requested = float(decoder_relative_jitter)
        retry_jitters = tuple(
            value
            for value in (1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
            if value != requested
        )
        for relative_jitter in retry_jitters:
            retried_D, retried_diagnostics = solve_decoder(
                current_A,
                relative_jitter_override=relative_jitter,
            )
            retried_loss, _ = _evaluate_quadratic_with_scale(
                objective, current_A, retried_D, mapping
            )
            if retried_loss < best_loss:
                best_D = retried_D
                best_diagnostics = retried_diagnostics
                best_loss = retried_loss
            if retried_loss <= current_loss + tolerance:
                print(
                    f"[RoutedOV] {boundary} decoder retry "
                    f"relative_jitter={relative_jitter:.1e} "
                    f"before={current_loss:.9g} after={retried_loss:.9g}",
                    flush=True,
                )
                return retried_D, retried_diagnostics, retried_loss
        refined_D, refined_diagnostics = solve_decoder(
            current_A,
            relative_jitter_override=0.0,
            linear_solve_dtype_override=torch.float64,
        )
        refined_loss, _ = _evaluate_quadratic_with_scale(
            objective, current_A, refined_D, mapping
        )
        if refined_loss <= current_loss + tolerance:
            print(
                f"[RoutedOV] {boundary} decoder retry "
                f"linear_solve_dtype=float64 before={current_loss:.9g} "
                f"after={refined_loss:.9g}",
                flush=True,
            )
            return refined_D, refined_diagnostics, refined_loss
        if refined_loss < best_loss:
            best_loss = refined_loss
        raise RuntimeError(
            f"{boundary} free-decoder retries increased the fitted objective: "
            f"before={current_loss:.9g} best_after={best_loss:.9g} "
            f"tolerance={tolerance:.9g}"
        )

    initial_loss = evaluate_quadratic(objective, A, D, mapping)
    emit_checkpoint("anchor", 0, initial_loss)
    D, first_decoder, decoder_only_loss = monotone_decoder_solve(
        A, D, initial_loss, boundary="initial"
    )
    emit_checkpoint("decoder_only", 0, decoder_only_loss)
    sweeps: list[RoutedOVSweep] = []
    redecoder_losses: dict[int, float] = {}
    current_loss = decoder_only_loss
    no_progress = 0
    for sweep in range(1, maximum_sweeps + 1):
        before_decoder = current_loss
        if sweep == 1:
            # ``A`` is unchanged since the initial decoder solve immediately
            # above, so solving the identical system again is redundant.
            decoder_diagnostics = first_decoder
            after_decoder = current_loss
        else:
            D, decoder_diagnostics, after_decoder = monotone_decoder_solve(
                A,
                D,
                before_decoder,
                boundary=f"sweep_{sweep}_redecoder",
            )
        if sweep > 1:
            redecoder_losses[sweep - 1] = after_decoder
            emit_checkpoint("after_redecoder", sweep - 1, after_decoder)
        decoder_grams = _decoder_cross_grams(D)
        per_head_gradient = _per_head_encoder_half_gradient(
            objective=objective,
            A_unique=A,
            D_heads=D,
            head_to_kv_group=mapping,
            decoder_grams=decoder_grams,
        )
        steps = []
        running_loss = after_decoder
        for group in selected_groups:
            heads = tuple(
                torch.nonzero(mapping == group, as_tuple=False).flatten().tolist()
            )
            half_gradient = encoder_group_half_gradient(per_head_gradient, heads)
            scale = _encoder_operator_scale(
                covariance=objective.covariance,
                decoder_grams=decoder_grams,
                head_indices=heads,
                head_dim=A.shape[1],
                rank=(
                    A.shape[2]
                    if ragged_layout is None
                    else ragged_layout.group_ranks[group]
                ),
            )

            def operator(value: torch.Tensor) -> torch.Tensor:
                return encoder_hessian_vector_product(
                    covariance=objective.covariance,
                    D_heads=D,
                    head_indices=heads,
                    delta=value,
                    decoder_grams=decoder_grams,
                )

            damping = encoder_relative_damping * scale
            old_loss = running_loss
            direction_diagnostics = None
            if encoder_direction_solver is None:
                delta, cg = conjugate_gradient_matrix(
                    operator,
                    -half_gradient,
                    relative_tolerance=cg_relative_tolerance,
                    max_iterations=cg_max_iterations,
                    absolute_damping=damping,
                    fixed_iterations=cg_fixed_iterations,
                )
                if ragged_layout is not None:
                    active_rank = ragged_layout.group_ranks[group]
                    delta[:, active_rank:] = 0
                h_delta = operator(delta)
                full_step_change = float(
                    2.0 * torch.sum(half_gradient * delta)
                    + torch.sum(delta * h_delta)
                )
                accepted_scale = 1.0
                retries = 0
                predicted = full_step_change
                tolerance = 1e-10 * max(abs(running_loss), 1.0)
                while predicted > tolerance and retries < maximum_backtracks:
                    accepted_scale *= 0.5
                    retries += 1
                    predicted = float(
                        2.0
                        * accepted_scale
                        * torch.sum(half_gradient * delta)
                        + accepted_scale**2 * torch.sum(delta * h_delta)
                    )
                if predicted > tolerance:
                    accepted_scale = 0.0
                    predicted = 0.0
                if accepted_scale:
                    accepted_delta = accepted_scale * delta
                    A[group].add_(accepted_delta)
                    _increment_encoder_gradients(
                        per_head_gradient=per_head_gradient,
                        covariance=objective.covariance,
                        delta=accepted_delta,
                        changed_heads=heads,
                        decoder_grams=decoder_grams,
                    )
                    running_loss += predicted
                else:
                    accepted_delta = torch.zeros_like(delta)
            else:
                proposal = encoder_direction_solver.propose(
                    half_gradient=half_gradient,
                    covariance=objective.covariance,
                    decoder_grams=decoder_grams,
                    head_indices=heads,
                )
                delta = proposal.direction.to(
                    device=A.device,
                    dtype=A.dtype,
                )
                if tuple(delta.shape) != tuple(half_gradient.shape):
                    raise ValueError(
                        "encoder direction does not match gradient shape"
                    )
                if not torch.isfinite(delta).all():
                    raise FloatingPointError(
                        "encoder direction contains non-finite values"
                    )
                h_delta = operator(delta)
                gradient_inner = float(torch.sum(half_gradient * delta))
                curvature = float(torch.sum(delta * h_delta))
                gradient_norm = float(torch.linalg.vector_norm(half_gradient))
                direction_norm = float(torch.linalg.vector_norm(delta))
                numerical_tolerance = (
                    1024.0
                    * torch.finfo(A.dtype).eps
                    * max(
                        abs(gradient_inner),
                        abs(curvature),
                        gradient_norm * direction_norm,
                        1.0,
                    )
                )
                if gradient_norm == 0.0:
                    eta_exact = 0.0
                    eta_used = 0.0
                else:
                    if gradient_inner >= -numerical_tolerance:
                        raise RuntimeError(
                            f"{proposal.solver_name} did not propose a "
                            f"descent direction for group {group}: "
                            f"<h,P>={gradient_inner:.6e}"
                        )
                    if curvature <= numerical_tolerance:
                        raise RuntimeError(
                            f"{proposal.solver_name} direction has "
                            f"non-positive exact curvature for group {group}: "
                            f"<P,HP>={curvature:.6e}"
                        )
                    eta_exact = -gradient_inner / curvature
                    eta_used = encoder_step_scale * eta_exact
                retries = 0
                accepted_scale = eta_used
                original_group = A[group].clone()
                predicted = float(
                    2.0 * eta_used * gradient_inner
                    + eta_used**2 * curvature
                )
                realized_loss = old_loss
                objective_tolerance = 1e-10 * max(abs(old_loss), 1.0)
                while eta_used:
                    A[group].copy_(original_group + eta_used * delta)
                    realized_loss = evaluate_quadratic(
                        objective,
                        A,
                        D,
                        mapping,
                    )
                    if realized_loss <= old_loss + objective_tolerance:
                        break
                    A[group].copy_(original_group)
                    retries += 1
                    if retries > maximum_backtracks:
                        raise RuntimeError(
                            f"{proposal.solver_name} exact line search failed "
                            f"for group {group} after {maximum_backtracks} "
                            "backtracks"
                        )
                    eta_used *= 0.5
                    accepted_scale = eta_used
                    predicted = float(
                        2.0 * eta_used * gradient_inner
                        + eta_used**2 * curvature
                    )
                if eta_used:
                    accepted_delta = eta_used * delta
                    predicted_loss = old_loss + predicted
                    verification_tolerance = 1e-9 * max(
                        abs(old_loss),
                        abs(predicted_loss),
                        abs(realized_loss),
                        1.0,
                    )
                    if abs(realized_loss - predicted_loss) > verification_tolerance:
                        raise RuntimeError(
                            "one-shot predicted objective change disagrees "
                            f"with FP64 realization: group={group} "
                            f"predicted={predicted_loss} "
                            f"realized={realized_loss}"
                        )
                    _increment_encoder_gradients(
                        per_head_gradient=per_head_gradient,
                        covariance=objective.covariance,
                        delta=accepted_delta,
                        changed_heads=heads,
                        decoder_grams=decoder_grams,
                    )
                    running_loss = realized_loss
                else:
                    A[group].copy_(original_group)
                    accepted_delta = torch.zeros_like(delta)
                    predicted = 0.0
                    running_loss = old_loss
                exact_residual = h_delta + half_gradient
                relative_residual = float(
                    torch.linalg.vector_norm(exact_residual)
                    / torch.linalg.vector_norm(half_gradient).clamp_min(
                        torch.finfo(A.dtype).tiny
                    )
                )
                maximum_angle, mean_angle = _subspace_movement(
                    anchor_A[group],
                    A[group],
                )
                movement = float(
                    torch.linalg.vector_norm(accepted_delta)
                    / torch.linalg.vector_norm(original_group).clamp_min(
                        torch.finfo(A.dtype).tiny
                    )
                )
                direction_diagnostics = EncoderDirectionDiagnostics(
                    solver_name=proposal.solver_name,
                    half_gradient_norm=gradient_norm,
                    direction_norm=direction_norm,
                    gradient_direction_inner_product=gradient_inner,
                    exact_hessian_curvature=curvature,
                    eta_exact=eta_exact,
                    eta_used=eta_used,
                    requested_step_scale=encoder_step_scale,
                    exact_hessian_relative_residual=relative_residual,
                    optimal_line_reduction=(
                        gradient_inner**2 / curvature
                        if curvature > 0
                        else 0.0
                    ),
                    relative_encoder_movement=movement,
                    maximum_anchor_principal_angle_radians=maximum_angle,
                    mean_anchor_principal_angle_radians=mean_angle,
                    solver_diagnostics=proposal.diagnostics,
                    complexity={
                        **proposal.complexity,
                        "exact_hessian_vector_products": 1,
                    },
                )
                cg = _zero_cg_diagnostics()
            if verify_encoder_step_objective:
                realized_loss = evaluate_quadratic(
                    objective,
                    A,
                    D,
                    mapping,
                )
                verification_tolerance = 1e-9 * max(
                    abs(old_loss),
                    abs(running_loss),
                    1.0,
                )
                if abs(realized_loss - running_loss) > verification_tolerance:
                    raise RuntimeError(
                        "predicted encoder-group quadratic change disagrees "
                        "with the realized objective: "
                        f"group={group} predicted_loss={running_loss} "
                        f"realized_loss={realized_loss}"
                    )
                running_loss = realized_loss
            active_rank = (
                A.shape[2]
                if ragged_layout is None
                else ragged_layout.group_ranks[group]
            )
            maximum_angle, mean_angle = _subspace_movement(
                anchor_A[group, :, :active_rank],
                A[group, :, :active_rank],
            )
            before_update = (
                A[group, :, :active_rank]
                - accepted_delta[:, :active_rank]
            )
            relative_movement = float(
                torch.linalg.vector_norm(accepted_delta[:, :active_rank])
                / torch.linalg.vector_norm(before_update).clamp_min(
                    torch.finfo(A.dtype).tiny
                )
            )
            steps.append(
                EncoderStepDiagnostics(
                    group_index=group,
                    old_loss=old_loss,
                    new_loss=running_loss,
                    predicted_change=predicted,
                    realized_change=running_loss - old_loss,
                    accepted_scale=accepted_scale,
                    retries=retries,
                    cg=cg,
                    direction=direction_diagnostics,
                    relative_encoder_movement=relative_movement,
                    maximum_anchor_principal_angle_radians=maximum_angle,
                    mean_anchor_principal_angle_radians=mean_angle,
                )
            )
        A, D, qr_error = canonicalize_factors(A, D)
        after_encoders = evaluate_quadratic(objective, A, D, mapping)
        encoder_tolerance = max(
            monotonicity_tolerance(after_decoder),
            1.0e-8 * max(abs(after_decoder), 1.0),
        )
        if after_encoders > after_decoder + encoder_tolerance:
            raise RuntimeError(
                "accepted encoder sweep increased the fitted objective: "
                f"before={after_decoder:.9g} after={after_encoders:.9g} "
                f"tolerance={encoder_tolerance:.9g}"
            )
        emit_checkpoint("after_encoder", sweep, after_encoders)
        improvement = (before_decoder - after_encoders) / max(abs(before_decoder), 1e-30)
        sweeps.append(
            RoutedOVSweep(
                sweep=sweep,
                loss_before_decoder=before_decoder,
                loss_after_decoder=after_decoder,
                loss_after_encoders=after_encoders,
                relative_improvement=improvement,
                decoder=decoder_diagnostics,
                encoder_steps=tuple(steps),
                maximum_qr_product_error=qr_error,
            )
        )
        current_loss = after_encoders
        if improvement <= relative_objective_tolerance:
            no_progress += 1
        else:
            no_progress = 0
        if sweep >= minimum_sweeps and no_progress >= patience:
            break
    legacy_final_loss = current_loss
    final_decoder_diagnostics = None
    if final_decoder_solve and sweeps:
        final_sweep = sweeps[-1].sweep
        D, final_decoder_diagnostics, final_loss = monotone_decoder_solve(
            A,
            D,
            current_loss,
            boundary="final",
        )
        redecoder_losses[final_sweep] = final_loss
        current_loss = final_loss
        emit_checkpoint("after_redecoder", final_sweep, final_loss)

    attribution_steps = tuple(
        RoutedOVAttributionStep(
            sweep=item.sweep,
            loss_before_encoder=item.loss_after_decoder,
            loss_after_encoder=item.loss_after_encoders,
            loss_after_redecoder=redecoder_losses.get(item.sweep),
            encoder_reduction=item.loss_after_decoder - item.loss_after_encoders,
            redecoder_reduction=(
                item.loss_after_encoders - redecoder_losses[item.sweep]
                if item.sweep in redecoder_losses
                else 0.0
            ),
        )
        for item in sweeps
    )
    initial_decoder_reduction = initial_loss - decoder_only_loss
    encoder_reduction = sum(item.encoder_reduction for item in attribution_steps)
    later_decoder_reduction = sum(
        item.redecoder_reduction for item in attribution_steps
    )
    decoder_reduction = initial_decoder_reduction + later_decoder_reduction
    total_reduction = initial_loss - current_loss
    denominator = max(abs(total_reduction), torch.finfo(work_dtype).tiny)
    identity_error = decoder_reduction + encoder_reduction - total_reduction
    attribution = RoutedOVAttribution(
        anchor_loss=initial_loss,
        decoder_only_loss=decoder_only_loss,
        endpoint_loss=current_loss,
        initial_decoder_reduction=initial_decoder_reduction,
        steps=attribution_steps,
        decoder_reduction=decoder_reduction,
        encoder_reduction=encoder_reduction,
        total_reduction=total_reduction,
        decoder_fraction=decoder_reduction / denominator,
        encoder_fraction=encoder_reduction / denominator,
        identity_error=identity_error,
    )
    return RoutedOVJointResult(
        A_unique=A.detach().cpu(),
        D_heads=D.detach().cpu(),
        initial_loss=initial_loss,
        decoder_only_loss=decoder_only_loss,
        final_loss=current_loss,
        sweeps=tuple(sweeps),
        initial_decoder=first_decoder,
        legacy_final_loss=legacy_final_loss,
        final_decoder=final_decoder_diagnostics,
        checkpoints=tuple(checkpoints),
        attribution=attribution,
    )


def extract_value_coordinate_factors(
    *,
    layout: GQAVOLayout,
    dense_v_proj_weight: torch.Tensor,
    compressed_v_proj_weight: torch.Tensor,
    compressed_o_proj_weight: torch.Tensor,
    work_dtype: torch.dtype = torch.float64,
    work_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Recover ``A_g`` and head-ordered ``D_h`` from a folded rank bank."""

    expected_dense = (layout.kv_width, layout.hidden_size)
    expected_v = (
        layout.num_key_value_heads * layout.rank,
        layout.hidden_size,
    )
    expected_o = (
        layout.hidden_size,
        layout.num_attention_heads * layout.rank,
    )
    if tuple(dense_v_proj_weight.shape) != expected_dense:
        raise ValueError("dense v_proj shape does not match layout")
    if tuple(compressed_v_proj_weight.shape) != expected_v:
        raise ValueError("compressed v_proj shape does not match layout")
    if tuple(compressed_o_proj_weight.shape) != expected_o:
        raise ValueError("compressed o_proj shape does not match layout")
    device = (
        torch.device(work_device)
        if work_device is not None
        else dense_v_proj_weight.device
    )
    dense = dense_v_proj_weight.detach().to(device=device, dtype=work_dtype)
    compressed = compressed_v_proj_weight.detach().to(device=device, dtype=work_dtype)
    A_groups = []
    residuals = []
    for group in range(layout.num_key_value_heads):
        dense_rows = slice(group * layout.head_dim, (group + 1) * layout.head_dim)
        compressed_rows = slice(group * layout.rank, (group + 1) * layout.rank)
        left = dense[dense_rows].transpose(0, 1)
        target = compressed[compressed_rows].transpose(0, 1)
        A = torch.linalg.lstsq(left, target).solution
        residuals.append(
            float(
                torch.linalg.vector_norm(left @ A - target)
                / torch.linalg.vector_norm(target).clamp_min(
                    torch.finfo(work_dtype).tiny
                )
            )
        )
        A_groups.append(A)
    o_row = compressed_o_proj_weight.detach().to(
        device=device,
        dtype=work_dtype,
    ).transpose(0, 1)
    D_heads = o_row.reshape(
        layout.num_attention_heads,
        layout.rank,
        layout.hidden_size,
    )
    A_unique = torch.stack(A_groups)
    A_unique, D_heads, qr_error = gauge_canonicalize(
        A_unique,
        D_heads,
        torch.arange(layout.num_attention_heads, device=device)
        // layout.query_heads_per_kv_group,
    )
    return A_unique, D_heads, {
        "maximum_value_encoder_recovery_error": max(residuals),
        "mean_value_encoder_recovery_error": sum(residuals) / len(residuals),
        "gauge_qr_product_error": qr_error,
    }


def fold_routed_ov_factors(
    *,
    layout: GQAVOLayout,
    dense_v_proj_weight: torch.Tensor,
    A_unique: torch.Tensor,
    D_heads: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    thin_qr: bool = True,
    output_dtype: torch.dtype | None = None,
) -> FoldedRoutedOVFactors:
    """Fold value-coordinate factors into serving-ready PyTorch weights."""

    dense = dense_v_proj_weight.detach().to(
        device=A_unique.device,
        dtype=A_unique.dtype,
    )
    mapping = torch.as_tensor(
        head_to_kv_group,
        dtype=torch.long,
        device=A_unique.device,
    )
    folded_v = []
    folded_D = D_heads.clone()
    maximum_qr_error = 0.0
    maximum_fold_error = 0.0
    for group in range(layout.num_key_value_heads):
        dense_rows = slice(group * layout.head_dim, (group + 1) * layout.head_dim)
        dense_math = dense[dense_rows].transpose(0, 1)
        value_math = dense_math @ A_unique[group]
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        before = value_math @ folded_D.index_select(0, heads)
        if thin_qr:
            q, r = torch.linalg.qr(value_math, mode="reduced")
            signs = torch.sign(torch.diagonal(r))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            q = q * signs.unsqueeze(0)
            r = signs.unsqueeze(1) * r
            adjusted = torch.einsum(
                "ab,hbo->hao",
                r,
                folded_D.index_select(0, heads),
            )
            after = q @ adjusted
            denominator = torch.linalg.vector_norm(before).clamp_min(
                torch.finfo(before.dtype).tiny
            )
            maximum_qr_error = max(
                maximum_qr_error,
                float(torch.linalg.vector_norm(after - before) / denominator),
            )
            folded_D.index_copy_(0, heads, adjusted)
            value_math = q
        after = value_math @ folded_D.index_select(0, heads)
        denominator = torch.linalg.vector_norm(before).clamp_min(
            torch.finfo(before.dtype).tiny
        )
        maximum_fold_error = max(
            maximum_fold_error,
            float(torch.linalg.vector_norm(after - before) / denominator),
        )
        folded_v.append(value_math.transpose(0, 1).contiguous())
    target_dtype = output_dtype or dense_v_proj_weight.dtype
    return FoldedRoutedOVFactors(
        v_proj_compressed_weight=torch.cat(folded_v, dim=0)
        .detach()
        .cpu()
        .to(target_dtype),
        o_decoder_weight=folded_D.reshape(
            layout.num_attention_heads * layout.rank,
            layout.hidden_size,
        )
        .transpose(0, 1)
        .contiguous()
        .detach()
        .cpu()
        .to(target_dtype),
        maximum_qr_product_error=maximum_qr_error,
        maximum_dense_fold_error=maximum_fold_error,
    )


def fold_ragged_routed_ov_factors(
    *,
    dense_v_proj_weight: torch.Tensor,
    A_unique: torch.Tensor,
    D_heads: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    group_ranks: Sequence[int],
    thin_qr: bool = True,
    output_dtype: torch.dtype | None = None,
) -> FoldedRaggedRoutedOVFactors:
    """Fold padded ragged factors into one V/O tensor pair per KV group."""

    layout = RaggedHeadLayout.from_group_ranks(
        group_ranks,
        head_to_kv_group,
        maximum_rank=A_unique.shape[2],
    )
    _validate_ragged_padded_factors(A_unique, D_heads, layout)
    mapping = torch.as_tensor(
        head_to_kv_group,
        dtype=torch.long,
        device=A_unique.device,
    )
    dense = dense_v_proj_weight.detach().to(
        device=A_unique.device,
        dtype=A_unique.dtype,
    )
    head_dim = A_unique.shape[1]
    hidden_size = dense.shape[1]
    if dense.shape[0] != len(layout.group_ranks) * head_dim:
        raise ValueError("dense V projection does not match ragged geometry")
    target_dtype = output_dtype or dense_v_proj_weight.dtype
    v_groups: list[torch.Tensor] = []
    o_groups: list[torch.Tensor] = []
    maximum_qr_error = 0.0
    maximum_fold_error = 0.0
    for group, rank in enumerate(layout.group_ranks):
        dense_rows = slice(group * head_dim, (group + 1) * head_dim)
        dense_math = dense[dense_rows].transpose(0, 1)
        active_A = A_unique[group, :, :rank]
        value_math = dense_math @ active_A
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        active_D = D_heads.index_select(0, heads)[:, :rank].clone()
        before = value_math @ active_D
        if thin_qr:
            q, r = torch.linalg.qr(value_math, mode="reduced")
            signs = torch.sign(torch.diagonal(r))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            q = q * signs.unsqueeze(0)
            r = signs.unsqueeze(1) * r
            active_D = torch.einsum("ab,hbo->hao", r, active_D)
            after = q @ active_D
            denominator = torch.linalg.vector_norm(before).clamp_min(
                torch.finfo(before.dtype).tiny
            )
            maximum_qr_error = max(
                maximum_qr_error,
                float(torch.linalg.vector_norm(after - before) / denominator),
            )
            value_math = q
        after = value_math @ active_D
        denominator = torch.linalg.vector_norm(before).clamp_min(
            torch.finfo(before.dtype).tiny
        )
        maximum_fold_error = max(
            maximum_fold_error,
            float(torch.linalg.vector_norm(after - before) / denominator),
        )
        v_groups.append(
            value_math.transpose(0, 1).contiguous().detach().cpu().to(target_dtype)
        )
        o_groups.append(
            active_D.reshape(len(heads) * rank, hidden_size)
            .transpose(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .to(target_dtype)
        )
    return FoldedRaggedRoutedOVFactors(
        v_group_weights=tuple(v_groups),
        o_group_weights=tuple(o_groups),
        maximum_qr_product_error=maximum_qr_error,
        maximum_dense_fold_error=maximum_fold_error,
    )


def covariance_minimum_eigenvalue(covariance: torch.Tensor) -> float:
    _validate_covariance_blocks(covariance)
    flat = flatten_covariance_blocks(_symmetric_blocks(covariance).double())
    return float(torch.linalg.eigvalsh(flat)[0])
