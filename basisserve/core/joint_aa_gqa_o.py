"""Joint activation-aware GQA-structured ``o_proj`` factorization.

The offline solver uses row-vector notation ``Y = Z @ W`` while checkpoint
weights follow ``nn.Linear`` convention.  One encoder is tied across all query
heads that share a KV head, and a single dense decoder is fitted against the
complete output residual using the full ``o_proj`` input covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from basisserve.core.gqa_vo_svdllm import GQAVOLayout


@dataclass(frozen=True)
class DecoderSolveDiagnostics:
    ridge: float
    covariance_scale: float
    cholesky_diag_condition_estimate: float


@dataclass(frozen=True)
class JointAAMetrics:
    objective: float
    data_loss: float
    expanded_data_loss: float
    target_energy: float
    relative_output_rmse: float
    relative_weight_error: float


@dataclass(frozen=True)
class JointAAIteration:
    iteration: int
    step_size: float
    backtracks: int
    ridge: float
    objective: float
    relative_output_rmse: float
    relative_improvement: float
    max_orthogonality_error: float


@dataclass(frozen=True)
class JointAAResult:
    layout: GQAVOLayout
    head_to_kv_group: torch.Tensor
    E_unique: torch.Tensor
    D_row: torch.Tensor
    decoder_diagnostics: DecoderSolveDiagnostics
    init_metrics: JointAAMetrics
    final_metrics: JointAAMetrics
    iteration_history: tuple[JointAAIteration, ...]

    @property
    def D_weight(self) -> torch.Tensor:
        return self.D_row.transpose(0, 1).contiguous()


@dataclass(frozen=True)
class FullAAOracle:
    rank: int
    metrics: JointAAMetrics


def resolve_head_to_kv_group(
    layout: GQAVOLayout,
    head_to_kv_group: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve and validate the query-head to KV-head mapping.

    Hugging Face Qwen3 repeats KV heads contiguously, so its mapping is
    ``query_head // num_key_value_groups``.  Callers may supply an explicit
    mapping for architectures with a different ordering.
    """

    if head_to_kv_group is None:
        mapping = torch.arange(layout.num_attention_heads, dtype=torch.long)
        mapping = mapping // layout.query_heads_per_kv_group
    else:
        mapping = torch.as_tensor(head_to_kv_group, dtype=torch.long).cpu().clone()
    if tuple(mapping.shape) != (layout.num_attention_heads,):
        raise ValueError(
            "head_to_kv_group must have shape "
            f"{(layout.num_attention_heads,)}, got {tuple(mapping.shape)}"
        )
    if torch.any(mapping < 0) or torch.any(mapping >= layout.num_key_value_heads):
        raise ValueError("head_to_kv_group contains an out-of-range KV group")
    counts = torch.bincount(mapping, minlength=layout.num_key_value_heads)
    expected = layout.query_heads_per_kv_group
    if torch.any(counts != expected):
        raise ValueError(
            "every KV group must serve the same number of query heads: "
            f"expected {expected}, got {counts.tolist()}"
        )
    return mapping


def materialize_structured_encoder(
    E_unique: torch.Tensor,
    head_to_kv_group: torch.Tensor,
) -> torch.Tensor:
    """Materialize the repeated block-diagonal encoder used offline."""

    if E_unique.ndim != 3:
        raise ValueError(f"E_unique must have shape [Hkv, Dh, Rv], got {E_unique.shape}")
    num_kv_heads, head_dim, rank = E_unique.shape
    mapping = torch.as_tensor(head_to_kv_group, dtype=torch.long, device=E_unique.device)
    if mapping.ndim != 1:
        raise ValueError("head_to_kv_group must be one-dimensional")
    if torch.any(mapping < 0) or torch.any(mapping >= num_kv_heads):
        raise ValueError("head_to_kv_group contains an out-of-range KV group")
    num_query_heads = mapping.numel()
    encoder = E_unique.new_zeros(num_query_heads * head_dim, num_query_heads * rank)
    for head_index, group_index in enumerate(mapping.tolist()):
        row = slice(head_index * head_dim, (head_index + 1) * head_dim)
        column = slice(head_index * rank, (head_index + 1) * rank)
        encoder[row, column] = E_unique[group_index]
    return encoder


def qr_retract_positive(matrix: torch.Tensor) -> torch.Tensor:
    """Thin-QR retraction with deterministic positive diagonal signs."""

    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def max_orthogonality_error(E_unique: torch.Tensor) -> float:
    rank = E_unique.shape[-1]
    identity = torch.eye(rank, device=E_unique.device, dtype=E_unique.dtype)
    errors = torch.linalg.matrix_norm(
        E_unique.transpose(-1, -2) @ E_unique - identity,
        ord="fro",
        dim=(-2, -1),
    )
    return float(errors.max())


def _validate_problem(
    layout: GQAVOLayout,
    covariance: torch.Tensor,
    Wo_row: torch.Tensor,
) -> None:
    expected_covariance = (layout.query_width, layout.query_width)
    expected_weight = (layout.query_width, layout.hidden_size)
    if tuple(covariance.shape) != expected_covariance:
        raise ValueError(
            f"covariance must have shape {expected_covariance}, got {tuple(covariance.shape)}"
        )
    if tuple(Wo_row.shape) != expected_weight:
        raise ValueError(f"Wo_row must have shape {expected_weight}, got {tuple(Wo_row.shape)}")


def _symmetric(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _matrix_scale(matrix: torch.Tensor) -> float:
    diagonal = torch.diagonal(matrix).abs()
    return max(float(diagonal.mean()), torch.finfo(matrix.dtype).tiny)


def damp_covariance(
    covariance: torch.Tensor,
    relative_damp: float,
) -> tuple[torch.Tensor, float]:
    """Add trace-scaled isotropic damping to an activation covariance."""

    if relative_damp < 0:
        raise ValueError(f"relative_damp must be non-negative, got {relative_damp}")
    symmetric = _symmetric(covariance)
    scale = _matrix_scale(symmetric)
    absolute_damp = relative_damp * scale
    if absolute_damp == 0:
        return symmetric, 0.0
    identity = torch.eye(
        symmetric.shape[0],
        device=symmetric.device,
        dtype=symmetric.dtype,
    )
    return symmetric + absolute_damp * identity, absolute_damp


def _ridge_candidates(relative_ridge: float) -> tuple[float, ...]:
    if relative_ridge < 0:
        raise ValueError(f"ridge must be non-negative, got {relative_ridge}")
    if relative_ridge > 1e-2:
        raise ValueError(f"ridge must not exceed 1e-2, got {relative_ridge}")
    if relative_ridge == 0:
        return (0.0,) + tuple(10.0**exponent for exponent in range(-12, -1))
    values = []
    candidate = relative_ridge
    while candidate <= 1e-2:
        values.append(candidate)
        candidate *= 10.0
    if values[-1] < 1e-2:
        values.append(1e-2)
    return tuple(values)


def solve_global_decoder(
    *,
    covariance: torch.Tensor,
    Wo_row: torch.Tensor,
    E_Q: torch.Tensor,
    relative_ridge: float = 1e-6,
) -> tuple[torch.Tensor, DecoderSolveDiagnostics]:
    """Solve the exact ridge decoder for a fixed structured encoder."""

    A_unregularized = _symmetric(E_Q.transpose(0, 1) @ covariance @ E_Q)
    B = E_Q.transpose(0, 1) @ covariance @ Wo_row
    # Keep the regularization scale independent of E_Q so line-search
    # candidates are compared under one fixed reduced objective.
    scale = _matrix_scale(covariance)
    identity = torch.eye(
        A_unregularized.shape[0],
        device=A_unregularized.device,
        dtype=A_unregularized.dtype,
    )
    for relative in _ridge_candidates(relative_ridge):
        ridge = relative * scale
        A = A_unregularized + ridge * identity
        chol, info = torch.linalg.cholesky_ex(A, check_errors=False)
        if int(info.max()) != 0:
            continue
        decoder = torch.cholesky_solve(B, chol)
        diagonal = torch.diagonal(chol).abs()
        condition_estimate = float(
            (diagonal.max() / diagonal.min().clamp_min(torch.finfo(chol.dtype).tiny)).square()
        )
        return decoder, DecoderSolveDiagnostics(
            ridge=float(ridge),
            covariance_scale=scale,
            cholesky_diag_condition_estimate=condition_estimate,
        )
    raise torch.linalg.LinAlgError(
        "global decoder Cholesky failed through relative ridge 1e-2"
    )


def evaluate_joint_objective(
    *,
    covariance: torch.Tensor,
    Wo_row: torch.Tensor,
    E_Q: torch.Tensor,
    D_row: torch.Tensor,
    ridge: float,
) -> JointAAMetrics:
    residual = Wo_row - E_Q @ D_row
    data_loss_tensor = torch.sum(residual * (covariance @ residual))
    target_energy_tensor = torch.sum(Wo_row * (covariance @ Wo_row))
    cross = torch.sum(D_row * (E_Q.transpose(0, 1) @ covariance @ Wo_row))
    quadratic = torch.sum(D_row * (E_Q.transpose(0, 1) @ covariance @ E_Q @ D_row))
    expanded = target_energy_tensor - 2.0 * cross + quadratic
    objective = data_loss_tensor + ridge * torch.sum(D_row.square())
    tiny = torch.finfo(covariance.dtype).tiny
    output_rmse = torch.sqrt(data_loss_tensor.clamp_min(0) / target_energy_tensor.clamp_min(tiny))
    weight_error = torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(Wo_row).clamp_min(tiny)
    return JointAAMetrics(
        objective=float(objective),
        data_loss=float(data_loss_tensor),
        expanded_data_loss=float(expanded),
        target_energy=float(target_energy_tensor),
        relative_output_rmse=float(output_rmse),
        relative_weight_error=float(weight_error),
    )


def initialize_unique_encoders(
    *,
    layout: GQAVOLayout,
    covariance: torch.Tensor,
    Wo_row: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    method: str = "pooled_activation_svd",
    covariance_ridge: float = 1e-6,
    seed: int = 1234,
) -> torch.Tensor:
    """Initialize one Euclidean-orthonormal encoder per KV group."""

    _validate_problem(layout, covariance, Wo_row)
    method = method.lower()
    if method not in {"pooled_activation_svd", "weight_svd", "random_orthogonal"}:
        raise ValueError(f"unsupported initialization method: {method}")
    generator = torch.Generator(device=covariance.device)
    generator.manual_seed(seed)
    encoders = []
    for group_index in range(layout.num_key_value_heads):
        heads = torch.nonzero(head_to_kv_group == group_index, as_tuple=False).flatten().tolist()
        weight_blocks = [
            Wo_row[
                head_index * layout.head_dim : (head_index + 1) * layout.head_dim,
                :,
            ]
            for head_index in heads
        ]
        weight_cat = torch.cat(weight_blocks, dim=1)
        if method == "random_orthogonal":
            raw = torch.randn(
                layout.head_dim,
                layout.rank,
                generator=generator,
                device=covariance.device,
                dtype=covariance.dtype,
            )
        elif method == "weight_svd":
            raw = torch.linalg.svd(weight_cat, full_matrices=False).U[:, : layout.rank]
        else:
            blocks = [
                covariance[
                    head_index * layout.head_dim : (head_index + 1) * layout.head_dim,
                    head_index * layout.head_dim : (head_index + 1) * layout.head_dim,
                ]
                for head_index in heads
            ]
            pooled = _symmetric(torch.stack(blocks).mean(dim=0))
            scale = _matrix_scale(pooled)
            identity = torch.eye(layout.head_dim, device=pooled.device, dtype=pooled.dtype)
            chol = None
            for relative in _ridge_candidates(covariance_ridge):
                candidate, info = torch.linalg.cholesky_ex(
                    pooled + relative * scale * identity,
                    check_errors=False,
                )
                if int(info.max()) == 0:
                    chol = candidate
                    break
            if chol is None:
                raise torch.linalg.LinAlgError(
                    f"pooled covariance Cholesky failed for KV group {group_index}"
                )
            whitened = chol.transpose(0, 1) @ weight_cat
            left = torch.linalg.svd(whitened, full_matrices=False).U[:, : layout.rank]
            raw = torch.linalg.solve_triangular(
                chol.transpose(0, 1),
                left,
                upper=True,
            )
        encoders.append(qr_retract_positive(raw))
    return torch.stack(encoders)


def structured_unique_gradient(
    *,
    covariance: torch.Tensor,
    Wo_row: torch.Tensor,
    E_unique: torch.Tensor,
    D_row: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    project_tangent: bool = True,
) -> torch.Tensor:
    """Return the tied unique-encoder gradient for fixed decoder ``D``."""

    E_Q = materialize_structured_encoder(E_unique, head_to_kv_group)
    full_gradient = 2.0 * covariance @ (E_Q @ D_row - Wo_row) @ D_row.transpose(0, 1)
    _, head_dim, rank = E_unique.shape
    unique_gradient = torch.zeros_like(E_unique)
    for head_index, group_index in enumerate(head_to_kv_group.tolist()):
        rows = slice(head_index * head_dim, (head_index + 1) * head_dim)
        columns = slice(head_index * rank, (head_index + 1) * rank)
        unique_gradient[group_index].add_(full_gradient[rows, columns])
    if not project_tangent:
        return unique_gradient
    product = E_unique.transpose(-1, -2) @ unique_gradient
    return unique_gradient - E_unique @ _symmetric(product)


def factorize_joint_aa_gqa_o(
    *,
    layout: GQAVOLayout,
    o_weight_pt: torch.Tensor,
    covariance: torch.Tensor,
    head_to_kv_group: torch.Tensor | None = None,
    init: str = "pooled_activation_svd",
    outer_iters: int = 10,
    initial_step_size: float = 1e-1,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 10,
    relative_improve_tol: float = 1e-6,
    patience: int = 2,
    decoder_ridge: float = 1e-6,
    covariance_ridge: float = 1e-6,
    global_covariance_damp: float = 0.0,
    weight_only: bool = False,
    seed: int = 1234,
    work_dtype: torch.dtype = torch.float64,
    work_device: torch.device | str | None = None,
    output_dtype: torch.dtype | None = None,
) -> JointAAResult:
    """Fit a GQA-tied encoder and global decoder from full sufficient statistics."""

    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("work_dtype must be float32 or float64")
    if outer_iters < 0 or max_backtracks < 0 or patience <= 0:
        raise ValueError("outer_iters/max_backtracks must be non-negative and patience positive")
    if initial_step_size <= 0 or not 0 < backtrack_factor < 1:
        raise ValueError("initial_step_size must be positive and backtrack_factor in (0, 1)")
    if relative_improve_tol < 0:
        raise ValueError("relative_improve_tol must be non-negative")
    device = torch.device(work_device) if work_device is not None else covariance.device
    S, _ = damp_covariance(
        covariance.detach().to(device=device, dtype=work_dtype),
        global_covariance_damp,
    )
    if weight_only:
        S = torch.eye(layout.query_width, device=device, dtype=work_dtype)
    Wo_row = o_weight_pt.detach().transpose(0, 1).to(device=device, dtype=work_dtype)
    _validate_problem(layout, S, Wo_row)
    mapping_cpu = resolve_head_to_kv_group(layout, head_to_kv_group)
    mapping = mapping_cpu.to(device)
    E_unique = initialize_unique_encoders(
        layout=layout,
        covariance=S,
        Wo_row=Wo_row,
        head_to_kv_group=mapping,
        method=init,
        covariance_ridge=covariance_ridge,
        seed=seed,
    )
    E_Q = materialize_structured_encoder(E_unique, mapping)
    D_row, diagnostics = solve_global_decoder(
        covariance=S,
        Wo_row=Wo_row,
        E_Q=E_Q,
        relative_ridge=decoder_ridge,
    )
    init_metrics = evaluate_joint_objective(
        covariance=S,
        Wo_row=Wo_row,
        E_Q=E_Q,
        D_row=D_row,
        ridge=diagnostics.ridge,
    )
    current_metrics = init_metrics
    history: list[JointAAIteration] = [
        JointAAIteration(
            iteration=0,
            step_size=0.0,
            backtracks=0,
            ridge=diagnostics.ridge,
            objective=init_metrics.objective,
            relative_output_rmse=init_metrics.relative_output_rmse,
            relative_improvement=0.0,
            max_orthogonality_error=max_orthogonality_error(E_unique),
        )
    ]
    no_progress = 0
    step_size = initial_step_size
    for iteration in range(1, outer_iters + 1):
        tangent = structured_unique_gradient(
            covariance=S,
            Wo_row=Wo_row,
            E_unique=E_unique,
            D_row=D_row,
            head_to_kv_group=mapping,
            project_tangent=True,
        )
        accepted = None
        trial_step = step_size
        for backtracks in range(max_backtracks + 1):
            candidate_E = torch.stack(
                [qr_retract_positive(E_unique[g] - trial_step * tangent[g]) for g in range(E_unique.shape[0])]
            )
            candidate_E_Q = materialize_structured_encoder(candidate_E, mapping)
            candidate_D, candidate_diagnostics = solve_global_decoder(
                covariance=S,
                Wo_row=Wo_row,
                E_Q=candidate_E_Q,
                relative_ridge=decoder_ridge,
            )
            candidate_metrics = evaluate_joint_objective(
                covariance=S,
                Wo_row=Wo_row,
                E_Q=candidate_E_Q,
                D_row=candidate_D,
                ridge=candidate_diagnostics.ridge,
            )
            improvement = (
                current_metrics.objective - candidate_metrics.objective
            ) / max(abs(current_metrics.objective), torch.finfo(work_dtype).tiny)
            if improvement > 0:
                accepted = (
                    candidate_E,
                    candidate_D,
                    candidate_diagnostics,
                    candidate_metrics,
                    float(improvement),
                    trial_step,
                    backtracks,
                )
                break
            trial_step *= backtrack_factor
        if accepted is None:
            no_progress += 1
            if no_progress >= patience:
                break
            step_size *= backtrack_factor
            continue
        (
            E_unique,
            D_row,
            diagnostics,
            current_metrics,
            relative_improvement,
            accepted_step,
            backtracks,
        ) = accepted
        history.append(
            JointAAIteration(
                iteration=iteration,
                step_size=accepted_step,
                backtracks=backtracks,
                ridge=diagnostics.ridge,
                objective=current_metrics.objective,
                relative_output_rmse=current_metrics.relative_output_rmse,
                relative_improvement=relative_improvement,
                max_orthogonality_error=max_orthogonality_error(E_unique),
            )
        )
        step_size = accepted_step
        if relative_improvement <= relative_improve_tol:
            no_progress += 1
        else:
            no_progress = 0
        if no_progress >= patience:
            break
    target_dtype = output_dtype or o_weight_pt.dtype
    return JointAAResult(
        layout=layout,
        head_to_kv_group=mapping_cpu,
        E_unique=E_unique.detach().cpu().to(target_dtype),
        D_row=D_row.detach().cpu().to(target_dtype),
        decoder_diagnostics=diagnostics,
        init_metrics=init_metrics,
        final_metrics=current_metrics,
        iteration_history=tuple(history),
    )


def full_activation_aware_svd_oracle(
    *,
    covariance: torch.Tensor,
    o_weight_pt: torch.Tensor,
    rank: int,
    covariance_ridge: float = 1e-6,
    work_dtype: torch.dtype = torch.float64,
    work_device: torch.device | str | None = None,
) -> FullAAOracle:
    """Compute the unconstrained activation-aware SVD oracle at rank ``rank``."""

    device = torch.device(work_device) if work_device is not None else covariance.device
    S = _symmetric(covariance.detach().to(device=device, dtype=work_dtype))
    W = o_weight_pt.detach().transpose(0, 1).to(device=device, dtype=work_dtype)
    if rank <= 0 or rank > min(W.shape):
        raise ValueError(f"oracle rank must be in [1, {min(W.shape)}], got {rank}")
    scale = _matrix_scale(S)
    identity = torch.eye(S.shape[0], device=device, dtype=work_dtype)
    chol = None
    for relative in _ridge_candidates(covariance_ridge):
        candidate, info = torch.linalg.cholesky_ex(
            S + relative * scale * identity,
            check_errors=False,
        )
        if int(info.max()) == 0:
            chol = candidate
            break
    if chol is None:
        raise torch.linalg.LinAlgError("oracle covariance Cholesky failed")
    U, singular_values, Vh = torch.linalg.svd(chol.transpose(0, 1) @ W, full_matrices=False)
    encoder = torch.linalg.solve_triangular(
        chol.transpose(0, 1),
        U[:, :rank],
        upper=True,
    )
    decoder = singular_values[:rank, None] * Vh[:rank, :]
    metrics = evaluate_joint_objective(
        covariance=S,
        Wo_row=W,
        E_Q=encoder,
        D_row=decoder,
        ridge=0.0,
    )
    return FullAAOracle(rank=rank, metrics=metrics)


def metrics_to_dict(metrics: JointAAMetrics) -> dict[str, float]:
    return {
        "objective": metrics.objective,
        "data_loss": metrics.data_loss,
        "expanded_data_loss": metrics.expanded_data_loss,
        "target_energy": metrics.target_energy,
        "relative_output_rmse": metrics.relative_output_rmse,
        "relative_weight_error": metrics.relative_weight_error,
    }


def iteration_to_dict(iteration: JointAAIteration) -> dict[str, float | int]:
    return {
        "iteration": iteration.iteration,
        "step_size": iteration.step_size,
        "backtracks": iteration.backtracks,
        "ridge": iteration.ridge,
        "objective": iteration.objective,
        "relative_output_rmse": iteration.relative_output_rmse,
        "relative_improvement": iteration.relative_improvement,
        "max_orthogonality_error": iteration.max_orthogonality_error,
    }
