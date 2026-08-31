"""Generic activation-aware C1 fitting for TP-source output projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from basisserve.core.gqa_routed_ov_joint import (
    SeparableEncoderSolver,
    covariance_with_trace_damping,
    evaluate_quadratic,
    fit_routed_ov_joint,
    initialize_group_pooled_routed_svd,
    quadratic_from_target,
)
from basisserve.core.tp_source_wo_c1 import (
    TPSourceWOLayout,
    covariance_to_source_blocks,
    identity_factors,
    weight_to_source_targets,
)


@dataclass(frozen=True)
class TPSourceWOFitConfig:
    """Numerical controls for one rectangular TP-source C1 fit."""

    encoder_sweeps: int = 1
    minimum_encoder_sweeps: int = 0
    covariance_damping: float = 1.0e-5
    encoder_relative_tolerance: float = 1.0e-5
    encoder_patience: int = 1
    maximum_backtracks: int = 8

    def validate(self) -> None:
        if self.encoder_sweeps < 0 or not 0 <= self.minimum_encoder_sweeps <= self.encoder_sweeps:
            raise ValueError("invalid encoder sweep range")
        if min(
            self.covariance_damping,
            self.encoder_relative_tolerance,
        ) < 0:
            raise ValueError("C1 damping and tolerances must be nonnegative")
        if min(
            self.encoder_patience,
            self.maximum_backtracks,
        ) <= 0:
            raise ValueError("C1 iteration controls must be positive")


@dataclass(frozen=True)
class TPSourceWOFitResult:
    source_encoders: Tensor
    source_decoders: Tensor
    fit_relative_mse: float
    heldout_relative_mse: float
    quantized_fit_relative_mse: float
    quantized_heldout_relative_mse: float
    selected_boundary: str
    selected_sweep: int
    checkpoints: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]


def _relative(loss: float, constant: Tensor) -> float:
    return float(loss) / max(abs(float(constant)), 1.0e-300)


class _HeldoutSelector:
    def __init__(self, objective: Any, mapping: Tensor, fit_constant: Tensor) -> None:
        self.objective = objective
        self.mapping = mapping
        self.fit_constant = fit_constant
        self.rows: list[dict[str, Any]] = []
        self.best_loss = float("inf")
        self.best_A: Tensor | None = None
        self.best_D: Tensor | None = None
        self.best_boundary = ""
        self.best_sweep = -1

    def __call__(self, checkpoint: Any, A: Tensor, D: Tensor) -> None:
        validation_loss = float(
            evaluate_quadratic(self.objective, A, D, self.mapping)
        )
        boundary = str(checkpoint.boundary)
        eligible = boundary in {"decoder_only", "after_redecoder"}
        self.rows.append(
            {
                "boundary": boundary,
                "sweep": int(checkpoint.sweep),
                "fit_relative_mse": _relative(
                    float(checkpoint.loss), self.fit_constant
                ),
                "heldout_relative_mse": _relative(
                    validation_loss, self.objective.constant
                ),
                "selection_eligible": eligible,
            }
        )
        if eligible and validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.best_A = A.detach().clone()
            self.best_D = D.detach().clone()
            self.best_boundary = boundary
            self.best_sweep = int(checkpoint.sweep)

    def selected(self) -> tuple[Tensor, Tensor]:
        if self.best_A is None or self.best_D is None:
            raise RuntimeError("C1 solver produced no decoder-closed checkpoint")
        return self.best_A, self.best_D


@torch.no_grad()
def fit_tp_source_wo_c1(
    weight: Tensor,
    fit_covariance: Tensor,
    heldout_covariance: Tensor,
    layout: TPSourceWOLayout,
    *,
    config: TPSourceWOFitConfig = TPSourceWOFitConfig(),
    work_device: torch.device | str | None = None,
    work_dtype: torch.dtype = torch.float32,
    factor_dtype: torch.dtype = torch.bfloat16,
    objective_name: str = "tp_source_wo_c1",
) -> TPSourceWOFitResult:
    """Fit one post-gate/output-projection C1 map and select on held-out MSE."""

    config.validate()
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("C1 work dtype must be float32 or float64")
    if factor_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("unsupported C1 factor dtype")
    device = torch.device(weight.device if work_device is None else work_device)
    if tuple(weight.shape) != (layout.output_width, layout.input_width):
        raise ValueError("C1 weight does not match its layout")
    expected_covariance = (layout.input_width, layout.input_width)
    if tuple(fit_covariance.shape) != expected_covariance or tuple(
        heldout_covariance.shape
    ) != expected_covariance:
        raise ValueError("C1 covariance does not match its layout")

    if layout.source_rank == layout.source_width:
        encoders, decoders = identity_factors(
            weight.to(device=device), layout, dtype=factor_dtype
        )
        return TPSourceWOFitResult(
            source_encoders=encoders.cpu().contiguous(),
            source_decoders=decoders.cpu().contiguous(),
            fit_relative_mse=0.0,
            heldout_relative_mse=0.0,
            quantized_fit_relative_mse=0.0,
            quantized_heldout_relative_mse=0.0,
            selected_boundary="identity",
            selected_sweep=0,
            checkpoints=(),
            diagnostics={"method": "analytic_identity"},
        )

    fit_raw = covariance_to_source_blocks(
        fit_covariance, layout, device=device, dtype=work_dtype
    )
    heldout_blocks = covariance_to_source_blocks(
        heldout_covariance, layout, device=device, dtype=work_dtype
    )
    fit_blocks, absolute_damping = covariance_with_trace_damping(
        fit_raw, relative_damping=config.covariance_damping
    )
    target = weight_to_source_targets(
        weight, layout, device=device, dtype=work_dtype
    )
    fit_objective = quadratic_from_target(
        covariance=fit_blocks,
        target=target,
        name=f"{objective_name}_fit",
        trace_normalize=False,
    )
    heldout_objective = quadratic_from_target(
        covariance=heldout_blocks,
        target=target,
        name=f"{objective_name}_heldout",
        trace_normalize=False,
    )
    mapping = torch.arange(layout.tp_size, device=device, dtype=torch.long)
    ranks = (layout.source_rank,) * layout.tp_size
    initialization = initialize_group_pooled_routed_svd(
        covariance=fit_objective.covariance,
        target=target,
        head_to_kv_group=mapping,
        group_ranks=ranks,
        covariance_ridge=0.0,
    )
    initial_D = torch.zeros(
        layout.tp_size,
        layout.source_rank,
        layout.output_width,
        device=device,
        dtype=work_dtype,
    )
    selector = _HeldoutSelector(
        heldout_objective, mapping, fit_objective.constant
    )
    solved = fit_routed_ov_joint(
        objective=fit_objective,
        initial_A=initialization.A_unique,
        initial_D=initial_D,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
        maximum_sweeps=config.encoder_sweeps,
        minimum_sweeps=config.minimum_encoder_sweeps,
        relative_objective_tolerance=config.encoder_relative_tolerance,
        patience=config.encoder_patience,
        decoder_relative_jitter=0.0,
        encoder_relative_damping=0.0,
        maximum_backtracks=config.maximum_backtracks,
        final_decoder_solve=True,
        group_ranks=ranks,
        checkpoint_callback=selector,
        encoder_direction_solver=SeparableEncoderSolver(
            left_relative_damping=0.0,
            right_relative_damping=0.0,
            name="exact_single_source_two_sided_cholesky",
        ),
        decoder_stationarity_override=lambda *_: 0.0,
        work_dtype=work_dtype,
        work_device=device,
    )
    selected_A, selected_D = selector.selected()

    fit_loss = float(
        evaluate_quadratic(fit_objective, selected_A, selected_D, mapping)
    )
    heldout_loss = float(
        evaluate_quadratic(heldout_objective, selected_A, selected_D, mapping)
    )
    stored_A = selected_A.to(dtype=factor_dtype)
    stored_D = selected_D.to(dtype=factor_dtype)
    quantized_A = stored_A.to(device=device, dtype=work_dtype)
    quantized_D = stored_D.to(device=device, dtype=work_dtype)
    quantized_fit = float(
        evaluate_quadratic(fit_objective, quantized_A, quantized_D, mapping)
    )
    quantized_heldout = float(
        evaluate_quadratic(
            heldout_objective, quantized_A, quantized_D, mapping
        )
    )
    encoder_solves = [
        {
            "sweep": int(sweep.sweep),
            "source": int(step.group_index),
            "solver": (
                step.direction.solver_name
                if step.direction is not None
                else "unknown"
            ),
            "relative_residual": (
                float(step.direction.exact_hessian_relative_residual)
                if step.direction is not None
                else float("nan")
            ),
            "two_sided_solve_relative_residual": (
                float(
                    step.direction.solver_diagnostics[
                        "two_sided_solve_relative_residual"
                    ]
                )
                if step.direction is not None
                else float("nan")
            ),
            "accepted_scale": float(step.accepted_scale),
            "backtracks": int(step.retries),
            "old_loss": float(step.old_loss),
            "new_loss": float(step.new_loss),
        }
        for sweep in solved.sweeps
        for step in sweep.encoder_steps
    ]
    accepted_encoder_solves = [
        row for row in encoder_solves if float(row["accepted_scale"]) > 0.0
    ]
    diagnostics = {
        "method": (
            "activation_weighted_initialization_plus_joint_c1_als_with_"
            "exact_single_source_two_sided_encoder_solves"
        ),
        "absolute_covariance_damping": absolute_damping,
        "encoder_sweeps_completed": len(solved.sweeps),
        "initial_relative_mse": _relative(
            solved.initial_loss, fit_objective.constant
        ),
        "decoder_only_relative_mse": _relative(
            solved.decoder_only_loss, fit_objective.constant
        ),
        "final_relative_mse": _relative(solved.final_loss, fit_objective.constant),
        "encoder_group_solves": len(encoder_solves),
        "encoder_maximum_relative_residual": max(
            (float(row["relative_residual"]) for row in encoder_solves),
            default=0.0,
        ),
        "encoder_maximum_accepted_relative_residual": max(
            (
                float(row["relative_residual"])
                for row in accepted_encoder_solves
            ),
            default=0.0,
        ),
        "encoder_maximum_backtracks": max(
            (int(row["backtracks"]) for row in encoder_solves),
            default=0,
        ),
        "encoder_solves": encoder_solves,
    }
    return TPSourceWOFitResult(
        source_encoders=stored_A.cpu().contiguous(),
        source_decoders=stored_D.cpu().contiguous(),
        fit_relative_mse=_relative(fit_loss, fit_objective.constant),
        heldout_relative_mse=_relative(heldout_loss, heldout_objective.constant),
        quantized_fit_relative_mse=_relative(
            quantized_fit, fit_objective.constant
        ),
        quantized_heldout_relative_mse=_relative(
            quantized_heldout, heldout_objective.constant
        ),
        selected_boundary=selector.best_boundary,
        selected_sweep=selector.best_sweep,
        checkpoints=tuple(selector.rows),
        diagnostics=diagnostics,
    )


__all__ = [
    "TPSourceWOFitConfig",
    "TPSourceWOFitResult",
    "fit_tp_source_wo_c1",
]
