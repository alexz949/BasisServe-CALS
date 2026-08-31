from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from basisserve.core.mlp_down_c1 import (
    MLPDownC1Linear,
    fit_mlp_down_c1,
    fit_mlp_down_c1_cross_moments,
    fit_mlp_down_c1_cross_moments_from_cholesky,
    fit_mlp_down_c1_fixed_encoder_decoder,
)


def test_cross_moment_fit_closes_sequential_teacher_target() -> None:
    generator = torch.Generator().manual_seed(23)
    student_inputs = torch.randn(
        2048, 7, generator=generator, dtype=torch.float64
    )
    left = torch.randn(5, 2, generator=generator, dtype=torch.float64)
    right = torch.randn(7, 2, generator=generator, dtype=torch.float64)
    teacher_targets = student_inputs @ right @ left.T
    rows = len(student_inputs)
    fitted = fit_mlp_down_c1_cross_moments(
        student_inputs.T @ student_inputs / rows,
        teacher_targets.T @ student_inputs / rows,
        teacher_targets.T @ teacher_targets / rows,
        rank=2,
        factor_dtype=torch.float64,
        work_dtype=torch.float64,
    )
    module = MLPDownC1Linear(fitted.input_factor, fitted.output_basis)
    torch.testing.assert_close(
        module(student_inputs), teacher_targets, rtol=1e-10, atol=1e-10
    )
    assert fitted.metrics["fit_relative_output_mse"] < 1e-12
    assert fitted.metrics["covariance_damping"] == 0.0

    _, input_r = torch.linalg.qr(student_inputs, mode="reduced")
    fitted_from_qr = fit_mlp_down_c1_cross_moments_from_cholesky(
        input_r.T / rows**0.5,
        teacher_targets.T @ student_inputs / rows,
        teacher_targets.T @ teacher_targets / rows,
        rank=2,
        factor_dtype=torch.float64,
        work_dtype=torch.float64,
    )
    qr_module = MLPDownC1Linear(
        fitted_from_qr.input_factor, fitted_from_qr.output_basis
    )
    torch.testing.assert_close(
        qr_module(student_inputs), teacher_targets, rtol=1e-10, atol=1e-10
    )
    assert "streaming_tsqr" in fitted_from_qr.metrics["solver"]


def test_ridge_cross_moment_metrics_match_direct_objective() -> None:
    generator = torch.Generator().manual_seed(37)
    student_inputs = torch.randn(
        1024, 7, generator=generator, dtype=torch.float64
    )
    teacher_targets = torch.randn(
        1024, 5, generator=generator, dtype=torch.float64
    )
    rows = len(student_inputs)
    c_xx = student_inputs.T @ student_inputs / rows
    c_yx = teacher_targets.T @ student_inputs / rows
    c_yy = teacher_targets.T @ teacher_targets / rows
    relative_damping = 0.1
    absolute_damping = relative_damping * torch.trace(c_xx).item() / len(c_xx)
    raw_cholesky = torch.linalg.cholesky(c_xx)
    solve_cholesky = torch.linalg.cholesky(
        c_xx + absolute_damping * torch.eye(7, dtype=torch.float64)
    )
    fitted = fit_mlp_down_c1_cross_moments_from_cholesky(
        solve_cholesky,
        c_yx,
        c_yy,
        objective_input_cholesky=raw_cholesky,
        absolute_damping=absolute_damping,
        relative_damping=relative_damping,
        rank=2,
        factor_dtype=torch.float64,
        work_dtype=torch.float64,
    )
    prediction = MLPDownC1Linear(
        fitted.input_factor, fitted.output_basis
    )(student_inputs)
    target_energy = torch.sum(teacher_targets * teacher_targets) / rows
    output_error = torch.sum((teacher_targets - prediction) ** 2) / rows
    weight = fitted.input_factor @ fitted.output_basis.T
    regularized = output_error + absolute_damping * torch.sum(weight * weight)
    assert fitted.metrics["fit_relative_output_mse"] == pytest.approx(
        (output_error / target_energy).item(), rel=1e-10, abs=1e-10
    )
    assert fitted.metrics["fit_relative_regularized_objective"] == pytest.approx(
        (regularized / target_energy).item(), rel=1e-10, abs=1e-10
    )
    assert fitted.metrics["relative_covariance_damping"] == relative_damping


def test_fixed_encoder_sequential_decoder_recovers_teacher() -> None:
    generator = torch.Generator().manual_seed(41)
    student_inputs = torch.randn(
        1536, 9, generator=generator, dtype=torch.float64
    )
    fixed_encoder = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    teacher_decoder = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    latent = student_inputs @ fixed_encoder
    teacher_targets = latent @ teacher_decoder.T
    rows = len(latent)
    _, latent_r = torch.linalg.qr(latent, mode="reduced")
    fitted = fit_mlp_down_c1_fixed_encoder_decoder(
        fixed_encoder,
        latent_r.T / rows**0.5,
        teacher_targets.T @ latent / rows,
        teacher_targets.T @ teacher_targets / rows,
        factor_dtype=torch.float64,
        work_dtype=torch.float64,
    )
    prediction = MLPDownC1Linear(
        fitted.input_factor, fitted.output_basis
    )(student_inputs)
    torch.testing.assert_close(
        prediction, teacher_targets, rtol=1e-10, atol=1e-10
    )
    assert fitted.metrics["fit_relative_output_mse"] < 1e-12
    assert fitted.metrics["covariance_damping"] == 0.0


def test_shared_decoder_fit_closes_complete_mlp_output() -> None:
    generator = torch.Generator().manual_seed(29)
    output_basis, _ = torch.linalg.qr(
        torch.randn(6, 2, generator=generator, dtype=torch.float64)
    )
    input_map = torch.randn(2, 9, generator=generator, dtype=torch.float64)
    weight = output_basis @ input_map
    fit_h = torch.randn(512, 9, generator=generator, dtype=torch.float64)
    heldout_h = torch.randn(173, 9, generator=generator, dtype=torch.float64)
    fit_y = F.linear(fit_h, weight)
    heldout_y = F.linear(heldout_h, weight)
    fitted = fit_mlp_down_c1(
        weight,
        fit_y.T @ fit_y / len(fit_y),
        heldout_y.T @ heldout_y / len(heldout_y),
        rank=2,
        factor_dtype=torch.float64,
        work_dtype=torch.float64,
    )
    module = MLPDownC1Linear(fitted.input_factor, fitted.output_basis)
    torch.testing.assert_close(module(heldout_h), heldout_y, rtol=1e-10, atol=1e-10)
    assert fitted.metrics["fit_relative_output_mse"] < 1e-12
    assert fitted.metrics["heldout_relative_output_mse"] < 1e-12
    assert fitted.metrics["als_sweeps"] == 0
    assert fitted.metrics["damping"] == 0.0


def test_tp_source_encoders_sum_in_the_shared_latent() -> None:
    generator = torch.Generator().manual_seed(31)
    weight = torch.randn(5, 12, generator=generator, dtype=torch.float64)
    hidden = torch.randn(7, 12, generator=generator, dtype=torch.float64)
    teacher = F.linear(hidden, weight)
    covariance = teacher.T @ teacher / len(teacher)
    fitted = fit_mlp_down_c1(
        weight,
        covariance,
        covariance,
        rank=3,
        factor_dtype=torch.float64,
        work_dtype=torch.float64,
    )
    source_latents = []
    for hidden_shard, encoder_shard in zip(hidden.chunk(4, dim=-1), fitted.input_factor.chunk(4, dim=0)):
        source_latents.append(hidden_shard @ encoder_shard)
    reduced = torch.stack(source_latents).sum(dim=0)
    direct = hidden @ fitted.input_factor
    torch.testing.assert_close(reduced, direct, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        reduced @ fitted.output_basis.T,
        MLPDownC1Linear(fitted.input_factor, fitted.output_basis)(hidden),
        rtol=1e-12,
        atol=1e-12,
    )
