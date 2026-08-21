"""Activation-aware source-private factors for Qwen3.5 GDN output wires."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor

from basisserve.core.qwen35_wo_wire import fit_activation_aware_wo_factors
from basisserve.core.routed_decoder_design import (
    prepare_trace_normalized_prior_ridge,
    solve_trace_normalized_prior_ridge,
)


@dataclass(frozen=True)
class GDNPrivateAGJointFactors:
    """One layer of fixed source-private encoders and one joint decoder."""

    private_encoders: Tensor
    joint_decoder_weight: Tensor
    metrics: dict[str, Any]


def _quadratic_relative_error(
    hessian: Tensor,
    rhs: Tensor,
    target_energy: Tensor,
    decoder: Tensor,
) -> float:
    loss = target_energy - 2.0 * (decoder * rhs).sum() + (
        decoder * (hessian @ decoder)
    ).sum()
    scale = float(target_energy)
    if not math.isfinite(scale) or scale <= 0:
        raise FloatingPointError("activation-aware target has invalid energy")
    value = float(loss) / scale
    tolerance = 2048.0 * torch.finfo(hessian.dtype).eps
    if value < -tolerance:
        raise FloatingPointError(
            f"activation-aware reconstruction objective is negative: {value:.6e}"
        )
    return max(value, 0.0)


@torch.no_grad()
def fit_gdn_private_ag_joint_factors(
    weight: Tensor,
    second_moment: Tensor,
    *,
    tp_size: int,
    local_rank: int,
    decoder_relative_ridge: float,
    relative_damping: float = 1.0e-5,
    factor_dtype: torch.dtype = torch.bfloat16,
) -> GDNPrivateAGJointFactors:
    """Fit equal-rank private encoders and a global output-MSE decoder.

    ``weight`` follows PyTorch's ``[output,input]`` convention.  Each source
    first receives its activation-aware local-output POD.  Those encoders are
    frozen, their codes are concatenated, and one trace-normalized centered
    ridge decoder is solved using the complete input second moment, including
    cross-source covariance.
    """

    if weight.ndim != 2 or second_moment.ndim != 2:
        raise ValueError("weight and second moment must be matrices")
    output_width, input_width = map(int, weight.shape)
    if tuple(second_moment.shape) != (input_width, input_width):
        raise ValueError("second moment does not match the GDN wire")
    if tp_size <= 1 or input_width % tp_size:
        raise ValueError("input width must divide across at least two TP sources")
    local_width = input_width // tp_size
    if not 0 < local_rank <= min(local_width, output_width):
        raise ValueError("local private rank is invalid")
    if not math.isfinite(decoder_relative_ridge) or decoder_relative_ridge < 0:
        raise ValueError("decoder ridge must be finite and nonnegative")
    if not math.isfinite(relative_damping) or relative_damping < 0:
        raise ValueError("moment damping must be finite and nonnegative")
    if factor_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("factor dtype must be BF16, FP16, or FP32")

    device = weight.device
    covariance = second_moment.detach().to(device=device, dtype=torch.float32)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    scale = covariance.diagonal().mean().clamp_min(torch.finfo(covariance.dtype).tiny)
    damping = scale * float(relative_damping)
    covariance.diagonal().add_(damping)
    logical = weight.detach().to(device=device, dtype=torch.float32).transpose(0, 1)

    encoder_blocks = []
    decoder_blocks = []
    local_metrics = []
    for source in range(tp_size):
        start = source * local_width
        stop = start + local_width
        local = fit_activation_aware_wo_factors(
            weight[:, start:stop],
            covariance[start:stop, start:stop],
            local_rank,
            relative_damping=0.0,
            factor_dtype=torch.float32,
        )
        encoder_blocks.append(local.input_factor)
        decoder_blocks.append(local.output_basis.transpose(0, 1))
        local_metrics.append(
            {
                "source": source,
                "retained_fraction": local.retained_fraction(local_rank),
                "total_weighted_energy": local.total_weighted_energy,
            }
        )

    total_rank = tp_size * local_rank
    structured_encoder = torch.zeros(
        input_width,
        total_rank,
        device=device,
        dtype=torch.float32,
    )
    for source, encoder in enumerate(encoder_blocks):
        input_start = source * local_width
        rank_start = source * local_rank
        structured_encoder[
            input_start : input_start + local_width,
            rank_start : rank_start + local_rank,
        ] = encoder
    center = torch.cat(decoder_blocks, dim=0).contiguous()

    covariance_encoder = covariance @ structured_encoder
    hessian = structured_encoder.transpose(0, 1) @ covariance_encoder
    hessian = 0.5 * (hessian + hessian.transpose(0, 1))
    covariance_target = covariance @ logical
    rhs = structured_encoder.transpose(0, 1) @ covariance_target
    target_energy = (logical * covariance_target).sum()
    workspace = prepare_trace_normalized_prior_ridge(
        hessian=hessian,
        rhs=rhs,
        center=center,
    )
    decoder, solver = solve_trace_normalized_prior_ridge(
        workspace,
        alpha=float(decoder_relative_ridge),
    )
    independent_error = _quadratic_relative_error(
        hessian,
        rhs,
        target_energy,
        center,
    )
    joint_error = _quadratic_relative_error(
        hessian,
        rhs,
        target_energy,
        decoder,
    )

    stored_encoders = torch.stack(encoder_blocks).to(dtype=factor_dtype)
    stored_decoder_weight = decoder.transpose(0, 1).to(dtype=factor_dtype)
    quantized_encoder = torch.zeros_like(structured_encoder)
    for source, encoder in enumerate(stored_encoders.float()):
        input_start = source * local_width
        rank_start = source * local_rank
        quantized_encoder[
            input_start : input_start + local_width,
            rank_start : rank_start + local_rank,
        ] = encoder
    quantized_decoder = stored_decoder_weight.float().transpose(0, 1)
    quantized_covariance_encoder = covariance @ quantized_encoder
    quantized_hessian = quantized_encoder.transpose(0, 1) @ quantized_covariance_encoder
    quantized_hessian = 0.5 * (
        quantized_hessian + quantized_hessian.transpose(0, 1)
    )
    quantized_rhs = quantized_encoder.transpose(0, 1) @ covariance_target
    quantized_error = _quadratic_relative_error(
        quantized_hessian,
        quantized_rhs,
        target_energy,
        quantized_decoder,
    )

    metrics = {
        "tp_size": tp_size,
        "input_width": input_width,
        "output_width": output_width,
        "local_width": local_width,
        "local_rank": local_rank,
        "total_private_rank": total_rank,
        "relative_damping": float(relative_damping),
        "absolute_damping": float(damping),
        "decoder_relative_ridge": float(decoder_relative_ridge),
        "independent_relative_output_mse": independent_error,
        "joint_relative_output_mse": joint_error,
        "joint_relative_reduction": 1.0 - joint_error / max(independent_error, 1.0e-30),
        "quantized_joint_relative_output_mse": quantized_error,
        "quantized_joint_relative_reduction": (
            1.0 - quantized_error / max(independent_error, 1.0e-30)
        ),
        "solver": asdict(solver),
        "local_fits": local_metrics,
    }
    return GDNPrivateAGJointFactors(
        private_encoders=stored_encoders.cpu().contiguous(),
        joint_decoder_weight=stored_decoder_weight.cpu().contiguous(),
        metrics=metrics,
    )


__all__ = [
    "GDNPrivateAGJointFactors",
    "fit_gdn_private_ag_joint_factors",
]
