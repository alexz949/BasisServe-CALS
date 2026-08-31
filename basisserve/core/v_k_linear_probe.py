"""Headwise linear probes from Value features to Key features.

The probes in this module are diagnostics, not serving implementations.  They
answer whether a resident Value representation linearly contains the
information needed to reconstruct a same-layer Key representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HeadwiseLinearProbe:
    """Independent affine maps for physical KV heads."""

    weight: Tensor
    bias: Tensor

    def validate(self, *, heads: int, input_dim: int, output_dim: int) -> None:
        expected_weight = (heads, input_dim, output_dim)
        expected_bias = (heads, output_dim)
        if tuple(self.weight.shape) != expected_weight:
            raise ValueError(
                f"probe weight must have shape {expected_weight}, got "
                f"{tuple(self.weight.shape)}"
            )
        if tuple(self.bias.shape) != expected_bias:
            raise ValueError(
                f"probe bias must have shape {expected_bias}, got "
                f"{tuple(self.bias.shape)}"
            )
        if not self.weight.is_floating_point() or not self.bias.is_floating_point():
            raise TypeError("probe factors must be floating point")


def _validate_training_pair(source: Tensor, target: Tensor) -> tuple[int, int, int]:
    if source.ndim != 4 or target.ndim != 4:
        raise ValueError("source and target must be [batch, heads, sequence, dim]")
    if tuple(source.shape[:3]) != tuple(target.shape[:3]):
        raise ValueError("source and target must share batch/head/sequence geometry")
    if not source.is_floating_point() or not target.is_floating_point():
        raise TypeError("source and target must be floating point")
    return int(source.shape[1]), int(source.shape[-1]), int(target.shape[-1])


def fit_headwise_linear_probe(
    source: Tensor,
    target: Tensor,
    *,
    relative_ridge: float = 1.0e-6,
) -> HeadwiseLinearProbe:
    """Fit centered headwise ridge regressions in FP64 on CPU.

    ``relative_ridge`` multiplies the mean diagonal of each centered Gram
    matrix, so its meaning is invariant to the scale of the source features.
    The returned bias restores the removed source and target means.
    """

    if relative_ridge < 0:
        raise ValueError("relative ridge must be nonnegative")
    heads, input_dim, output_dim = _validate_training_pair(source, target)
    source_by_head = source.detach().permute(1, 0, 2, 3).reshape(
        heads, -1, input_dim
    )
    target_by_head = target.detach().permute(1, 0, 2, 3).reshape(
        heads, -1, output_dim
    )
    weights = []
    biases = []
    for head in range(heads):
        design = source_by_head[head].to(device="cpu", dtype=torch.float64)
        response = target_by_head[head].to(device="cpu", dtype=torch.float64)
        design_mean = design.mean(dim=0)
        response_mean = response.mean(dim=0)
        centered_design = design - design_mean
        centered_response = response - response_mean
        gram = centered_design.mT @ centered_design
        scale = gram.diagonal().mean()
        if not torch.isfinite(scale) or float(scale) <= 0.0:
            raise ValueError(f"source head {head} has no finite centered energy")
        regularized = gram + (
            relative_ridge * scale * torch.eye(input_dim, dtype=torch.float64)
        )
        weight = torch.linalg.solve(
            regularized,
            centered_design.mT @ centered_response,
        )
        bias = response_mean - design_mean @ weight
        weights.append(weight.float())
        biases.append(bias.float())
    result = HeadwiseLinearProbe(
        weight=torch.stack(weights),
        bias=torch.stack(biases),
    )
    result.validate(heads=heads, input_dim=input_dim, output_dim=output_dim)
    return result


def apply_headwise_linear_probe(source: Tensor, probe: HeadwiseLinearProbe) -> Tensor:
    """Apply a headwise probe to ``[batch, heads, sequence, input_dim]``."""

    if source.ndim != 4 or not source.is_floating_point():
        raise ValueError("source must be floating [batch, heads, sequence, dim]")
    _, heads, _, input_dim = map(int, source.shape)
    output_dim = int(probe.weight.shape[-1])
    probe.validate(heads=heads, input_dim=input_dim, output_dim=output_dim)
    weight = probe.weight.to(device=source.device, dtype=torch.float32)
    bias = probe.bias.to(device=source.device, dtype=torch.float32)
    return torch.einsum("bhsi,hio->bhso", source.float(), weight) + bias[
        None, :, None, :
    ]


def centered_relative_squared_error(reference: Tensor, candidate: Tensor) -> float:
    """Return error energy divided by per-head centered reference energy."""

    if tuple(reference.shape) != tuple(candidate.shape) or reference.ndim != 4:
        raise ValueError("reference and candidate must share four-dimensional geometry")
    reference_fp64 = reference.detach().to(device="cpu", dtype=torch.float64)
    candidate_fp64 = candidate.detach().to(device="cpu", dtype=torch.float64)
    head_mean = reference_fp64.mean(dim=(0, 2), keepdim=True)
    energy = (reference_fp64 - head_mean).square().sum()
    error = (candidate_fp64 - reference_fp64).square().sum()
    return float(error / energy.clamp_min(torch.finfo(torch.float64).tiny))
