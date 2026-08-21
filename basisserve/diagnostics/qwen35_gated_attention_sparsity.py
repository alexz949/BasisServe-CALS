"""Offline effective-sparsity diagnostics for Qwen3.5 gated attention.

The routines in this module never alter model weights.  They reconstruct the
actual bfloat16 sigmoid-and-multiply path saved by the gated-Wo collector and
then evaluate token-dependent coordinate subsets before ``o_proj``.  Metric
reductions use float64; operator products remain float32.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


SCORE_METHODS = (
    "gate_abs",
    "pre_gate_activation_abs",
    "gated_activation_abs",
    "pre_gate_column_norm_weighted",
    "column_norm_weighted",
)


@dataclass(frozen=True)
class RuntimeActivations:
    """Values produced by the checkpoint's bfloat16 post-attention gate."""

    h_pre_gate: Tensor
    gate_logits: Tensor
    a_post_sigmoid: Tensor
    c_post_gate: Tensor


@dataclass(frozen=True)
class GreedyCheckpoint:
    """A direct-GEMM audit point along a fixed-k greedy path."""

    selected_k: int
    selected_indices: Tensor
    prediction: Tensor
    direct_residual_energy: Tensor
    recurrence_residual_energy: Tensor
    maximum_recurrence_absolute_error: float
    maximum_recurrence_relative_error: float


@dataclass(frozen=True)
class GreedyResidualResult:
    """Nested target-aware greedy selections and their recurrence audits."""

    ranking: Tensor
    selected_gains: Tensor
    teacher: Tensor
    checkpoints: Mapping[int, GreedyCheckpoint]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class RandomExpectationStatistics:
    """Per-token sufficient statistics shared by every random keep ratio."""

    teacher_energy_per_token: Tensor
    component_energy_per_token: Tensor
    num_tokens: int
    width: int


def _require_matrix(name: str, value: Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got shape {tuple(value.shape)}")


def _require_same_device(*values: Tensor) -> None:
    devices = {value.device for value in values}
    if len(devices) != 1:
        raise ValueError(f"tensors must share one device, got {sorted(map(str, devices))}")


def reconstruct_bf16_runtime(h_pre_gate: Tensor, gate_logits: Tensor) -> RuntimeActivations:
    """Reconstruct sigmoid and multiplication before any float32 conversion.

    The collector stores both inputs as bfloat16.  Requiring that dtype here
    prevents an apparently innocuous float32 sigmoid from changing the
    executed Qwen3.5 path.
    """

    _require_matrix("h_pre_gate", h_pre_gate)
    _require_matrix("gate_logits", gate_logits)
    if h_pre_gate.shape != gate_logits.shape:
        raise ValueError("h_pre_gate and gate_logits shapes differ")
    _require_same_device(h_pre_gate, gate_logits)
    if h_pre_gate.dtype != torch.bfloat16 or gate_logits.dtype != torch.bfloat16:
        raise TypeError("runtime reconstruction requires bfloat16 snapshots")
    if not bool(torch.isfinite(h_pre_gate).all()) or not bool(
        torch.isfinite(gate_logits).all()
    ):
        raise ValueError("runtime snapshots contain non-finite values")
    a_post_sigmoid = torch.sigmoid(gate_logits)
    c_post_gate = h_pre_gate * a_post_sigmoid
    return RuntimeActivations(
        h_pre_gate=h_pre_gate,
        gate_logits=gate_logits,
        a_post_sigmoid=a_post_sigmoid,
        c_post_gate=c_post_gate,
    )


def reconstruct_gdn_silu_runtime(
    raw_core: Tensor,
    gate_preactivation: Tensor,
    norm_weight: Tensor,
    *,
    num_value_heads: int,
    value_head_dim: int,
    rms_norm_eps: float,
) -> RuntimeActivations:
    """Reconstruct the Qwen3.5 GDN post-state RMSNorm/SiLU wire.

    Qwen3.5 evaluates RMSNorm independently on every value head, rounds the
    normalized core to the model dtype, multiplies it by an FP32 SiLU gate,
    and rounds the product back to the model dtype before ``out_proj``.  The
    returned ``RuntimeActivations`` uses the historical field
    ``a_post_sigmoid`` for the *SiLU* value so the shared sparsity diagnostics
    can remain gate-family agnostic.  No recurrent-state tensor is changed.
    """

    _require_matrix("raw_core", raw_core)
    _require_matrix("gate_preactivation", gate_preactivation)
    if raw_core.shape != gate_preactivation.shape:
        raise ValueError("GDN raw_core and gate_preactivation shapes differ")
    _require_same_device(raw_core, gate_preactivation, norm_weight)
    if raw_core.dtype != torch.bfloat16 or gate_preactivation.dtype != torch.bfloat16:
        raise TypeError("GDN runtime reconstruction requires bfloat16 snapshots")
    if norm_weight.ndim != 1 or norm_weight.dtype != torch.float32:
        raise TypeError("GDN norm_weight must be a float32 vector")
    if num_value_heads <= 0 or value_head_dim <= 0:
        raise ValueError("GDN value-head geometry must be positive")
    if int(raw_core.shape[1]) != num_value_heads * value_head_dim:
        raise ValueError("GDN snapshot width differs from value-head geometry")
    if tuple(norm_weight.shape) != (value_head_dim,):
        raise ValueError("GDN norm_weight differs from value-head dimension")
    if not math.isfinite(float(rms_norm_eps)) or float(rms_norm_eps) <= 0.0:
        raise ValueError("GDN RMSNorm epsilon must be finite and positive")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (raw_core, gate_preactivation, norm_weight)
    ):
        raise ValueError("GDN runtime snapshots contain non-finite values")

    rows = int(raw_core.shape[0])
    core_heads = raw_core.reshape(rows, num_value_heads, value_head_dim)
    work = core_heads.float()
    variance = work.pow(2).mean(dim=-1, keepdim=True)
    normalized_heads = work * torch.rsqrt(variance + float(rms_norm_eps))
    normalized_heads = (
        norm_weight.to(dtype=raw_core.dtype).reshape(1, 1, value_head_dim)
        * normalized_heads.to(dtype=raw_core.dtype)
    )
    gate_activation = F.silu(gate_preactivation.float())
    normalized = normalized_heads.reshape_as(raw_core).contiguous()
    post_gate = (normalized * gate_activation).to(dtype=raw_core.dtype)
    return RuntimeActivations(
        h_pre_gate=normalized,
        gate_logits=gate_preactivation,
        # Historical name retained for compatibility with generic scoring,
        # energy, random, and greedy routines.  This tensor is FP32 SiLU.
        a_post_sigmoid=gate_activation,
        c_post_gate=post_gate,
    )


def _validate_weight(runtime: RuntimeActivations, weight: Tensor) -> None:
    _require_matrix("weight", weight)
    if int(weight.shape[1]) != int(runtime.c_post_gate.shape[1]):
        raise ValueError("o_proj input width differs from snapshot width")
    if weight.device != runtime.c_post_gate.device:
        raise ValueError("weight and snapshots must share a device")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("weight contains non-finite values")


def score_channels(
    runtime: RuntimeActivations,
    weight: Tensor,
    method: str,
) -> Tensor:
    """Return one corrected selection score per token and wire channel."""

    _validate_weight(runtime, weight)
    if method not in SCORE_METHODS:
        raise ValueError(f"unsupported score method {method!r}")
    h = runtime.h_pre_gate.float().abs()
    a = runtime.a_post_sigmoid.float().abs()
    c = runtime.c_post_gate.float().abs()
    if method == "gate_abs":
        return a
    if method == "pre_gate_activation_abs":
        return h
    if method == "gated_activation_abs":
        return c
    column_norm = torch.linalg.vector_norm(weight.float(), dim=0)
    if method == "pre_gate_column_norm_weighted":
        return h * column_norm
    return c * column_norm


def stable_descending_ranking(scores: Tensor) -> Tensor:
    """Sort descending, resolving exact ties toward the lower channel index."""

    _require_matrix("scores", scores)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores contain non-finite values")
    return torch.argsort(scores, dim=1, descending=True, stable=True)


def retained_k(keep_ratio: float, width: int) -> int:
    """Convert a ratio to fixed cardinality using round-half-up semantics."""

    ratio = float(keep_ratio)
    if width <= 0:
        raise ValueError("width must be positive")
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("keep_ratio must lie in [0, 1]")
    return min(width, max(0, int(math.floor(ratio * width + 0.5))))


def selected_indices(ranking: Tensor, selected_k: int) -> Tensor:
    _require_matrix("ranking", ranking)
    width = int(ranking.shape[1])
    if ranking.dtype != torch.long:
        raise TypeError("ranking must use torch.long indices")
    if not 0 <= int(selected_k) <= width:
        raise ValueError(f"selected_k must lie in [0, {width}]")
    return ranking[:, : int(selected_k)]


def scatter_selected_to_dense(values: Tensor, indices: Tensor) -> Tensor:
    """Scatter token-dependent coordinate values into a dense GEMM operand."""

    _require_matrix("values", values)
    _require_matrix("indices", indices)
    if indices.dtype != torch.long:
        raise TypeError("indices must use torch.long")
    if int(indices.shape[0]) != int(values.shape[0]):
        raise ValueError("indices and values batch sizes differ")
    if indices.device != values.device:
        raise ValueError("indices and values must share a device")
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= int(values.shape[1])
    ):
        raise ValueError("selected index is out of range")
    dense = torch.zeros_like(values)
    if indices.shape[1]:
        dense.scatter_(1, indices, values.gather(1, indices))
    return dense


def selected_output(c_post_gate: Tensor, weight: Tensor, indices: Tensor) -> Tensor:
    """Evaluate a token-dependent subset by dense scatter followed by GEMM."""

    _require_matrix("weight", weight)
    if int(weight.shape[1]) != int(c_post_gate.shape[1]):
        raise ValueError("weight and c_post_gate widths differ")
    _require_same_device(c_post_gate, weight, indices)
    dense = scatter_selected_to_dense(c_post_gate, indices)
    return dense.float() @ weight.float().transpose(0, 1)


def _quantiles(values: Tensor) -> dict[str, float | None]:
    if values.numel() == 0:
        return {name: None for name in ("mean", "median", "p90", "p95", "p99")}
    values = values.double()
    return {
        "mean": float(values.mean()),
        "median": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
    }


def exact_linear_quantiles(values: Tensor, levels: Tensor) -> Tensor:
    """Exact linear quantiles without ``torch.quantile``'s 2**24 limit.

    ``torch.quantile`` rejects a flattened 8192x4096 population on currently
    deployed CPU and CUDA builds.  Sorting followed by the documented linear
    interpolation gives the same result and has no such indexing guard.
    """

    if values.ndim != 1 or levels.ndim != 1:
        raise ValueError("values and quantile levels must be vectors")
    if values.numel() == 0:
        raise ValueError("cannot compute quantiles of an empty vector")
    if values.device != levels.device:
        raise ValueError("values and quantile levels must share a device")
    if not bool(torch.isfinite(values).all()) or not bool(
        torch.isfinite(levels).all()
    ):
        raise ValueError("quantile inputs contain non-finite values")
    if bool((levels < 0).any()) or bool((levels > 1).any()):
        raise ValueError("quantile levels must lie in [0, 1]")
    ordered = torch.sort(values).values
    lower, upper, fraction = _linear_quantile_positions(
        levels,
        num_values=int(values.numel()),
        interpolation_dtype=ordered.dtype,
    )
    return torch.lerp(ordered[lower], ordered[upper], fraction)


def _linear_quantile_positions(
    levels: Tensor,
    *,
    num_values: int,
    interpolation_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute endpoint-safe ranks even when ``num_values > 2**24``."""

    if levels.ndim != 1 or num_values <= 0:
        raise ValueError("invalid quantile position geometry")
    maximum = num_values - 1
    positions = (levels.double() * maximum).clamp_(0.0, float(maximum))
    lower = torch.floor(positions).long().clamp_(0, maximum)
    upper = torch.ceil(positions).long().clamp_(0, maximum)
    fraction = (positions - lower.double()).to(dtype=interpolation_dtype)
    return lower, upper, fraction


def output_metrics(
    teacher: Tensor,
    prediction: Tensor,
    *,
    per_token_teacher_epsilon: float = 1.0e-24,
) -> dict[str, Any]:
    """Compute teacher-normalized output metrics with float64 reductions."""

    _require_matrix("teacher", teacher)
    _require_matrix("prediction", prediction)
    if teacher.shape != prediction.shape:
        raise ValueError("teacher and prediction shapes differ")
    _require_same_device(teacher, prediction)
    if not bool(torch.isfinite(teacher).all()) or not bool(
        torch.isfinite(prediction).all()
    ):
        raise ValueError("teacher or prediction contains non-finite values")
    if per_token_teacher_epsilon < 0:
        raise ValueError("per_token_teacher_epsilon must be nonnegative")

    teacher64 = teacher.double()
    prediction64 = prediction.double()
    teacher_per_token = teacher64.square().sum(dim=1)
    prediction_per_token = prediction64.square().sum(dim=1)
    cross_per_token = (prediction64 * teacher64).sum(dim=1)
    residual_per_token = (teacher64 - prediction64).square().sum(dim=1)
    teacher_energy = teacher_per_token.sum()
    if not float(teacher_energy) > 0.0:
        raise ValueError("aggregate teacher energy must be positive")
    prediction_energy_raw = prediction_per_token.sum()
    cross_raw = cross_per_token.sum()
    residual_raw = residual_per_token.sum()
    normalized_cross = cross_raw / teacher_energy
    prediction_teacher_energy = prediction_energy_raw / teacher_energy
    relative_mse_direct = residual_raw / teacher_energy
    relative_mse_identity = 1.0 + prediction_teacher_energy - 2.0 * normalized_cross
    denominator = torch.sqrt(teacher_energy * prediction_energy_raw)
    cosine = cross_raw / denominator if float(denominator) > 0.0 else denominator.new_zeros(())

    valid = teacher_per_token > float(per_token_teacher_epsilon)
    per_token = residual_per_token[valid] / teacher_per_token[valid]
    result: dict[str, Any] = {
        "relative_mse": float(relative_mse_direct),
        "normalized_cross": float(normalized_cross),
        "prediction_teacher_energy": float(prediction_teacher_energy),
        "cosine_similarity": float(cosine),
        "relative_mse_from_identity": float(relative_mse_identity),
        "relative_mse_identity_absolute_error": float(
            torch.abs(relative_mse_direct - relative_mse_identity)
        ),
        "per_token_relative_squared_error": _quantiles(per_token),
        # Backward-compatible spelling while explicitly documenting "squared".
        "per_token_relative_error": _quantiles(per_token),
        "near_zero_teacher_tokens": int((~valid).sum()),
        "valid_per_token_count": int(valid.sum()),
        "teacher_energy": float(teacher_energy),
        "prediction_energy": float(prediction_energy_raw),
        "cross_inner_product": float(cross_raw),
        "residual_energy": float(residual_raw),
    }
    return result


def retained_energy_metrics(
    c_post_gate: Tensor,
    a_post_sigmoid: Tensor,
    indices: Tensor,
) -> dict[str, float]:
    """Return selected post-gate-input and gate squared-energy fractions."""

    if c_post_gate.shape != a_post_sigmoid.shape:
        raise ValueError("c_post_gate and a_post_sigmoid shapes differ")
    selected_c = scatter_selected_to_dense(c_post_gate, indices).double()
    selected_a = scatter_selected_to_dense(a_post_sigmoid, indices).double()
    c64 = c_post_gate.double()
    a64 = a_post_sigmoid.double()
    c_denominator = c64.square().sum()
    a_denominator = a64.square().sum()
    if float(c_denominator) <= 0.0 or float(a_denominator) <= 0.0:
        raise ValueError("activation and gate energies must be positive")
    return {
        "retained_input_energy": float(selected_c.square().sum() / c_denominator),
        "retained_gate_energy": float(selected_a.square().sum() / a_denominator),
    }


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def random_rankings(
    num_tokens: int,
    width: int,
    *,
    seed: int,
    row_offset: int = 0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate independent exact random permutations, stable across chunks."""

    if num_tokens < 0 or width <= 0 or row_offset < 0:
        raise ValueError("invalid random-ranking geometry")
    target_device = torch.device(device)
    result = torch.empty((num_tokens, width), dtype=torch.long, device=target_device)
    generator = torch.Generator(device=target_device)
    # Hash seed and row non-commutatively.  ``seed ^ row`` would make nearby
    # seeds reuse the same multiset of row permutations (only reassigned to
    # different tokens), which invalidates Monte-Carlo replication.
    base = _splitmix64((int(seed) & 0xFFFFFFFFFFFFFFFF) ^ 0xD1B54A32D192ED03)
    for local_row in range(num_tokens):
        global_row = int(row_offset + local_row)
        row_state = (
            base
            + ((global_row + 1) * 0x9E3779B97F4A7C15)
            + 0x94D049BB133111EB
        ) & 0xFFFFFFFFFFFFFFFF
        mixed = _splitmix64(row_state) & 0x7FFFFFFFFFFFFFFF
        generator.manual_seed(mixed)
        result[local_row] = torch.randperm(
            width,
            generator=generator,
            device=target_device,
        )
    return result


def selection_frequency(indices: Tensor, width: int, *, normalize: bool = True) -> Tensor:
    _require_matrix("indices", indices)
    if width <= 0:
        raise ValueError("width must be positive")
    if indices.dtype != torch.long:
        raise TypeError("indices must use torch.long")
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= width):
        raise ValueError("selected index is out of range")
    counts = torch.bincount(indices.reshape(-1), minlength=width)
    if not normalize:
        return counts
    if int(indices.shape[0]) == 0:
        return counts.double()
    return counts.double() / int(indices.shape[0])


def selection_stability(
    indices: Tensor, width: int
) -> dict[str, float | int | str | None]:
    """Pairwise overlap plus chance-adjusted intersection on a fixed subset."""

    _require_matrix("indices", indices)
    if indices.dtype != torch.long:
        raise TypeError("indices must use torch.long")
    num_tokens, selected_k = map(int, indices.shape)
    if width <= 0 or selected_k > width:
        raise ValueError("invalid selection geometry")
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= width):
        raise ValueError("selected index is out of range")
    if num_tokens < 2:
        return {
            "sampled_tokens": num_tokens,
            "pair_count": 0,
            "mean_pairwise_intersection": None,
            "mean_pairwise_jaccard": None,
            "chance_expected_intersection": float(selected_k * selected_k / width),
            "chance_adjusted_intersection": None,
            "chance_adjusted_endpoint_convention": "insufficient_token_pairs",
        }
    mask = torch.zeros((num_tokens, width), dtype=torch.float64, device=indices.device)
    if selected_k:
        mask.scatter_(1, indices, 1.0)
    intersections = mask @ mask.transpose(0, 1)
    pair_mask = torch.triu(
        torch.ones_like(intersections, dtype=torch.bool), diagonal=1
    )
    pair_intersections = intersections[pair_mask]
    mean_intersection = pair_intersections.mean()
    if selected_k == 0:
        jaccard = mean_intersection.new_ones(pair_intersections.shape)
    else:
        jaccard = pair_intersections / (2.0 * selected_k - pair_intersections)
    expected_intersection = float(selected_k * selected_k / width)
    if selected_k in {0, width}:
        adjusted = None
        endpoint_convention = "undefined_zero_denominator_reported_as_null"
    else:
        adjusted = float(
            (mean_intersection - expected_intersection)
            / (selected_k - expected_intersection)
        )
        endpoint_convention = "not_an_endpoint"
    return {
        "sampled_tokens": num_tokens,
        "pair_count": int(pair_intersections.numel()),
        "mean_pairwise_intersection": float(mean_intersection),
        "mean_pairwise_jaccard": float(jaccard.mean()),
        "chance_expected_intersection": expected_intersection,
        "chance_adjusted_intersection": adjusted,
        "chance_adjusted_endpoint_convention": endpoint_convention,
    }


def random_expectation_statistics(
    c_post_gate: Tensor,
    weight: Tensor,
) -> RandomExpectationStatistics:
    """Build random-subset sufficient statistics once for all cardinalities."""

    _require_matrix("c_post_gate", c_post_gate)
    _require_matrix("weight", weight)
    _require_same_device(c_post_gate, weight)
    num_tokens, width = map(int, c_post_gate.shape)
    if int(weight.shape[1]) != width:
        raise ValueError("weight and c_post_gate widths differ")
    c = c_post_gate.float()
    w = weight.float()
    teacher = c @ w.transpose(0, 1)
    teacher_per_token = teacher.double().square().sum(dim=1)
    diagonal = w.double().square().sum(dim=0)
    component_energy = (c.double().square() * diagonal).sum(dim=1)
    return RandomExpectationStatistics(
        teacher_energy_per_token=teacher_per_token,
        component_energy_per_token=component_energy,
        num_tokens=num_tokens,
        width=width,
    )


def analytic_random_expected_metrics(
    c_post_gate: Tensor,
    weight: Tensor,
    selected_k: int,
    *,
    a_post_sigmoid: Tensor | None = None,
    statistics: RandomExpectationStatistics | None = None,
    per_token_teacher_epsilon: float = 1.0e-24,
) -> dict[str, Any]:
    """Exact first-moment output metrics for a uniform fixed-k subset.

    Cosine is the ratio formed from expected aggregate cross and prediction
    energy; it is not the expectation of the nonlinear sample cosine.
    """

    _require_matrix("c_post_gate", c_post_gate)
    _require_matrix("weight", weight)
    _require_same_device(c_post_gate, weight)
    num_tokens, width = map(int, c_post_gate.shape)
    if int(weight.shape[1]) != width:
        raise ValueError("weight and c_post_gate widths differ")
    if not 0 <= int(selected_k) <= width:
        raise ValueError(f"selected_k must lie in [0, {width}]")
    if a_post_sigmoid is not None and a_post_sigmoid.shape != c_post_gate.shape:
        raise ValueError("gate and c_post_gate shapes differ")

    if statistics is None:
        statistics = random_expectation_statistics(c_post_gate, weight)
    if (
        statistics.num_tokens != num_tokens
        or statistics.width != width
        or tuple(statistics.teacher_energy_per_token.shape) != (num_tokens,)
        or tuple(statistics.component_energy_per_token.shape) != (num_tokens,)
        or statistics.teacher_energy_per_token.device != c_post_gate.device
        or statistics.component_energy_per_token.device != c_post_gate.device
    ):
        raise ValueError("random-expectation statistics are incompatible")
    teacher_per_token = statistics.teacher_energy_per_token
    component_energy = statistics.component_energy_per_token
    k = int(selected_k)
    q = width - k
    if width == 1:
        expected_residual = teacher_per_token if k == 0 else torch.zeros_like(teacher_per_token)
        expected_prediction = torch.zeros_like(teacher_per_token) if k == 0 else teacher_per_token
    else:
        denominator = float(width * (width - 1))
        expected_residual = (
            q * (q - 1) / denominator * teacher_per_token
            + k * q / denominator * component_energy
        )
        expected_prediction = (
            k * (k - 1) / denominator * teacher_per_token
            + k * q / denominator * component_energy
        )
    expected_cross = (k / width) * teacher_per_token
    total_teacher = teacher_per_token.sum()
    if not float(total_teacher) > 0.0:
        raise ValueError("aggregate teacher energy must be positive")
    total_residual = expected_residual.sum()
    total_prediction = expected_prediction.sum()
    total_cross = expected_cross.sum()
    relative_mse = total_residual / total_teacher
    prediction_teacher_energy = total_prediction / total_teacher
    normalized_cross = total_cross / total_teacher
    cosine_denominator = torch.sqrt(total_teacher * total_prediction)
    cosine = (
        total_cross / cosine_denominator
        if float(cosine_denominator) > 0.0
        else total_cross.new_zeros(())
    )
    valid = teacher_per_token > float(per_token_teacher_epsilon)
    per_token = expected_residual[valid] / teacher_per_token[valid]
    realized_ratio = k / width
    result: dict[str, Any] = {
        "relative_mse": float(relative_mse),
        "normalized_cross": float(normalized_cross),
        "prediction_teacher_energy": float(prediction_teacher_energy),
        "cosine_similarity": float(cosine),
        "cosine_definition": "ratio_of_expected_aggregate_cross_and_energy",
        "relative_mse_from_identity": float(
            1.0 + prediction_teacher_energy - 2.0 * normalized_cross
        ),
        "relative_mse_identity_absolute_error": float(
            torch.abs(
                relative_mse
                - (1.0 + prediction_teacher_energy - 2.0 * normalized_cross)
            )
        ),
        "per_token_relative_squared_error": _quantiles(per_token),
        "per_token_relative_error": _quantiles(per_token),
        "near_zero_teacher_tokens": int((~valid).sum()),
        "valid_per_token_count": int(valid.sum()),
        "teacher_energy": float(total_teacher),
        "prediction_energy": float(total_prediction),
        "cross_inner_product": float(total_cross),
        "residual_energy": float(total_residual),
        "retained_input_energy": realized_ratio,
        "retained_gate_energy": realized_ratio if a_post_sigmoid is not None else None,
        "analytic_expectation": True,
    }
    return result


def greedy_candidate_gains(
    c_post_gate: Tensor,
    weight: Tensor,
    residual: Tensor,
) -> Tensor:
    """Evaluate the exact single-coordinate residual-energy reductions."""

    _require_matrix("c_post_gate", c_post_gate)
    _require_matrix("weight", weight)
    _require_matrix("residual", residual)
    if int(weight.shape[1]) != int(c_post_gate.shape[1]):
        raise ValueError("weight and c_post_gate widths differ")
    if tuple(residual.shape) != (int(c_post_gate.shape[0]), int(weight.shape[0])):
        raise ValueError("residual shape is incompatible")
    _require_same_device(c_post_gate, weight, residual)
    c = c_post_gate.float()
    w = weight.float()
    correlations = residual.float() @ w
    column_energy = w.square().sum(dim=0)
    return 2.0 * c * correlations - c.square() * column_energy


def _checkpoint_from_path(
    c: Tensor,
    weight: Tensor,
    teacher: Tensor,
    ranking: Tensor,
    selected_k: int,
    recurrence_energy: Tensor,
) -> GreedyCheckpoint:
    indices = selected_indices(ranking, selected_k).clone()
    prediction = selected_output(c, weight, indices)
    direct_energy = (teacher.double() - prediction.double()).square().sum(dim=1)
    recurrence64 = recurrence_energy.double().clone()
    absolute = torch.abs(direct_energy - recurrence64)
    relative = absolute / direct_energy.abs().clamp_min(1.0e-30)
    return GreedyCheckpoint(
        selected_k=selected_k,
        selected_indices=indices,
        prediction=prediction,
        direct_residual_energy=direct_energy,
        recurrence_residual_energy=recurrence64,
        maximum_recurrence_absolute_error=float(absolute.max()) if absolute.numel() else 0.0,
        maximum_recurrence_relative_error=float(relative.max()) if relative.numel() else 0.0,
    )


def greedy_residual_reference(
    c_post_gate: Tensor,
    weight: Tensor,
    checkpoint_ks: Iterable[int],
) -> GreedyResidualResult:
    """Run one nested fixed-k target-aware greedy path.

    This is a privileged reference, not a globally optimal cardinality oracle.
    Negative gains are intentionally allowed: destructive cancellation can
    make the exact-k residual increase before all coordinates are restored.
    """

    _require_matrix("c_post_gate", c_post_gate)
    _require_matrix("weight", weight)
    _require_same_device(c_post_gate, weight)
    num_tokens, width = map(int, c_post_gate.shape)
    if int(weight.shape[1]) != width:
        raise ValueError("weight and c_post_gate widths differ")
    checkpoints_requested = tuple(sorted(set(map(int, checkpoint_ks))))
    if not checkpoints_requested:
        raise ValueError("at least one greedy checkpoint is required")
    if checkpoints_requested[0] < 0 or checkpoints_requested[-1] > width:
        raise ValueError(f"greedy checkpoints must lie in [0, {width}]")

    c = c_post_gate.float()
    w = weight.float()
    teacher = c @ w.transpose(0, 1)
    gram = w.transpose(0, 1) @ w
    gram_diagonal = torch.diagonal(gram)
    # Match the actual dense-target definition at the start of the path.
    # ``c @ (W.T @ W)`` is algebraically equal but can pick a different
    # near-tied candidate after FP32 reassociation.
    correlations = teacher @ w
    recurrence_energy = teacher.double().square().sum(dim=1)
    max_k = checkpoints_requested[-1]
    ranking = torch.empty((num_tokens, max_k), dtype=torch.long, device=c.device)
    selected_gains = torch.empty((num_tokens, max_k), dtype=torch.float32, device=c.device)
    chosen = torch.zeros((num_tokens, width), dtype=torch.bool, device=c.device)
    checkpoints: dict[int, GreedyCheckpoint] = {}
    if 0 in checkpoints_requested:
        checkpoints[0] = _checkpoint_from_path(
            c, w, teacher, ranking, 0, recurrence_energy
        )

    row_index = torch.arange(num_tokens, device=c.device)
    for step in range(max_k):
        gains = 2.0 * c * correlations - c.square() * gram_diagonal
        gains.masked_fill_(chosen, -torch.inf)
        selected_gain, selected_channel = torch.max(gains, dim=1)
        ranking[:, step] = selected_channel
        selected_gains[:, step] = selected_gain
        chosen[row_index, selected_channel] = True
        selected_value = c[row_index, selected_channel]
        correlations = correlations - selected_value[:, None] * gram[selected_channel, :]
        recurrence_energy = recurrence_energy - selected_gain.double()
        selected_count = step + 1
        if selected_count in checkpoints_requested:
            checkpoints[selected_count] = _checkpoint_from_path(
                c,
                w,
                teacher,
                ranking,
                selected_count,
                recurrence_energy,
            )

    if not bool(torch.isfinite(selected_gains).all()):
        raise RuntimeError("greedy path produced a non-finite selected gain")
    if max_k == 0:
        first_nonpositive: list[int | None] = [None] * num_tokens
    else:
        nonpositive_by_token = selected_gains <= 0
        has_nonpositive = nonpositive_by_token.any(dim=1)
        first_nonpositive_tensor = (
            nonpositive_by_token.to(torch.int64).argmax(dim=1) + 1
        )
        first_nonpositive_cpu = first_nonpositive_tensor.cpu().tolist()
        has_nonpositive_cpu = has_nonpositive.cpu().tolist()
        first_nonpositive = [
            int(step) if bool(present) else None
            for step, present in zip(
                first_nonpositive_cpu, has_nonpositive_cpu, strict=True
            )
        ]
    flat_gains = selected_gains.reshape(-1)
    negative = flat_gains < 0
    nonpositive = flat_gains <= 0
    recurrence_errors = [
        checkpoint.maximum_recurrence_absolute_error
        for checkpoint in checkpoints.values()
    ]
    diagnostics: dict[str, Any] = {
        "reference_kind": "privileged_target_aware_greedy_not_global_oracle",
        "fixed_k": True,
        "negative_gains_allowed": True,
        "first_nonpositive_step_by_token": first_nonpositive,
        "tokens_with_nonpositive_gain": sum(value is not None for value in first_nonpositive),
        "negative_gain_count": int(negative.sum()),
        "nonpositive_gain_count": int(nonpositive.sum()),
        "minimum_selected_gain": float(flat_gains.min()) if flat_gains.numel() else None,
        "maximum_residual_increase": (
            float((-flat_gains[negative]).max()) if bool(negative.any()) else 0.0
        ),
        "maximum_checkpoint_recurrence_absolute_error": max(recurrence_errors, default=0.0),
    }
    return GreedyResidualResult(
        ranking=ranking,
        selected_gains=selected_gains,
        teacher=teacher,
        checkpoints=checkpoints,
        diagnostics=diagnostics,
    )


__all__ = [
    "GreedyCheckpoint",
    "GreedyResidualResult",
    "RandomExpectationStatistics",
    "RuntimeActivations",
    "SCORE_METHODS",
    "analytic_random_expected_metrics",
    "exact_linear_quantiles",
    "greedy_candidate_gains",
    "greedy_residual_reference",
    "output_metrics",
    "random_rankings",
    "random_expectation_statistics",
    "reconstruct_bf16_runtime",
    "retained_energy_metrics",
    "retained_k",
    "scatter_selected_to_dense",
    "score_channels",
    "selected_indices",
    "selected_output",
    "selection_frequency",
    "selection_stability",
    "stable_descending_ranking",
]
