"""State, RMSNorm, and SiLU-gate foldability oracles for Qwen3.5 GDN.

For one value head, the deployable projected-state ordering is

    z = c @ V
    rho = sqrt(sum(z ** 2) / D + eps)
    c_hat_norm = (z @ V.T) * gamma / rho
    y = Wo [SiLU(g) * c_hat_norm]

where ``D`` is the original value-head width, not the latent rank.  With an
orthonormal ``V``, the latent expression for ``rho`` is exactly the RMS of the
decoded projected core.  The only learned dynamic approximation in this file
is therefore a centered headwise PCA expansion of the post-SiLU gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


SNAPSHOT_FORMAT = "basisserve.qwen35.gdn_foldability_snapshots.v1"
RESULT_FORMAT = "basisserve.qwen35.gdn_u_identity_foldability.v1"
BASIS_KINDS = (
    "pre_norm_core_pca",
    "post_gate_pca",
    "coordinate_postgate_energy",
)


def _field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source[name]
    return getattr(source, name)


@dataclass(frozen=True)
class Qwen35GDNFoldabilityGeometry:
    hidden_size: int
    num_value_heads: int
    value_head_dim: int
    rms_norm_eps: float
    layer_types: tuple[str, ...]

    @classmethod
    def from_config(cls, config: Any) -> "Qwen35GDNFoldabilityGeometry":
        text = (
            config.get("text_config", config)
            if isinstance(config, Mapping)
            else getattr(config, "text_config", config)
        )
        geometry = cls(
            hidden_size=int(_field(text, "hidden_size")),
            num_value_heads=int(_field(text, "linear_num_value_heads")),
            value_head_dim=int(_field(text, "linear_value_head_dim")),
            rms_norm_eps=float(_field(text, "rms_norm_eps")),
            layer_types=tuple(map(str, _field(text, "layer_types"))),
        )
        geometry.validate()
        return geometry

    def validate(self) -> None:
        if min(self.hidden_size, self.num_value_heads, self.value_head_dim) <= 0:
            raise ValueError("GDN foldability dimensions must be positive")
        if self.wire_width != self.hidden_size:
            raise ValueError("Qwen3.5 GDN value wire must equal hidden size")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("GDN RMSNorm epsilon must be positive")
        if not self.layer_types or any(
            item not in {"full_attention", "linear_attention"}
            for item in self.layer_types
        ):
            raise ValueError("Qwen3.5 layer types are missing or unsupported")

    @property
    def wire_width(self) -> int:
        return self.num_value_heads * self.value_head_dim

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, layer_type in enumerate(self.layer_types)
            if layer_type == "linear_attention"
        )


def parse_state_ranks(
    raw: str | Iterable[int],
    maximum: int,
) -> tuple[int, ...]:
    values = (
        [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
        if isinstance(raw, str)
        else list(map(int, raw))
    )
    result = tuple(sorted(set(values)))
    if not result or any(rank <= 0 or rank > maximum for rank in result):
        raise ValueError(f"state ranks must lie in [1, {maximum}]")
    return result


def parse_gate_ranks(
    raw: str | Iterable[int],
    maximum: int,
) -> tuple[int, ...]:
    values = (
        [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
        if isinstance(raw, str)
        else list(map(int, raw))
    )
    result = tuple(sorted(set(values)))
    if not result or any(rank < 0 or rank > maximum for rank in result):
        raise ValueError(f"gate ranks must lie in [0, {maximum}]")
    return result


def validate_snapshot_matrices(
    raw_core: Tensor,
    gate_preactivation: Tensor,
    geometry: Qwen35GDNFoldabilityGeometry,
) -> None:
    if raw_core.ndim != 2 or gate_preactivation.ndim != 2:
        raise ValueError("GDN raw-core and gate snapshots must be matrices")
    if raw_core.shape != gate_preactivation.shape:
        raise ValueError("GDN raw-core and gate snapshot shapes differ")
    if int(raw_core.shape[1]) != geometry.wire_width:
        raise ValueError("GDN snapshot width does not match geometry")
    if not bool(torch.isfinite(raw_core).all()) or not bool(
        torch.isfinite(gate_preactivation).all()
    ):
        raise FloatingPointError("GDN snapshots contain non-finite values")


def silu_gate(gate_preactivation: Tensor) -> Tensor:
    return F.silu(gate_preactivation.float())


def _heads(values: Tensor, geometry: Qwen35GDNFoldabilityGeometry) -> Tensor:
    if values.ndim != 2 or int(values.shape[1]) != geometry.wire_width:
        raise ValueError("GDN values do not match the value wire")
    return values.float().reshape(
        int(values.shape[0]),
        geometry.num_value_heads,
        geometry.value_head_dim,
    )


def rms_normalize_heads(
    core_heads: Tensor,
    norm_weight: Tensor,
    geometry: Qwen35GDNFoldabilityGeometry,
) -> tuple[Tensor, Tensor]:
    expected = (
        int(core_heads.shape[0]),
        geometry.num_value_heads,
        geometry.value_head_dim,
    )
    if core_heads.ndim != 3 or tuple(core_heads.shape) != expected:
        raise ValueError("GDN core heads do not match geometry")
    if tuple(norm_weight.shape) != (geometry.value_head_dim,):
        raise ValueError("GDN RMSNorm weight does not match one value head")
    work = core_heads.float()
    rms = torch.sqrt(
        work.square().mean(dim=-1) + geometry.rms_norm_eps
    )
    normalized = (
        work
        / rms.unsqueeze(-1)
        * norm_weight.float().reshape(1, 1, geometry.value_head_dim)
    )
    return normalized, rms


def latent_rms(
    latent: Tensor,
    geometry: Qwen35GDNFoldabilityGeometry,
) -> Tensor:
    if latent.ndim != 3 or int(latent.shape[1]) != geometry.num_value_heads:
        raise ValueError("GDN latent values do not match the head layout")
    # RMSNorm still averages over the decoded D-dimensional head, not rank R.
    return torch.sqrt(
        latent.float().square().sum(dim=-1) / geometry.value_head_dim
        + geometry.rms_norm_eps
    )


def _deterministic_column_signs(vectors: Tensor) -> Tensor:
    pivots = torch.argmax(vectors.abs(), dim=-2, keepdim=True)
    pivot_values = torch.gather(vectors, -2, pivots).squeeze(-2)
    signs = torch.where(
        pivot_values < 0,
        -torch.ones_like(pivot_values),
        torch.ones_like(pivot_values),
    )
    return vectors * signs.unsqueeze(-2)


def _nested_eigenbasis(covariance: Tensor) -> Tensor:
    if covariance.ndim != 3 or covariance.shape[-1] != covariance.shape[-2]:
        raise ValueError("headwise covariance must contain square matrices")
    symmetric = 0.5 * (covariance.float() + covariance.float().transpose(-1, -2))
    _, vectors = torch.linalg.eigh(symmetric.to(dtype=torch.float64))
    vectors = torch.flip(vectors, dims=(-1,))
    return _deterministic_column_signs(vectors).to(dtype=torch.float32).contiguous()


def headwise_statistics(
    raw_core: Tensor,
    gate_preactivation: Tensor,
    output_weight: Tensor,
    norm_weight: Tensor,
    geometry: Qwen35GDNFoldabilityGeometry,
) -> dict[str, Tensor]:
    """Return train-only state and post-SiLU gate statistics."""

    validate_snapshot_matrices(raw_core, gate_preactivation, geometry)
    if tuple(output_weight.shape) != (geometry.hidden_size, geometry.wire_width):
        raise ValueError("GDN out_proj weight does not match geometry")
    core = _heads(raw_core, geometry)
    gate = _heads(silu_gate(gate_preactivation), geometry)
    normalized, _ = rms_normalize_heads(core, norm_weight, geometry)
    post = normalized * gate
    rows = max(int(core.shape[0]), 1)
    pre_covariance = torch.einsum("nhd,nhe->hde", core, core) / rows
    post_covariance = torch.einsum("nhd,nhe->hde", post, post) / rows
    gate_mean = gate.mean(dim=0)
    gate_centered = gate - gate_mean.unsqueeze(0)
    gate_covariance = (
        torch.einsum("nhd,nhe->hde", gate_centered, gate_centered) / rows
    )
    column_energy = (
        output_weight.float().square().sum(dim=0).reshape(
            geometry.num_value_heads,
            geometry.value_head_dim,
        )
    )
    coordinate_score = post.square().sum(dim=0) * column_energy / rows
    return {
        "pre_norm_core_covariance": pre_covariance,
        "post_gate_covariance": post_covariance,
        "coordinate_postgate_output_energy": coordinate_score,
        "gate_mean": gate_mean.contiguous(),
        "gate_covariance": gate_covariance,
    }


def state_bases_from_statistics(
    statistics: Mapping[str, Tensor],
    *,
    basis_kind: str,
) -> Tensor:
    if basis_kind not in BASIS_KINDS:
        raise ValueError(f"unsupported GDN basis kind {basis_kind!r}")
    if basis_kind == "coordinate_postgate_energy":
        scores = statistics["coordinate_postgate_output_energy"].float()
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        identity = torch.eye(
            scores.shape[-1],
            device=scores.device,
            dtype=torch.float32,
        )
        return torch.stack(
            [identity.index_select(1, indices) for indices in order],
            dim=0,
        ).contiguous()
    key = (
        "pre_norm_core_covariance"
        if basis_kind == "pre_norm_core_pca"
        else "post_gate_covariance"
    )
    return _nested_eigenbasis(statistics[key])


def gate_modes_from_statistics(statistics: Mapping[str, Tensor]) -> Tensor:
    return _nested_eigenbasis(statistics["gate_covariance"])


def _basis_prefix(
    bases: Tensor,
    geometry: Qwen35GDNFoldabilityGeometry,
    rank: int,
) -> Tensor:
    expected = (
        geometry.num_value_heads,
        geometry.value_head_dim,
        geometry.value_head_dim,
    )
    if tuple(bases.shape) != expected or not 0 < rank <= geometry.value_head_dim:
        raise ValueError("GDN headwise bases or state rank are invalid")
    return bases[..., :rank]


def encode_headwise(
    values: Tensor,
    bases: Tensor,
    geometry: Qwen35GDNFoldabilityGeometry,
    rank: int,
) -> Tensor:
    by_head = _heads(values, geometry)
    prefix = _basis_prefix(bases, geometry, rank)
    return torch.einsum("nhd,hdr->nhr", by_head, prefix).contiguous()


def project_headwise(
    values: Tensor,
    bases: Tensor,
    geometry: Qwen35GDNFoldabilityGeometry,
    rank: int,
) -> Tensor:
    latent = encode_headwise(values, bases, geometry, rank)
    prefix = _basis_prefix(bases, geometry, rank)
    return torch.einsum("nhr,hdr->nhd", latent, prefix).contiguous()


def reconstruct_gate_modes(
    gate_values: Tensor,
    gate_mean: Tensor,
    gate_modes: Tensor,
    geometry: Qwen35GDNFoldabilityGeometry,
    rank: int,
) -> tuple[Tensor, Tensor]:
    if not 0 <= rank <= geometry.value_head_dim:
        raise ValueError("GDN gate-mode rank is invalid")
    gate = _heads(gate_values, geometry)
    expected_mean = (geometry.num_value_heads, geometry.value_head_dim)
    expected_modes = (
        geometry.num_value_heads,
        geometry.value_head_dim,
        geometry.value_head_dim,
    )
    if tuple(gate_mean.shape) != expected_mean or tuple(gate_modes.shape) != expected_modes:
        raise ValueError("GDN gate mean or modes do not match geometry")
    centered = gate - gate_mean.float().unsqueeze(0)
    if rank == 0:
        coefficients = centered.new_empty(
            centered.shape[0],
            geometry.num_value_heads,
            0,
        )
        reconstructed = gate_mean.float().unsqueeze(0).expand_as(gate)
    else:
        prefix = gate_modes[..., :rank].float()
        coefficients = torch.einsum("nhd,hdr->nhr", centered, prefix)
        reconstructed = gate_mean.float().unsqueeze(0) + torch.einsum(
            "nhr,hdr->nhd",
            coefficients,
            prefix,
        )
    return reconstructed.contiguous(), coefficients.contiguous()


def _linear_output(values: Tensor, output_weight: Tensor) -> Tensor:
    if values.ndim != 3:
        raise ValueError("GDN output values must be [rows, heads, head_dim]")
    return values.flatten(1) @ output_weight.float().transpose(0, 1)


def prepare_state_paths(
    raw_core: Tensor,
    gate_preactivation: Tensor,
    output_weight: Tensor,
    norm_weight: Tensor,
    state_bases: Tensor,
    geometry: Qwen35GDNFoldabilityGeometry,
    state_rank: int,
) -> dict[str, Tensor]:
    """Build teacher and exact-gate state/RMS attribution paths."""

    validate_snapshot_matrices(raw_core, gate_preactivation, geometry)
    core = _heads(raw_core, geometry)
    exact_gate = _heads(silu_gate(gate_preactivation), geometry)
    teacher_normalized, teacher_rms = rms_normalize_heads(
        core,
        norm_weight,
        geometry,
    )
    prefix = _basis_prefix(state_bases, geometry, state_rank)
    latent = torch.einsum("nhd,hdr->nhr", core, prefix)
    projected_core = torch.einsum("nhr,hdr->nhd", latent, prefix)
    projected_normalized, projected_rms = rms_normalize_heads(
        projected_core,
        norm_weight,
        geometry,
    )
    latent_rms_value = latent_rms(latent, geometry)
    gamma = norm_weight.float().reshape(1, 1, geometry.value_head_dim)
    frozen_rms_normalized = (
        projected_core / teacher_rms.unsqueeze(-1) * gamma
    )
    teacher_post = teacher_normalized * exact_gate
    postgate_projected = project_headwise(
        teacher_post.flatten(1),
        state_bases,
        geometry,
        state_rank,
    )
    postnorm_projected = project_headwise(
        teacher_normalized.flatten(1),
        state_bases,
        geometry,
        state_rank,
    )
    return {
        "core": core,
        "exact_gate": exact_gate,
        "teacher_normalized": teacher_normalized,
        "projected_normalized": projected_normalized,
        "frozen_rms_normalized": frozen_rms_normalized,
        "latent": latent,
        "rms_scaled_latent": (
            latent / latent_rms_value.unsqueeze(-1)
        ).contiguous(),
        "normalized_latent": torch.einsum(
            "nhd,hdr->nhr",
            teacher_normalized,
            prefix,
        ).contiguous(),
        "projected_core": projected_core,
        "teacher_rms": teacher_rms,
        "projected_rms": projected_rms,
        "latent_rms": latent_rms_value,
        "teacher": _linear_output(teacher_post, output_weight),
        "post_gate_projected": _linear_output(
            postgate_projected,
            output_weight,
        ),
        "post_norm_projected_then_gate": _linear_output(
            postnorm_projected * exact_gate,
            output_weight,
        ),
        "projected_frozen_rms_exact_gate": _linear_output(
            frozen_rms_normalized * exact_gate,
            output_weight,
        ),
        "projected_rms_exact_gate": _linear_output(
            projected_normalized * exact_gate,
            output_weight,
        ),
    }


def gate_mode_outputs(
    state_paths: Mapping[str, Tensor],
    reconstructed_gate: Tensor,
    output_weight: Tensor,
) -> dict[str, Tensor]:
    if reconstructed_gate.ndim != 3:
        raise ValueError("reconstructed GDN gate must be headwise")
    return {
        "gate_modes_teacher_core": _linear_output(
            state_paths["teacher_normalized"] * reconstructed_gate,
            output_weight,
        ),
        "combined": _linear_output(
            state_paths["projected_normalized"] * reconstructed_gate,
            output_weight,
        ),
    }


def dynamic_bridge_factors(
    output_weight: Tensor,
    norm_weight: Tensor,
    state_bases: Tensor,
    state_rank: int,
    gate_mean: Tensor,
    gate_modes: Tensor,
    gate_rank: int,
    geometry: Qwen35GDNFoldabilityGeometry,
) -> tuple[Tensor, Tensor]:
    """Materialize the exact static bridge and gate-mode bridge slices.

    This reference is intended for correctness tests and small-rank studies.
    Full production-sized dynamic slices should be streamed or factorized
    rather than materialized as ``[H, gate_rank, hidden, state_rank]``.
    """

    if tuple(output_weight.shape) != (geometry.hidden_size, geometry.wire_width):
        raise ValueError("GDN out_proj weight does not match bridge geometry")
    if tuple(norm_weight.shape) != (geometry.value_head_dim,):
        raise ValueError("GDN norm weight does not match bridge geometry")
    if tuple(gate_mean.shape) != (
        geometry.num_value_heads,
        geometry.value_head_dim,
    ):
        raise ValueError("GDN gate mean does not match bridge geometry")
    if not 0 <= gate_rank <= geometry.value_head_dim:
        raise ValueError("GDN gate rank is invalid for bridge construction")
    basis = _basis_prefix(state_bases, geometry, state_rank).float()
    modes = gate_modes[..., :gate_rank].float()
    weight_by_head = output_weight.float().reshape(
        geometry.hidden_size,
        geometry.num_value_heads,
        geometry.value_head_dim,
    ).permute(1, 0, 2)
    gamma = norm_weight.float().reshape(1, geometry.value_head_dim)
    static_bridge = torch.einsum(
        "hod,hd,hdr->hor",
        weight_by_head,
        gate_mean.float() * gamma,
        basis,
    ).contiguous()
    dynamic_bridge = torch.einsum(
        "hod,hdl,hd,hdr->hlor",
        weight_by_head,
        modes,
        gamma.expand(geometry.num_value_heads, -1),
        basis,
    ).contiguous()
    return static_bridge, dynamic_bridge


def apply_dynamic_bridge(
    rms_scaled_latent: Tensor,
    gate_coefficients: Tensor,
    static_bridge: Tensor,
    dynamic_bridge: Tensor,
) -> Tensor:
    """Apply materialized GDN gate-mode bridge factors in row convention."""

    if rms_scaled_latent.ndim != 3 or gate_coefficients.ndim != 3:
        raise ValueError("GDN bridge inputs must be headwise tensors")
    rows, heads, state_rank = map(int, rms_scaled_latent.shape)
    if tuple(gate_coefficients.shape[:2]) != (rows, heads):
        raise ValueError("GDN bridge coefficient rows or heads differ")
    gate_rank = int(gate_coefficients.shape[-1])
    if (
        static_bridge.ndim != 3
        or tuple(static_bridge.shape[:1]) != (heads,)
        or int(static_bridge.shape[-1]) != state_rank
        or tuple(dynamic_bridge.shape) != (
            heads,
            gate_rank,
            int(static_bridge.shape[1]),
            state_rank,
        )
    ):
        raise ValueError("GDN bridge factors do not match latent inputs")
    output = torch.einsum(
        "nhr,hor->no",
        rms_scaled_latent.float(),
        static_bridge.float(),
    )
    if gate_rank:
        output = output + torch.einsum(
            "nhl,nhr,hlor->no",
            gate_coefficients.float(),
            rms_scaled_latent.float(),
            dynamic_bridge.float(),
        )
    return output.contiguous()


def relative_tensor_mse(target: Tensor, prediction: Tensor) -> float:
    if target.shape != prediction.shape or target.numel() == 0:
        raise ValueError("relative tensor MSE inputs must be equal and non-empty")
    difference = target.float() - prediction.float()
    numerator = float(torch.sum(difference.square(), dtype=torch.float64))
    denominator = float(torch.sum(target.float().square(), dtype=torch.float64))
    return numerator / max(denominator, 1e-300)


__all__ = [
    "BASIS_KINDS",
    "Qwen35GDNFoldabilityGeometry",
    "RESULT_FORMAT",
    "SNAPSHOT_FORMAT",
    "apply_dynamic_bridge",
    "dynamic_bridge_factors",
    "encode_headwise",
    "gate_mode_outputs",
    "gate_modes_from_statistics",
    "headwise_statistics",
    "latent_rms",
    "parse_gate_ranks",
    "parse_state_ranks",
    "prepare_state_paths",
    "project_headwise",
    "reconstruct_gate_modes",
    "relative_tensor_mse",
    "rms_normalize_heads",
    "silu_gate",
    "state_bases_from_statistics",
    "validate_snapshot_matrices",
]
