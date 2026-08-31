"""U=I foldability oracles for Qwen3.5 gated full-attention outputs.

The deployable ordering first projects the ungated attention result and then
applies the channel gate.  The post-gate control applies the same projector
after the gate.  Their gap isolates whether a groupwise Value basis can move
across Qwen3.5's output gate; no dynamic-residual model is involved here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor


SNAPSHOT_FORMAT = "basisserve.qwen35.gated_wo_foldability_snapshots.v1"
RESULT_FORMAT = "basisserve.qwen35.gated_wo_u_identity_foldability.v1"
BASIS_KINDS = (
    "pre_gate_pca",
    "post_gate_pca",
    "coordinate_postgate_energy",
)


def _field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source[name]
    return getattr(source, name)


@dataclass(frozen=True)
class Qwen35FullAttentionGeometry:
    hidden_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    layer_types: tuple[str, ...]

    @classmethod
    def from_config(cls, config: Any) -> "Qwen35FullAttentionGeometry":
        text = (
            config.get("text_config", config)
            if isinstance(config, Mapping)
            else getattr(config, "text_config", config)
        )
        geometry = cls(
            hidden_size=int(_field(text, "hidden_size")),
            num_query_heads=int(_field(text, "num_attention_heads")),
            num_kv_heads=int(_field(text, "num_key_value_heads")),
            head_dim=int(_field(text, "head_dim")),
            layer_types=tuple(map(str, _field(text, "layer_types"))),
        )
        geometry.validate()
        return geometry

    def validate(self) -> None:
        if min(
            self.hidden_size,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
        ) <= 0:
            raise ValueError("full-attention geometry dimensions must be positive")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.wire_width != self.hidden_size:
            raise ValueError("Qwen3.5 full-attention output width must equal hidden size")
        if not self.layer_types or any(
            item not in {"full_attention", "linear_attention"}
            for item in self.layer_types
        ):
            raise ValueError("Qwen3.5 layer_types are missing or unsupported")

    @property
    def wire_width(self) -> int:
        return self.num_query_heads * self.head_dim

    @property
    def query_heads_per_kv_group(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, layer_type in enumerate(self.layer_types)
            if layer_type == "full_attention"
        )

    def head_to_kv_group(self, *, device: torch.device | str = "cpu") -> Tensor:
        return torch.arange(
            self.num_query_heads,
            dtype=torch.long,
            device=device,
        ) // self.query_heads_per_kv_group


def parse_ranks(raw: str | Iterable[int], maximum: int) -> tuple[int, ...]:
    values = (
        [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
        if isinstance(raw, str)
        else list(map(int, raw))
    )
    result = tuple(sorted(set(values)))
    if not result or any(rank <= 0 or rank > maximum for rank in result):
        raise ValueError(f"ranks must lie in [1, {maximum}]")
    return result


def validate_snapshot_matrices(
    h_pre_gate: Tensor,
    gate_logits: Tensor,
    geometry: Qwen35FullAttentionGeometry,
) -> None:
    if h_pre_gate.ndim != 2 or gate_logits.ndim != 2:
        raise ValueError("pre-gate and gate snapshots must be matrices")
    if h_pre_gate.shape != gate_logits.shape:
        raise ValueError("pre-gate and gate snapshot shapes differ")
    if int(h_pre_gate.shape[1]) != geometry.wire_width:
        raise ValueError("snapshot width does not match full-attention geometry")
    if not bool(torch.isfinite(h_pre_gate).all()) or not bool(
        torch.isfinite(gate_logits).all()
    ):
        raise FloatingPointError("full-attention snapshots contain non-finite values")


def post_sigmoid_gate(gate_logits: Tensor) -> Tensor:
    """Match the snapshot dtype's sigmoid before returning FP32 work values."""

    return torch.sigmoid(gate_logits).float()


def groupwise_statistics(
    h_pre_gate: Tensor,
    gate_logits: Tensor,
    output_weight: Tensor,
    geometry: Qwen35FullAttentionGeometry,
) -> dict[str, Tensor]:
    """Return train-only statistics for nested groupwise foldable bases."""

    validate_snapshot_matrices(h_pre_gate, gate_logits, geometry)
    if tuple(output_weight.shape) != (geometry.hidden_size, geometry.wire_width):
        raise ValueError("o_proj weight does not match full-attention geometry")
    rows = int(h_pre_gate.shape[0])
    heads = geometry.num_query_heads
    width = geometry.head_dim
    groups = geometry.num_kv_heads
    mapping = geometry.head_to_kv_group(device=h_pre_gate.device)
    pre = h_pre_gate.float().reshape(rows, heads, width)
    gate = post_sigmoid_gate(gate_logits).reshape(rows, heads, width)
    post = pre * gate
    pre_covariance = pre.new_zeros(groups, width, width)
    post_covariance = pre.new_zeros(groups, width, width)
    coordinate_score = pre.new_zeros(groups, width)
    column_energy = output_weight.float().square().sum(dim=0).reshape(heads, width)
    counts = pre.new_zeros(groups)
    for group in range(groups):
        head_indices = torch.nonzero(mapping == group, as_tuple=False).flatten()
        group_pre = pre.index_select(1, head_indices).reshape(-1, width)
        group_post = post.index_select(1, head_indices).reshape(-1, width)
        pre_covariance[group] = group_pre.transpose(0, 1) @ group_pre
        post_covariance[group] = group_post.transpose(0, 1) @ group_post
        selected_energy = post.square().sum(dim=0).index_select(0, head_indices)
        selected_columns = column_energy.index_select(0, head_indices)
        coordinate_score[group] = (selected_energy * selected_columns).sum(dim=0)
        counts[group] = float(group_pre.shape[0])
    scale = counts.clamp_min(1).reshape(groups, 1, 1)
    return {
        "pre_gate_covariance": pre_covariance / scale,
        "post_gate_covariance": post_covariance / scale,
        "coordinate_postgate_output_energy": coordinate_score / counts.clamp_min(1).unsqueeze(1),
        "gate_mean": gate.mean(dim=0).reshape(-1).float().contiguous(),
    }


def _deterministic_column_signs(vectors: Tensor) -> Tensor:
    pivots = torch.argmax(vectors.abs(), dim=-2, keepdim=True)
    pivot_values = torch.gather(vectors, -2, pivots).squeeze(-2)
    signs = torch.where(pivot_values < 0, -torch.ones_like(pivot_values), torch.ones_like(pivot_values))
    return vectors * signs.unsqueeze(-2)


def bases_from_statistics(
    statistics: Mapping[str, Tensor],
    *,
    basis_kind: str,
) -> Tensor:
    """Build a full nested orthonormal basis for every physical KV group."""

    if basis_kind not in BASIS_KINDS:
        raise ValueError(f"unsupported basis kind {basis_kind!r}")
    if basis_kind == "coordinate_postgate_energy":
        scores = statistics["coordinate_postgate_output_energy"]
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        identity = torch.eye(scores.shape[1], device=scores.device, dtype=torch.float32)
        return torch.stack(
            [identity.index_select(1, indices) for indices in order],
            dim=0,
        ).contiguous()
    key = (
        "pre_gate_covariance"
        if basis_kind == "pre_gate_pca"
        else "post_gate_covariance"
    )
    covariance = statistics[key].float()
    symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
    # These 256 x 256 matrices are cheap to diagonalize in FP64.  cuSOLVER's
    # FP32 eigenvectors can lose enough orthogonality on the highly conditioned
    # activation covariance to make the nominal full-rank projector measurably
    # non-identity (around 1e-4 orthogonality error on Qwen3.5).  Solve the small
    # rank-side problem in FP64, then store the basis in FP32 for activation work.
    _, vectors = torch.linalg.eigh(symmetric.to(dtype=torch.float64))
    vectors = torch.flip(vectors, dims=(-1,))
    return _deterministic_column_signs(vectors).to(dtype=torch.float32).contiguous()


def _basis_by_head(
    group_bases: Tensor,
    geometry: Qwen35FullAttentionGeometry,
    rank: int,
) -> Tensor:
    expected = (geometry.num_kv_heads, geometry.head_dim, geometry.head_dim)
    if tuple(group_bases.shape) != expected:
        raise ValueError(f"group bases must have shape {expected}")
    if not 0 < rank <= geometry.head_dim:
        raise ValueError("basis rank is outside the per-head dimension")
    mapping = geometry.head_to_kv_group(device=group_bases.device)
    return group_bases.index_select(0, mapping)[..., :rank]


def encode_groupwise(
    values: Tensor,
    group_bases: Tensor,
    geometry: Qwen35FullAttentionGeometry,
    rank: int,
) -> Tensor:
    if values.ndim != 2 or int(values.shape[1]) != geometry.wire_width:
        raise ValueError("values do not match the full-attention wire")
    rows = int(values.shape[0])
    by_head = values.float().reshape(rows, geometry.num_query_heads, geometry.head_dim)
    bases = _basis_by_head(group_bases, geometry, rank)
    latent = torch.einsum("nhd,hdr->nhr", by_head, bases)
    return latent.reshape(rows, geometry.num_query_heads * rank).contiguous()


def project_groupwise(
    values: Tensor,
    group_bases: Tensor,
    geometry: Qwen35FullAttentionGeometry,
    rank: int,
) -> Tensor:
    latent = encode_groupwise(values, group_bases, geometry, rank)
    rows = int(values.shape[0])
    by_head = latent.reshape(rows, geometry.num_query_heads, rank)
    bases = _basis_by_head(group_bases, geometry, rank)
    reconstructed = torch.einsum("nhr,hdr->nhd", by_head, bases)
    return reconstructed.reshape(rows, geometry.wire_width).contiguous()


def foldability_outputs(
    h_pre_gate: Tensor,
    gate_logits: Tensor,
    output_weight: Tensor,
    group_bases: Tensor,
    geometry: Qwen35FullAttentionGeometry,
    rank: int,
    gate_mean: Tensor,
) -> dict[str, Tensor]:
    """Evaluate dense, post-gate, foldable-dynamic, and mean-static paths."""

    validate_snapshot_matrices(h_pre_gate, gate_logits, geometry)
    if tuple(output_weight.shape) != (geometry.hidden_size, geometry.wire_width):
        raise ValueError("o_proj weight does not match full-attention geometry")
    if tuple(gate_mean.shape) != (geometry.wire_width,):
        raise ValueError("gate mean does not match the full-attention wire")
    pre = h_pre_gate.float()
    gate = post_sigmoid_gate(gate_logits)
    post = pre * gate
    projected_pre = project_groupwise(pre, group_bases, geometry, rank)
    projected_post = project_groupwise(post, group_bases, geometry, rank)
    weight_t = output_weight.float().transpose(0, 1)
    return {
        "teacher": post @ weight_t,
        "post_gate_projected": projected_post @ weight_t,
        "foldable_dynamic": (projected_pre * gate) @ weight_t,
        "mean_gate_static": (projected_pre * gate_mean.float()) @ weight_t,
    }


def output_metrics(target: Tensor, prediction: Tensor) -> dict[str, float]:
    if target.ndim != 2 or target.shape != prediction.shape or target.numel() == 0:
        raise ValueError("target and prediction must be equal non-empty matrices")
    work_target = target.float()
    work_prediction = prediction.float()
    difference = work_prediction - work_target
    target_row_energy = work_target.square().sum(dim=1, dtype=torch.float64)
    error_row_energy = difference.square().sum(dim=1, dtype=torch.float64)
    target_energy = float(target_row_energy.sum())
    prediction_energy = float(work_prediction.square().sum(dtype=torch.float64))
    error_energy = float(error_row_energy.sum())
    cross = float((work_target * work_prediction).sum(dtype=torch.float64))
    tiny = torch.finfo(torch.float64).tiny
    token_relative = error_row_energy / target_row_energy.clamp_min(tiny)
    quantiles = torch.quantile(
        token_relative,
        torch.tensor([0.5, 0.9, 0.95, 0.99], device=token_relative.device, dtype=token_relative.dtype),
    )
    return {
        "rows": int(target.shape[0]),
        "relative_mse": error_energy / max(target_energy, 1e-300),
        "normalized_cross": cross / max(target_energy, 1e-300),
        "prediction_to_teacher_energy": prediction_energy / max(target_energy, 1e-300),
        "cosine_similarity": cross / max(math.sqrt(target_energy * prediction_energy), 1e-300),
        "token_relative_mse_p50": float(quantiles[0]),
        "token_relative_mse_p90": float(quantiles[1]),
        "token_relative_mse_p95": float(quantiles[2]),
        "token_relative_mse_p99": float(quantiles[3]),
        "token_relative_mse_max": float(token_relative.max()),
    }


def fit_activation_static_ridge(
    train_latent: Tensor,
    train_target: Tensor,
    dev_latent: Tensor,
    dev_target: Tensor,
    *,
    relative_ridges: Sequence[float],
) -> tuple[Tensor, dict[str, float]]:
    """Fit the best shared static bridge by damped normal equations."""

    if (
        train_latent.ndim != 2
        or train_target.ndim != 2
        or dev_latent.ndim != 2
        or dev_target.ndim != 2
        or train_latent.shape[0] != train_target.shape[0]
        or dev_latent.shape[0] != dev_target.shape[0]
        or train_latent.shape[1] != dev_latent.shape[1]
        or train_target.shape[1] != dev_target.shape[1]
    ):
        raise ValueError("static ridge matrices have incompatible shapes")
    ridges = tuple(sorted(set(map(float, relative_ridges))))
    if not ridges or any(value <= 0 for value in ridges):
        raise ValueError("relative ridge grid must contain positive values")
    z_train = train_latent.float()
    y_train = train_target.float()
    z_dev = dev_latent.float()
    y_dev = dev_target.float()
    gram = z_train.transpose(0, 1) @ z_train
    cross = z_train.transpose(0, 1) @ y_train
    scale = float(torch.diagonal(gram).mean().clamp_min(torch.finfo(gram.dtype).tiny))
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    best_decoder: Tensor | None = None
    best_record: dict[str, float] | None = None
    for relative in ridges:
        absolute = relative * scale
        cholesky, info = torch.linalg.cholesky_ex(
            0.5 * (gram + gram.transpose(0, 1)) + absolute * identity,
            check_errors=False,
        )
        if int(info.max()) != 0:
            continue
        decoder = torch.cholesky_solve(cross, cholesky)
        dev_relative_mse = output_metrics(y_dev, z_dev @ decoder)["relative_mse"]
        record = {
            "relative_ridge": relative,
            "absolute_ridge": absolute,
            "dev_relative_mse": dev_relative_mse,
        }
        if best_record is None or dev_relative_mse < best_record["dev_relative_mse"]:
            best_decoder = decoder
            best_record = record
    if best_decoder is None or best_record is None:
        raise torch.linalg.LinAlgError("every activation-static ridge solve failed")
    return best_decoder, best_record


__all__ = [
    "BASIS_KINDS",
    "Qwen35FullAttentionGeometry",
    "RESULT_FORMAT",
    "SNAPSHOT_FORMAT",
    "bases_from_statistics",
    "encode_groupwise",
    "fit_activation_static_ridge",
    "foldability_outputs",
    "groupwise_statistics",
    "output_metrics",
    "parse_ranks",
    "post_sigmoid_gate",
    "project_groupwise",
    "validate_snapshot_matrices",
]
