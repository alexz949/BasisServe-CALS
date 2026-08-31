"""GQA-tied weight-only and activation-whitened V/O factorization utilities.

The factorization keeps Q/K unchanged.  Each KV head gets one shared value
basis, while every query head in that KV group gets its own output decoder.
All tensor shapes in this module follow ``torch.nn.Linear`` weight convention:
``[out_features, in_features]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GQAVOLayout:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rank: int

    def __post_init__(self) -> None:
        values = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rank": self.rank,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads: "
                f"{self.num_attention_heads} vs {self.num_key_value_heads}"
            )
        if self.rank > self.head_dim:
            raise ValueError(f"rank must not exceed head_dim: {self.rank} vs {self.head_dim}")

    @property
    def query_heads_per_kv_group(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def compressed_query_width(self) -> int:
        return self.num_attention_heads * self.rank

    @property
    def compressed_kv_width(self) -> int:
        return self.num_key_value_heads * self.rank

    def query_head_indices(self, group_index: int) -> tuple[int, ...]:
        if not 0 <= group_index < self.num_key_value_heads:
            raise IndexError(f"KV group index out of range: {group_index}")
        start = group_index * self.query_heads_per_kv_group
        return tuple(range(start, start + self.query_heads_per_kv_group))


@dataclass(frozen=True)
class GQAVOGroupFactors:
    group_index: int
    query_head_indices: tuple[int, ...]
    A_pt: torch.Tensor
    Wv_compressed_pt: torch.Tensor
    D_pt_list: tuple[torch.Tensor, ...]
    v_bias_compressed_pt: torch.Tensor | None
    relative_frobenius_error: float
    relative_whitened_error: float
    covariance_rows: int | tuple[int, ...] | None = None
    initial_relative_whitened_error: float | None = None
    optimization_steps: int = 0


@dataclass(frozen=True)
class GQAVOLayerFactors:
    layout: GQAVOLayout
    groups: tuple[GQAVOGroupFactors, ...]
    o_decoder_bias_pt: torch.Tensor | None
    joint_initial_relative_output_error: float | None = None
    joint_final_relative_output_error: float | None = None
    joint_optimization_steps: int = 0

    @property
    def v_proj_compressed_weight(self) -> torch.Tensor:
        return torch.cat([group.Wv_compressed_pt for group in self.groups], dim=0)

    @property
    def v_proj_compressed_bias(self) -> torch.Tensor | None:
        biases = [group.v_bias_compressed_pt for group in self.groups]
        if all(bias is None for bias in biases):
            return None
        if any(bias is None for bias in biases):
            raise RuntimeError("inconsistent compressed V bias state across KV groups")
        return torch.cat([bias for bias in biases if bias is not None], dim=0)

    @property
    def o_decoder_weight(self) -> torch.Tensor:
        decoders: list[torch.Tensor] = []
        for group in self.groups:
            decoders.extend(group.D_pt_list)
        return torch.cat(decoders, dim=1)


@dataclass(frozen=True)
class GlobalOWhitenedError:
    rank: int
    relative_whitened_error: float
    energy_retained: float


class GQAOInputCovariance:
    """Streaming pooled covariance for query heads in each KV group."""

    def __init__(self, layout: GQAVOLayout) -> None:
        self.layout = layout
        self._gram = torch.zeros(
            layout.num_key_value_heads,
            layout.head_dim,
            layout.head_dim,
            dtype=torch.float64,
        )
        self._rows = torch.zeros(layout.num_key_value_heads, dtype=torch.int64)

    @torch.no_grad()
    def update(self, o_proj_input: torch.Tensor) -> None:
        if o_proj_input.shape[-1] != self.layout.query_width:
            raise ValueError(
                "o_proj input width mismatch: "
                f"expected {self.layout.query_width}, got {o_proj_input.shape[-1]}"
            )
        activations = o_proj_input.detach().reshape(
            -1,
            self.layout.num_attention_heads,
            self.layout.head_dim,
        )
        heads_per_group = self.layout.query_heads_per_kv_group
        grouped = activations.reshape(
            -1,
            self.layout.num_key_value_heads,
            heads_per_group,
            self.layout.head_dim,
        )
        grouped = grouped.permute(1, 0, 2, 3).reshape(
            self.layout.num_key_value_heads,
            -1,
            self.layout.head_dim,
        )
        grouped = grouped.to(torch.float32)
        gram = torch.bmm(grouped.transpose(1, 2), grouped)
        self._gram.add_(gram.cpu().to(torch.float64))
        self._rows.add_(grouped.shape[1])

    @property
    def rows(self) -> torch.Tensor:
        return self._rows.clone()

    def covariances(self) -> torch.Tensor:
        if torch.any(self._rows <= 0):
            missing = torch.nonzero(self._rows <= 0, as_tuple=False).flatten().tolist()
            raise RuntimeError(f"no activation rows collected for KV groups {missing}")
        covariance = self._gram / self._rows.to(torch.float64).view(-1, 1, 1)
        return 0.5 * (covariance + covariance.transpose(-1, -2))


class GQAIdentityOInputCovariance:
    """Identity metric for ordinary weight-only KV-group SVD."""

    def __init__(self, layout: GQAVOLayout) -> None:
        self.layout = layout

    @property
    def rows(self) -> None:
        return None

    def covariances(self) -> torch.Tensor:
        eye = torch.eye(self.layout.head_dim, dtype=torch.float64)
        return eye.unsqueeze(0).repeat(self.layout.num_key_value_heads, 1, 1)


class GQAHeadwiseOInputCovariance:
    """Streaming covariance for every query-head attention output.

    The returned tensor is grouped as ``[Hkv, Hq/Hkv, D, D]`` so each query
    head keeps its own activation metric while the factorizer can still tie a
    single value basis across the query heads that share one KV head.
    """

    def __init__(self, layout: GQAVOLayout) -> None:
        self.layout = layout
        self._gram = torch.zeros(
            layout.num_attention_heads,
            layout.head_dim,
            layout.head_dim,
            dtype=torch.float64,
        )
        self._rows = torch.zeros(layout.num_attention_heads, dtype=torch.int64)

    @torch.no_grad()
    def update(self, o_proj_input: torch.Tensor) -> None:
        if o_proj_input.shape[-1] != self.layout.query_width:
            raise ValueError(
                "o_proj input width mismatch: "
                f"expected {self.layout.query_width}, got {o_proj_input.shape[-1]}"
            )
        activations = o_proj_input.detach().reshape(
            -1,
            self.layout.num_attention_heads,
            self.layout.head_dim,
        )
        by_head = activations.permute(1, 0, 2).to(torch.float32)
        gram = torch.bmm(by_head.transpose(1, 2), by_head)
        self._gram.add_(gram.cpu().to(torch.float64))
        self._rows.add_(by_head.shape[1])

    @property
    def rows(self) -> torch.Tensor:
        return self._rows.reshape(
            self.layout.num_key_value_heads,
            self.layout.query_heads_per_kv_group,
        ).clone()

    def covariances(self) -> torch.Tensor:
        if torch.any(self._rows <= 0):
            missing = torch.nonzero(self._rows <= 0, as_tuple=False).flatten().tolist()
            raise RuntimeError(f"no activation rows collected for query heads {missing}")
        covariance = self._gram / self._rows.to(torch.float64).view(-1, 1, 1)
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        return covariance.reshape(
            self.layout.num_key_value_heads,
            self.layout.query_heads_per_kv_group,
            self.layout.head_dim,
            self.layout.head_dim,
        )


class OInputActivationSamples:
    """Collect a bounded, evenly distributed sample of o_proj inputs."""

    def __init__(self, width: int, *, max_rows: int, rows_per_update: int) -> None:
        if width <= 0 or max_rows <= 0 or rows_per_update <= 0:
            raise ValueError("width, max_rows, and rows_per_update must be positive")
        self.width = int(width)
        self.max_rows = int(max_rows)
        self.rows_per_update = int(rows_per_update)
        self._samples: list[torch.Tensor] = []
        self._rows = 0

    @torch.no_grad()
    def update(self, o_proj_input: torch.Tensor) -> None:
        if o_proj_input.shape[-1] != self.width:
            raise ValueError(
                f"o_proj input width mismatch: expected {self.width}, got {o_proj_input.shape[-1]}"
            )
        remaining = self.max_rows - self._rows
        if remaining <= 0:
            return
        rows = o_proj_input.detach().reshape(-1, self.width)
        count = min(self.rows_per_update, remaining, rows.shape[0])
        if count <= 0:
            return
        if count == rows.shape[0]:
            selected = rows
        else:
            indices = torch.linspace(
                0,
                rows.shape[0] - 1,
                steps=count,
                device=rows.device,
            ).round().to(torch.long)
            selected = rows.index_select(0, indices)
        self._samples.append(selected.to(device="cpu", dtype=torch.bfloat16).clone())
        self._rows += count

    @property
    def rows(self) -> int:
        return self._rows

    def samples(self) -> torch.Tensor:
        if not self._samples:
            raise RuntimeError("no o_proj input samples collected")
        return torch.cat(self._samples, dim=0)

    def clear(self) -> None:
        self._samples.clear()
        self._rows = 0


class FullOInputCovariance:
    """Streaming full covariance of the concatenated attention output."""

    def __init__(self, width: int) -> None:
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        self.width = int(width)
        self._gram: torch.Tensor | None = None
        self._rows = 0

    @torch.no_grad()
    def update(self, o_proj_input: torch.Tensor) -> None:
        if o_proj_input.shape[-1] != self.width:
            raise ValueError(
                f"o_proj input width mismatch: expected {self.width}, got {o_proj_input.shape[-1]}"
            )
        activations = o_proj_input.detach().reshape(-1, self.width).to(torch.float32)
        if self._gram is None:
            self._gram = torch.zeros(
                self.width,
                self.width,
                dtype=torch.float32,
                device=activations.device,
            )
        elif self._gram.device != activations.device:
            raise RuntimeError(
                "o_proj activation device changed during covariance collection: "
                f"{self._gram.device} -> {activations.device}"
            )
        self._gram.addmm_(activations.transpose(0, 1), activations)
        self._rows += activations.shape[0]

    @property
    def rows(self) -> int:
        return self._rows

    def covariance(self) -> torch.Tensor:
        if self._gram is None or self._rows <= 0:
            raise RuntimeError("no activation rows collected")
        covariance = self._gram / self._rows
        return 0.5 * (covariance + covariance.transpose(0, 1))


def validate_projection_shapes(
    layout: GQAVOLayout,
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
) -> None:
    expected = {
        "q_proj.weight": (layout.query_width, layout.hidden_size),
        "k_proj.weight": (layout.kv_width, layout.hidden_size),
        "v_proj.weight": (layout.kv_width, layout.hidden_size),
        "o_proj.weight": (layout.hidden_size, layout.query_width),
    }
    actual = {
        "q_proj.weight": tuple(q_weight.shape),
        "k_proj.weight": tuple(k_weight.shape),
        "v_proj.weight": tuple(v_weight.shape),
        "o_proj.weight": tuple(o_weight.shape),
    }
    mismatches = [
        f"{name}: expected {expected[name]}, got {actual[name]}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise ValueError("invalid GQA projection shapes: " + "; ".join(mismatches))


def _cholesky_with_damping(
    covariance: torch.Tensor,
    *,
    damp: float,
) -> tuple[torch.Tensor, float]:
    if damp < 0:
        raise ValueError(f"damp must be non-negative, got {damp}")
    if covariance.dtype not in (torch.float32, torch.float64):
        covariance = covariance.to(torch.float32)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    diag_mean = covariance.diagonal().abs().mean().clamp_min(1e-12)
    eye = torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device)
    jitter = max(float(damp * diag_mean), float(torch.finfo(covariance.dtype).eps * diag_mean))
    for _ in range(10):
        factor, info = torch.linalg.cholesky_ex(covariance + jitter * eye)
        if int(info.item()) == 0:
            return factor, jitter
        jitter *= 10.0
    raise RuntimeError("activation covariance remained non-PSD after 10 damping attempts")


@torch.no_grad()
def factorize_gqa_vo_activation_whitened(
    *,
    layout: GQAVOLayout,
    o_weight: torch.Tensor,
    v_weight: torch.Tensor,
    covariances: torch.Tensor,
    v_bias: torch.Tensor | None = None,
    o_bias: torch.Tensor | None = None,
    covariance_rows: torch.Tensor | None = None,
    act_damp: float = 1e-4,
    orthonormalize_a: bool = False,
    output_dtype: torch.dtype | None = None,
    factor_device: str | torch.device | None = None,
    work_dtype: torch.dtype = torch.float32,
) -> GQAVOLayerFactors:
    """Factor one attention layer with pooled per-KV-group whitening.

    ``covariances[g]`` is the covariance pooled over all query heads tied to
    KV group ``g``.  This is the shared-covariance approximation described by
    the GQA V/O proposal; it is not a per-query-head covariance objective.
    """

    if tuple(o_weight.shape) != (layout.hidden_size, layout.query_width):
        raise ValueError(
            f"o_weight must have shape {(layout.hidden_size, layout.query_width)}, "
            f"got {tuple(o_weight.shape)}"
        )
    if tuple(v_weight.shape) != (layout.kv_width, layout.hidden_size):
        raise ValueError(
            f"v_weight must have shape {(layout.kv_width, layout.hidden_size)}, "
            f"got {tuple(v_weight.shape)}"
        )
    if tuple(covariances.shape) != (
        layout.num_key_value_heads,
        layout.head_dim,
        layout.head_dim,
    ):
        raise ValueError(
            "covariances must have shape "
            f"{(layout.num_key_value_heads, layout.head_dim, layout.head_dim)}, "
            f"got {tuple(covariances.shape)}"
        )
    if v_bias is not None and tuple(v_bias.shape) != (layout.kv_width,):
        raise ValueError(f"v_bias must have shape {(layout.kv_width,)}, got {tuple(v_bias.shape)}")
    if o_bias is not None and tuple(o_bias.shape) != (layout.hidden_size,):
        raise ValueError(
            f"o_bias must have shape {(layout.hidden_size,)}, got {tuple(o_bias.shape)}"
        )
    if covariance_rows is not None and tuple(covariance_rows.shape) != (
        layout.num_key_value_heads,
    ):
        raise ValueError(
            f"covariance_rows must have shape {(layout.num_key_value_heads,)}, "
            f"got {tuple(covariance_rows.shape)}"
        )

    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError(f"work_dtype must be float32 or float64, got {work_dtype}")
    target_dtype = output_dtype or o_weight.dtype
    work_device = torch.device(factor_device) if factor_device is not None else o_weight.device
    work_o = o_weight.detach().to(device=work_device, dtype=work_dtype)
    work_v = v_weight.detach().to(device=work_device, dtype=work_dtype)
    work_v_bias = (
        None if v_bias is None else v_bias.detach().to(device=work_device, dtype=work_dtype)
    )
    groups: list[GQAVOGroupFactors] = []

    for group_index in range(layout.num_key_value_heads):
        query_head_indices = layout.query_head_indices(group_index)
        weight_rows = []
        for head_index in query_head_indices:
            start = head_index * layout.head_dim
            weight_rows.append(work_o[:, start : start + layout.head_dim].transpose(0, 1))
        weight_cat_row = torch.cat(weight_rows, dim=1)

        cholesky, _ = _cholesky_with_damping(
            covariances[group_index].to(device=work_device, dtype=work_dtype),
            damp=act_damp,
        )
        whitening = cholesky.transpose(0, 1)
        whitened_weight = whitening @ weight_cat_row
        U, singular_values, Vh = torch.linalg.svd(whitened_weight, full_matrices=False)
        U_r = U[:, : layout.rank]
        D_cat_row = singular_values[: layout.rank, None] * Vh[: layout.rank, :]
        A_row = torch.linalg.solve_triangular(whitening, U_r, upper=True)

        if orthonormalize_a:
            A_row, triangular = torch.linalg.qr(A_row, mode="reduced")
            D_cat_row = triangular @ D_cat_row

        reconstruction = A_row @ D_cat_row
        relative_error = torch.linalg.vector_norm(weight_cat_row - reconstruction) / torch.linalg.vector_norm(
            weight_cat_row
        ).clamp_min(1e-30)
        relative_whitened_error = torch.linalg.vector_norm(
            whitening @ (weight_cat_row - reconstruction)
        ) / torch.linalg.vector_norm(whitened_weight).clamp_min(1e-30)

        A_pt = A_row.transpose(0, 1).contiguous()
        D_pt_list = []
        for local_head_index in range(layout.query_heads_per_kv_group):
            start = local_head_index * layout.hidden_size
            D_row = D_cat_row[:, start : start + layout.hidden_size]
            D_pt_list.append(D_row.transpose(0, 1).contiguous().to(device="cpu", dtype=target_dtype))

        v_start = group_index * layout.head_dim
        Wv_group = work_v[v_start : v_start + layout.head_dim, :]
        Wv_compressed_pt = A_pt @ Wv_group
        v_bias_compressed_pt = None
        if work_v_bias is not None:
            v_bias_compressed_pt = A_pt @ work_v_bias[v_start : v_start + layout.head_dim]

        row_count = None
        if covariance_rows is not None:
            row_count = int(covariance_rows[group_index].item())
        groups.append(
            GQAVOGroupFactors(
                group_index=group_index,
                query_head_indices=query_head_indices,
                A_pt=A_pt.to(device="cpu", dtype=target_dtype),
                Wv_compressed_pt=Wv_compressed_pt.to(device="cpu", dtype=target_dtype),
                D_pt_list=tuple(D_pt_list),
                v_bias_compressed_pt=(
                    None
                    if v_bias_compressed_pt is None
                    else v_bias_compressed_pt.to(device="cpu", dtype=target_dtype)
                ),
                relative_frobenius_error=float(relative_error),
                relative_whitened_error=float(relative_whitened_error),
                covariance_rows=row_count,
            )
        )

    return GQAVOLayerFactors(
        layout=layout,
        groups=tuple(groups),
        o_decoder_bias_pt=(None if o_bias is None else o_bias.detach().cpu().to(target_dtype)),
    )


def _headwise_optimal_decoders(
    A_row: torch.Tensor,
    covariances: tuple[torch.Tensor, ...],
    weight_rows: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """Solve every decoder and return residual and total weighted energies."""

    decoders = []
    residual_energy = A_row.new_zeros(())
    total_energy = A_row.new_zeros(())
    for covariance, weight_row in zip(covariances, weight_rows, strict=True):
        covariance_weight = covariance @ weight_row
        gram = A_row.transpose(0, 1) @ covariance @ A_row
        gram = 0.5 * (gram + gram.transpose(0, 1))
        rhs = A_row.transpose(0, 1) @ covariance_weight
        decoder = torch.linalg.solve(gram, rhs)
        energy = (weight_row * covariance_weight).sum()
        captured = (rhs * decoder).sum()
        residual_energy = residual_energy + (energy - captured).clamp_min(0.0)
        total_energy = total_energy + energy
        decoders.append(decoder)
    return tuple(decoders), residual_energy, total_energy


def _headwise_basis_gradient(
    A_row: torch.Tensor,
    covariances: tuple[torch.Tensor, ...],
    weight_rows: tuple[torch.Tensor, ...],
    decoders: tuple[torch.Tensor, ...],
    total_energy: torch.Tensor,
) -> torch.Tensor:
    gradient = torch.zeros_like(A_row)
    for covariance, weight_row, decoder in zip(
        covariances,
        weight_rows,
        decoders,
        strict=True,
    ):
        covariance_weight = covariance @ weight_row
        gradient.add_(
            2.0
            * (
                covariance @ A_row @ (decoder @ decoder.transpose(0, 1))
                - covariance_weight @ decoder.transpose(0, 1)
            )
        )
    gradient.div_(total_energy.clamp_min(torch.finfo(total_energy.dtype).tiny))
    symmetric = 0.5 * (
        A_row.transpose(0, 1) @ gradient + gradient.transpose(0, 1) @ A_row
    )
    return gradient - A_row @ symmetric


def _headwise_residual_energy(
    A_row: torch.Tensor,
    covariances: tuple[torch.Tensor, ...],
    weight_rows: tuple[torch.Tensor, ...],
    decoders: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_energy = A_row.new_zeros(())
    total_energy = A_row.new_zeros(())
    for covariance, weight_row, decoder in zip(
        covariances,
        weight_rows,
        decoders,
        strict=True,
    ):
        residual = weight_row - A_row @ decoder
        residual_energy = residual_energy + (residual * (covariance @ residual)).sum()
        total_energy = total_energy + (weight_row * (covariance @ weight_row)).sum()
    return residual_energy.clamp_min(0.0), total_energy


@torch.no_grad()
def factorize_gqa_vo_headwise_activation_whitened(
    *,
    layout: GQAVOLayout,
    o_weight: torch.Tensor,
    v_weight: torch.Tensor,
    covariances: torch.Tensor,
    v_bias: torch.Tensor | None = None,
    o_bias: torch.Tensor | None = None,
    covariance_rows: torch.Tensor | None = None,
    act_damp: float = 1e-4,
    optimization_steps: int = 20,
    step_size: float = 0.25,
    tolerance: float = 1e-7,
    output_dtype: torch.dtype | None = None,
    factor_device: str | torch.device | None = None,
    work_dtype: torch.dtype = torch.float32,
) -> GQAVOLayerFactors:
    """Factor one layer with a head-specific metric and a KV-group-tied basis.

    A pooled-covariance SVD initializes each shared basis.  The solver then
    alternates closed-form head decoders with monotonic, QR-retracted updates
    of the common basis.  The output tensors use the same folded V/O runtime
    format as :func:`factorize_gqa_vo_activation_whitened`.
    """

    expected_covariance_shape = (
        layout.num_key_value_heads,
        layout.query_heads_per_kv_group,
        layout.head_dim,
        layout.head_dim,
    )
    if tuple(o_weight.shape) != (layout.hidden_size, layout.query_width):
        raise ValueError(
            f"o_weight must have shape {(layout.hidden_size, layout.query_width)}, "
            f"got {tuple(o_weight.shape)}"
        )
    if tuple(v_weight.shape) != (layout.kv_width, layout.hidden_size):
        raise ValueError(
            f"v_weight must have shape {(layout.kv_width, layout.hidden_size)}, "
            f"got {tuple(v_weight.shape)}"
        )
    if tuple(covariances.shape) != expected_covariance_shape:
        raise ValueError(
            f"covariances must have shape {expected_covariance_shape}, "
            f"got {tuple(covariances.shape)}"
        )
    if v_bias is not None and tuple(v_bias.shape) != (layout.kv_width,):
        raise ValueError(f"v_bias must have shape {(layout.kv_width,)}, got {tuple(v_bias.shape)}")
    if o_bias is not None and tuple(o_bias.shape) != (layout.hidden_size,):
        raise ValueError(
            f"o_bias must have shape {(layout.hidden_size,)}, got {tuple(o_bias.shape)}"
        )
    if covariance_rows is not None and tuple(covariance_rows.shape) != (
        layout.num_key_value_heads,
        layout.query_heads_per_kv_group,
    ):
        raise ValueError(
            "covariance_rows must have shape "
            f"{(layout.num_key_value_heads, layout.query_heads_per_kv_group)}, "
            f"got {tuple(covariance_rows.shape)}"
        )
    if optimization_steps < 0:
        raise ValueError(f"optimization_steps must be non-negative, got {optimization_steps}")
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError(f"work_dtype must be float32 or float64, got {work_dtype}")

    target_dtype = output_dtype or o_weight.dtype
    work_device = torch.device(factor_device) if factor_device is not None else o_weight.device
    work_o = o_weight.detach().to(device=work_device, dtype=work_dtype)
    work_v = v_weight.detach().to(device=work_device, dtype=work_dtype)
    work_v_bias = (
        None if v_bias is None else v_bias.detach().to(device=work_device, dtype=work_dtype)
    )
    groups: list[GQAVOGroupFactors] = []

    for group_index in range(layout.num_key_value_heads):
        query_head_indices = layout.query_head_indices(group_index)
        weight_rows = tuple(
            work_o[
                :,
                head_index * layout.head_dim : (head_index + 1) * layout.head_dim,
            ].transpose(0, 1)
            for head_index in query_head_indices
        )

        stable_covariances = []
        for local_head_index in range(layout.query_heads_per_kv_group):
            cholesky, _ = _cholesky_with_damping(
                covariances[group_index, local_head_index].to(
                    device=work_device,
                    dtype=work_dtype,
                ),
                damp=act_damp,
            )
            stable_covariances.append(cholesky @ cholesky.transpose(0, 1))
        stable_covariance_tuple = tuple(stable_covariances)

        pooled_covariance = torch.stack(stable_covariances).mean(dim=0)
        pooled_cholesky, _ = _cholesky_with_damping(pooled_covariance, damp=0.0)
        whitening = pooled_cholesky.transpose(0, 1)
        concatenated_weight = torch.cat(weight_rows, dim=1)
        U, _, _ = torch.linalg.svd(
            whitening @ concatenated_weight,
            full_matrices=False,
        )
        A_row = torch.linalg.solve_triangular(
            whitening,
            U[:, : layout.rank],
            upper=True,
        )
        A_row = torch.linalg.qr(A_row, mode="reduced").Q

        decoders, residual_energy, total_energy = _headwise_optimal_decoders(
            A_row,
            stable_covariance_tuple,
            weight_rows,
        )
        exact_initial_residual, exact_initial_total = _headwise_residual_energy(
            A_row,
            stable_covariance_tuple,
            weight_rows,
            decoders,
        )
        initial_relative_error = torch.sqrt(
            exact_initial_residual
            / exact_initial_total.clamp_min(torch.finfo(work_dtype).tiny)
        )
        accepted_steps = 0
        for _ in range(optimization_steps):
            gradient = _headwise_basis_gradient(
                A_row,
                stable_covariance_tuple,
                weight_rows,
                decoders,
                total_energy,
            )
            gradient_norm = torch.linalg.vector_norm(gradient)
            if not torch.isfinite(gradient_norm) or float(gradient_norm) <= tolerance:
                break
            direction = gradient / gradient_norm
            trial_step = step_size
            accepted = False
            for _ in range(10):
                candidate = torch.linalg.qr(A_row - trial_step * direction, mode="reduced").Q
                candidate_decoders, candidate_residual, candidate_total = (
                    _headwise_optimal_decoders(
                        candidate,
                        stable_covariance_tuple,
                        weight_rows,
                    )
                )
                if float(candidate_residual) < float(residual_energy):
                    relative_improvement = float(
                        (residual_energy - candidate_residual)
                        / residual_energy.clamp_min(torch.finfo(work_dtype).tiny)
                    )
                    A_row = candidate
                    decoders = candidate_decoders
                    residual_energy = candidate_residual
                    total_energy = candidate_total
                    accepted_steps += 1
                    accepted = True
                    break
                trial_step *= 0.5
            if not accepted or relative_improvement <= tolerance:
                break

        exact_residual_energy, exact_total_energy = _headwise_residual_energy(
            A_row,
            stable_covariance_tuple,
            weight_rows,
            decoders,
        )
        relative_whitened_error = torch.sqrt(
            exact_residual_energy
            / exact_total_energy.clamp_min(torch.finfo(work_dtype).tiny)
        )
        residuals = [
            weight_row - A_row @ decoder
            for weight_row, decoder in zip(weight_rows, decoders, strict=True)
        ]
        frobenius_numerator = torch.stack(
            [torch.linalg.vector_norm(residual).square() for residual in residuals]
        ).sum()
        frobenius_denominator = torch.stack(
            [torch.linalg.vector_norm(weight_row).square() for weight_row in weight_rows]
        ).sum().clamp_min(torch.finfo(work_dtype).tiny)
        relative_frobenius_error = torch.sqrt(
            frobenius_numerator / frobenius_denominator
        )

        A_pt = A_row.transpose(0, 1).contiguous()
        v_start = group_index * layout.head_dim
        Wv_group = work_v[v_start : v_start + layout.head_dim, :]
        Wv_compressed_pt = A_pt @ Wv_group
        v_bias_compressed_pt = None
        if work_v_bias is not None:
            v_bias_compressed_pt = A_pt @ work_v_bias[v_start : v_start + layout.head_dim]
        rows = None
        if covariance_rows is not None:
            rows = tuple(int(value) for value in covariance_rows[group_index].tolist())

        groups.append(
            GQAVOGroupFactors(
                group_index=group_index,
                query_head_indices=query_head_indices,
                A_pt=A_pt.to(device="cpu", dtype=target_dtype),
                Wv_compressed_pt=Wv_compressed_pt.to(device="cpu", dtype=target_dtype),
                D_pt_list=tuple(
                    decoder.transpose(0, 1).contiguous().to(device="cpu", dtype=target_dtype)
                    for decoder in decoders
                ),
                v_bias_compressed_pt=(
                    None
                    if v_bias_compressed_pt is None
                    else v_bias_compressed_pt.to(device="cpu", dtype=target_dtype)
                ),
                relative_frobenius_error=float(relative_frobenius_error),
                relative_whitened_error=float(relative_whitened_error),
                covariance_rows=rows,
                initial_relative_whitened_error=float(initial_relative_error),
                optimization_steps=accepted_steps,
            )
        )

    return GQAVOLayerFactors(
        layout=layout,
        groups=tuple(groups),
        o_decoder_bias_pt=(None if o_bias is None else o_bias.detach().cpu().to(target_dtype)),
    )


def refine_gqa_vo_global_joint_linear(
    *,
    initial_factors: GQAVOLayerFactors,
    o_weight: torch.Tensor,
    v_weight: torch.Tensor,
    activation_samples: torch.Tensor,
    v_bias: torch.Tensor | None = None,
    steps: int = 100,
    batch_size: int = 128,
    basis_learning_rate: float = 1e-2,
    decoder_learning_rate: float = 1e-3,
    eval_interval: int = 10,
    output_dtype: torch.dtype | None = None,
    factor_device: str | torch.device | None = None,
    work_dtype: torch.dtype = torch.float32,
) -> GQAVOLayerFactors:
    """Jointly refine all group bases and the global decoder on O outputs.

    The structured projection remains block diagonal and repeats one basis for
    every query head tied to a KV head.  Consequently the refined bases remain
    foldable into V and preserve the original compressed-cache geometry.
    """

    layout = initial_factors.layout
    if steps < 0 or batch_size <= 0 or eval_interval <= 0:
        raise ValueError("steps must be non-negative; batch_size and eval_interval must be positive")
    if basis_learning_rate <= 0 or decoder_learning_rate <= 0:
        raise ValueError("joint learning rates must be positive")
    if tuple(o_weight.shape) != (layout.hidden_size, layout.query_width):
        raise ValueError(
            f"o_weight must have shape {(layout.hidden_size, layout.query_width)}, "
            f"got {tuple(o_weight.shape)}"
        )
    if tuple(v_weight.shape) != (layout.kv_width, layout.hidden_size):
        raise ValueError(
            f"v_weight must have shape {(layout.kv_width, layout.hidden_size)}, "
            f"got {tuple(v_weight.shape)}"
        )
    if activation_samples.ndim != 2 or activation_samples.shape[1] != layout.query_width:
        raise ValueError(
            f"activation_samples must have shape [rows, {layout.query_width}], "
            f"got {tuple(activation_samples.shape)}"
        )
    if activation_samples.shape[0] <= 0:
        raise ValueError("activation_samples must contain at least one row")
    if v_bias is not None and tuple(v_bias.shape) != (layout.kv_width,):
        raise ValueError(f"v_bias must have shape {(layout.kv_width,)}, got {tuple(v_bias.shape)}")
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError(f"work_dtype must be float32 or float64, got {work_dtype}")

    target_dtype = output_dtype or o_weight.dtype
    work_device = torch.device(factor_device) if factor_device is not None else o_weight.device

    # Conversion runs under inference_mode; clone inside a disabled scope so
    # the optimizer receives normal tensors that can track gradients.
    with torch.inference_mode(False):
        samples = activation_samples.detach().to(
            device=work_device,
            dtype=work_dtype,
        ).clone()
        work_o = o_weight.detach().to(device=work_device, dtype=work_dtype).clone()
        work_v = v_weight.detach().to(device=work_device, dtype=work_dtype).clone()
        work_v_bias = (
            None
            if v_bias is None
            else v_bias.detach().to(device=work_device, dtype=work_dtype).clone()
        )
        basis_seed = torch.stack(
            [group.A_pt.transpose(0, 1) for group in initial_factors.groups]
        ).to(device=work_device, dtype=work_dtype).clone()
        decoder_seed = torch.stack(
            [
                torch.stack(
                    [decoder.transpose(0, 1) for decoder in group.D_pt_list]
                )
                for group in initial_factors.groups
            ]
        ).to(device=work_device, dtype=work_dtype).clone()

        # Normalize each seed basis while exactly compensating its decoder.
        for group_index in range(layout.num_key_value_heads):
            q_basis, transform = torch.linalg.qr(basis_seed[group_index], mode="reduced")
            basis_seed[group_index] = q_basis
            decoder_seed[group_index] = torch.einsum(
                "rs,qso->qro",
                transform,
                decoder_seed[group_index],
            )

        basis = torch.nn.Parameter(basis_seed)
        decoder = torch.nn.Parameter(decoder_seed)
        optimizer = torch.optim.Adam(
            [
                {"params": [basis], "lr": basis_learning_rate},
                {"params": [decoder], "lr": decoder_learning_rate},
            ]
        )

        def predict(rows: torch.Tensor) -> torch.Tensor:
            grouped = rows.reshape(
                -1,
                layout.num_key_value_heads,
                layout.query_heads_per_kv_group,
                layout.head_dim,
            )
            latent = torch.einsum("ngqd,gdr->ngqr", grouped, basis)
            return torch.einsum("ngqr,gqro->no", latent, decoder)

        @torch.no_grad()
        def relative_output_error() -> float:
            numerator = torch.zeros((), device=work_device, dtype=work_dtype)
            denominator = torch.zeros((), device=work_device, dtype=work_dtype)
            evaluation_batch = max(batch_size, 256)
            for start in range(0, samples.shape[0], evaluation_batch):
                rows = samples[start : start + evaluation_batch]
                target = rows @ work_o.transpose(0, 1)
                residual = target - predict(rows)
                numerator.add_(residual.square().sum())
                denominator.add_(target.square().sum())
            return float(
                torch.sqrt(
                    numerator / denominator.clamp_min(torch.finfo(work_dtype).tiny)
                )
            )

        initial_output_error = relative_output_error()
        best_output_error = initial_output_error
        best_basis = basis.detach().clone()
        best_decoder = decoder.detach().clone()
        completed_steps = 0

        for step in range(steps):
            start = (step * batch_size) % samples.shape[0]
            end = start + min(batch_size, samples.shape[0])
            if end <= samples.shape[0]:
                rows = samples[start:end]
            else:
                rows = torch.cat([samples[start:], samples[: end - samples.shape[0]]], dim=0)
            target = rows @ work_o.transpose(0, 1)
            prediction = predict(rows)
            loss = (prediction - target).square().sum() / target.square().sum().clamp_min(
                torch.finfo(work_dtype).tiny
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([basis, decoder], max_norm=1.0)
            optimizer.step()

            # QR retraction keeps every A_g conditioned.  Multiplying each
            # decoder by R preserves the represented linear map exactly.
            with torch.no_grad():
                for group_index in range(layout.num_key_value_heads):
                    q_basis, transform = torch.linalg.qr(basis[group_index], mode="reduced")
                    basis[group_index].copy_(q_basis)
                    decoder[group_index].copy_(
                        torch.einsum(
                            "rs,qso->qro",
                            transform,
                            decoder[group_index],
                        )
                    )
            completed_steps = step + 1

            if completed_steps % eval_interval == 0 or completed_steps == steps:
                candidate_error = relative_output_error()
                if candidate_error < best_output_error:
                    best_output_error = candidate_error
                    best_basis = basis.detach().clone()
                    best_decoder = decoder.detach().clone()

        basis_final = best_basis
        decoder_final = best_decoder
        grouped_samples = samples.reshape(
            -1,
            layout.num_key_value_heads,
            layout.query_heads_per_kv_group,
            layout.head_dim,
        )
        refined_groups = []
        for group_index, initial_group in enumerate(initial_factors.groups):
            A_row = basis_final[group_index]
            decoders = tuple(decoder_final[group_index, local_head_index] for local_head_index in range(
                layout.query_heads_per_kv_group
            ))
            frobenius_numerator = torch.zeros((), device=work_device, dtype=work_dtype)
            frobenius_denominator = torch.zeros((), device=work_device, dtype=work_dtype)
            weighted_numerator = torch.zeros((), device=work_device, dtype=work_dtype)
            weighted_denominator = torch.zeros((), device=work_device, dtype=work_dtype)
            for local_head_index, head_index in enumerate(initial_group.query_head_indices):
                start = head_index * layout.head_dim
                weight_row = work_o[:, start : start + layout.head_dim].transpose(0, 1)
                residual = weight_row - A_row @ decoders[local_head_index]
                frobenius_numerator.add_(residual.square().sum())
                frobenius_denominator.add_(weight_row.square().sum())
                head_rows = grouped_samples[:, group_index, local_head_index, :]
                weighted_numerator.add_((head_rows @ residual).square().sum())
                weighted_denominator.add_((head_rows @ weight_row).square().sum())

            A_pt = A_row.transpose(0, 1).contiguous()
            v_start = group_index * layout.head_dim
            Wv_compressed = A_pt @ work_v[v_start : v_start + layout.head_dim, :]
            compressed_bias = None
            if work_v_bias is not None:
                compressed_bias = A_pt @ work_v_bias[v_start : v_start + layout.head_dim]
            refined_groups.append(
                GQAVOGroupFactors(
                    group_index=group_index,
                    query_head_indices=initial_group.query_head_indices,
                    A_pt=A_pt.to(device="cpu", dtype=target_dtype),
                    Wv_compressed_pt=Wv_compressed.to(device="cpu", dtype=target_dtype),
                    D_pt_list=tuple(
                        value.transpose(0, 1).contiguous().to(device="cpu", dtype=target_dtype)
                        for value in decoders
                    ),
                    v_bias_compressed_pt=(
                        None
                        if compressed_bias is None
                        else compressed_bias.to(device="cpu", dtype=target_dtype)
                    ),
                    relative_frobenius_error=float(
                        torch.sqrt(
                            frobenius_numerator
                            / frobenius_denominator.clamp_min(torch.finfo(work_dtype).tiny)
                        )
                    ),
                    relative_whitened_error=float(
                        torch.sqrt(
                            weighted_numerator
                            / weighted_denominator.clamp_min(torch.finfo(work_dtype).tiny)
                        )
                    ),
                    covariance_rows=initial_group.covariance_rows,
                    initial_relative_whitened_error=(
                        initial_group.initial_relative_whitened_error
                    ),
                    optimization_steps=initial_group.optimization_steps,
                )
            )

    return GQAVOLayerFactors(
        layout=layout,
        groups=tuple(refined_groups),
        o_decoder_bias_pt=initial_factors.o_decoder_bias_pt,
        joint_initial_relative_output_error=initial_output_error,
        joint_final_relative_output_error=best_output_error,
        joint_optimization_steps=completed_steps,
    )


@torch.no_grad()
def global_o_activation_whitened_errors(
    *,
    o_weight: torch.Tensor,
    covariance: torch.Tensor,
    ranks: list[int] | tuple[int, ...],
    act_damp: float = 1e-4,
    factor_device: str | torch.device | None = None,
    work_dtype: torch.dtype = torch.float32,
) -> tuple[GlobalOWhitenedError, ...]:
    """Return optimal global o_proj errors under one activation covariance.

    Only the singular-value spectrum of ``R @ o_weight.T`` is needed, so this
    comparison does not materialize or export global low-rank factors.
    """

    if o_weight.ndim != 2:
        raise ValueError(f"o_weight must be a matrix, got shape {tuple(o_weight.shape)}")
    out_features, in_features = o_weight.shape
    if tuple(covariance.shape) != (in_features, in_features):
        raise ValueError(
            f"covariance must have shape {(in_features, in_features)}, "
            f"got {tuple(covariance.shape)}"
        )
    if not ranks:
        raise ValueError("at least one global rank is required")
    max_rank = min(in_features, out_features)
    unique_ranks = sorted(set(int(rank) for rank in ranks))
    if unique_ranks[0] <= 0 or unique_ranks[-1] > max_rank:
        raise ValueError(f"global ranks must be in [1, {max_rank}], got {unique_ranks}")
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError(f"work_dtype must be float32 or float64, got {work_dtype}")

    work_device = torch.device(factor_device) if factor_device is not None else o_weight.device
    work_weight_row = o_weight.detach().transpose(0, 1).to(
        device=work_device,
        dtype=work_dtype,
    )
    cholesky, _ = _cholesky_with_damping(
        covariance.to(device=work_device, dtype=work_dtype),
        damp=act_damp,
    )
    whitening = cholesky.transpose(0, 1)
    singular_values = torch.linalg.svdvals(whitening @ work_weight_row)
    squared = singular_values.square()
    total = squared.sum().clamp_min(torch.finfo(squared.dtype).tiny)

    results = []
    for rank in unique_ranks:
        retained = squared[:rank].sum() / total
        relative_error = torch.sqrt((1.0 - retained).clamp_min(0.0))
        results.append(
            GlobalOWhitenedError(
                rank=rank,
                relative_whitened_error=float(relative_error),
                energy_retained=float(retained),
            )
        )
    return tuple(results)


@torch.no_grad()
def activation_whitened_o_reconstruction_error(
    *,
    o_weight: torch.Tensor,
    reconstructed_o_weight: torch.Tensor,
    covariance: torch.Tensor,
    act_damp: float = 1e-4,
    factor_device: str | torch.device | None = None,
    work_dtype: torch.dtype = torch.float32,
) -> float:
    """Measure one O reconstruction under the full activation covariance."""

    if o_weight.ndim != 2 or reconstructed_o_weight.shape != o_weight.shape:
        raise ValueError(
            "o_weight and reconstructed_o_weight must have identical matrix shapes, got "
            f"{tuple(o_weight.shape)} and {tuple(reconstructed_o_weight.shape)}"
        )
    out_features, in_features = o_weight.shape
    if tuple(covariance.shape) != (in_features, in_features):
        raise ValueError(
            f"covariance must have shape {(in_features, in_features)}, "
            f"got {tuple(covariance.shape)}"
        )
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError(f"work_dtype must be float32 or float64, got {work_dtype}")

    work_device = torch.device(factor_device) if factor_device is not None else o_weight.device
    weight_row = o_weight.detach().transpose(0, 1).to(device=work_device, dtype=work_dtype)
    reconstructed_row = reconstructed_o_weight.detach().transpose(0, 1).to(
        device=work_device,
        dtype=work_dtype,
    )
    cholesky, _ = _cholesky_with_damping(
        covariance.to(device=work_device, dtype=work_dtype),
        damp=act_damp,
    )
    whitening = cholesky.transpose(0, 1)
    denominator = torch.linalg.vector_norm(whitening @ weight_row).clamp_min(
        torch.finfo(work_dtype).tiny
    )
    numerator = torch.linalg.vector_norm(whitening @ (weight_row - reconstructed_row))
    return float(numerator / denominator)
