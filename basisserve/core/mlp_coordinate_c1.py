"""Coordinate-selected C1 decoders for row-parallel MLP outputs.

For post-SwiGLU activations ``A``, fixed coordinates ``S``, and the dense
teacher output ``Y = A @ W_down.T``, the offline solver fits

``D_S = argmin_D E[||Y - A[:, S] @ D.T||^2]``.

The selected coordinates can be exchanged in TP source-rank order and the
compact decoder can be output-row sharded.  The single-process runtime below
is the exact quality equivalent of that topology; it does not benchmark the
sparse exchange itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MLPCoordinateDecoderFit:
    decoder_weight: Tensor
    metrics: Mapping[str, Any]


@torch.no_grad()
def fit_mlp_coordinate_decoder(
    selected_input_second_moment: Tensor,
    target_selected_input_cross_moment: Tensor,
    target_second_moment_trace: Tensor | float,
    *,
    input_width: int,
    factor_dtype: torch.dtype = torch.bfloat16,
    work_dtype: torch.dtype = torch.float64,
    relative_damping: float = 0.0,
) -> MLPCoordinateDecoderFit:
    """Fit the free decoder for a fixed coordinate encoder.

    ``target_selected_input_cross_moment`` follows linear-weight layout and is
    ``E[Y.T @ A_S]`` with shape ``[output_width, selected_width]``.
    Relative damping, when requested, is scaled by the mean diagonal of the
    selected-input second moment.
    """

    if (
        selected_input_second_moment.ndim != 2
        or selected_input_second_moment.shape[0]
        != selected_input_second_moment.shape[1]
    ):
        raise ValueError("selected-input second moment must be square")
    selected_width = int(selected_input_second_moment.shape[0])
    if (
        target_selected_input_cross_moment.ndim != 2
        or int(target_selected_input_cross_moment.shape[1]) != selected_width
    ):
        raise ValueError("target/input cross moment has incompatible width")
    output_width = int(target_selected_input_cross_moment.shape[0])
    if input_width < selected_width:
        raise ValueError("full input width is below the selected width")
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("work dtype must be float32 or float64")
    if factor_dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError("unsupported coordinate decoder factor dtype")
    if relative_damping < 0:
        raise ValueError("coordinate decoder damping must be nonnegative")

    device = selected_input_second_moment.device
    c_zz = selected_input_second_moment.detach().to(
        device=device, dtype=work_dtype
    )
    c_zz = 0.5 * (c_zz + c_zz.T)
    c_yz = target_selected_input_cross_moment.detach().to(
        device=device, dtype=work_dtype
    )
    target_energy = torch.as_tensor(
        target_second_moment_trace, device=device, dtype=work_dtype
    )
    if target_energy.ndim != 0 or not bool(torch.isfinite(target_energy)):
        raise ValueError("target second-moment trace must be a finite scalar")
    if float(target_energy.item()) <= 0.0:
        raise ValueError("target second-moment trace must be positive")
    if not bool(torch.isfinite(c_zz).all()) or not bool(torch.isfinite(c_yz).all()):
        raise FloatingPointError("coordinate decoder moments must be finite")

    mean_diagonal = torch.trace(c_zz) / selected_width
    absolute_damping = relative_damping * float(mean_diagonal.item())
    solve_gram = c_zz
    if absolute_damping > 0.0:
        solve_gram = c_zz + absolute_damping * torch.eye(
            selected_width, device=device, dtype=work_dtype
        )
    cholesky, info = torch.linalg.cholesky_ex(solve_gram, check_errors=False)
    if int(info.item()) != 0:
        raise torch.linalg.LinAlgError(
            "selected-input covariance is not positive definite at "
            f"relative damping {relative_damping:g}"
        )
    decoder_coefficient = torch.cholesky_solve(c_yz.T.contiguous(), cholesky)
    decoder_weight = decoder_coefficient.T.contiguous()

    prediction_energy = torch.sum(decoder_coefficient * (c_zz @ decoder_coefficient))
    target_prediction_cross = torch.sum(decoder_weight * c_yz)
    output_residual = (
        target_energy - 2.0 * target_prediction_cross + prediction_energy
    ).clamp_min(0.0)
    regularized_residual = output_residual + absolute_damping * torch.sum(
        decoder_weight * decoder_weight
    )
    stored_decoder = decoder_weight.to(dtype=factor_dtype)
    stored_work = stored_decoder.to(dtype=work_dtype)
    stored_coefficient = stored_work.T.contiguous()
    stored_prediction_energy = torch.sum(
        stored_coefficient * (c_zz @ stored_coefficient)
    )
    stored_cross = torch.sum(stored_work * c_yz)
    stored_residual = (
        target_energy - 2.0 * stored_cross + stored_prediction_energy
    ).clamp_min(0.0)
    diagonal = cholesky.diagonal().abs()
    return MLPCoordinateDecoderFit(
        decoder_weight=stored_decoder.cpu().contiguous(),
        metrics={
            "algorithm": "coordinate_selected_free_decoder_least_squares",
            "objective": "dense_mlp_output_from_selected_post_swiglu_coordinates",
            "solver": (
                "zero_damping_fp64_normal_equation_cholesky"
                if relative_damping == 0.0
                else "ridge_fp64_normal_equation_cholesky"
            ),
            "input_width": input_width,
            "selected_width": selected_width,
            "output_width": output_width,
            "relative_covariance_damping": relative_damping,
            "absolute_covariance_damping": absolute_damping,
            "fit_relative_output_mse": float((output_residual / target_energy).item()),
            "stored_factor_relative_output_mse": float(
                (stored_residual / target_energy).item()
            ),
            "fit_relative_regularized_objective": float(
                (regularized_residual / target_energy).item()
            ),
            "cholesky_minimum_abs_diagonal": float(diagonal.min().item()),
            "cholesky_maximum_abs_diagonal": float(diagonal.max().item()),
            "cholesky_diagonal_ratio": float(
                (diagonal.min() / diagonal.max()).item()
            ),
            "decoder_frobenius_norm": float(torch.linalg.vector_norm(decoder_weight).item()),
        },
    )


class MLPCoordinateC1Linear(nn.Module):
    """Single-process quality equivalent of coordinate exchange plus decode."""

    def __init__(
        self,
        input_indices: Tensor,
        decoder_weight: Tensor,
        *,
        input_width: int,
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        indices = input_indices.detach().to(dtype=torch.int64).flatten()
        if indices.numel() == 0:
            raise ValueError("coordinate decoder must select at least one input")
        if int(indices.min()) < 0 or int(indices.max()) >= input_width:
            raise ValueError("coordinate decoder index is outside the input width")
        if int(torch.unique(indices).numel()) != int(indices.numel()):
            raise ValueError("coordinate decoder indices must be unique")
        if decoder_weight.ndim != 2 or int(decoder_weight.shape[1]) != int(
            indices.numel()
        ):
            raise ValueError("coordinate decoder weight has incompatible width")
        self.input_width = int(input_width)
        self.register_buffer("input_indices", indices.contiguous(), persistent=True)
        self.decoder_weight = nn.Parameter(
            decoder_weight.detach().contiguous(), requires_grad=False
        )
        if bias is None:
            self.register_parameter("bias", None)
        else:
            if tuple(bias.shape) != (int(decoder_weight.shape[0]),):
                raise ValueError("coordinate decoder bias has the wrong shape")
            self.bias = nn.Parameter(bias.detach().contiguous(), requires_grad=False)

    @property
    def in_features(self) -> int:
        return self.input_width

    @property
    def out_features(self) -> int:
        return int(self.decoder_weight.shape[0])

    @property
    def selected_features(self) -> int:
        return int(self.input_indices.numel())

    def forward(self, hidden_states: Tensor) -> Tensor:
        if int(hidden_states.shape[-1]) != self.input_width:
            raise ValueError("coordinate decoder received the wrong input width")
        selected = hidden_states.index_select(-1, self.input_indices)
        return F.linear(selected, self.decoder_weight, self.bias)


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    candidates = (
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
    )
    for candidate in candidates:
        if isinstance(candidate, nn.ModuleList):
            return candidate
    raise ValueError("could not locate language-model decoder layers")


class CoordinateMLPDecoderRuntime:
    """Temporarily replace all MLP down projections by coordinate decoders."""

    def __init__(
        self,
        model: nn.Module,
        indices: Mapping[int, Tensor],
        decoder_weights: Mapping[int, Tensor],
        *,
        tp_size: int = 8,
    ) -> None:
        if tp_size <= 0:
            raise ValueError("TP size must be positive")
        self.model = model
        self.indices = {int(key): value.detach().cpu() for key, value in indices.items()}
        self.decoder_weights = {
            int(key): value.detach().cpu() for key, value in decoder_weights.items()
        }
        self.tp_size = int(tp_size)
        self.kept_per_source: int | None = None
        self._originals: dict[int, nn.Module] = {}

    @property
    def installed(self) -> bool:
        return bool(self._originals)

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("coordinate MLP runtime is already installed")
        layers = _decoder_layers(self.model)
        expected = set(range(len(layers)))
        if set(self.indices) != expected or set(self.decoder_weights) != expected:
            raise ValueError("coordinate decoder factors must cover every layer")
        try:
            kept_per_source: int | None = None
            for layer_index, layer in enumerate(layers):
                original = getattr(getattr(layer, "mlp", None), "down_proj", None)
                if not isinstance(original, nn.Linear):
                    raise TypeError(f"layer {layer_index} MLP down projection is not Linear")
                input_width = int(original.in_features)
                if input_width % self.tp_size:
                    raise ValueError(f"layer {layer_index} input width is not TP divisible")
                local_width = input_width // self.tp_size
                selected = self.indices[layer_index].to(dtype=torch.int64).flatten()
                counts = torch.bincount(
                    torch.div(selected, local_width, rounding_mode="floor"),
                    minlength=self.tp_size,
                )
                if counts.numel() != self.tp_size or not torch.equal(
                    counts, counts[:1].expand_as(counts)
                ):
                    raise ValueError(f"layer {layer_index} coordinates are not TP balanced")
                layer_kept = int(counts[0])
                if kept_per_source is None:
                    kept_per_source = layer_kept
                elif kept_per_source != layer_kept:
                    raise ValueError("coordinate decoder geometry is not uniform")
                decoder = self.decoder_weights[layer_index]
                replacement = MLPCoordinateC1Linear(
                    selected.to(device=original.weight.device),
                    decoder.to(device=original.weight.device, dtype=original.weight.dtype),
                    input_width=input_width,
                    bias=original.bias,
                ).eval()
                self._originals[layer_index] = original
                layer.mlp.down_proj = replacement
            self.kept_per_source = kept_per_source
        except Exception:
            self.restore()
            raise

    def restore(self) -> None:
        if not self._originals:
            return
        layers = _decoder_layers(self.model)
        for layer_index, original in self._originals.items():
            layers[layer_index].mlp.down_proj = original
        self._originals.clear()
        self.kept_per_source = None

    def __enter__(self) -> "CoordinateMLPDecoderRuntime":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


__all__ = [
    "CoordinateMLPDecoderRuntime",
    "MLPCoordinateC1Linear",
    "MLPCoordinateDecoderFit",
    "fit_mlp_coordinate_decoder",
]
