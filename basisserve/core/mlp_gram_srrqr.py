"""Static TP-local MLP channel selection from contribution Gram matrices.

For post-SwiGLU activations ``A`` and down-projection columns ``w_i``, the
calibration contribution associated with channel ``i`` is
``vec(A[:, i] outer w_i)``.  Its Gram matrix is available without forming that
large matrix:

``G = E[A.T @ A] * (W_down.T @ W_down)``.

The selector below takes a square-root POD sketch of ``G`` and applies the
repository's Gu--Eisenstat strong RRQR implementation to select real channels.
The runtime then applies those fixed, per-layer masks as a dense zero-fill
quality oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from basisserve.core.strong_rrqr import StrongRRQRDiagnostics, strong_rrqr_basis


@dataclass(frozen=True)
class GramSRRQRSelection:
    indices: Tensor
    diagnostics: StrongRRQRDiagnostics
    pod_dimension: int
    pod_energy_fraction: float
    gram_trace: float
    gram_min_eigenvalue: float
    gram_negative_eigenvalue_fraction: float
    gram_positive_eigenvalue_count: int
    gram_effective_rank_fp32: int


@dataclass(frozen=True)
class GramSubsetReweighting:
    coefficients: Tensor
    relative_residual: float
    coefficient_minimum: float
    coefficient_maximum: float
    coefficient_mean: float
    coefficient_l2: float


def contribution_gram(
    activation_second_moment: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """Return the channel-contribution Gram in FP32.

    ``activation_second_moment`` is ``E[a a^T]`` for one TP-local source
    shard. ``down_weight`` follows ``nn.Linear`` storage and is
    ``[output_width, local_source_width]``.
    """

    if activation_second_moment.ndim != 2:
        raise ValueError("activation second moment must be a matrix")
    width = int(activation_second_moment.shape[0])
    if tuple(activation_second_moment.shape) != (width, width):
        raise ValueError("activation second moment must be square")
    if down_weight.ndim != 2 or int(down_weight.shape[1]) != width:
        raise ValueError("down-projection weight width does not match the moment")
    if not torch.isfinite(activation_second_moment).all() or not torch.isfinite(
        down_weight
    ).all():
        raise ValueError("contribution Gram inputs must be finite")

    moment = activation_second_moment.detach().to(dtype=torch.float32)
    weight = down_weight.detach().to(device=moment.device, dtype=torch.float32)
    weight_gram = weight.transpose(0, 1) @ weight
    gram = moment * weight_gram
    return ((gram + gram.transpose(0, 1)) * 0.5).contiguous()


def gram_srrqr_coordinates(
    gram: Tensor,
    selected_width: int,
    *,
    pod_oversample: int = 64,
    bound: float = 2.0,
    max_swaps: int = 64,
) -> GramSRRQRSelection:
    """Select real coordinates from a positive-semidefinite contribution Gram.

    A top-``selected_width + pod_oversample`` eigenspace is converted to a
    square-root feature matrix whose channel Gram equals the truncated ``G``.
    Strong RRQR then selects columns of that feature matrix.
    """

    if gram.ndim != 2 or int(gram.shape[0]) != int(gram.shape[1]):
        raise ValueError("Gram matrix must be square")
    width = int(gram.shape[0])
    if not 0 < selected_width < width:
        raise ValueError("selected width must lie in [1, width)")
    if pod_oversample < 0:
        raise ValueError("POD oversampling must be nonnegative")
    if not torch.isfinite(gram).all():
        raise ValueError("Gram matrix contains non-finite values")

    work = gram.detach().to(dtype=torch.float32)
    work = (work + work.transpose(0, 1)) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(work)
    scale = max(
        float(eigenvalues[-1].abs().detach().cpu()),
        torch.finfo(torch.float32).tiny,
    )
    effective_tolerance = torch.finfo(torch.float32).eps * width * scale
    negative_tolerance = 32.0 * effective_tolerance
    materially_negative = eigenvalues < -negative_tolerance
    negative_fraction = float(materially_negative.float().mean().detach().cpu())
    minimum = float(eigenvalues[0].detach().cpu())
    clipped = eigenvalues.clamp_min(0.0)
    total = float(clipped.sum(dtype=torch.float64).detach().cpu())
    if total <= 0.0:
        raise ValueError("contribution Gram has no positive energy")

    pod_dimension = min(width, selected_width + pod_oversample)
    values = clipped[-pod_dimension:]
    vectors = eigenvectors[:, -pod_dimension:]
    positive = values > 0.0
    if int(positive.sum()) < selected_width:
        raise ValueError(
            "contribution Gram numerical rank is below selected width: "
            f"rank={int(positive.sum())}, selected_width={selected_width}"
        )
    feature = values.sqrt().unsqueeze(1) * vectors.transpose(0, 1)
    rrqr = strong_rrqr_basis(
        feature,
        selected_width,
        bound=bound,
        max_swaps=max_swaps,
    )
    indices = torch.tensor(
        rrqr.diagnostics.selected_columns,
        dtype=torch.int64,
    ).sort().values
    return GramSRRQRSelection(
        indices=indices,
        diagnostics=rrqr.diagnostics,
        pod_dimension=pod_dimension,
        pod_energy_fraction=float(values.sum(dtype=torch.float64).cpu()) / total,
        gram_trace=total,
        gram_min_eigenvalue=minimum,
        gram_negative_eigenvalue_fraction=negative_fraction,
        gram_positive_eigenvalue_count=int((clipped > 0.0).sum().detach().cpu()),
        gram_effective_rank_fp32=int(
            (clipped > effective_tolerance).sum().detach().cpu()
        ),
    )


def gram_subset_reweighting(gram: Tensor, indices: Tensor) -> GramSubsetReweighting:
    """Fit fixed selected-channel scales for the full output contribution.

    For contribution atoms with Gram ``G``, this solves
    ``G[S,S] c = G[S,:] 1`` in FP64.  The resulting approximation is
    ``sum_{i in S} c_i a_i w_i`` and requires neither omitted activations nor a
    dense reconstructed activation vector at inference time.
    """

    if gram.ndim != 2 or int(gram.shape[0]) != int(gram.shape[1]):
        raise ValueError("Gram matrix must be square")
    width = int(gram.shape[0])
    selected = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
    if selected.numel() == 0 or int(selected.min()) < 0 or int(selected.max()) >= width:
        raise ValueError("selected reweighting indices are out of range")
    if int(torch.unique(selected).numel()) != int(selected.numel()):
        raise ValueError("selected reweighting indices must be unique")
    if not torch.isfinite(gram).all():
        raise ValueError("reweighting Gram contains non-finite values")

    import numpy as np
    from scipy.linalg import cho_factor, cho_solve

    work = np.asarray(
        gram.detach().to(device="cpu", dtype=torch.float64).numpy(),
        order="F",
    )
    chosen = selected.numpy()
    selected_gram = np.asarray(work[np.ix_(chosen, chosen)], order="F")
    right_hand_side = np.sum(work[chosen, :], axis=1)
    factor = cho_factor(selected_gram, lower=False, check_finite=False)
    coefficients = cho_solve(factor, right_hand_side, check_finite=False)
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("Gram subset reweighting produced non-finite coefficients")

    ones = np.ones(width, dtype=np.float64)
    target_energy = float(ones @ work @ ones)
    residual = target_energy - 2.0 * float(coefficients @ right_hand_side)
    residual += float(coefficients @ selected_gram @ coefficients)
    roundoff = 256.0 * np.finfo(np.float64).eps * max(abs(target_energy), 1.0)
    if residual < -roundoff:
        raise FloatingPointError(
            f"Gram subset reweighting has a negative residual: {residual:.6e}"
        )
    residual = max(residual, 0.0)
    result = torch.from_numpy(np.array(coefficients, dtype=np.float64, copy=True)).float()
    return GramSubsetReweighting(
        coefficients=result,
        relative_residual=residual / max(abs(target_energy), np.finfo(float).tiny),
        coefficient_minimum=float(np.min(coefficients)),
        coefficient_maximum=float(np.max(coefficients)),
        coefficient_mean=float(np.mean(coefficients)),
        coefficient_l2=float(np.linalg.norm(coefficients)),
    )


@dataclass
class _StaticProfile:
    input_energy: Tensor
    retained_energy: Tensor
    transformed_energy: Tensor
    calls: int = 0
    vectors: int = 0


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidates = (
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
    )
    for candidate in candidates:
        if isinstance(candidate, nn.ModuleList):
            return candidate
    raise ValueError("could not locate language-model decoder layers")


class StaticMLPMaskRuntime:
    """Apply fixed per-layer MLP masks before every ``down_proj``."""

    def __init__(
        self,
        model: nn.Module,
        masks: Mapping[int, Tensor],
        *,
        scales: Mapping[int, Tensor] | None = None,
        tp_size: int = 8,
        profile: bool = True,
    ) -> None:
        if tp_size <= 0:
            raise ValueError("TP size must be positive")
        self.model = model
        self.tp_size = int(tp_size)
        self.profile_enabled = bool(profile)
        self._source_masks = {int(key): value.detach().cpu() for key, value in masks.items()}
        self._source_scales = (
            None
            if scales is None
            else {int(key): value.detach().cpu() for key, value in scales.items()}
        )
        self._device_masks: dict[int, Tensor] = {}
        self._device_scales: dict[int, Tensor] = {}
        self._profiles: dict[int, _StaticProfile] = {}
        self._handles: dict[int, Any] = {}
        self._modules: dict[int, nn.Linear] = {}
        self.kept_per_source: int | None = None
        self.width: int | None = None

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    def _hook(self, layer_index: int):
        def apply_mask(module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
            del module
            if not args or not isinstance(args[0], Tensor):
                raise TypeError(f"layer {layer_index} down projection input is not a tensor")
            source = args[0]
            mask = self._device_masks[layer_index]
            if int(source.shape[-1]) != int(mask.numel()):
                raise ValueError(f"layer {layer_index} MLP mask width mismatch")
            selected = source * mask
            transformed = selected
            if self._source_scales is not None:
                transformed = selected * self._device_scales[layer_index]
            if self.profile_enabled:
                state = self._profiles[layer_index]
                state.calls += 1
                state.vectors += int(source.numel() // source.shape[-1])
                state.input_energy.add_(source.float().square().sum())
                state.retained_energy.add_(selected.float().square().sum())
                state.transformed_energy.add_(transformed.float().square().sum())
            return (transformed, *args[1:])

        return apply_mask

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("static MLP mask runtime is already installed")
        layers = _decoder_layers(self.model)
        if set(self._source_masks) != set(range(len(layers))):
            raise ValueError("static MLP masks must cover every decoder layer exactly")
        if self._source_scales is not None and set(self._source_scales) != set(range(len(layers))):
            raise ValueError("static MLP scales must cover every decoder layer exactly")
        try:
            kept_per_source: int | None = None
            width: int | None = None
            for layer_index, layer in enumerate(layers):
                projection = getattr(getattr(layer, "mlp", None), "down_proj", None)
                if not isinstance(projection, nn.Linear):
                    raise TypeError(f"layer {layer_index} MLP down projection is not Linear")
                layer_width = int(projection.in_features)
                if layer_width % self.tp_size:
                    raise ValueError(f"layer {layer_index} MLP width is not divisible by TP size")
                source = self._source_masks[layer_index]
                if source.ndim != 1 or int(source.numel()) != layer_width:
                    raise ValueError(f"layer {layer_index} static mask has the wrong shape")
                boolean = source.to(dtype=torch.bool)
                counts = boolean.reshape(self.tp_size, -1).sum(dim=1)
                if not torch.equal(counts, counts[:1].expand_as(counts)):
                    raise ValueError(f"layer {layer_index} mask is not TP-source balanced")
                layer_kept = int(counts[0])
                if kept_per_source is None:
                    kept_per_source = layer_kept
                    width = layer_width
                elif kept_per_source != layer_kept or width != layer_width:
                    raise ValueError("static MLP masks must have uniform geometry")
                device_mask = boolean.to(device=projection.weight.device)
                self._device_masks[layer_index] = device_mask
                if self._source_scales is not None:
                    scale = self._source_scales[layer_index]
                    if scale.ndim != 1 or int(scale.numel()) != layer_width:
                        raise ValueError(f"layer {layer_index} static scale has the wrong shape")
                    if not torch.isfinite(scale).all() or torch.count_nonzero(
                        scale[~boolean]
                    ):
                        raise ValueError(
                            f"layer {layer_index} scale must be finite and zero outside the mask"
                        )
                    self._device_scales[layer_index] = scale.to(
                        device=projection.weight.device,
                        dtype=projection.weight.dtype,
                    )
                if self.profile_enabled:
                    self._profiles[layer_index] = _StaticProfile(
                        input_energy=torch.zeros((), device=projection.weight.device),
                        retained_energy=torch.zeros((), device=projection.weight.device),
                        transformed_energy=torch.zeros((), device=projection.weight.device),
                    )
                self._modules[layer_index] = projection
                self._handles[layer_index] = projection.register_forward_pre_hook(
                    self._hook(layer_index)
                )
            self.kept_per_source = kept_per_source
            self.width = width
        except Exception:
            self.restore()
            raise

    def profile_snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer_index, state in sorted(self._profiles.items()):
            source = float(state.input_energy.detach().cpu())
            retained = float(state.retained_energy.detach().cpu())
            transformed = float(state.transformed_energy.detach().cpu())
            rows.append(
                {
                    "layer": layer_index,
                    "calls": state.calls,
                    "vectors": state.vectors,
                    "retained_input_energy": retained / max(source, 1e-30),
                    "transformed_input_energy": transformed / max(source, 1e-30),
                }
            )
        return rows

    def restore(self) -> None:
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()
        self._modules.clear()
        self._device_masks.clear()
        self._device_scales.clear()
        self._profiles.clear()
        self.kept_per_source = None
        self.width = None

    def __enter__(self) -> "StaticMLPMaskRuntime":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


__all__ = [
    "GramSRRQRSelection",
    "GramSubsetReweighting",
    "StaticMLPMaskRuntime",
    "contribution_gram",
    "gram_subset_reweighting",
    "gram_srrqr_coordinates",
]
