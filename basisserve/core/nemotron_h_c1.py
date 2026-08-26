"""Nemotron-H hybrid-layer adapters for TP-source C1 output projections.

Nemotron-H exposes a heterogeneous decoder stack.  Full-attention blocks end
in ``mixer.o_proj`` while Mamba-2 blocks end in ``mixer.out_proj``.  Both are
row-parallel output boundaries in a tensor-parallel deployment, so the same
source-private C1 factorization applies once the exact projection input has
been identified.

This module does not import Transformers and does not modify Mamba state or
attention KV caches.  It only discovers and validates the output projections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from torch import nn

from basisserve.core.tp_source_wo_c1 import TPSourceWOLayout


NemotronC1LayerKind = Literal["linear_attention", "full_attention"]
SUPPORTED_KINDS: tuple[NemotronC1LayerKind, ...] = (
    "linear_attention",
    "full_attention",
)


@dataclass(frozen=True)
class NemotronC1Target:
    """One validated Nemotron-H output-projection boundary."""

    layer_index: int
    layer_kind: NemotronC1LayerKind
    projection_name: str
    input_width: int
    output_width: int

    def layout(
        self,
        *,
        tp_size: int,
        source_rank: int,
        dtype_bytes: int = 2,
    ) -> TPSourceWOLayout:
        return TPSourceWOLayout(
            input_width=self.input_width,
            output_width=self.output_width,
            tp_size=tp_size,
            source_rank=source_rank,
            dtype_bytes=dtype_bytes,
        )


def nemotron_h_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder stack in base and causal-LM Nemotron-H wrappers."""

    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
    )
    for candidate in candidates:
        if isinstance(candidate, nn.ModuleList):
            return candidate
    raise ValueError("could not locate Nemotron-H decoder layers")


def nemotron_h_projection(layer: nn.Module) -> tuple[NemotronC1LayerKind, str, nn.Linear]:
    """Return the C1-eligible projection for one hybrid decoder block."""

    kind = str(getattr(layer, "block_type", ""))
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"Nemotron-H block type {kind!r} has no C1-Wo target")
    mixer = getattr(layer, "mixer", None)
    if mixer is None:
        raise ValueError(f"Nemotron-H {kind} block has no mixer")
    projection_name = "out_proj" if kind == "linear_attention" else "o_proj"
    projection = getattr(mixer, projection_name, None)
    if not isinstance(projection, nn.Linear):
        raise TypeError(
            f"Nemotron-H {kind} mixer {projection_name} is not torch.nn.Linear"
        )
    return kind, projection_name, projection


def discover_nemotron_h_c1_targets(
    model: nn.Module,
    *,
    kinds: Sequence[NemotronC1LayerKind] = SUPPORTED_KINDS,
) -> tuple[NemotronC1Target, ...]:
    """Discover every selected Mamba/attention C1 output boundary."""

    selected = tuple(dict.fromkeys(kinds))
    if not selected or any(kind not in SUPPORTED_KINDS for kind in selected):
        raise ValueError(f"unsupported Nemotron-H C1 layer kinds: {selected}")
    result: list[NemotronC1Target] = []
    for layer_index, layer in enumerate(nemotron_h_decoder_layers(model)):
        kind = str(getattr(layer, "block_type", ""))
        if kind not in selected:
            continue
        layer_kind, projection_name, projection = nemotron_h_projection(layer)
        result.append(
            NemotronC1Target(
                layer_index=layer_index,
                layer_kind=layer_kind,
                projection_name=projection_name,
                input_width=int(projection.in_features),
                output_width=int(projection.out_features),
            )
        )
    missing = sorted(set(selected) - {target.layer_kind for target in result})
    if missing:
        raise ValueError(f"Nemotron-H checkpoint has no selected layer kinds: {missing}")
    return tuple(result)


def first_nemotron_h_c1_target_per_kind(
    model: nn.Module,
) -> tuple[NemotronC1Target, NemotronC1Target]:
    """Return the first Mamba and first full-attention target, in that order."""

    targets = discover_nemotron_h_c1_targets(model)
    by_kind = {
        kind: next(target for target in targets if target.layer_kind == kind)
        for kind in SUPPORTED_KINDS
    }
    return by_kind["linear_attention"], by_kind["full_attention"]


def projection_for_target(model: nn.Module, target: NemotronC1Target) -> nn.Linear:
    """Resolve a previously discovered target and revalidate its geometry."""

    layers = nemotron_h_decoder_layers(model)
    if not 0 <= target.layer_index < len(layers):
        raise ValueError(f"Nemotron-H target layer {target.layer_index} is absent")
    kind, name, projection = nemotron_h_projection(layers[target.layer_index])
    observed = (
        kind,
        name,
        int(projection.in_features),
        int(projection.out_features),
    )
    expected = (
        target.layer_kind,
        target.projection_name,
        target.input_width,
        target.output_width,
    )
    if observed != expected:
        raise ValueError(
            f"Nemotron-H target geometry changed: observed={observed}, expected={expected}"
        )
    return projection


__all__ = [
    "NemotronC1LayerKind",
    "NemotronC1Target",
    "SUPPORTED_KINDS",
    "discover_nemotron_h_c1_targets",
    "first_nemotron_h_c1_target_per_kind",
    "nemotron_h_decoder_layers",
    "nemotron_h_projection",
    "projection_for_target",
]
