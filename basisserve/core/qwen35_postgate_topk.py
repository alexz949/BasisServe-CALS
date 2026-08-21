"""Runtime post-gate Top-K interventions for Qwen3.5 hybrid layers.

The adapter installs forward pre-hooks on the exact output-projection inputs:

* full attention: ``self_attn.o_proj`` after the sigmoid output gate;
* Gated DeltaNet: ``linear_attn.out_proj`` after gated RMSNorm/SiLU.

Only the transient post-gate wire is sparsified.  Checkpoint weights, attention
states, and GDN recurrent states remain unchanged.  The implementation is a
quality oracle: it materializes a dense zero-filled tensor before the existing
PyTorch linear layer and therefore makes no latency claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from basisserve.core.qwen35_gdn_runtime import qwen35_decoder_layers


LayerKind = Literal["full_attention", "gdn"]
Intervention = Literal["full", "gdn", "both"]
SelectionScope = Literal["global", "source_local"]


def retained_count(width: int, keep_ratio: float) -> int:
    """Return the nearest fixed-K endpoint, matching the existing oracle."""

    if width <= 0:
        raise ValueError("Top-K width must be positive")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep ratio must lie in (0,1]")
    return min(width, max(1, int(round(width * keep_ratio))))


@dataclass(frozen=True)
class PostGateTopKRecord:
    layer_index: int
    layer_kind: LayerKind
    width: int
    keep_ratio: float
    selection_scope: SelectionScope
    tp_size: int
    kept_per_vector: int
    kept_per_source: int | None

    @property
    def realized_ratio(self) -> float:
        return self.kept_per_vector / self.width


@dataclass
class _ProfileState:
    frequency: Tensor
    input_energy: Tensor
    retained_energy: Tensor
    calls: int = 0
    vectors: int = 0


class Qwen35PostGateTopKRuntime:
    """Temporarily sparsify Qwen3.5 post-gate output-projection inputs."""

    def __init__(
        self,
        model: nn.Module,
        *,
        intervention: Intervention,
        keep_ratio: float,
        selection_scope: SelectionScope = "source_local",
        tp_size: int = 8,
        profile: bool = True,
    ) -> None:
        if intervention not in {"full", "gdn", "both"}:
            raise ValueError(f"unsupported intervention {intervention!r}")
        if selection_scope not in {"global", "source_local"}:
            raise ValueError(f"unsupported selection scope {selection_scope!r}")
        if tp_size <= 0:
            raise ValueError("TP size must be positive")
        retained_count(1, keep_ratio)
        self.model = model
        self.intervention = intervention
        self.keep_ratio = float(keep_ratio)
        self.selection_scope = selection_scope
        self.tp_size = int(tp_size)
        self.profile_enabled = bool(profile)
        self._handles: dict[int, Any] = {}
        self._modules: dict[int, nn.Linear] = {}
        self._profiles: dict[int, _ProfileState] = {}
        self.records: tuple[PostGateTopKRecord, ...] = ()

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    def _targeted(self, kind: LayerKind) -> bool:
        return self.intervention == "both" or (
            self.intervention == "full" and kind == "full_attention"
        ) or (self.intervention == "gdn" and kind == "gdn")

    def _record(
        self,
        *,
        layer_index: int,
        layer_kind: LayerKind,
        module: nn.Linear,
    ) -> PostGateTopKRecord:
        width = int(module.in_features)
        if width <= 0 or module.weight.ndim != 2 or module.weight.shape[1] != width:
            raise ValueError(f"layer {layer_index} has an invalid output projection")
        kept_per_source: int | None = None
        if self.selection_scope == "source_local":
            if width % self.tp_size:
                raise ValueError(
                    f"layer {layer_index} width {width} is not divisible by TP{self.tp_size}"
                )
            kept_per_source = retained_count(width // self.tp_size, self.keep_ratio)
            kept_per_vector = kept_per_source * self.tp_size
        else:
            kept_per_vector = retained_count(width, self.keep_ratio)
        return PostGateTopKRecord(
            layer_index=layer_index,
            layer_kind=layer_kind,
            width=width,
            keep_ratio=self.keep_ratio,
            selection_scope=self.selection_scope,
            tp_size=self.tp_size,
            kept_per_vector=kept_per_vector,
            kept_per_source=kept_per_source,
        )

    def _sparsify(
        self,
        source: Tensor,
        record: PostGateTopKRecord,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if source.ndim < 1 or int(source.shape[-1]) != record.width:
            raise ValueError(
                f"layer {record.layer_index} post-gate input has shape "
                f"{tuple(source.shape)}, expected final width {record.width}"
            )
        flat = source.reshape(-1, record.width)
        if self.selection_scope == "global":
            indices = torch.topk(
                flat.float().abs(),
                k=record.kept_per_vector,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            selected = flat.gather(1, indices)
            sparse = torch.zeros_like(flat)
            sparse.scatter_(1, indices, selected)
            global_indices = indices
        else:
            local_width = record.width // self.tp_size
            assert record.kept_per_source is not None
            by_source = flat.reshape(-1, self.tp_size, local_width)
            local_indices = torch.topk(
                by_source.float().abs(),
                k=record.kept_per_source,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            selected = by_source.gather(2, local_indices)
            sparse_by_source = torch.zeros_like(by_source)
            sparse_by_source.scatter_(2, local_indices, selected)
            sparse = sparse_by_source.reshape_as(flat)
            offsets = (
                torch.arange(self.tp_size, device=source.device, dtype=local_indices.dtype)
                * local_width
            )
            global_indices = local_indices + offsets.view(1, self.tp_size, 1)
        if self.profile_enabled:
            state = self._profiles[record.layer_index]
            state.calls += 1
            state.vectors += int(flat.shape[0])
            state.frequency.add_(
                torch.bincount(
                    global_indices.reshape(-1),
                    minlength=record.width,
                )
            )
            state.input_energy.add_(flat.float().square().sum())
            state.retained_energy.add_(selected.float().square().sum())
        return sparse.reshape_as(source), global_indices, selected

    def _hook(self, record: PostGateTopKRecord):
        def apply_topk(module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
            del module
            if not args or not isinstance(args[0], Tensor):
                raise TypeError(
                    f"layer {record.layer_index} output projection did not receive a tensor"
                )
            sparse, _, _ = self._sparsify(args[0], record)
            return (sparse, *args[1:])

        return apply_topk

    def install(self) -> tuple[PostGateTopKRecord, ...]:
        if self.installed:
            raise RuntimeError("post-gate Top-K runtime is already installed")
        layers = qwen35_decoder_layers(self.model)
        records: list[PostGateTopKRecord] = []
        try:
            for layer_index, layer in enumerate(layers):
                candidates: tuple[tuple[LayerKind, Any, str], ...] = (
                    ("full_attention", getattr(layer, "self_attn", None), "o_proj"),
                    ("gdn", getattr(layer, "linear_attn", None), "out_proj"),
                )
                present = [item for item in candidates if item[1] is not None]
                if len(present) != 1:
                    raise ValueError(
                        f"decoder layer {layer_index} must expose exactly one full/GDN module"
                    )
                layer_kind, owner, projection_name = present[0]
                if not self._targeted(layer_kind):
                    continue
                projection = getattr(owner, projection_name, None)
                if not isinstance(projection, nn.Linear):
                    raise TypeError(
                        f"layer {layer_index} {layer_kind} output projection is not Linear"
                    )
                if getattr(projection, "_basisserve_postgate_topk", False):
                    raise RuntimeError(f"layer {layer_index} already has a Top-K runtime")
                record = self._record(
                    layer_index=layer_index,
                    layer_kind=layer_kind,
                    module=projection,
                )
                if self.profile_enabled:
                    device = projection.weight.device
                    self._profiles[layer_index] = _ProfileState(
                        frequency=torch.zeros(record.width, dtype=torch.int64, device=device),
                        input_energy=torch.zeros((), dtype=torch.float32, device=device),
                        retained_energy=torch.zeros((), dtype=torch.float32, device=device),
                    )
                handle = projection.register_forward_pre_hook(self._hook(record))
                projection._basisserve_postgate_topk = True
                self._handles[layer_index] = handle
                self._modules[layer_index] = projection
                records.append(record)
        except Exception:
            self.restore()
            raise
        if not records:
            self.restore()
            raise ValueError(f"intervention {self.intervention!r} selected no layers")
        self.records = tuple(records)
        return self.records

    def reset_profile(self) -> None:
        if not self.profile_enabled:
            return
        for state in self._profiles.values():
            state.frequency.zero_()
            state.input_energy.zero_()
            state.retained_energy.zero_()
            state.calls = 0
            state.vectors = 0

    def profile_snapshot(self) -> tuple[list[dict[str, Any]], dict[str, Tensor]]:
        if not self.profile_enabled:
            return [], {}
        records = {record.layer_index: record for record in self.records}
        summaries: list[dict[str, Any]] = []
        tensors: dict[str, Tensor] = {}
        for layer_index, state in sorted(self._profiles.items()):
            record = records[layer_index]
            input_energy = float(state.input_energy.detach().cpu())
            retained_energy = float(state.retained_energy.detach().cpu())
            frequency = state.frequency.detach().cpu()
            probability = frequency.float() / max(state.vectors, 1)
            tensors[f"layer_{layer_index:02d}_{record.layer_kind}_selection_probability"] = (
                probability.contiguous()
            )
            summaries.append(
                {
                    "layer": layer_index,
                    "layer_kind": record.layer_kind,
                    "calls": state.calls,
                    "vectors": state.vectors,
                    "width": record.width,
                    "kept_per_vector": record.kept_per_vector,
                    "kept_per_source": record.kept_per_source,
                    "realized_ratio": record.realized_ratio,
                    "retained_input_energy": retained_energy / max(input_energy, 1e-30),
                    "mean_channel_selection_probability": float(probability.mean()),
                    "selection_probability_std": float(probability.std(unbiased=False)),
                    "selection_probability_max": float(probability.max()),
                }
            )
        return summaries, tensors

    def restore(self) -> None:
        for layer_index, handle in tuple(self._handles.items()):
            handle.remove()
            projection = self._modules[layer_index]
            if hasattr(projection, "_basisserve_postgate_topk"):
                delattr(projection, "_basisserve_postgate_topk")
        self._handles.clear()
        self._modules.clear()
        self._profiles.clear()
        self.records = ()

    def __enter__(self) -> "Qwen35PostGateTopKRuntime":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


__all__ = [
    "PostGateTopKRecord",
    "Qwen35PostGateTopKRuntime",
    "retained_count",
]
