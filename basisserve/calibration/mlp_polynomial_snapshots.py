"""Exact Qwen3.5 MLP tensors for top-k and polynomial diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.calibration.mlp_snapshots import SPLITS


SNAPSHOT_FORMAT = "basisserve.qwen35.mlp_topk_chebyshev_snapshots.v1"
SNAPSHOT_SIGNALS = (
    "X",
    "A",
    "B",
    "C",
    "Y",
)


class MLPPolynomialSnapshotCollector:
    """Capture exact rows across one checkpoint MLP operation sequence.

    Signals use row-major notation:

        A = gate_proj(X)
        B = up_proj(X)
        C = act_fn(A) * B
        Y = down_proj(C)
    """

    def __init__(
        self,
        *,
        layer_index: int,
        mlp: nn.Module,
        storage_dtype: torch.dtype,
    ) -> None:
        self.layer_index = int(layer_index)
        self.mlp = mlp
        self.gate_proj = getattr(mlp, "gate_proj", None)
        self.up_proj = getattr(mlp, "up_proj", None)
        self.down_proj = getattr(mlp, "down_proj", None)
        self.act_fn: Callable[[Tensor], Tensor] | None = getattr(mlp, "act_fn", None)
        if not all(
            isinstance(module, nn.Module)
            for module in (self.gate_proj, self.up_proj, self.down_proj)
        ) or not callable(self.act_fn):
            raise TypeError("MLP does not expose gate/up/down projections and act_fn")
        gate_weight = getattr(self.gate_proj, "weight", None)
        up_weight = getattr(self.up_proj, "weight", None)
        down_weight = getattr(self.down_proj, "weight", None)
        if not all(
            isinstance(weight, Tensor) and weight.ndim == 2
            for weight in (gate_weight, up_weight, down_weight)
        ):
            raise TypeError("MLP projection weights must be matrices")
        assert isinstance(gate_weight, Tensor)
        assert isinstance(up_weight, Tensor)
        assert isinstance(down_weight, Tensor)
        if gate_weight.shape != up_weight.shape:
            raise ValueError("MLP gate/up projection shapes differ")
        self.intermediate_size = int(gate_weight.shape[0])
        self.hidden_size = int(gate_weight.shape[1])
        if tuple(down_weight.shape) != (self.hidden_size, self.intermediate_size):
            raise ValueError("MLP down projection shape is incompatible")
        self.storage_dtype = storage_dtype
        self._split: str | None = None
        self._indices: Tensor | None = None
        self._pending_device: dict[str, Tensor] = {}
        self._pending_cpu: dict[str, Tensor] = {}
        self._chunks: dict[str, dict[str, list[Tensor]]] = {
            split: {name: [] for name in SNAPSHOT_SIGNALS} for split in SPLITS
        }
        self._diagnostics: dict[str, dict[str, float | int]] = {
            split: {
                "calls": 0,
                "post_swiglu_error_energy": 0.0,
                "post_swiglu_teacher_energy": 0.0,
                "post_swiglu_max_abs": 0.0,
                "direct_down_error_energy": 0.0,
                "direct_down_teacher_energy": 0.0,
                "direct_down_max_abs": 0.0,
            }
            for split in SPLITS
        }
        assert isinstance(self.gate_proj, nn.Module)
        assert isinstance(self.up_proj, nn.Module)
        assert isinstance(self.down_proj, nn.Module)
        self.handles = (
            mlp.register_forward_pre_hook(self._mlp_pre_hook),
            self.gate_proj.register_forward_hook(self._gate_hook),
            self.up_proj.register_forward_hook(self._up_hook),
            self.down_proj.register_forward_pre_hook(self._down_pre_hook),
            self.down_proj.register_forward_hook(self._down_output_hook),
        )

    def begin(self, split: str, flat_indices: Tensor) -> None:
        if split not in SPLITS or self._split is not None:
            raise RuntimeError("snapshot collector begin state is invalid")
        if flat_indices.ndim != 1 or flat_indices.dtype != torch.long:
            raise ValueError("flat indices must be a long vector")
        self._split = split
        self._indices = flat_indices.detach().cpu()

    def _selected(self, value: Tensor, width: int) -> Tensor:
        if self._indices is None or int(value.shape[-1]) != width:
            raise ValueError("snapshot hook tensor or collector state is invalid")
        rows = value.reshape(-1, width)
        return rows.index_select(0, self._indices.to(rows.device)).detach()

    def _store(self, name: str, value: Tensor) -> None:
        self._pending_device[name] = value
        self._pending_cpu[name] = value.to(
            device="cpu",
            dtype=self.storage_dtype,
        ).contiguous()

    @torch.no_grad()
    def _mlp_pre_hook(self, module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1 or not isinstance(inputs[0], Tensor):
            raise TypeError("MLP must receive one tensor input")
        self._store("X", self._selected(inputs[0], self.hidden_size))

    @torch.no_grad()
    def _gate_hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not isinstance(output, Tensor):
            raise TypeError("gate projection output must be a tensor")
        self._store("A", self._selected(output, self.intermediate_size))

    @torch.no_grad()
    def _up_hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not isinstance(output, Tensor):
            raise TypeError("up projection output must be a tensor")
        self._store("B", self._selected(output, self.intermediate_size))

    @torch.no_grad()
    def _down_pre_hook(self, module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1 or not isinstance(inputs[0], Tensor):
            raise TypeError("down projection must receive one tensor input")
        if "A" not in self._pending_device or "B" not in self._pending_device:
            raise RuntimeError(
                "gate/up projections were not observed before down projection"
            )
        exact = self._selected(inputs[0], self.intermediate_size)
        assert self.act_fn is not None
        reconstructed = (
            self.act_fn(self._pending_device["A"]) * self._pending_device["B"]
        )
        difference = reconstructed.float() - exact.float()
        assert self._split is not None
        diagnostics = self._diagnostics[self._split]
        diagnostics["post_swiglu_error_energy"] = float(
            diagnostics["post_swiglu_error_energy"]
        ) + float(difference.double().square().sum())
        diagnostics["post_swiglu_teacher_energy"] = float(
            diagnostics["post_swiglu_teacher_energy"]
        ) + float(exact.double().square().sum())
        diagnostics["post_swiglu_max_abs"] = max(
            float(diagnostics["post_swiglu_max_abs"]),
            float(difference.abs().max()),
        )
        self._store("C", exact)

    @torch.no_grad()
    def _down_output_hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not isinstance(output, Tensor) or "C" not in self._pending_device:
            raise TypeError("down projection output hook received invalid state")
        selected = self._selected(output, self.hidden_size)
        weight = getattr(self.down_proj, "weight", None)
        bias = getattr(self.down_proj, "bias", None)
        if not isinstance(weight, Tensor):
            raise TypeError("down projection has no tensor weight")
        direct = F.linear(self._pending_device["C"], weight, bias)
        difference = direct.float() - selected.float()
        assert self._split is not None
        diagnostics = self._diagnostics[self._split]
        diagnostics["direct_down_error_energy"] = float(
            diagnostics["direct_down_error_energy"]
        ) + float(difference.double().square().sum())
        diagnostics["direct_down_teacher_energy"] = float(
            diagnostics["direct_down_teacher_energy"]
        ) + float(selected.double().square().sum())
        diagnostics["direct_down_max_abs"] = max(
            float(diagnostics["direct_down_max_abs"]),
            float(difference.abs().max()),
        )
        self._store("Y", selected)

    def finish(self) -> None:
        if self._split is None or set(self._pending_cpu) != set(SNAPSHOT_SIGNALS):
            raise RuntimeError("snapshot collector did not observe a complete MLP")
        split = self._split
        for name in SNAPSHOT_SIGNALS:
            self._chunks[split][name].append(self._pending_cpu[name])
        self._diagnostics[split]["calls"] = int(self._diagnostics[split]["calls"]) + 1
        self._split = None
        self._indices = None
        self._pending_device.clear()
        self._pending_cpu.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def tensors(self, split: str, target_rows: int) -> dict[str, Tensor]:
        if split not in SPLITS or target_rows <= 0:
            raise ValueError("invalid snapshot split/row request")
        if not self._chunks[split]["X"]:
            raise RuntimeError(f"layer {self.layer_index} collected no {split} rows")
        return {
            name: torch.cat(chunks, dim=0)[:target_rows].contiguous()
            for name, chunks in self._chunks[split].items()
        }

    def diagnostics(self, split: str) -> dict[str, float | int]:
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        source = self._diagnostics[split]
        return {
            "calls": int(source["calls"]),
            "post_swiglu_relative_mse": float(source["post_swiglu_error_energy"])
            / max(float(source["post_swiglu_teacher_energy"]), 1.0e-300),
            "post_swiglu_max_abs": float(source["post_swiglu_max_abs"]),
            "direct_down_relative_mse": float(source["direct_down_error_energy"])
            / max(float(source["direct_down_teacher_energy"]), 1.0e-300),
            "direct_down_max_abs": float(source["direct_down_max_abs"]),
        }


__all__ = [
    "MLPPolynomialSnapshotCollector",
    "SNAPSHOT_FORMAT",
    "SNAPSHOT_SIGNALS",
]
