"""Private-AllGather runtime for Qwen3.5 gated full-attention outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from basisserve.core.qwen35_gdn_private_ag_runtime import Qwen35PrivateAGOutput
from basisserve.core.qwen35_gdn_runtime import qwen35_decoder_layers


FACTOR_FORMAT = "basisserve.qwen35.full_attention_private_ag_joint_factors.v1"


def load_qwen35_full_attention_private_ag_factors(
    path: str | Path,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != FACTOR_FORMAT
        or int(payload.get("schema_version", -1)) != 1
    ):
        raise ValueError(f"unsupported Qwen3.5 full-attention factors: {source}")
    return payload


@dataclass(frozen=True)
class Qwen35FullAttentionPrivateAGRecord:
    layer_index: int
    tp_size: int
    local_width: int
    local_rank: int
    total_private_rank: int


class Qwen35FullAttentionPrivateAGRuntime:
    """Temporarily replace selected full-attention ``o_proj`` modules."""

    def __init__(
        self,
        model: nn.Module,
        factors: Mapping[str, Any] | str | Path,
        *,
        factor_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.factors = (
            load_qwen35_full_attention_private_ag_factors(factors)
            if isinstance(factors, (str, Path))
            else dict(factors)
        )
        if (
            self.factors.get("format") != FACTOR_FORMAT
            or int(self.factors.get("schema_version", -1)) != 1
        ):
            raise ValueError("invalid Qwen3.5 full-attention factor payload")
        self.factor_dtype = factor_dtype
        self._originals: dict[int, nn.Module] = {}
        self._installed_modules: dict[int, Qwen35PrivateAGOutput] = {}
        self.records: tuple[Qwen35FullAttentionPrivateAGRecord, ...] = ()

    @property
    def installed(self) -> bool:
        return bool(self._originals)

    def _layers_by_index(self) -> dict[int, Mapping[str, Any]]:
        result: dict[int, Mapping[str, Any]] = {}
        for layer in self.factors["layers"]:
            layer_index = int(layer["layer_index"])
            if layer_index in result:
                raise ValueError(f"duplicate full-attention layer {layer_index}")
            result[layer_index] = layer
        if not result:
            raise ValueError("full-attention factor bank contains no layers")
        return result

    def install(self) -> tuple[Qwen35FullAttentionPrivateAGRecord, ...]:
        if self.installed:
            raise RuntimeError("full-attention Private AG runtime is already installed")
        decoder_layers = qwen35_decoder_layers(self.model)
        factor_layers = self._layers_by_index()
        records: list[Qwen35FullAttentionPrivateAGRecord] = []
        try:
            for layer_index, layer_factors in sorted(factor_layers.items()):
                if not 0 <= layer_index < len(decoder_layers):
                    raise ValueError(
                        f"full-attention layer {layer_index} is absent from model"
                    )
                attention = getattr(decoder_layers[layer_index], "self_attn", None)
                if attention is None:
                    raise ValueError(f"layer {layer_index} is not full attention")
                if getattr(attention, "_basisserve_full_private_ag", False):
                    raise RuntimeError(
                        f"layer {layer_index} already has full-attention Private AG"
                    )
                original = getattr(attention, "o_proj", None)
                weight = getattr(original, "weight", None)
                if not isinstance(weight, Tensor) or weight.ndim != 2:
                    raise TypeError(f"layer {layer_index} o_proj has no matrix weight")
                encoders = layer_factors["private_encoders"]
                decoder_weight = layer_factors["joint_decoder_weight"]
                if encoders.ndim != 3 or decoder_weight.ndim != 2:
                    raise ValueError(
                        f"layer {layer_index} full-attention factors are malformed"
                    )
                tp_size, local_width, local_rank = map(int, encoders.shape)
                if tp_size * local_width != int(weight.shape[1]):
                    raise ValueError(
                        f"layer {layer_index} private encoders do not cover o_proj input"
                    )
                expected_decoder = (int(weight.shape[0]), tp_size * local_rank)
                if tuple(decoder_weight.shape) != expected_decoder:
                    raise ValueError(
                        f"layer {layer_index} joint decoder has invalid shape"
                    )
                target_device = weight.device
                target_dtype = self.factor_dtype or weight.dtype
                bias = getattr(original, "bias", None)
                replacement = Qwen35PrivateAGOutput(
                    encoders.to(device=target_device, dtype=target_dtype),
                    decoder_weight.to(device=target_device, dtype=target_dtype),
                    bias=(
                        None
                        if bias is None
                        else bias.to(device=target_device, dtype=target_dtype)
                    ),
                )
                self._originals[layer_index] = original
                self._installed_modules[layer_index] = replacement
                attention.o_proj = replacement
                attention._basisserve_full_private_ag = True
                records.append(
                    Qwen35FullAttentionPrivateAGRecord(
                        layer_index=layer_index,
                        tp_size=tp_size,
                        local_width=local_width,
                        local_rank=local_rank,
                        total_private_rank=tp_size * local_rank,
                    )
                )
        except Exception:
            self.restore()
            raise
        self.records = tuple(records)
        if len(self.records) != len(factor_layers):
            self.restore()
            raise RuntimeError("not all full-attention factors were installed")
        return self.records

    def restore(self) -> None:
        if not self._originals:
            self._installed_modules.clear()
            self.records = ()
            return
        decoder_layers = qwen35_decoder_layers(self.model)
        for layer_index, original in self._originals.items():
            attention = decoder_layers[layer_index].self_attn
            attention.o_proj = original
            if hasattr(attention, "_basisserve_full_private_ag"):
                delattr(attention, "_basisserve_full_private_ag")
        self._originals.clear()
        self._installed_modules.clear()
        self.records = ()

    def __enter__(self) -> "Qwen35FullAttentionPrivateAGRuntime":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


__all__ = [
    "FACTOR_FORMAT",
    "Qwen35FullAttentionPrivateAGRecord",
    "Qwen35FullAttentionPrivateAGRuntime",
    "load_qwen35_full_attention_private_ag_factors",
]
