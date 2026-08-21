"""Single-process equivalent runtime for Qwen3.5 GDN Private AllGather."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.core.qwen35_gdn_runtime import qwen35_decoder_layers


FACTOR_FORMAT = "basisserve.qwen35.gdn_private_ag_joint_factors.v1"


def load_qwen35_gdn_private_ag_factors(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != FACTOR_FORMAT
        or int(payload.get("schema_version", -1)) != 1
    ):
        raise ValueError(f"unsupported Qwen3.5 GDN Private AG factors: {source}")
    return payload


class Qwen35PrivateAGOutput(nn.Module):
    """Evaluate source-local encoders, concatenation, and a joint decoder."""

    def __init__(
        self,
        private_encoders: Tensor,
        joint_decoder_weight: Tensor,
        *,
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        if private_encoders.ndim != 3:
            raise ValueError("private encoders must have shape [TP,local_width,rank]")
        if joint_decoder_weight.ndim != 2:
            raise ValueError("joint decoder weight must be a matrix")
        tp_size, local_width, local_rank = map(int, private_encoders.shape)
        output_width, total_rank = map(int, joint_decoder_weight.shape)
        if tp_size <= 1 or total_rank != tp_size * local_rank:
            raise ValueError("Private AG encoder/decoder rank geometry is inconsistent")
        if private_encoders.device != joint_decoder_weight.device:
            raise ValueError("Private AG factors must share a device")
        if private_encoders.dtype != joint_decoder_weight.dtype:
            raise ValueError("Private AG factors must share a dtype")
        self.private_projection_weight = nn.Parameter(
            private_encoders.detach().transpose(1, 2).contiguous(),
            requires_grad=False,
        )
        self.joint_decoder_weight = nn.Parameter(
            joint_decoder_weight.detach().contiguous(),
            requires_grad=False,
        )
        if bias is None:
            self.register_parameter("bias", None)
        else:
            if tuple(bias.shape) != (output_width,):
                raise ValueError("Private AG bias shape is inconsistent")
            self.bias = nn.Parameter(bias.detach().contiguous(), requires_grad=False)
        self.tp_size = tp_size
        self.local_width = local_width
        self.local_rank = local_rank

    @property
    def in_features(self) -> int:
        return self.tp_size * self.local_width

    @property
    def out_features(self) -> int:
        return int(self.joint_decoder_weight.shape[0])

    @property
    def total_private_rank(self) -> int:
        return self.tp_size * self.local_rank

    def forward(self, hidden_states: Tensor) -> Tensor:
        if torch.is_grad_enabled() and hidden_states.requires_grad:
            raise RuntimeError("Qwen35PrivateAGOutput is an inference-only prototype")
        if int(hidden_states.shape[-1]) != self.in_features:
            raise ValueError(
                f"expected GDN wire width {self.in_features}, got {hidden_states.shape[-1]}"
            )
        codes = [
            F.linear(
                hidden_states[
                    ...,
                    source * self.local_width : (source + 1) * self.local_width,
                ],
                self.private_projection_weight[source],
                None,
            )
            for source in range(self.tp_size)
        ]
        gathered = torch.cat(codes, dim=-1)
        return F.linear(gathered, self.joint_decoder_weight, self.bias)


@dataclass(frozen=True)
class Qwen35PrivateAGRecord:
    layer_index: int
    tp_size: int
    local_width: int
    local_rank: int
    total_private_rank: int


class Qwen35PrivateAGRuntime:
    """Temporarily replace all selected GDN ``out_proj`` modules."""

    def __init__(
        self,
        model: nn.Module,
        factors: Mapping[str, Any] | str | Path,
        *,
        factor_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.factors = (
            load_qwen35_gdn_private_ag_factors(factors)
            if isinstance(factors, (str, Path))
            else dict(factors)
        )
        if (
            self.factors.get("format") != FACTOR_FORMAT
            or int(self.factors.get("schema_version", -1)) != 1
        ):
            raise ValueError("invalid Qwen3.5 GDN Private AG factor payload")
        self.factor_dtype = factor_dtype
        self._originals: dict[int, nn.Module] = {}
        self._installed_modules: dict[int, Qwen35PrivateAGOutput] = {}
        self.records: tuple[Qwen35PrivateAGRecord, ...] = ()

    @property
    def installed(self) -> bool:
        return bool(self._originals)

    def _layers_by_index(self) -> dict[int, Mapping[str, Any]]:
        result = {}
        for layer in self.factors["layers"]:
            layer_index = int(layer["layer_index"])
            if layer_index in result:
                raise ValueError(f"duplicate Private AG layer {layer_index}")
            result[layer_index] = layer
        if not result:
            raise ValueError("Private AG factor bank contains no layers")
        return result

    def install(self) -> tuple[Qwen35PrivateAGRecord, ...]:
        if self.installed:
            raise RuntimeError("Qwen3.5 Private AG runtime is already installed")
        decoder_layers = qwen35_decoder_layers(self.model)
        factor_layers = self._layers_by_index()
        records = []
        try:
            for layer_index, layer_factors in sorted(factor_layers.items()):
                if not 0 <= layer_index < len(decoder_layers):
                    raise ValueError(f"Private AG layer {layer_index} is absent from model")
                gdn = getattr(decoder_layers[layer_index], "linear_attn", None)
                if gdn is None:
                    raise ValueError(f"layer {layer_index} is not a GDN layer")
                if getattr(gdn, "_basisserve_gdn_private_ag", False):
                    raise RuntimeError(f"layer {layer_index} already has Private AG installed")
                original = getattr(gdn, "out_proj", None)
                weight = getattr(original, "weight", None)
                if not isinstance(weight, Tensor) or weight.ndim != 2:
                    raise TypeError(f"layer {layer_index} out_proj has no matrix weight")
                encoders = layer_factors["private_encoders"]
                decoder_weight = layer_factors["joint_decoder_weight"]
                if encoders.ndim != 3 or decoder_weight.ndim != 2:
                    raise ValueError(f"layer {layer_index} Private AG factors are malformed")
                tp_size, local_width, local_rank = map(int, encoders.shape)
                expected_decoder = (int(weight.shape[0]), tp_size * local_rank)
                if tp_size * local_width != int(weight.shape[1]):
                    raise ValueError(f"layer {layer_index} encoders do not cover the wire")
                if tuple(decoder_weight.shape) != expected_decoder:
                    raise ValueError(f"layer {layer_index} joint decoder shape is invalid")
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
                gdn.out_proj = replacement
                gdn._basisserve_gdn_private_ag = True
                records.append(
                    Qwen35PrivateAGRecord(
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
        return self.records

    def restore(self) -> None:
        if not self._originals:
            self._installed_modules.clear()
            self.records = ()
            return
        decoder_layers = qwen35_decoder_layers(self.model)
        for layer_index, original in self._originals.items():
            gdn = decoder_layers[layer_index].linear_attn
            gdn.out_proj = original
            if hasattr(gdn, "_basisserve_gdn_private_ag"):
                delattr(gdn, "_basisserve_gdn_private_ag")
        self._originals.clear()
        self._installed_modules.clear()
        self.records = ()

    def __enter__(self) -> "Qwen35PrivateAGRuntime":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


__all__ = [
    "FACTOR_FORMAT",
    "Qwen35PrivateAGOutput",
    "Qwen35PrivateAGRecord",
    "Qwen35PrivateAGRuntime",
    "load_qwen35_gdn_private_ag_factors",
]
