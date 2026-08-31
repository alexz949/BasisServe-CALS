"""Static-scaled FP8 communication simulation for Qwen3.5 Private AllGather."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.core.qwen35_full_attention_private_ag_runtime import (
    load_qwen35_full_attention_private_ag_factors,
)
from basisserve.core.qwen35_gdn_private_ag_runtime import (
    load_qwen35_gdn_private_ag_factors,
)
from basisserve.core.qwen35_gdn_runtime import qwen35_decoder_layers


FP8_E4M3_DTYPE = torch.float8_e4m3fn
FP8_E4M3_MAX = float(torch.finfo(FP8_E4M3_DTYPE).max)
RuntimeMode = Literal["passthrough", "observe", "quantize"]


class Qwen35FP8PrivateAGOutput(nn.Module):
    """Private encoders with optional static-scaled E4M3 wire quantization."""

    def __init__(
        self,
        private_encoders: Tensor,
        joint_decoder_weight: Tensor,
        *,
        mode: RuntimeMode,
        source_scales: Tensor | None = None,
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        if private_encoders.ndim != 3:
            raise ValueError("private encoders must have shape [TP,local_width,rank]")
        if joint_decoder_weight.ndim != 2:
            raise ValueError("joint decoder weight must be a matrix")
        if mode not in ("passthrough", "observe", "quantize"):
            raise ValueError(f"unsupported FP8 Private-AG mode {mode!r}")
        tp_size, local_width, local_rank = map(int, private_encoders.shape)
        output_width, total_rank = map(int, joint_decoder_weight.shape)
        if tp_size <= 1 or total_rank != tp_size * local_rank:
            raise ValueError("Private-AG encoder/decoder rank geometry is inconsistent")
        if private_encoders.device != joint_decoder_weight.device:
            raise ValueError("Private-AG factors must share a device")
        if private_encoders.dtype != joint_decoder_weight.dtype:
            raise ValueError("Private-AG factors must share a dtype")
        if mode == "quantize":
            if source_scales is None or tuple(source_scales.shape) != (tp_size,):
                raise ValueError("quantize mode requires one scale per TP source")
            work_scales = source_scales.detach().to(
                device=private_encoders.device,
                dtype=torch.float32,
            )
            if not torch.isfinite(work_scales).all() or bool((work_scales <= 0).any()):
                raise ValueError("FP8 source scales must be finite and positive")
        elif source_scales is not None:
            raise ValueError("source scales are valid only in quantize mode")
        else:
            work_scales = torch.empty(
                0,
                device=private_encoders.device,
                dtype=torch.float32,
            )

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
                raise ValueError("Private-AG bias shape is inconsistent")
            self.bias = nn.Parameter(bias.detach().contiguous(), requires_grad=False)
        self.register_buffer("source_scales", work_scales, persistent=True)
        self.register_buffer(
            "source_amax",
            torch.zeros(tp_size, device=private_encoders.device, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "wire_elements",
            torch.zeros((), device=private_encoders.device, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "clipped_elements",
            torch.zeros((), device=private_encoders.device, dtype=torch.int64),
            persistent=True,
        )
        self.tp_size = tp_size
        self.local_width = local_width
        self.local_rank = local_rank
        self.mode: RuntimeMode = mode

    @property
    def in_features(self) -> int:
        return self.tp_size * self.local_width

    @property
    def out_features(self) -> int:
        return int(self.joint_decoder_weight.shape[0])

    def latent_components(self, hidden_states: Tensor) -> Tensor:
        if int(hidden_states.shape[-1]) != self.in_features:
            raise ValueError(
                f"expected wire width {self.in_features}, got {hidden_states.shape[-1]}"
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
        return torch.stack(codes, dim=-2)

    @torch.no_grad()
    def _update_amax(self, codes: Tensor) -> None:
        observed = (
            codes.detach()
            .float()
            .movedim(-2, 0)
            .reshape(self.tp_size, -1)
            .abs()
            .amax(dim=1)
        )
        torch.maximum(self.source_amax, observed, out=self.source_amax)

    def _quantize(self, codes: Tensor) -> Tensor:
        self._update_amax(codes)
        shape = (1,) * (codes.ndim - 2) + (self.tp_size, 1)
        scales = self.source_scales.reshape(shape)
        normalized = codes.float() / scales
        clipped = normalized.abs() > FP8_E4M3_MAX
        self.wire_elements.add_(codes.numel())
        self.clipped_elements.add_(clipped.count_nonzero())
        quantized = normalized.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(
            FP8_E4M3_DTYPE
        )
        return (quantized.float() * scales).to(dtype=codes.dtype)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if torch.is_grad_enabled() and hidden_states.requires_grad:
            raise RuntimeError("Qwen35 FP8 Private-AG is inference-only")
        codes = self.latent_components(hidden_states)
        if self.mode == "observe":
            self._update_amax(codes)
        elif self.mode == "quantize":
            codes = self._quantize(codes)
        gathered = codes.flatten(start_dim=-2)
        return F.linear(gathered, self.joint_decoder_weight, self.bias)

    def calibrated_scale(self) -> Tensor:
        if self.mode != "observe":
            raise RuntimeError("calibrated scales are available only in observe mode")
        tiny = torch.finfo(torch.float32).tiny
        return (self.source_amax / FP8_E4M3_MAX).clamp_min(tiny)


@dataclass(frozen=True)
class Qwen35FP8PrivateAGRecord:
    layer_index: int
    block_type: str
    tp_size: int
    local_width: int
    local_rank: int
    total_private_rank: int


class Qwen35FP8PrivateAGRuntime:
    """Install passthrough, calibration, or FP8 Private-AG Wo replacements."""

    def __init__(
        self,
        model: nn.Module,
        gdn_factors: Mapping[str, Any] | str | Path,
        full_factors: Mapping[str, Any] | str | Path,
        *,
        mode: RuntimeMode,
        scales: Mapping[int | str, Tensor] | None = None,
        factor_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.gdn_factors = (
            load_qwen35_gdn_private_ag_factors(gdn_factors)
            if isinstance(gdn_factors, (str, Path))
            else dict(gdn_factors)
        )
        self.full_factors = (
            load_qwen35_full_attention_private_ag_factors(full_factors)
            if isinstance(full_factors, (str, Path))
            else dict(full_factors)
        )
        if mode not in ("passthrough", "observe", "quantize"):
            raise ValueError(f"unsupported FP8 Private-AG mode {mode!r}")
        if mode == "quantize" and scales is None:
            raise ValueError("quantize mode requires calibrated scales")
        if mode != "quantize" and scales is not None:
            raise ValueError("calibrated scales are valid only in quantize mode")
        self.mode: RuntimeMode = mode
        self.scales = (
            None
            if scales is None
            else {int(layer): value for layer, value in scales.items()}
        )
        self.factor_dtype = factor_dtype
        self._originals: dict[int, tuple[nn.Module, str, nn.Module]] = {}
        self._installed: dict[int, Qwen35FP8PrivateAGOutput] = {}
        self.records: tuple[Qwen35FP8PrivateAGRecord, ...] = ()

    @property
    def installed(self) -> bool:
        return bool(self._originals)

    def _factor_layers(self) -> dict[int, tuple[str, Mapping[str, Any]]]:
        result: dict[int, tuple[str, Mapping[str, Any]]] = {}
        for block_type, factors in (
            ("linear_attention", self.gdn_factors),
            ("full_attention", self.full_factors),
        ):
            for layer in factors.get("layers", ()):
                index = int(layer["layer_index"])
                if index in result:
                    raise ValueError(f"duplicate FP8 Private-AG layer {index}")
                result[index] = (block_type, layer)
        if not result:
            raise ValueError("FP8 Private-AG factor banks contain no layers")
        if self.scales is not None and set(self.scales) != set(result):
            raise ValueError("FP8 scale coverage differs from factor coverage")
        return result

    @staticmethod
    def _projection(layer: nn.Module, block_type: str) -> tuple[nn.Module, str]:
        if block_type == "linear_attention":
            return layer.linear_attn, "out_proj"
        if block_type == "full_attention":
            return layer.self_attn, "o_proj"
        raise ValueError(f"unsupported Qwen3.5 block type {block_type!r}")

    def install(self) -> tuple[Qwen35FP8PrivateAGRecord, ...]:
        if self.installed:
            raise RuntimeError("Qwen3.5 FP8 Private-AG runtime is already installed")
        decoder_layers = qwen35_decoder_layers(self.model)
        records = []
        try:
            for layer_index, (block_type, factors) in sorted(
                self._factor_layers().items()
            ):
                if not 0 <= layer_index < len(decoder_layers):
                    raise ValueError(f"FP8 Private-AG layer {layer_index} is absent")
                parent, attribute = self._projection(
                    decoder_layers[layer_index], block_type
                )
                original = getattr(parent, attribute)
                weight = getattr(original, "weight", None)
                if not isinstance(weight, Tensor) or weight.ndim != 2:
                    raise TypeError(f"layer {layer_index} projection has no matrix weight")
                encoders = factors["private_encoders"]
                decoder = factors["joint_decoder_weight"]
                if encoders.ndim != 3 or decoder.ndim != 2:
                    raise ValueError(f"layer {layer_index} Private-AG factors are malformed")
                tp_size, local_width, local_rank = map(int, encoders.shape)
                if tp_size * local_width != int(weight.shape[1]):
                    raise ValueError(f"layer {layer_index} encoders do not cover the wire")
                if tuple(decoder.shape) != (
                    int(weight.shape[0]),
                    tp_size * local_rank,
                ):
                    raise ValueError(f"layer {layer_index} decoder shape is invalid")
                dtype = self.factor_dtype or weight.dtype
                bias = getattr(original, "bias", None)
                replacement = Qwen35FP8PrivateAGOutput(
                    encoders.to(device=weight.device, dtype=dtype),
                    decoder.to(device=weight.device, dtype=dtype),
                    mode=self.mode,
                    source_scales=(
                        None if self.scales is None else self.scales[layer_index]
                    ),
                    bias=(
                        None
                        if bias is None
                        else bias.to(device=weight.device, dtype=dtype)
                    ),
                )
                setattr(parent, attribute, replacement)
                self._originals[layer_index] = (parent, attribute, original)
                self._installed[layer_index] = replacement
                records.append(
                    Qwen35FP8PrivateAGRecord(
                        layer_index=layer_index,
                        block_type=block_type,
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

    def scale_state(self) -> dict[int, dict[str, Tensor | int | str]]:
        if self.mode != "observe" or not self.installed:
            raise RuntimeError("scale state requires an installed observe runtime")
        return {
            layer: {
                "layer_index": layer,
                "block_type": next(
                    record.block_type
                    for record in self.records
                    if record.layer_index == layer
                ),
                "local_rank": module.local_rank,
                "source_amax": module.source_amax.detach().cpu().contiguous(),
                "source_scales": module.calibrated_scale()
                .detach()
                .cpu()
                .contiguous(),
            }
            for layer, module in sorted(self._installed.items())
        }

    def quantization_profile(self) -> dict[int, dict[str, Tensor | int | float]]:
        if self.mode != "quantize" or not self.installed:
            raise RuntimeError("quantization profile requires an installed FP8 runtime")
        result = {}
        for layer, module in sorted(self._installed.items()):
            elements = int(module.wire_elements.item())
            clipped = int(module.clipped_elements.item())
            result[layer] = {
                "layer_index": layer,
                "local_rank": module.local_rank,
                "wire_elements": elements,
                "clipped_elements": clipped,
                "clipped_fraction": clipped / max(elements, 1),
                "evaluation_source_amax": module.source_amax.detach()
                .cpu()
                .contiguous(),
                "source_scales": module.source_scales.detach().cpu().contiguous(),
            }
        return result

    def restore(self) -> None:
        for parent, attribute, original in self._originals.values():
            setattr(parent, attribute, original)
        self._originals.clear()
        self._installed.clear()
        self.records = ()

    def __enter__(self) -> "Qwen35FP8PrivateAGRuntime":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


__all__ = [
    "FP8_E4M3_DTYPE",
    "FP8_E4M3_MAX",
    "Qwen35FP8PrivateAGOutput",
    "Qwen35FP8PrivateAGRecord",
    "Qwen35FP8PrivateAGRuntime",
]
