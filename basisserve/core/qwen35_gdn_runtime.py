"""No-cache projected-state runtime for Qwen3.5 Gated DeltaNet.

The adapter wraps each GDN chunk kernel at its value interface.  It projects
``v`` from ``D`` channels to ``R`` channels, lets the existing FLA kernel
evolve the closed recurrent state ``S E``, then decodes the current readout
before Qwen3.5's gated RMSNorm.  This first runtime intentionally supports
teacher-forced, ``use_cache=False`` evaluation only; decode-cache layout is a
separate serving integration step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn


SPECTRA_FORMAT = "basisserve.qwen35.gdn_spectra.v1"


def qwen35_decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidates = (
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
    )
    for candidate in candidates:
        if isinstance(candidate, nn.ModuleList):
            return candidate
    raise ValueError("could not locate Qwen3.5 language-model decoder layers")


def load_qwen35_gdn_spectra(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("format") != SPECTRA_FORMAT or int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported Qwen3.5 GDN spectra: {source}")
    return payload


@dataclass(frozen=True)
class ProjectedGDNRecord:
    layer_index: int
    rank: int
    value_head_dim: int
    num_value_heads: int
    signal: str
    basis_kind: str

    @property
    def state_fraction(self) -> float:
        return self.rank / self.value_head_dim


class Qwen35ProjectedGDNRuntime:
    """Temporarily install projected value/state GDN chunk kernels."""

    def __init__(
        self,
        model: nn.Module,
        spectra: Mapping[str, Any] | str | Path,
        *,
        rank: int,
        signal: str = "core",
        factor_dtype: torch.dtype | None = None,
    ) -> None:
        if rank <= 0:
            raise ValueError("projected GDN rank must be positive")
        self.model = model
        self.spectra = (
            load_qwen35_gdn_spectra(spectra)
            if isinstance(spectra, (str, Path))
            else dict(spectra)
        )
        if self.spectra.get("format") != SPECTRA_FORMAT:
            raise ValueError("invalid projected GDN spectra payload")
        self.rank = int(rank)
        self.signal = str(signal)
        self.factor_dtype = factor_dtype
        self._originals: dict[int, Callable[..., tuple[Tensor, Tensor | None]]] = {}
        self._factors: dict[int, tuple[Tensor, Tensor]] = {}
        self.records: tuple[ProjectedGDNRecord, ...] = ()

    @property
    def installed(self) -> bool:
        return bool(self._originals)

    def _basis_by_layer(self) -> dict[int, Tensor]:
        result: dict[int, Tensor] = {}
        for layer in self.spectra["layers"]:
            layer_index = int(layer["layer_index"])
            signals = layer["signals"]
            if self.signal not in signals:
                raise ValueError(f"layer {layer_index} has no {self.signal!r} spectrum")
            vectors = signals[self.signal]["eigenvectors"]
            if vectors.ndim != 3 or self.rank > vectors.shape[-1]:
                raise ValueError(
                    f"layer {layer_index} spectrum cannot supply rank {self.rank}"
                )
            if self.rank == vectors.shape[-1]:
                # Make the full-rank endpoint an identity control.  This avoids
                # attributing BF16 rotate/decode roundoff to state projection.
                identity = torch.eye(vectors.shape[-1], dtype=vectors.dtype)
                result[layer_index] = identity.expand(vectors.shape[0], -1, -1).clone()
            else:
                result[layer_index] = vectors[..., : self.rank].contiguous()
        return result

    @staticmethod
    def _wrapper(
        original: Callable[..., tuple[Tensor, Tensor | None]],
        encoder: Tensor,
        decoder: Tensor,
        *,
        layer_index: int,
    ) -> Callable[..., tuple[Tensor, Tensor | None]]:
        def projected_chunk(
            query: Tensor,
            key: Tensor,
            value: Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[Tensor, Tensor | None]:
            initial_state = kwargs.get("initial_state")
            output_final_state = bool(kwargs.get("output_final_state", False))
            if initial_state is not None or output_final_state:
                raise RuntimeError(
                    f"layer {layer_index} projected GDN runtime supports use_cache=False only"
                )
            local_encoder = encoder.to(device=value.device, dtype=value.dtype)
            local_decoder = decoder.to(device=value.device, dtype=value.dtype)
            latent_value = torch.einsum("bthv,hvr->bthr", value, local_encoder)
            latent_output, final_state = original(
                query,
                key,
                latent_value,
                *args,
                **kwargs,
            )
            decoded_output = torch.einsum(
                "bthr,hrv->bthv", latent_output, local_decoder
            )
            return decoded_output.to(value.dtype), final_state

        return projected_chunk

    def install(self) -> tuple[ProjectedGDNRecord, ...]:
        if self.installed:
            raise RuntimeError("projected GDN runtime is already installed")
        layers = qwen35_decoder_layers(self.model)
        bases = self._basis_by_layer()
        records: list[ProjectedGDNRecord] = []
        try:
            for layer_index, basis in sorted(bases.items()):
                if layer_index >= len(layers):
                    raise ValueError(f"spectrum layer {layer_index} is absent from model")
                module = getattr(layers[layer_index], "linear_attn", None)
                if module is None:
                    raise ValueError(f"spectrum layer {layer_index} is not a GDN layer")
                if getattr(module, "_basisserve_projected_gdn", False):
                    raise RuntimeError(f"layer {layer_index} already has a projected runtime")
                value_head_dim = int(getattr(module, "head_v_dim"))
                num_value_heads = int(getattr(module, "num_v_heads"))
                expected = (num_value_heads, value_head_dim, self.rank)
                if tuple(basis.shape) != expected:
                    raise ValueError(
                        f"layer {layer_index} basis has {tuple(basis.shape)}, expected {expected}"
                    )
                target_dtype = self.factor_dtype or module.out_proj.weight.dtype
                encoder = basis.to(
                    device=module.out_proj.weight.device,
                    dtype=target_dtype,
                ).contiguous()
                decoder = encoder.transpose(-1, -2).contiguous()
                original = module.chunk_gated_delta_rule
                self._originals[layer_index] = original
                self._factors[layer_index] = (encoder, decoder)
                module.chunk_gated_delta_rule = self._wrapper(
                    original,
                    encoder,
                    decoder,
                    layer_index=layer_index,
                )
                module._basisserve_projected_gdn = True
                records.append(
                    ProjectedGDNRecord(
                        layer_index=layer_index,
                        rank=self.rank,
                        value_head_dim=value_head_dim,
                        num_value_heads=num_value_heads,
                        signal=self.signal,
                        basis_kind=(
                            "identity_full_rank"
                            if self.rank == value_head_dim
                            else "headwise_pca_prefix"
                        ),
                    )
                )
        except Exception:
            self.restore()
            raise
        self.records = tuple(records)
        if len(self.records) != len(bases):
            self.restore()
            raise RuntimeError("not all projected GDN layers were installed")
        return self.records

    def restore(self) -> None:
        if not self._originals:
            self.records = ()
            self._factors.clear()
            return
        layers = qwen35_decoder_layers(self.model)
        for layer_index, original in self._originals.items():
            module = layers[layer_index].linear_attn
            module.chunk_gated_delta_rule = original
            if hasattr(module, "_basisserve_projected_gdn"):
                delattr(module, "_basisserve_projected_gdn")
        self._originals.clear()
        self._factors.clear()
        self.records = ()

    def __enter__(self) -> "Qwen35ProjectedGDNRuntime":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


__all__ = [
    "ProjectedGDNRecord",
    "Qwen35ProjectedGDNRuntime",
    "SPECTRA_FORMAT",
    "load_qwen35_gdn_spectra",
    "qwen35_decoder_layers",
]
