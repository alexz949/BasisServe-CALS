"""Global activation-aware SVD for Qwen3.5 TP output AllReduce."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.core.qwen35_gdn_runtime import qwen35_decoder_layers


FACTOR_FORMAT = "basisserve.qwen35.global_aa_svd_allreduce_factors.v1"


def _canonicalize_columns(matrix: Tensor) -> Tensor:
    pivots = matrix.abs().argmax(dim=0)
    columns = torch.arange(matrix.shape[1], device=matrix.device)
    signs = matrix[pivots, columns].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return matrix * signs.unsqueeze(0)


def _relative_output_mse(weight: Tensor, approximation: Tensor, covariance: Tensor) -> float:
    residual = weight - approximation
    numerator = ((residual @ covariance) * residual).sum().clamp_min(0)
    denominator = ((weight @ covariance) * weight).sum().clamp_min(
        torch.finfo(weight.dtype).tiny
    )
    return float(numerator / denominator)


@dataclass(frozen=True)
class GlobalAASVDFactors:
    """One global common-code encoder and replicated output decoder."""

    input_factor: Tensor
    output_basis: Tensor
    singular_values: Tensor
    metrics: dict[str, Any]


@torch.no_grad()
def fit_global_activation_aware_svd(
    weight: Tensor,
    fit_second_moment: Tensor,
    heldout_second_moment: Tensor,
    *,
    rank: int,
    tp_size: int,
    covariance_damping: float = 1.0e-5,
    work_dtype: torch.dtype = torch.float32,
    factor_dtype: torch.dtype = torch.float16,
) -> GlobalAASVDFactors:
    """Fit the optimal damped activation-weighted global rank subspace.

    For ``C_lambda = C + lambda * trace(C) / H * I``, the leading
    eigenspace of ``W C_lambda W.T`` is the optimal output subspace.  The
    balanced factors reconstruct ``U U.T W`` and therefore implement

        z = AllReduce_s(x_s @ A_s),  y = z @ D.T.
    """

    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("Qwen3.5 output weight must be a square matrix")
    width = int(weight.shape[0])
    if tuple(fit_second_moment.shape) != (width, width) or tuple(
        heldout_second_moment.shape
    ) != (width, width):
        raise ValueError("activation moments do not match the output weight")
    if not 0 < rank <= width:
        raise ValueError("global rank must lie within the output width")
    if tp_size <= 1 or width % tp_size:
        raise ValueError("output width must divide across a nontrivial TP size")
    if covariance_damping < 0:
        raise ValueError("covariance damping must be nonnegative")
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("work dtype must be float32 or float64")
    if factor_dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError("unsupported factor dtype")

    device = weight.device
    work_weight = weight.detach().to(device=device, dtype=work_dtype)
    fit_raw = fit_second_moment.to(device=device, dtype=work_dtype)
    fit_raw = 0.5 * (fit_raw + fit_raw.T)
    heldout = heldout_second_moment.to(device=device, dtype=work_dtype)
    heldout = 0.5 * (heldout + heldout.T)
    if not all(
        torch.isfinite(value).all() for value in (work_weight, fit_raw, heldout)
    ):
        raise FloatingPointError("weight and activation moments must be finite")

    trace_scale = float(torch.trace(fit_raw)) / width
    absolute_damping = covariance_damping * trace_scale
    damped = fit_raw.clone()
    if absolute_damping:
        damped.diagonal().add_(absolute_damping)

    output_gram = work_weight @ damped @ work_weight.T
    output_gram = 0.5 * (output_gram + output_gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(output_gram)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    retained_eigenvalues = eigenvalues.index_select(0, order).clamp_min(0)
    output_subspace = _canonicalize_columns(eigenvectors.index_select(1, order))
    singular_values = retained_eigenvalues.sqrt()
    balance = singular_values.sqrt().clamp_min(torch.finfo(work_dtype).tiny)
    output_basis = output_subspace * balance.unsqueeze(0)
    input_factor = (work_weight.T @ output_subspace) / balance.unsqueeze(0)
    approximation = output_basis @ input_factor.T

    fit_damped_mse = _relative_output_mse(work_weight, approximation, damped)
    fit_raw_mse = _relative_output_mse(work_weight, approximation, fit_raw)
    heldout_mse = _relative_output_mse(work_weight, approximation, heldout)
    stored_input = input_factor.to(dtype=factor_dtype)
    stored_output = output_basis.to(dtype=factor_dtype)
    quantized = stored_output.to(work_dtype) @ stored_input.to(work_dtype).T
    quantized_fit_raw_mse = _relative_output_mse(work_weight, quantized, fit_raw)
    quantized_heldout_mse = _relative_output_mse(work_weight, quantized, heldout)

    dense_ring_elements = 2.0 * (tp_size - 1) / tp_size * width
    compressed_ring_elements = 2.0 * (tp_size - 1) / tp_size * rank
    return GlobalAASVDFactors(
        input_factor=stored_input.cpu().contiguous(),
        output_basis=stored_output.cpu().contiguous(),
        singular_values=singular_values.float().cpu().contiguous(),
        metrics={
            "algorithm": "global_activation_aware_svd",
            "collective": "common_code_allreduce",
            "input_width": width,
            "output_width": width,
            "global_rank": rank,
            "tp_size": tp_size,
            "local_input_width": width // tp_size,
            "covariance_damping": covariance_damping,
            "absolute_covariance_damping": absolute_damping,
            "fit_damped_relative_output_mse": fit_damped_mse,
            "fit_raw_relative_output_mse": fit_raw_mse,
            "heldout_relative_output_mse": heldout_mse,
            "quantized_fit_raw_relative_output_mse": quantized_fit_raw_mse,
            "quantized_heldout_relative_output_mse": quantized_heldout_mse,
            "dense_allreduce_ring_elements_per_token_per_rank": dense_ring_elements,
            "compressed_allreduce_ring_elements_per_token_per_rank": (
                compressed_ring_elements
            ),
            "communication_fraction_of_dense_allreduce": rank / width,
            "communication_reduction_fraction": 1.0 - rank / width,
        },
    )


def load_qwen35_global_aa_svd_factors(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != FACTOR_FORMAT
        or int(payload.get("schema_version", -1)) != 1
    ):
        raise ValueError(f"unsupported Qwen3.5 global AA-SVD factors: {source}")
    return payload


class Qwen35GlobalAASVDOutput(nn.Module):
    """Single-process TP8 simulation of common-code AllReduce and decode."""

    def __init__(
        self,
        input_factor: Tensor,
        output_basis: Tensor,
        *,
        tp_size: int,
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        if input_factor.ndim != 2 or output_basis.ndim != 2:
            raise ValueError("global AA-SVD factors must be matrices")
        input_width, rank = map(int, input_factor.shape)
        if int(output_basis.shape[1]) != rank:
            raise ValueError("global AA-SVD factors use different ranks")
        if tp_size <= 1 or input_width % tp_size:
            raise ValueError("input width must divide across TP sources")
        output_width = int(output_basis.shape[0])
        if bias is not None and tuple(bias.shape) != (output_width,):
            raise ValueError("bias shape disagrees with output width")
        local_width = input_width // tp_size
        grouped = input_factor.reshape(tp_size, local_width, rank)
        self.local_projection_weight = nn.Parameter(
            grouped.permute(0, 2, 1).contiguous(),
            requires_grad=False,
        )
        self.output_basis_weight = nn.Parameter(
            output_basis.detach().contiguous(),
            requires_grad=False,
        )
        self.bias = (
            None
            if bias is None
            else nn.Parameter(bias.detach().contiguous(), requires_grad=False)
        )
        self.tp_size = tp_size
        self.local_width = local_width

    @property
    def rank(self) -> int:
        return int(self.output_basis_weight.shape[1])

    @property
    def in_features(self) -> int:
        return self.tp_size * self.local_width

    @property
    def out_features(self) -> int:
        return int(self.output_basis_weight.shape[0])

    def latent_components(self, hidden_states: Tensor) -> Tensor:
        if int(hidden_states.shape[-1]) != self.in_features:
            raise ValueError(
                f"expected input width {self.in_features}, got {hidden_states.shape[-1]}"
            )
        grouped = hidden_states.reshape(
            *hidden_states.shape[:-1], self.tp_size, self.local_width
        )
        return torch.einsum(
            "...si,sri->...sr",
            grouped,
            self.local_projection_weight,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if torch.is_grad_enabled() and hidden_states.requires_grad:
            raise RuntimeError("Qwen35GlobalAASVDOutput is inference-only")
        reduced = self.latent_components(hidden_states).sum(dim=-2)
        return F.linear(reduced, self.output_basis_weight, self.bias)


@dataclass(frozen=True)
class Qwen35GlobalAASVDRecord:
    layer_index: int
    block_type: str
    tp_size: int
    input_width: int
    global_rank: int


class Qwen35GlobalAASVDRuntime:
    """Install global AA-SVD factors on Qwen3.5 GDN and full-attention wires."""

    def __init__(
        self,
        model: nn.Module,
        factors: Mapping[str, Any] | str | Path,
        *,
        factor_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.factors = (
            load_qwen35_global_aa_svd_factors(factors)
            if isinstance(factors, (str, Path))
            else dict(factors)
        )
        if (
            self.factors.get("format") != FACTOR_FORMAT
            or int(self.factors.get("schema_version", -1)) != 1
        ):
            raise ValueError("invalid Qwen3.5 global AA-SVD factor payload")
        self.factor_dtype = factor_dtype
        self._originals: dict[int, tuple[nn.Module, str, nn.Module]] = {}
        self._installed: dict[int, Qwen35GlobalAASVDOutput] = {}
        self.records: tuple[Qwen35GlobalAASVDRecord, ...] = ()

    @property
    def installed(self) -> bool:
        return bool(self._originals)

    def _layers_by_index(self) -> dict[int, Mapping[str, Any]]:
        result: dict[int, Mapping[str, Any]] = {}
        for layer in self.factors.get("layers", ()):
            index = int(layer["layer_index"])
            if index in result:
                raise ValueError(f"duplicate global AA-SVD layer {index}")
            result[index] = layer
        if not result:
            raise ValueError("global AA-SVD factor bank contains no layers")
        return result

    @staticmethod
    def _projection(layer: nn.Module, block_type: str) -> tuple[nn.Module, str]:
        if block_type == "linear_attention":
            return layer.linear_attn, "out_proj"
        if block_type == "full_attention":
            return layer.self_attn, "o_proj"
        raise ValueError(f"unsupported Qwen3.5 block type {block_type!r}")

    def install(self) -> tuple[Qwen35GlobalAASVDRecord, ...]:
        if self.installed:
            raise RuntimeError("Qwen3.5 global AA-SVD runtime is already installed")
        decoder_layers = qwen35_decoder_layers(self.model)
        tp_size = int(self.factors.get("tp_size", 0))
        if tp_size <= 1:
            raise ValueError("global AA-SVD payload has no valid TP size")
        records = []
        try:
            for layer_index, layer_factors in sorted(self._layers_by_index().items()):
                if not 0 <= layer_index < len(decoder_layers):
                    raise ValueError(f"global AA-SVD layer {layer_index} is absent")
                block_type = str(layer_factors["block_type"])
                parent, attribute = self._projection(
                    decoder_layers[layer_index], block_type
                )
                original = getattr(parent, attribute)
                weight = getattr(original, "weight", None)
                if not isinstance(weight, Tensor) or weight.ndim != 2:
                    raise TypeError(f"layer {layer_index} projection has no matrix weight")
                input_factor = layer_factors["input_factor"]
                output_basis = layer_factors["output_basis"]
                if not isinstance(input_factor, Tensor) or not isinstance(
                    output_basis, Tensor
                ):
                    raise TypeError("global AA-SVD factors must be tensors")
                expected_input = (int(weight.shape[1]), int(output_basis.shape[1]))
                if tuple(input_factor.shape) != expected_input:
                    raise ValueError(f"layer {layer_index} input factor is malformed")
                if int(output_basis.shape[0]) != int(weight.shape[0]):
                    raise ValueError(f"layer {layer_index} output basis is malformed")
                dtype = self.factor_dtype or weight.dtype
                bias = getattr(original, "bias", None)
                replacement = Qwen35GlobalAASVDOutput(
                    input_factor.to(device=weight.device, dtype=dtype),
                    output_basis.to(device=weight.device, dtype=dtype),
                    tp_size=tp_size,
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
                    Qwen35GlobalAASVDRecord(
                        layer_index=layer_index,
                        block_type=block_type,
                        tp_size=tp_size,
                        input_width=replacement.in_features,
                        global_rank=replacement.rank,
                    )
                )
        except Exception:
            self.restore()
            raise
        self.records = tuple(records)
        return self.records

    def restore(self) -> None:
        for parent, attribute, original in self._originals.values():
            setattr(parent, attribute, original)
        self._originals.clear()
        self._installed.clear()
        self.records = ()

    def __enter__(self) -> "Qwen35GlobalAASVDRuntime":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


__all__ = [
    "FACTOR_FORMAT",
    "GlobalAASVDFactors",
    "Qwen35GlobalAASVDOutput",
    "Qwen35GlobalAASVDRecord",
    "Qwen35GlobalAASVDRuntime",
    "fit_global_activation_aware_svd",
    "load_qwen35_global_aa_svd_factors",
]
