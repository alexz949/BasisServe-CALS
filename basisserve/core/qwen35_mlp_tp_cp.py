"""Logical-group TP-coupled latent MLPs for Qwen3.5.

The single-GPU reference in this module deliberately preserves the operation
order required by a future tensor-parallel deployment.  For every logical
group ``g`` it computes a local bilinear latent, then sums those local latents:

    s = sum_g (gate_g @ B_g) * (up_g @ C_g)
    y = s @ A.T

Keeping the group-local products explicit prevents a single-GPU experiment
from accidentally introducing cross-group interactions that cannot be
reproduced with one latent all-reduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.core.qwen35_gdn_runtime import qwen35_decoder_layers


FACTOR_FORMAT = "basisserve.qwen35.mlp_tp_latent_factors.v1"
LatentMethod = Literal["cp", "matrix_lr"]


def _as_grouped_factor(
    factor: Tensor,
    *,
    logical_groups: int,
    name: str,
) -> Tensor:
    """Return a factor with shape ``[groups, group_width, rank]``."""

    if logical_groups <= 0:
        raise ValueError("logical_groups must be positive")
    if factor.ndim == 3:
        if int(factor.shape[0]) != logical_groups:
            raise ValueError(
                f"{name} has {factor.shape[0]} groups, expected {logical_groups}"
            )
        return factor
    if factor.ndim != 2:
        raise ValueError(f"{name} must be a matrix or grouped rank-3 tensor")
    intermediate_size, rank = map(int, factor.shape)
    if intermediate_size % logical_groups:
        raise ValueError(
            f"{name} width {intermediate_size} is not divisible by "
            f"{logical_groups} logical groups"
        )
    return factor.reshape(logical_groups, intermediate_size // logical_groups, rank)


def logical_group_bilinear_components(
    gate: Tensor,
    up: Tensor,
    gate_factor: Tensor,
    up_factor: Tensor,
) -> Tensor:
    """Return every logical group's local CP latent.

    Args:
        gate, up: tensors with identical shape ``[..., intermediate_size]``.
        gate_factor, up_factor: tensors with shape
            ``[logical_groups, group_width, rank]``.

    Returns:
        Tensor with shape ``[..., logical_groups, rank]``.  The group axis is
        intentionally retained so callers can emulate different physical TP
        assignments without changing the numerical model.
    """

    if gate.shape != up.shape:
        raise ValueError(
            f"gate and up shapes differ: {tuple(gate.shape)} != {tuple(up.shape)}"
        )
    if gate.ndim < 1:
        raise ValueError("gate and up must have a feature dimension")
    if gate_factor.ndim != 3 or up_factor.ndim != 3:
        raise ValueError("CP factors must have shape [groups, group_width, rank]")
    if gate_factor.shape != up_factor.shape:
        raise ValueError("gate and up CP factor shapes must match")
    groups, group_width, _ = map(int, gate_factor.shape)
    if int(gate.shape[-1]) != groups * group_width:
        raise ValueError(
            f"activation width {gate.shape[-1]} does not match grouped factor "
            f"width {groups * group_width}"
        )
    grouped_gate = gate.reshape(*gate.shape[:-1], groups, group_width)
    grouped_up = up.reshape(*up.shape[:-1], groups, group_width)
    gate_latent = torch.einsum("...gm,gmr->...gr", grouped_gate, gate_factor)
    up_latent = torch.einsum("...gm,gmr->...gr", grouped_up, up_factor)
    return gate_latent * up_latent


def logical_group_bilinear_latent(
    gate: Tensor,
    up: Tensor,
    gate_factor: Tensor,
    up_factor: Tensor,
) -> Tensor:
    """Compute ``sum_g (B_g^T gate_g) * (C_g^T up_g)``."""

    return logical_group_bilinear_components(
        gate,
        up,
        gate_factor,
        up_factor,
    ).sum(dim=-2)


def logical_group_matrix_lr_components(
    hidden: Tensor,
    input_factor: Tensor,
) -> Tensor:
    """Return per-group matrix-LR latents for the communication-matched baseline."""

    if hidden.ndim < 1 or input_factor.ndim != 3:
        raise ValueError("hidden must end in features and input_factor must be grouped")
    groups, group_width, _ = map(int, input_factor.shape)
    if int(hidden.shape[-1]) != groups * group_width:
        raise ValueError(
            f"activation width {hidden.shape[-1]} does not match grouped factor "
            f"width {groups * group_width}"
        )
    grouped = hidden.reshape(*hidden.shape[:-1], groups, group_width)
    return torch.einsum("...gm,gmr->...gr", grouped, input_factor)


class TPCoupledCPDecoder(nn.Module):
    """Decode gate/up activations through logical-group coupled CP factors."""

    def __init__(
        self,
        gate_factor: Tensor,
        up_factor: Tensor,
        output_basis: Tensor,
        *,
        logical_groups: int,
        bias: Tensor | None = None,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        grouped_gate = _as_grouped_factor(
            gate_factor,
            logical_groups=logical_groups,
            name="gate_factor",
        )
        grouped_up = _as_grouped_factor(
            up_factor,
            logical_groups=logical_groups,
            name="up_factor",
        )
        if grouped_gate.shape != grouped_up.shape:
            raise ValueError("gate_factor and up_factor shapes must match")
        if output_basis.ndim != 2:
            raise ValueError("output_basis must have shape [hidden_size, rank]")
        if int(output_basis.shape[1]) != int(grouped_gate.shape[2]):
            raise ValueError("output_basis rank does not match CP factor rank")
        if bias is not None and tuple(bias.shape) != (int(output_basis.shape[0]),):
            raise ValueError("bias shape does not match output width")

        self.logical_groups = int(logical_groups)
        self.gate_factor = nn.Parameter(grouped_gate, requires_grad=trainable)
        self.up_factor = nn.Parameter(grouped_up, requires_grad=trainable)
        self.output_basis = nn.Parameter(output_basis, requires_grad=trainable)
        self.bias = (
            None if bias is None else nn.Parameter(bias, requires_grad=trainable)
        )

    @property
    def rank(self) -> int:
        return int(self.output_basis.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.gate_factor.shape[0] * self.gate_factor.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.output_basis.shape[0])

    def latent_components(self, gate: Tensor, up: Tensor) -> Tensor:
        return logical_group_bilinear_components(
            gate,
            up,
            self.gate_factor,
            self.up_factor,
        )

    def latent(self, gate: Tensor, up: Tensor) -> Tensor:
        return self.latent_components(gate, up).sum(dim=-2)

    def forward(self, gate: Tensor, up: Tensor) -> Tensor:
        return F.linear(self.latent(gate, up), self.output_basis, self.bias)


class GroupedMatrixLRDecoder(nn.Module):
    """Communication-matched matrix low-rank decoder."""

    def __init__(
        self,
        input_factor: Tensor,
        output_basis: Tensor,
        *,
        logical_groups: int,
        bias: Tensor | None = None,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        grouped_input = _as_grouped_factor(
            input_factor,
            logical_groups=logical_groups,
            name="input_factor",
        )
        if output_basis.ndim != 2:
            raise ValueError("output_basis must have shape [hidden_size, rank]")
        if int(output_basis.shape[1]) != int(grouped_input.shape[2]):
            raise ValueError("output_basis rank does not match input factor rank")
        if bias is not None and tuple(bias.shape) != (int(output_basis.shape[0]),):
            raise ValueError("bias shape does not match output width")
        self.logical_groups = int(logical_groups)
        self.input_factor = nn.Parameter(grouped_input, requires_grad=trainable)
        self.output_basis = nn.Parameter(output_basis, requires_grad=trainable)
        self.bias = (
            None if bias is None else nn.Parameter(bias, requires_grad=trainable)
        )

    @property
    def rank(self) -> int:
        return int(self.output_basis.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.input_factor.shape[0] * self.input_factor.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.output_basis.shape[0])

    def latent_components(self, hidden: Tensor) -> Tensor:
        return logical_group_matrix_lr_components(hidden, self.input_factor)

    def latent(self, hidden: Tensor) -> Tensor:
        return self.latent_components(hidden).sum(dim=-2)

    def forward(self, hidden: Tensor) -> Tensor:
        return F.linear(self.latent(hidden), self.output_basis, self.bias)


class Qwen35TPCoupledCPMLP(nn.Module):
    """Drop-in Qwen3.5 MLP using a TP-coupled CP down path."""

    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        act_fn: Callable[[Tensor], Tensor],
        decoder: TPCoupledCPDecoder,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.act_fn = act_fn
        self.decoder = decoder

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.decoder(gate, up)


class Qwen35MatrixLRMLP(nn.Module):
    """Drop-in Qwen3.5 MLP for the communication-matched LR baseline."""

    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        act_fn: Callable[[Tensor], Tensor],
        decoder: GroupedMatrixLRDecoder,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.act_fn = act_fn
        self.decoder = decoder

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.decoder(hidden)


@dataclass(frozen=True)
class OutputPCAOracle:
    """Best rank-R output subspace for a matrix of teacher activations."""

    output_basis: Tensor
    singular_values: Tensor
    relative_mse: float

    def project(self, outputs: Tensor) -> Tensor:
        if outputs.ndim != 2 or int(outputs.shape[1]) != int(self.output_basis.shape[0]):
            raise ValueError("outputs must have shape [tokens, hidden_size]")
        return (outputs @ self.output_basis) @ self.output_basis.transpose(0, 1)


@torch.no_grad()
def fit_output_pca_oracle(outputs: Tensor, rank: int) -> OutputPCAOracle:
    """Fit the exact, uncentered output-PCA oracle for a moderate matrix."""

    if outputs.ndim != 2:
        raise ValueError("outputs must have shape [tokens, hidden_size]")
    maximum_rank = min(map(int, outputs.shape))
    if not 0 < rank <= maximum_rank:
        raise ValueError(f"rank must lie in [1,{maximum_rank}]")
    _, singular_values, vh = torch.linalg.svd(outputs.float(), full_matrices=False)
    basis = vh[:rank].transpose(0, 1).contiguous()
    total = singular_values.square().sum().clamp_min(1e-30)
    discarded = singular_values[rank:].square().sum()
    return OutputPCAOracle(
        output_basis=basis,
        singular_values=singular_values,
        relative_mse=float((discarded / total).item()),
    )


@torch.no_grad()
def output_subspace_relative_mse(outputs: Tensor, output_basis: Tensor) -> float:
    """Measure oracle projection error for a fixed orthonormal basis."""

    if outputs.ndim != 2 or output_basis.ndim != 2:
        raise ValueError("outputs and output_basis must be matrices")
    if int(outputs.shape[1]) != int(output_basis.shape[0]):
        raise ValueError("output widths do not match")
    work = outputs.float()
    basis = output_basis.float()
    residual = work - (work @ basis) @ basis.transpose(0, 1)
    return float(
        (residual.square().sum() / work.square().sum().clamp_min(1e-30)).item()
    )


def load_qwen35_mlp_latent_factors(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != FACTOR_FORMAT
        or int(payload.get("schema_version", -1)) != 1
    ):
        raise ValueError(f"unsupported Qwen3.5 MLP latent factors: {source}")
    return payload


@dataclass(frozen=True)
class MLPLatentRuntimeRecord:
    layer_index: int
    method: LatentMethod
    rank: int
    logical_groups: int
    hidden_size: int
    intermediate_size: int

    @property
    def communication_ratio(self) -> float:
        return self.rank / self.hidden_size

    @property
    def payload_reduction(self) -> float:
        return self.hidden_size / self.rank


class Qwen35MLPLatentRuntime:
    """Temporarily install activation-aware CP or matrix-LR Qwen3.5 MLPs."""

    def __init__(
        self,
        model: nn.Module,
        factors: Mapping[str, Any] | str | Path,
        *,
        factor_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.factors = (
            load_qwen35_mlp_latent_factors(factors)
            if isinstance(factors, (str, Path))
            else dict(factors)
        )
        if (
            self.factors.get("format") != FACTOR_FORMAT
            or int(self.factors.get("schema_version", -1)) != 1
        ):
            raise ValueError("invalid Qwen3.5 MLP latent factor payload")
        method = str(self.factors.get("method"))
        if method not in ("cp", "matrix_lr"):
            raise ValueError(f"unsupported MLP latent method {method!r}")
        self.method: LatentMethod = method  # type: ignore[assignment]
        self.logical_groups = int(self.factors.get("logical_groups", 0))
        if self.logical_groups <= 0:
            raise ValueError("factor payload has no positive logical group count")
        self.rank = int(self.factors.get("rank", 0))
        if self.rank <= 0:
            raise ValueError("factor payload has no positive rank")
        self.factor_dtype = factor_dtype
        self._originals: dict[int, nn.Module] = {}
        self._installed: dict[int, nn.Module] = {}
        self.records: tuple[MLPLatentRuntimeRecord, ...] = ()

    @property
    def installed(self) -> bool:
        return bool(self._originals)

    def _layers_by_index(self) -> dict[int, Mapping[str, Any]]:
        result: dict[int, Mapping[str, Any]] = {}
        for layer in self.factors.get("layers", []):
            layer_index = int(layer["layer_index"])
            if layer_index in result:
                raise ValueError(f"duplicate MLP factors for layer {layer_index}")
            result[layer_index] = layer
        if not result:
            raise ValueError("MLP latent factor payload contains no layers")
        return result

    @staticmethod
    def _tensor(
        value: Any,
        *,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} is not a tensor")
        return value.to(device=device, dtype=dtype).contiguous()

    def install(self) -> tuple[MLPLatentRuntimeRecord, ...]:
        if self.installed:
            raise RuntimeError("Qwen3.5 MLP latent runtime is already installed")
        decoder_layers = qwen35_decoder_layers(self.model)
        records: list[MLPLatentRuntimeRecord] = []
        try:
            for layer_index, layer_factors in sorted(self._layers_by_index().items()):
                if not 0 <= layer_index < len(decoder_layers):
                    raise ValueError(f"factor layer {layer_index} is absent from model")
                layer = decoder_layers[layer_index]
                original = getattr(layer, "mlp", None)
                if not isinstance(original, nn.Module):
                    raise TypeError(f"layer {layer_index} has no MLP module")
                if getattr(original, "_basisserve_mlp_tp_latent", False):
                    raise RuntimeError(f"layer {layer_index} already has a latent MLP")
                gate_proj = getattr(original, "gate_proj", None)
                up_proj = getattr(original, "up_proj", None)
                down_proj = getattr(original, "down_proj", None)
                act_fn = getattr(original, "act_fn", None)
                gate_weight = getattr(gate_proj, "weight", None)
                up_weight = getattr(up_proj, "weight", None)
                down_weight = getattr(down_proj, "weight", None)
                if not all(
                    isinstance(weight, Tensor)
                    for weight in (gate_weight, up_weight, down_weight)
                ) or not callable(act_fn):
                    raise TypeError(f"layer {layer_index} is not a supported gated MLP")
                assert isinstance(gate_weight, Tensor)
                assert isinstance(up_weight, Tensor)
                assert isinstance(down_weight, Tensor)
                intermediate_size = int(gate_weight.shape[0])
                hidden_size = int(down_weight.shape[0])
                if tuple(up_weight.shape) != tuple(gate_weight.shape):
                    raise ValueError(f"layer {layer_index} gate/up shapes differ")
                if tuple(down_weight.shape) != (hidden_size, intermediate_size):
                    raise ValueError(f"layer {layer_index} down projection shape is invalid")
                if intermediate_size % self.logical_groups:
                    raise ValueError(
                        f"layer {layer_index} intermediate width is not divisible by "
                        f"{self.logical_groups}"
                    )
                target_device = down_weight.device
                target_dtype = self.factor_dtype or down_weight.dtype
                output_basis = self._tensor(
                    layer_factors.get("output_basis"),
                    name=f"layer {layer_index} output_basis",
                    device=target_device,
                    dtype=target_dtype,
                )
                expected_basis = (hidden_size, self.rank)
                if tuple(output_basis.shape) != expected_basis:
                    raise ValueError(
                        f"layer {layer_index} output_basis has {tuple(output_basis.shape)}, "
                        f"expected {expected_basis}"
                    )
                bias = getattr(down_proj, "bias", None)
                if isinstance(bias, Tensor):
                    bias = bias.to(device=target_device, dtype=target_dtype)

                if self.method == "cp":
                    gate_factor = self._tensor(
                        layer_factors.get("gate_factor"),
                        name=f"layer {layer_index} gate_factor",
                        device=target_device,
                        dtype=target_dtype,
                    )
                    up_factor = self._tensor(
                        layer_factors.get("up_factor"),
                        name=f"layer {layer_index} up_factor",
                        device=target_device,
                        dtype=target_dtype,
                    )
                    expected_factor = (intermediate_size, self.rank)
                    if tuple(gate_factor.shape) != expected_factor:
                        raise ValueError(
                            f"layer {layer_index} gate_factor has {tuple(gate_factor.shape)}, "
                            f"expected {expected_factor}"
                        )
                    if tuple(up_factor.shape) != expected_factor:
                        raise ValueError(
                            f"layer {layer_index} up_factor has {tuple(up_factor.shape)}, "
                            f"expected {expected_factor}"
                        )
                    decoder = TPCoupledCPDecoder(
                        gate_factor,
                        up_factor,
                        output_basis,
                        logical_groups=self.logical_groups,
                        bias=bias,
                    )
                    replacement: nn.Module = Qwen35TPCoupledCPMLP(
                        gate_proj,
                        up_proj,
                        act_fn,
                        decoder,
                    )
                else:
                    input_factor = self._tensor(
                        layer_factors.get("input_factor"),
                        name=f"layer {layer_index} input_factor",
                        device=target_device,
                        dtype=target_dtype,
                    )
                    expected_factor = (intermediate_size, self.rank)
                    if tuple(input_factor.shape) != expected_factor:
                        raise ValueError(
                            f"layer {layer_index} input_factor has {tuple(input_factor.shape)}, "
                            f"expected {expected_factor}"
                        )
                    decoder = GroupedMatrixLRDecoder(
                        input_factor,
                        output_basis,
                        logical_groups=self.logical_groups,
                        bias=bias,
                    )
                    replacement = Qwen35MatrixLRMLP(
                        gate_proj,
                        up_proj,
                        act_fn,
                        decoder,
                    )

                replacement.train(original.training)
                replacement._basisserve_mlp_tp_latent = True
                self._originals[layer_index] = original
                self._installed[layer_index] = replacement
                layer.mlp = replacement
                records.append(
                    MLPLatentRuntimeRecord(
                        layer_index=layer_index,
                        method=self.method,
                        rank=self.rank,
                        logical_groups=self.logical_groups,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                    )
                )
        except Exception:
            self.restore()
            raise
        self.records = tuple(records)
        return self.records

    def restore(self) -> None:
        if not self._originals:
            self._installed.clear()
            self.records = ()
            return
        decoder_layers = qwen35_decoder_layers(self.model)
        for layer_index, original in self._originals.items():
            decoder_layers[layer_index].mlp = original
        self._originals.clear()
        self._installed.clear()
        self.records = ()

    def __enter__(self) -> "Qwen35MLPLatentRuntime":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


__all__ = [
    "FACTOR_FORMAT",
    "GroupedMatrixLRDecoder",
    "LatentMethod",
    "MLPLatentRuntimeRecord",
    "OutputPCAOracle",
    "Qwen35MLPLatentRuntime",
    "Qwen35MatrixLRMLP",
    "Qwen35TPCoupledCPMLP",
    "TPCoupledCPDecoder",
    "fit_output_pca_oracle",
    "load_qwen35_mlp_latent_factors",
    "logical_group_bilinear_components",
    "logical_group_bilinear_latent",
    "logical_group_matrix_lr_components",
    "output_subspace_relative_mse",
]
