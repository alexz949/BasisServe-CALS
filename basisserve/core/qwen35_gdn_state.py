"""Projected recurrent-state algebra for Qwen3.5 Gated DeltaNet.

Qwen3.5 stores one recurrent matrix ``S`` with shape ``[d_k, d_v]`` per
value head.  A value-side encoder ``E`` gives a closed projected recurrence

    Z = S E,

because both the state update and readout commute with right multiplication
by ``E``.  Only the current head output needs a decoder before Qwen3.5's
token-dependent gated RMSNorm.

This module is deliberately independent of Transformers and serving caches.
It supplies correctness-first tensor references used by offline fitting,
unit tests, and later runtime adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class Qwen35GDNGeometry:
    num_value_heads: int
    num_key_heads: int
    key_head_dim: int
    value_head_dim: int

    def __post_init__(self) -> None:
        values = (
            self.num_value_heads,
            self.num_key_heads,
            self.key_head_dim,
            self.value_head_dim,
        )
        if any(int(value) <= 0 for value in values):
            raise ValueError("GDN geometry values must be positive")
        if self.num_value_heads % self.num_key_heads:
            raise ValueError("value-head count must be divisible by key-head count")

    @property
    def query_key_repeat(self) -> int:
        return self.num_value_heads // self.num_key_heads

    @property
    def dense_state_elements(self) -> int:
        return self.num_value_heads * self.key_head_dim * self.value_head_dim

    @classmethod
    def from_config(cls, config: Any) -> "Qwen35GDNGeometry":
        text = getattr(config, "text_config", config)
        return cls(
            num_value_heads=int(getattr(text, "linear_num_value_heads")),
            num_key_heads=int(getattr(text, "linear_num_key_heads")),
            key_head_dim=int(getattr(text, "linear_key_head_dim")),
            value_head_dim=int(getattr(text, "linear_value_head_dim")),
        )


@dataclass(frozen=True)
class HeadwiseStateCodec:
    """Value-side state encoder and current-output decoder.

    Factors may be shared across heads (``[D,R]`` and ``[R,D]``) or distinct
    per head (``[H,D,R]`` and ``[H,R,D]``).
    """

    encoder: Tensor
    decoder: Tensor

    @property
    def rank(self) -> int:
        return int(self.encoder.shape[-1])

    @property
    def value_dim(self) -> int:
        return int(self.encoder.shape[-2])

    def validate(self, *, num_heads: int | None = None) -> None:
        if self.encoder.ndim not in (2, 3):
            raise ValueError("state encoder must have shape [D,R] or [H,D,R]")
        if self.decoder.ndim != self.encoder.ndim:
            raise ValueError("state encoder and decoder must have equal rank")
        if self.encoder.shape[-1] != self.decoder.shape[-2]:
            raise ValueError("state codec latent dimensions differ")
        if self.encoder.shape[-2] != self.decoder.shape[-1]:
            raise ValueError("state codec value dimensions differ")
        if self.encoder.ndim == 3:
            if self.encoder.shape[0] != self.decoder.shape[0]:
                raise ValueError("state codec head counts differ")
            if num_heads is not None and self.encoder.shape[0] != num_heads:
                raise ValueError("state codec head count does not match inputs")
        if not torch.isfinite(self.encoder).all() or not torch.isfinite(self.decoder).all():
            raise ValueError("state codec contains non-finite values")


def identity_state_codec(
    value_dim: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> HeadwiseStateCodec:
    identity = torch.eye(value_dim, device=device, dtype=dtype)
    return HeadwiseStateCodec(identity, identity)


def _l2_normalize(value: Tensor, *, eps: float = 1e-6) -> Tensor:
    return value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + eps)


def _validate_recurrence_inputs(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    log_decay: Tensor,
    beta: Tensor,
) -> tuple[int, int, int, int, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B,T,H,D]")
    if query.shape != key.shape:
        raise ValueError("query and key shapes differ")
    if query.shape[:3] != value.shape[:3]:
        raise ValueError("query/key and value batch-token-head shapes differ")
    expected_scalars = tuple(query.shape[:3])
    if tuple(log_decay.shape) != expected_scalars or tuple(beta.shape) != expected_scalars:
        raise ValueError("log_decay and beta must have shape [B,T,H]")
    tensors = (query, key, value, log_decay, beta)
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("recurrent inputs contain non-finite values")
    batch, tokens, heads, key_dim = map(int, query.shape)
    value_dim = int(value.shape[-1])
    return batch, tokens, heads, key_dim, value_dim


def gated_delta_recurrent_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    log_decay: Tensor,
    beta: Tensor,
    *,
    initial_state: Tensor | None = None,
    normalize_query_key: bool = True,
    work_dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Sequential reference matching Qwen3.5's recurrent GDN update."""

    batch, tokens, heads, key_dim, value_dim = _validate_recurrence_inputs(
        query, key, value, log_decay, beta
    )
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("recurrent work dtype must be float32 or float64")
    output_dtype = query.dtype
    q = query.to(work_dtype)
    k = key.to(work_dtype)
    v = value.to(work_dtype)
    decay = log_decay.to(work_dtype)
    step = beta.to(work_dtype)
    if normalize_query_key:
        q = _l2_normalize(q)
        k = _l2_normalize(k)
    q = q * (1.0 / math.sqrt(key_dim))
    if initial_state is None:
        state = torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            device=query.device,
            dtype=work_dtype,
        )
    else:
        expected = (batch, heads, key_dim, value_dim)
        if tuple(initial_state.shape) != expected:
            raise ValueError(
                f"initial state has shape {tuple(initial_state.shape)}, expected {expected}"
            )
        state = initial_state.to(device=query.device, dtype=work_dtype).clone()

    outputs: list[Tensor] = []
    for token in range(tokens):
        q_t = q[:, token]
        k_t = k[:, token]
        v_t = v[:, token]
        state.mul_(decay[:, token].exp().unsqueeze(-1).unsqueeze(-1))
        memory = torch.einsum("bhkv,bhk->bhv", state, k_t)
        delta = (v_t - memory) * step[:, token].unsqueeze(-1)
        state.add_(k_t.unsqueeze(-1) * delta.unsqueeze(-2))
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q_t))
    output = torch.stack(outputs, dim=1).to(output_dtype)
    return output, state


def _encode_values(value: Tensor, encoder: Tensor) -> Tensor:
    if encoder.ndim == 2:
        return value @ encoder
    return torch.einsum("bthv,hvr->bthr", value, encoder)


def _decode_values(value: Tensor, decoder: Tensor) -> Tensor:
    if decoder.ndim == 2:
        return value @ decoder
    return torch.einsum("bthr,hrv->bthv", value, decoder)


def project_dense_state(state: Tensor, encoder: Tensor) -> Tensor:
    if state.ndim != 4:
        raise ValueError("dense state must have shape [B,H,K,V]")
    if encoder.ndim == 2:
        return state @ encoder
    if encoder.ndim != 3 or state.shape[1] != encoder.shape[0]:
        raise ValueError("headwise encoder does not match dense state")
    return torch.einsum("bhkv,hvr->bhkr", state, encoder)


def projected_gated_delta_recurrent_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    log_decay: Tensor,
    beta: Tensor,
    codec: HeadwiseStateCodec,
    *,
    initial_projected_state: Tensor | None = None,
    normalize_query_key: bool = True,
    work_dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the closed value-projected recurrence and decode current outputs.

    Returns ``(decoded_output, latent_output, final_projected_state)``.
    """

    _, _, heads, _, value_dim = _validate_recurrence_inputs(
        query, key, value, log_decay, beta
    )
    codec.validate(num_heads=heads)
    if codec.value_dim != value_dim:
        raise ValueError("state codec value dimension does not match inputs")
    encoder = codec.encoder.to(device=value.device, dtype=work_dtype)
    decoder = codec.decoder.to(device=value.device, dtype=work_dtype)
    projected_value = _encode_values(value.to(work_dtype), encoder)
    latent, projected_state = gated_delta_recurrent_reference(
        query,
        key,
        projected_value,
        log_decay,
        beta,
        initial_state=initial_projected_state,
        normalize_query_key=normalize_query_key,
        work_dtype=work_dtype,
    )
    decoded = _decode_values(latent.to(work_dtype), decoder).to(value.dtype)
    return decoded, latent, projected_state


def qwen35_gated_rmsnorm(
    hidden_states: Tensor,
    gate: Tensor,
    weight: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Framework-independent reference for ``Qwen3_5RMSNormGated``."""

    if hidden_states.shape != gate.shape:
        raise ValueError("hidden states and gate must have equal shapes")
    if hidden_states.shape[-1] != weight.numel():
        raise ValueError("gated RMSNorm weight has the wrong width")
    input_dtype = hidden_states.dtype
    work = hidden_states.float()
    variance = work.square().mean(dim=-1, keepdim=True)
    normalized = work * torch.rsqrt(variance + float(eps))
    normalized = normalized.to(input_dtype) * weight.to(
        device=hidden_states.device, dtype=input_dtype
    )
    return (normalized * F.silu(gate.float())).to(input_dtype)


class HeadwiseSecondMoment:
    """Streaming per-head sums and unnormalized second moments."""

    def __init__(
        self,
        num_heads: int,
        width: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if num_heads <= 0 or width <= 0:
            raise ValueError("moment geometry must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("moment storage dtype must be float32 or float64")
        self.num_heads = int(num_heads)
        self.width = int(width)
        self.device = torch.device(device)
        self.dtype = dtype
        self.rows = 0
        self.sum = torch.zeros(
            self.num_heads, self.width, dtype=dtype, device=self.device
        )
        self.gram = torch.zeros(
            self.num_heads,
            self.width,
            self.width,
            dtype=dtype,
            device=self.device,
        )

    @torch.no_grad()
    def update(self, rows: Tensor) -> None:
        if rows.ndim != 3 or tuple(rows.shape[1:]) != (
            self.num_heads,
            self.width,
        ):
            raise ValueError(
                f"moment rows must have shape [N,{self.num_heads},{self.width}]"
            )
        if rows.shape[0] == 0:
            return
        work = rows.detach().to(dtype=torch.float32)
        if not torch.isfinite(work).all():
            raise ValueError("moment rows contain non-finite values")
        self.sum.add_(work.sum(dim=0).to(device=self.device, dtype=self.dtype))
        gram = torch.einsum("nhd,nhe->hde", work, work)
        self.gram.add_(gram.to(device=self.device, dtype=self.dtype))
        self.rows += int(rows.shape[0])

    def second_moment(self) -> Tensor:
        if self.rows <= 0:
            raise ValueError("cannot normalize empty moments")
        result = self.gram / float(self.rows)
        return 0.5 * (result + result.transpose(-1, -2))

    def covariance(self) -> Tensor:
        mean = self.mean()
        result = self.second_moment() - torch.einsum("hd,he->hde", mean, mean)
        return 0.5 * (result + result.transpose(-1, -2))

    def mean(self) -> Tensor:
        if self.rows <= 0:
            raise ValueError("cannot normalize empty moments")
        return self.sum / float(self.rows)

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_heads": self.num_heads,
            "width": self.width,
            "storage_dtype": str(self.dtype),
            "rows": self.rows,
            "sum": self.sum.detach().cpu(),
            "gram": self.gram.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "HeadwiseSecondMoment":
        result = cls(int(payload["num_heads"]), int(payload["width"]))
        result.rows = int(payload["rows"])
        result.sum.copy_(payload["sum"].to(dtype=torch.float64, device="cpu"))
        result.gram.copy_(payload["gram"].to(dtype=torch.float64, device="cpu"))
        return result


@dataclass(frozen=True)
class NestedHeadwisePCA:
    eigenvalues: Tensor
    eigenvectors: Tensor
    centered: bool

    @property
    def maximum_rank(self) -> int:
        return int(self.eigenvalues.shape[-1])

    def codec(self, rank: int, *, dtype: torch.dtype = torch.float32) -> HeadwiseStateCodec:
        if not 0 < rank <= self.maximum_rank:
            raise ValueError(f"rank must lie in [1,{self.maximum_rank}]")
        encoder = self.eigenvectors[..., :rank].to(dtype=dtype).contiguous()
        decoder = encoder.transpose(-1, -2).contiguous()
        result = HeadwiseStateCodec(encoder=encoder, decoder=decoder)
        result.validate(num_heads=int(encoder.shape[0]))
        return result

    def retained_fraction(self, rank: int) -> Tensor:
        if not 0 < rank <= self.maximum_rank:
            raise ValueError(f"rank must lie in [1,{self.maximum_rank}]")
        energy = self.eigenvalues.clamp_min(0)
        return energy[..., :rank].sum(-1) / energy.sum(-1).clamp_min(
            torch.finfo(energy.dtype).tiny
        )


@torch.no_grad()
def fit_nested_headwise_pca(
    moments: HeadwiseSecondMoment,
    *,
    centered: bool,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> NestedHeadwisePCA:
    matrix = moments.covariance() if centered else moments.second_moment()
    if dtype is not None and dtype not in (torch.float32, torch.float64):
        raise ValueError("PCA work dtype must be float32 or float64")
    matrix = matrix.to(
        device=matrix.device if device is None else device,
        dtype=matrix.dtype if dtype is None else dtype,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    order = torch.arange(
        eigenvalues.shape[-1] - 1,
        -1,
        -1,
        device=eigenvalues.device,
    )
    eigenvalues = eigenvalues.index_select(-1, order).clamp_min(0).contiguous()
    eigenvectors = eigenvectors.index_select(-1, order).contiguous()
    return NestedHeadwisePCA(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        centered=bool(centered),
    )


__all__ = [
    "HeadwiseSecondMoment",
    "HeadwiseStateCodec",
    "NestedHeadwisePCA",
    "Qwen35GDNGeometry",
    "fit_nested_headwise_pca",
    "gated_delta_recurrent_reference",
    "identity_state_codec",
    "project_dense_state",
    "projected_gated_delta_recurrent_reference",
    "qwen35_gated_rmsnorm",
]
