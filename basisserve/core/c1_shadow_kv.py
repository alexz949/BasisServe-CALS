"""Reference shadow-Key quantization and transactional C1 KV cache.

The cache keeps target-computed C1 Values as the only committed Value state.
Drafting reads a quantized view of committed post-RoPE Keys, optionally
replacing a recent suffix with exact Keys.  Target verification never mutates
committed state until an explicit prefix commit.

This module is a correctness oracle.  Logical four-bit accounting assumes a
packed production representation, while the reference implementation stores
four-bit integers in an ``int8`` tensor.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterator, Literal

import torch
from torch import Tensor
from transformers.cache_utils import Cache, CacheLayerMixin


RuntimeMode = Literal["idle", "target_prefill", "draft", "target_verify"]


@dataclass(frozen=True)
class ShadowKeyConfig:
    bits: int
    group_size: int
    recent_exact_window: int
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.bits not in (4, 8, 16):
            raise ValueError("shadow Key bits must be 4, 8, or 16")
        if self.group_size <= 0:
            raise ValueError("shadow Key group size must be positive")
        if self.recent_exact_window < 0:
            raise ValueError("recent exact window must be non-negative")
        if not self.symmetric:
            raise NotImplementedError(
                "only symmetric shadow-Key quantization is supported"
            )


@dataclass(frozen=True)
class QuantizedKeys:
    """One reference quantized tensor and the scales needed to decode it."""

    values: Tensor
    scales: Tensor | None
    bits: int
    group_size: int
    original_shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return math.prod(self.original_shape)

    def logical_storage_bytes(self) -> int:
        if self.bits == 16:
            return self.values.numel() * self.values.element_size()
        value_bytes = (self.numel * self.bits + 7) // 8
        scale_bytes = (
            0
            if self.scales is None
            else self.scales.numel() * self.scales.element_size()
        )
        return value_bytes + scale_bytes

    def physical_storage_bytes(self) -> int:
        value_bytes = self.values.numel() * self.values.element_size()
        scale_bytes = (
            0
            if self.scales is None
            else self.scales.numel() * self.scales.element_size()
        )
        return value_bytes + scale_bytes


def quantize(keys: Tensor, config: ShadowKeyConfig) -> QuantizedKeys:
    """Quantize floating post-RoPE Keys along their final/head dimension."""

    if not keys.is_floating_point():
        raise TypeError("shadow Keys must be floating point")
    if keys.ndim < 1 or int(keys.shape[-1]) <= 0:
        raise ValueError("shadow Keys must have a non-empty head dimension")
    head_dim = int(keys.shape[-1])
    if head_dim % config.group_size:
        raise ValueError(
            "Key head dimension must be divisible by shadow group size: "
            f"{head_dim} vs {config.group_size}"
        )
    if config.bits == 16:
        return QuantizedKeys(
            values=keys,
            scales=None,
            bits=16,
            group_size=config.group_size,
            original_shape=tuple(keys.shape),
        )

    groups = keys.reshape(
        *keys.shape[:-1], head_dim // config.group_size, config.group_size
    )
    maximum = groups.abs().amax(dim=-1)
    quantization_maximum = (1 << (config.bits - 1)) - 1
    scales = maximum / quantization_maximum
    safe_scales = torch.where(maximum == 0, torch.ones_like(scales), scales)
    values = torch.round(groups / safe_scales.unsqueeze(-1)).clamp(
        -quantization_maximum,
        quantization_maximum,
    )
    return QuantizedKeys(
        values=values.to(torch.int8).reshape(keys.shape).contiguous(),
        scales=scales.contiguous(),
        bits=config.bits,
        group_size=config.group_size,
        original_shape=tuple(keys.shape),
    )


def dequantize(
    quantized_keys: QuantizedKeys,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Tensor:
    """Materialize reference Keys in the requested compute dtype/device."""

    target_device = torch.device(device)
    if quantized_keys.bits == 16:
        return quantized_keys.values.to(device=target_device, dtype=dtype)
    if quantized_keys.scales is None:
        raise ValueError("quantized shadow Keys are missing scales")
    head_dim = quantized_keys.original_shape[-1]
    values = quantized_keys.values.to(device=target_device, dtype=dtype).reshape(
        *quantized_keys.original_shape[:-1],
        head_dim // quantized_keys.group_size,
        quantized_keys.group_size,
    )
    scales = quantized_keys.scales.to(device=target_device, dtype=dtype)
    return (values * scales.unsqueeze(-1)).reshape(quantized_keys.original_shape)


def logical_storage_bytes(quantized_keys: QuantizedKeys) -> int:
    return quantized_keys.logical_storage_bytes()


def physical_storage_bytes(quantized_keys: QuantizedKeys) -> int:
    return quantized_keys.physical_storage_bytes()


def _tensor_bytes(value: Tensor | None) -> int:
    return 0 if value is None else value.numel() * value.element_size()


def _append_sequence(previous: Tensor | None, current: Tensor) -> Tensor:
    detached = current.detach()
    return detached if previous is None else torch.cat((previous, detached), dim=-2)


def _sequence_length(value: Tensor | None) -> int:
    return 0 if value is None else int(value.shape[-2])


class C1ShadowCacheLayer(CacheLayerMixin):
    """Transactional state for one attention layer."""

    supports_early_init = False
    is_sliding = False

    def __init__(self, config: ShadowKeyConfig) -> None:
        super().__init__()
        self.config = config
        self.runtime_mode: RuntimeMode = "idle"
        self.exact_key_committed: Tensor | None = None
        self.shadow_key_committed: QuantizedKeys | None = None
        self.c1_value_committed: Tensor | None = None
        self.draft_key_provisional: Tensor | None = None
        self.draft_c1_value_provisional: Tensor | None = None
        self.target_key_pending: Tensor | None = None
        self.target_c1_value_pending: Tensor | None = None

    def lazy_initialization(self, key_states: Tensor, value_states: Tensor) -> None:
        self._validate_update(key_states, value_states)
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    @property
    def committed_length(self) -> int:
        return _sequence_length(self.exact_key_committed)

    @property
    def draft_length(self) -> int:
        return _sequence_length(self.draft_key_provisional)

    @property
    def verify_length(self) -> int:
        return _sequence_length(self.target_key_pending)

    def _validate_update(self, key_states: Tensor, value_states: Tensor) -> None:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError(
                "C1 shadow cache expects K/V shaped [batch, heads, tokens, dim]"
            )
        if tuple(key_states.shape[:-1]) != tuple(value_states.shape[:-1]):
            raise ValueError(
                "C1 shadow cache K/V batch, head, and token axes must match"
            )
        if not key_states.is_floating_point() or not value_states.is_floating_point():
            raise TypeError("C1 shadow cache K/V tensors must be floating point")
        if (
            key_states.device != value_states.device
            or key_states.dtype != value_states.dtype
        ):
            raise ValueError("C1 shadow cache K/V tensors must share dtype and device")
        if int(key_states.shape[-1]) % self.config.group_size:
            raise ValueError(
                "Key head dimension is incompatible with shadow group size"
            )
        reference_key = self.exact_key_committed
        reference_value = self.c1_value_committed
        if reference_key is not None:
            if (
                tuple(key_states.shape[:2]) != tuple(reference_key.shape[:2])
                or int(key_states.shape[-1]) != int(reference_key.shape[-1])
                or key_states.dtype != reference_key.dtype
                or key_states.device != reference_key.device
            ):
                raise ValueError("new Keys are incompatible with committed Keys")
        if reference_value is not None:
            if (
                tuple(value_states.shape[:2]) != tuple(reference_value.shape[:2])
                or int(value_states.shape[-1]) != int(reference_value.shape[-1])
                or value_states.dtype != reference_value.dtype
                or value_states.device != reference_value.device
            ):
                raise ValueError("new C1 Values are incompatible with committed Values")

    def _refresh_shadow(self) -> None:
        self.shadow_key_committed = (
            None
            if self.exact_key_committed is None
            else quantize(self.exact_key_committed, self.config)
        )
        self.keys = self.exact_key_committed
        self.values = self.c1_value_committed

    def _committed_shadow_view(self, prototype: Tensor) -> Tensor:
        if self.shadow_key_committed is None or self.exact_key_committed is None:
            return prototype.new_empty(
                prototype.shape[0],
                prototype.shape[1],
                0,
                prototype.shape[-1],
            )
        recent = min(self.config.recent_exact_window, self.committed_length)
        if recent == self.committed_length:
            return self.exact_key_committed
        approximate = dequantize(
            self.shadow_key_committed,
            dtype=prototype.dtype,
            device=prototype.device,
        )
        if recent == 0:
            return approximate
        return torch.cat(
            (
                approximate[..., : self.committed_length - recent, :],
                self.exact_key_committed[..., self.committed_length - recent :, :],
            ),
            dim=-2,
        )

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        del args, kwargs
        self._validate_update(key_states, value_states)
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if self.runtime_mode == "target_prefill":
            self.exact_key_committed = _append_sequence(
                self.exact_key_committed,
                key_states,
            )
            self.c1_value_committed = _append_sequence(
                self.c1_value_committed,
                value_states,
            )
            self._refresh_shadow()
            return self.exact_key_committed, self.c1_value_committed

        if self.runtime_mode == "draft":
            self.draft_key_provisional = _append_sequence(
                self.draft_key_provisional,
                key_states,
            )
            self.draft_c1_value_provisional = _append_sequence(
                self.draft_c1_value_provisional,
                value_states,
            )
            shadow = self._committed_shadow_view(key_states)
            keys = torch.cat((shadow, self.draft_key_provisional), dim=-2)
            committed_values = self.c1_value_committed
            values = (
                self.draft_c1_value_provisional
                if committed_values is None
                else torch.cat(
                    (committed_values, self.draft_c1_value_provisional), dim=-2
                )
            )
            return keys, values

        if self.runtime_mode == "target_verify":
            if (
                self.target_key_pending is not None
                or self.target_c1_value_pending is not None
            ):
                raise RuntimeError(
                    "one target-verification block per layer is supported"
                )
            self.target_key_pending = key_states.detach()
            self.target_c1_value_pending = value_states.detach()
            keys = (
                self.target_key_pending
                if self.exact_key_committed is None
                else torch.cat(
                    (self.exact_key_committed, self.target_key_pending), dim=-2
                )
            )
            values = (
                self.target_c1_value_pending
                if self.c1_value_committed is None
                else torch.cat(
                    (self.c1_value_committed, self.target_c1_value_pending), dim=-2
                )
            )
            return keys, values

        raise RuntimeError("C1 shadow cache update requires an active runtime mode")

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + int(query_length), 0

    def get_seq_length(self) -> int:
        if self.runtime_mode == "draft":
            return self.committed_length + self.draft_length
        return self.committed_length

    def get_max_length(self) -> int:
        return -1

    def _truncate_committed(self, new_length: int) -> None:
        if not 0 <= new_length <= self.committed_length:
            raise ValueError("committed truncation must stay within the valid prefix")
        if self.exact_key_committed is not None:
            self.exact_key_committed = self.exact_key_committed[..., :new_length, :]
        if self.c1_value_committed is not None:
            self.c1_value_committed = self.c1_value_committed[..., :new_length, :]
        self._refresh_shadow()

    def _truncate_draft(self, new_length: int) -> None:
        if not 0 <= new_length <= self.draft_length:
            raise ValueError("draft truncation must stay within the valid prefix")
        if self.draft_key_provisional is not None:
            self.draft_key_provisional = self.draft_key_provisional[..., :new_length, :]
        if self.draft_c1_value_provisional is not None:
            self.draft_c1_value_provisional = self.draft_c1_value_provisional[
                ..., :new_length, :
            ]
        if new_length == 0:
            self.draft_key_provisional = None
            self.draft_c1_value_provisional = None

    def _clear_pending(self) -> None:
        self.target_key_pending = None
        self.target_c1_value_pending = None

    def commit_pending_prefix(self, accepted_length: int) -> None:
        if not 0 <= accepted_length <= self.verify_length:
            raise ValueError(
                "accepted target prefix exceeds pending verification state"
            )
        if accepted_length:
            if self.target_key_pending is None or self.target_c1_value_pending is None:
                raise RuntimeError("target pending state is incomplete")
            self.exact_key_committed = _append_sequence(
                self.exact_key_committed,
                self.target_key_pending[..., :accepted_length, :],
            )
            self.c1_value_committed = _append_sequence(
                self.c1_value_committed,
                self.target_c1_value_pending[..., :accepted_length, :],
            )
        self._clear_pending()
        self._truncate_draft(0)
        self._refresh_shadow()

    def reset(self) -> None:
        self.exact_key_committed = None
        self.shadow_key_committed = None
        self.c1_value_committed = None
        self.draft_key_provisional = None
        self.draft_c1_value_provisional = None
        self.target_key_pending = None
        self.target_c1_value_pending = None
        self.keys = None
        self.values = None
        self.is_initialized = False
        self.runtime_mode = "idle"

    def storage_summary(self) -> dict[str, int]:
        shadow_logical = (
            0
            if self.shadow_key_committed is None
            else self.shadow_key_committed.logical_storage_bytes()
        )
        shadow_physical = (
            0
            if self.shadow_key_committed is None
            else self.shadow_key_committed.physical_storage_bytes()
        )
        recent = min(self.config.recent_exact_window, self.committed_length)
        recent_bytes = 0
        if self.exact_key_committed is not None:
            rows = self.exact_key_committed.numel() // max(self.committed_length, 1)
            recent_bytes = rows * recent * self.exact_key_committed.element_size()
        return {
            "exact_key_bytes": _tensor_bytes(self.exact_key_committed),
            "shadow_logical_bytes": shadow_logical,
            "shadow_physical_bytes": shadow_physical,
            "c1_value_bytes": _tensor_bytes(self.c1_value_committed),
            "recent_exact_key_bytes": recent_bytes,
            "draft_scratch_bytes": _tensor_bytes(self.draft_key_provisional)
            + _tensor_bytes(self.draft_c1_value_provisional),
            "target_pending_bytes": _tensor_bytes(self.target_key_pending)
            + _tensor_bytes(self.target_c1_value_pending),
        }


class C1ShadowKeyValueCache(Cache):
    """Transformers-compatible, multi-layer transactional C1 cache."""

    def __init__(self, *, num_layers: int, config: ShadowKeyConfig) -> None:
        if int(num_layers) <= 0:
            raise ValueError("C1 shadow cache requires at least one layer")
        self.shadow_config = config
        layers = [C1ShadowCacheLayer(config) for _ in range(int(num_layers))]
        super().__init__(layers=layers)
        self.runtime_mode: RuntimeMode = "idle"
        self._forward_active = False
        self._expected_query_length = 0
        self._updated_layers: set[int] = set()
        self._forward_lengths_before: tuple[int, ...] = ()

    @property
    def shadow_layers(self) -> tuple[C1ShadowCacheLayer, ...]:
        return tuple(self.layers)  # type: ignore[return-value]

    @property
    def committed_length(self) -> int:
        lengths = self._lengths("committed")
        if len(set(lengths)) != 1:
            raise RuntimeError(f"C1 committed lengths differ across layers: {lengths}")
        return lengths[0]

    def _lengths(self, state: str) -> tuple[int, ...]:
        if state == "committed":
            return tuple(layer.committed_length for layer in self.shadow_layers)
        if state == "draft":
            return tuple(layer.draft_length for layer in self.shadow_layers)
        if state == "verify":
            return tuple(layer.verify_length for layer in self.shadow_layers)
        raise ValueError(f"unknown C1 cache state {state!r}")

    def _require_idle_forward(self) -> None:
        if self._forward_active:
            raise RuntimeError("cannot change C1 cache mode during a model forward")

    def _set_mode(self, mode: RuntimeMode) -> None:
        self.runtime_mode = mode
        for layer in self.shadow_layers:
            layer.runtime_mode = mode

    def begin_target_prefill(self) -> None:
        self._require_idle_forward()
        if (
            self.runtime_mode != "idle"
            or any(self._lengths("draft"))
            or any(self._lengths("verify"))
        ):
            raise RuntimeError("target prefill requires idle cache state")
        self._set_mode("target_prefill")

    def finish_target_prefill(self) -> None:
        self._require_idle_forward()
        if self.runtime_mode != "target_prefill":
            raise RuntimeError("target prefill is not active")
        self._validate_synchronized("committed")
        self._set_mode("idle")

    def begin_draft(self) -> None:
        self._require_idle_forward()
        if (
            self.runtime_mode != "idle"
            or any(self._lengths("draft"))
            or any(self._lengths("verify"))
        ):
            raise RuntimeError("drafting requires idle cache state")
        self._set_mode("draft")

    def begin_verify(self) -> None:
        self._require_idle_forward()
        if self.runtime_mode not in ("idle", "draft"):
            raise RuntimeError("target verification requires idle or draft cache state")
        if any(self._lengths("verify")):
            raise RuntimeError("target pending state must be empty before verification")
        self._validate_synchronized("committed")
        self._validate_synchronized("draft")
        self._set_mode("target_verify")

    @contextmanager
    def forward_pass(self, *, query_length: int) -> Iterator[None]:
        """Validate one complete model forward across every cache layer."""

        if self.runtime_mode == "idle":
            raise RuntimeError("begin a C1 cache runtime mode before model forward")
        if self._forward_active:
            raise RuntimeError("nested C1 cache model forwards are unsupported")
        requested = int(query_length)
        if requested <= 0:
            raise ValueError("C1 cache query length must be positive")
        state = {
            "target_prefill": "committed",
            "draft": "draft",
            "target_verify": "verify",
        }[self.runtime_mode]
        self._forward_active = True
        self._expected_query_length = requested
        self._updated_layers.clear()
        self._forward_lengths_before = self._lengths(state)
        try:
            yield
            expected_layers = set(range(len(self.layers)))
            if self._updated_layers != expected_layers:
                raise RuntimeError(
                    "C1 cache forward did not update every layer exactly once: "
                    f"updated={sorted(self._updated_layers)}, expected={sorted(expected_layers)}"
                )
            after = self._lengths(state)
            expected = tuple(
                value + requested for value in self._forward_lengths_before
            )
            if after != expected:
                raise RuntimeError(
                    f"C1 cache {state} lengths changed unexpectedly: {after} vs {expected}"
                )
            self._validate_synchronized(state)
        except BaseException:
            self._restore_forward_state(state)
            raise
        finally:
            self._forward_active = False
            self._expected_query_length = 0
            self._updated_layers.clear()
            self._forward_lengths_before = ()

    def _restore_forward_state(self, state: str) -> None:
        for layer, before in zip(self.shadow_layers, self._forward_lengths_before):
            if state == "committed":
                layer._truncate_committed(before)
            elif state == "draft":
                layer._truncate_draft(before)
            elif state == "verify":
                layer._clear_pending()

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        *args: object,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        if not self._forward_active:
            raise RuntimeError(
                "C1 cache update occurred outside a validated model forward"
            )
        index = int(layer_idx)
        if not 0 <= index < len(self.layers):
            raise ValueError(f"C1 cache layer index {index} is out of range")
        if index in self._updated_layers:
            raise RuntimeError(
                f"C1 cache layer {index} was updated twice in one forward"
            )
        if int(key_states.shape[-2]) != self._expected_query_length:
            raise ValueError(
                "C1 cache update query length differs from the forward contract: "
                f"{key_states.shape[-2]} vs {self._expected_query_length}"
            )
        result = self.shadow_layers[index].update(
            key_states,
            value_states,
            *args,
            **kwargs,
        )
        self._updated_layers.add(index)
        return result

    def _validate_synchronized(self, state: str) -> None:
        lengths = self._lengths(state)
        if len(set(lengths)) != 1:
            raise RuntimeError(f"C1 {state} lengths differ across layers: {lengths}")

    def commit_pending_target_prefix(self, accepted_length: int) -> None:
        self._require_idle_forward()
        if self.runtime_mode != "target_verify":
            raise RuntimeError("target prefix commit requires verification mode")
        self._validate_synchronized("verify")
        self._validate_synchronized("draft")
        committed_before = self.committed_length
        requested = int(accepted_length)
        verify_length = self._lengths("verify")[0]
        if not 0 <= requested <= verify_length:
            raise ValueError(
                f"accepted prefix {requested} is outside pending length {verify_length}"
            )
        for layer in self.shadow_layers:
            layer.commit_pending_prefix(requested)
        self._set_mode("idle")
        self._validate_synchronized("committed")
        self._validate_synchronized("draft")
        self._validate_synchronized("verify")
        if self.committed_length != committed_before + requested:
            raise RuntimeError(
                "target prefix commit advanced the canonical cache incorrectly: "
                f"{self.committed_length} vs {committed_before + requested}"
            )
        if any(self._lengths("draft")) or any(self._lengths("verify")):
            raise RuntimeError(
                "target prefix commit left provisional cache state alive"
            )

    def rollback_draft(self) -> None:
        self._require_idle_forward()
        if self.runtime_mode != "draft":
            raise RuntimeError("draft rollback requires draft mode")
        for layer in self.shadow_layers:
            layer._truncate_draft(0)
        self._set_mode("idle")
        self._validate_synchronized("draft")

    def abort_transaction(self) -> None:
        """Discard every uncommitted tensor after a failed draft/verify call."""

        self._require_idle_forward()
        for layer in self.shadow_layers:
            layer._truncate_draft(0)
            layer._clear_pending()
        self._set_mode("idle")
        self._validate_synchronized("committed")

    def truncate_committed(self, new_length: int) -> None:
        self._require_idle_forward()
        if self.runtime_mode != "idle":
            raise RuntimeError("committed truncation requires idle cache state")
        requested = int(new_length)
        for layer in self.shadow_layers:
            layer._truncate_committed(requested)
            layer._truncate_draft(0)
            layer._clear_pending()
        self._validate_synchronized("committed")

    def reset(self) -> None:
        self._require_idle_forward()
        for layer in self.shadow_layers:
            layer.reset()
        self._set_mode("idle")

    def storage_summary(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for layer in self.shadow_layers:
            for name, value in layer.storage_summary().items():
                result[name] = result.get(name, 0) + value
        return result


__all__ = [
    "C1ShadowCacheLayer",
    "C1ShadowKeyValueCache",
    "QuantizedKeys",
    "ShadowKeyConfig",
    "dequantize",
    "logical_storage_bytes",
    "physical_storage_bytes",
    "quantize",
]
