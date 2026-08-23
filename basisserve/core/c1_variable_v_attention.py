"""Rank-local C1 attention with a variable-width Value cache.

This module is the correctness reference for the serving boundary that a
future CUDA kernel will implement.  One TP process owns one physical KV head,
its query-head group, and one C1 encoder ``A_s``.  Folding ``A_s`` into the
local dense Value projection changes the cache width from ``head_dim`` to
``r_s`` without changing Q/K scores::

    W_v_compact = A_s.T @ W_v
    softmax(Q @ K.T) @ (V @ A_s)
        == (softmax(Q @ K.T) @ V) @ A_s

The resulting attention coordinates are laid out as
``[batch, tokens, local_query_heads * r_s]``.  That is exactly the local wire
block expected by the C1 ragged collective and compact global decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.core.c1_tp_decode import PackedC1TPLayer


def fold_local_c1_value_projection(
    dense_weight: Tensor,
    encoder: Tensor,
    dense_bias: Tensor | None = None,
    *,
    output_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Fold one physical KV head's ``A_s`` into its dense Value projection.

    Args:
        dense_weight: Local ``nn.Linear`` weight with shape
            ``[head_dim, hidden_size]``.
        encoder: C1 Value encoder with shape ``[head_dim, source_rank]``.
        dense_bias: Optional dense Value bias with shape ``[head_dim]``.
        output_dtype: Stored serving dtype.  It defaults to ``dense_weight``'s
            dtype; the one-time fold itself uses at least float32 arithmetic.

    Returns:
        ``(compact_weight, compact_bias)`` with shapes
        ``[source_rank, hidden_size]`` and ``[source_rank]``.
    """

    if dense_weight.ndim != 2:
        raise ValueError("dense Value weight must be [head_dim, hidden_size]")
    if encoder.ndim != 2:
        raise ValueError("C1 encoder must be [head_dim, source_rank]")
    if int(dense_weight.shape[0]) != int(encoder.shape[0]):
        raise ValueError(
            "dense Value and C1 encoder head dimensions differ: "
            f"{dense_weight.shape[0]} != {encoder.shape[0]}"
        )
    if dense_bias is not None and tuple(dense_bias.shape) != (
        int(dense_weight.shape[0]),
    ):
        raise ValueError(
            "dense Value bias must have shape [head_dim], got "
            f"{tuple(dense_bias.shape)}"
        )
    tensors = (dense_weight, encoder) + (() if dense_bias is None else (dense_bias,))
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("C1 Value projection factors must be floating point")
    if any(tensor.device != dense_weight.device for tensor in tensors):
        raise ValueError("C1 Value projection factors must be on one device")

    target_dtype = dense_weight.dtype if output_dtype is None else output_dtype
    if not torch.empty((), dtype=target_dtype).is_floating_point():
        raise TypeError("folded C1 Value projection must use a floating dtype")
    work_dtype = (
        torch.float64
        if any(tensor.dtype == torch.float64 for tensor in tensors)
        else torch.float32
    )
    weight_math = dense_weight.detach().to(work_dtype)
    encoder_math = encoder.detach().to(work_dtype)
    compact_weight = torch.mm(encoder_math.transpose(0, 1), weight_math)
    compact_bias = None
    if dense_bias is not None:
        compact_bias = torch.mv(encoder_math.transpose(0, 1), dense_bias.detach().to(work_dtype))
    return (
        compact_weight.to(target_dtype).contiguous(),
        None if compact_bias is None else compact_bias.to(target_dtype).contiguous(),
    )


class C1StaticKVCache(nn.Module):
    """Fixed-capacity inference cache for one TP rank, layer, and KV head.

    Keys retain the dense Q/K head dimension.  Values use this rank's static
    C1 width ``r_s``.  The cache intentionally supports only append-only
    prefill/decode because that is the serving path being measured.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        capacity: int,
        head_dim: int,
        value_head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        dimensions = (batch_size, capacity, head_dim, value_head_dim)
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError(f"C1 cache dimensions must be positive, got {dimensions}")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("C1 cache dtype must be floating point")
        self.register_buffer(
            "key_storage",
            torch.empty(
                int(batch_size),
                int(capacity),
                int(head_dim),
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "value_storage",
            torch.empty(
                int(batch_size),
                int(capacity),
                int(value_head_dim),
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )
        self._sequence_length = 0

    @property
    def batch_size(self) -> int:
        return int(self.key_storage.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.key_storage.shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.key_storage.shape[2])

    @property
    def value_head_dim(self) -> int:
        return int(self.value_storage.shape[2])

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def key_bytes(self) -> int:
        return self.key_storage.numel() * self.key_storage.element_size()

    @property
    def value_bytes(self) -> int:
        return self.value_storage.numel() * self.value_storage.element_size()

    @property
    def allocated_bytes(self) -> int:
        return self.key_bytes + self.value_bytes

    def current(self) -> tuple[Tensor, Tensor]:
        """Return valid K/V views with shapes ``[B, S, D_k]`` and ``[B, S, r_s]``."""

        stop = self._sequence_length
        return self.key_storage[:, :stop], self.value_storage[:, :stop]

    @torch.no_grad()
    def append(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        """Append one prefill chunk or decode step and return all valid states."""

        if key_states.ndim != 3 or value_states.ndim != 3:
            raise ValueError("cache updates must be [batch, tokens, feature]")
        if tuple(key_states.shape[:2]) != tuple(value_states.shape[:2]):
            raise ValueError("key and Value cache updates must share batch/token dimensions")
        expected_key = (self.batch_size, int(key_states.shape[1]), self.head_dim)
        expected_value = (
            self.batch_size,
            int(value_states.shape[1]),
            self.value_head_dim,
        )
        if tuple(key_states.shape) != expected_key:
            raise ValueError(
                f"key update must have shape {expected_key}, got {tuple(key_states.shape)}"
            )
        if tuple(value_states.shape) != expected_value:
            raise ValueError(
                "Value update must have shape "
                f"{expected_value}, got {tuple(value_states.shape)}"
            )
        if key_states.dtype != self.key_storage.dtype or value_states.dtype != self.value_storage.dtype:
            raise ValueError("cache updates must match cache dtype")
        if key_states.device != self.key_storage.device or value_states.device != self.value_storage.device:
            raise ValueError("cache updates must match cache device")
        token_count = int(key_states.shape[1])
        if token_count <= 0:
            raise ValueError("cache update must contain at least one token")
        start = self._sequence_length
        stop = start + token_count
        if stop > self.capacity:
            raise RuntimeError(
                f"C1 cache capacity exceeded: requested length {stop}, capacity {self.capacity}"
            )
        self.key_storage[:, start:stop].copy_(key_states)
        self.value_storage[:, start:stop].copy_(value_states)
        self._sequence_length = stop
        return self.current()

    def reset(self) -> None:
        """Logically clear the cache without reallocating its storage."""

        self._sequence_length = 0

    def truncate(self, sequence_length: int) -> None:
        """Rewind to an already-written prefix without moving cache data."""

        requested = int(sequence_length)
        if not 0 <= requested <= self._sequence_length:
            raise ValueError(
                "cache truncation must stay within the valid prefix: "
                f"requested {requested}, current length {self._sequence_length}"
            )
        self._sequence_length = requested


def reference_grouped_query_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    *,
    attention_mask: Tensor | None = None,
    is_causal: bool = True,
    scaling: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Reference one-physical-KV-head grouped-query attention.

    Args:
        query_states: ``[batch, local_query_heads, query_tokens, head_dim]``.
        key_states: ``[batch, sequence, head_dim]``.
        value_states: ``[batch, sequence, value_head_dim]``.  The last
            dimension may be dense ``head_dim`` or this source's ``r_s``.

    Returns:
        Attention output ``[batch, query_tokens, local_query_heads,
        value_head_dim]`` and weights ``[batch, local_query_heads,
        query_tokens, sequence]``.
    """

    if query_states.ndim != 4:
        raise ValueError("query states must be [batch, heads, tokens, head_dim]")
    if key_states.ndim != 3 or value_states.ndim != 3:
        raise ValueError("key and Value states must be [batch, sequence, feature]")
    batch, _, query_tokens, head_dim = map(int, query_states.shape)
    if batch != int(key_states.shape[0]) or batch != int(value_states.shape[0]):
        raise ValueError("Q/K/V batch dimensions differ")
    sequence = int(key_states.shape[1])
    if sequence <= 0 or query_tokens <= 0:
        raise ValueError("attention requires non-empty query and cache sequences")
    if sequence != int(value_states.shape[1]):
        raise ValueError("K/V sequence lengths differ")
    if head_dim != int(key_states.shape[2]):
        raise ValueError("Q/K head dimensions differ")
    if query_tokens > sequence and is_causal:
        raise ValueError("causal query chunk cannot be longer than the visible K/V sequence")
    if not (
        query_states.dtype == key_states.dtype == value_states.dtype
        and query_states.device == key_states.device == value_states.device
    ):
        raise ValueError("Q/K/V must share dtype and device")

    scale = head_dim**-0.5 if scaling is None else float(scaling)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"attention scaling must be finite and positive, got {scale}")
    scores = torch.matmul(
        query_states,
        key_states.transpose(1, 2).unsqueeze(1),
    ) * scale

    if is_causal:
        query_positions = torch.arange(
            sequence - query_tokens,
            sequence,
            device=query_states.device,
        )
        key_positions = torch.arange(sequence, device=query_states.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(
            ~causal_mask.view(1, 1, query_tokens, sequence),
            torch.finfo(scores.dtype).min,
        )
    if attention_mask is not None:
        if attention_mask.device != scores.device:
            raise ValueError("attention mask must be on the Q/K/V device")
        try:
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
            elif attention_mask.is_floating_point():
                scores = scores + attention_mask.to(scores.dtype)
            else:
                raise TypeError("attention mask must be boolean or floating point")
        except RuntimeError as error:
            raise ValueError(
                f"attention mask shape {tuple(attention_mask.shape)} is not broadcastable "
                f"to scores {tuple(scores.shape)}"
            ) from error

    attention_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    head_output = torch.matmul(attention_weights, value_states.unsqueeze(1))
    return head_output.transpose(1, 2).contiguous(), attention_weights


@dataclass(frozen=True)
class C1RankLocalAttentionOutput:
    """Outputs at the local compact-attention/ragged-collective boundary."""

    local_coordinates: Tensor
    head_coordinates: Tensor
    attention_weights: Tensor
    projected_values: Tensor


class C1RankLocalAttentionReference(nn.Module):
    """End-to-end rank-local compact-V prefill/decode correctness path."""

    def __init__(
        self,
        *,
        compact_v_weight: Tensor,
        compact_v_bias: Tensor | None,
        query_heads: int,
    ) -> None:
        super().__init__()
        if compact_v_weight.ndim != 2 or int(compact_v_weight.shape[0]) <= 0:
            raise ValueError("compact Value weight must be [source_rank, hidden_size]")
        if int(query_heads) <= 0:
            raise ValueError("local query-head count must be positive")
        if compact_v_bias is not None and tuple(compact_v_bias.shape) != (
            int(compact_v_weight.shape[0]),
        ):
            raise ValueError("compact Value bias must have shape [source_rank]")
        if compact_v_bias is not None and (
            compact_v_bias.dtype != compact_v_weight.dtype
            or compact_v_bias.device != compact_v_weight.device
        ):
            raise ValueError("compact Value weight and bias must share dtype/device")
        self.query_heads = int(query_heads)
        self.register_buffer("compact_v_weight", compact_v_weight.detach().contiguous())
        self.register_buffer(
            "compact_v_bias",
            None if compact_v_bias is None else compact_v_bias.detach().contiguous(),
        )

    @property
    def source_rank(self) -> int:
        return int(self.compact_v_weight.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.compact_v_weight.shape[1])

    @classmethod
    def from_dense_projection(
        cls,
        dense_v_projection: nn.Linear,
        factors: PackedC1TPLayer,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> "C1RankLocalAttentionReference":
        """Build the local serving module using loaded TP ownership and ``A_s``."""

        if int(dense_v_projection.out_features) != int(factors.local_encoder.shape[0]):
            raise ValueError(
                "local dense v_proj output width must equal C1 head_dim: "
                f"{dense_v_projection.out_features} != {factors.local_encoder.shape[0]}"
            )
        compact_weight, compact_bias = fold_local_c1_value_projection(
            dense_v_projection.weight,
            factors.local_encoder,
            dense_v_projection.bias,
            output_dtype=output_dtype,
        )
        return cls(
            compact_v_weight=compact_weight,
            compact_v_bias=compact_bias,
            query_heads=factors.ownership.query_head_count,
        )

    def project_values(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden states must be [batch, tokens, hidden_size]")
        if int(hidden_states.shape[2]) != self.hidden_size:
            raise ValueError(
                f"hidden width must be {self.hidden_size}, got {hidden_states.shape[2]}"
            )
        if (
            hidden_states.dtype != self.compact_v_weight.dtype
            or hidden_states.device != self.compact_v_weight.device
        ):
            raise ValueError("hidden states must match compact v_proj dtype/device")
        return F.linear(hidden_states, self.compact_v_weight, self.compact_v_bias)

    def forward(
        self,
        hidden_states: Tensor,
        query_states: Tensor,
        key_states: Tensor,
        cache: C1StaticKVCache,
        *,
        attention_mask: Tensor | None = None,
        is_causal: bool = True,
        scaling: float | None = None,
    ) -> C1RankLocalAttentionOutput:
        """Project new compact V, append K/V, and attend over the valid cache."""

        if hidden_states.ndim != 3:
            raise ValueError("hidden states must be [batch, tokens, hidden_size]")
        batch, tokens = map(int, hidden_states.shape[:2])
        expected_query = (batch, self.query_heads, tokens, cache.head_dim)
        expected_key = (batch, tokens, cache.head_dim)
        if tuple(query_states.shape) != expected_query:
            raise ValueError(
                f"query states must have shape {expected_query}, got {tuple(query_states.shape)}"
            )
        if tuple(key_states.shape) != expected_key:
            raise ValueError(
                f"key states must have shape {expected_key}, got {tuple(key_states.shape)}"
            )
        if cache.value_head_dim != self.source_rank:
            raise ValueError(
                "cache Value width differs from local C1 source rank: "
                f"{cache.value_head_dim} != {self.source_rank}"
            )
        projected_values = self.project_values(hidden_states)
        cached_keys, cached_values = cache.append(key_states, projected_values)
        head_coordinates, attention_weights = reference_grouped_query_attention(
            query_states,
            cached_keys,
            cached_values,
            attention_mask=attention_mask,
            is_causal=is_causal,
            scaling=scaling,
        )
        local_coordinates = head_coordinates.reshape(
            batch,
            tokens,
            self.query_heads * self.source_rank,
        )
        return C1RankLocalAttentionOutput(
            local_coordinates=local_coordinates,
            head_coordinates=head_coordinates,
            attention_weights=attention_weights,
            projected_values=projected_values,
        )
