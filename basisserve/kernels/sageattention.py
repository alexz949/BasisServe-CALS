"""Optional SageAttention bridge for compressed-Value attention.

SageAttention 2.2.0 uses one shared kernel head dimension for Q, K, V, and
the output.  BASISServe keeps Q/K at the model head dimension while storing V
at a smaller rank.  This bridge preserves the compressed V representation,
zero-pads it only at the kernel boundary, and slices the padded output before
the compressed output projection.

The padding path is intended as a correctness and quality prototype.  It does
not reduce the SageAttention PV arithmetic.  A native split-dimension kernel
is required to realize the full compute benefit of compressed V.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.nn import functional as F


class SageAttentionUnavailableError(ImportError):
    """Raised when the optional SageAttention package is not installed."""


def _load_sageattention_apis() -> tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
    try:
        from sageattention import (
            sageattn,
            sageattn_qk_int8_pv_fp16_triton,
        )
    except ImportError as exc:
        raise SageAttentionUnavailableError(
            "SageAttention is required for the 'sage' compressed-attention "
            "backend. Install thu-ml/SageAttention 2.2.0 in the active "
            "environment before running this backend."
        ) from exc
    return sageattn, sageattn_qk_int8_pv_fp16_triton


def _validate_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> int:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("SageAttention bridge expects rank-4 HND Q/K/V tensors")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Q/K/V batch sizes must match")
    if key.shape[1] != value.shape[1] or key.shape[2] != value.shape[2]:
        raise ValueError("K/V head counts and sequence lengths must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Q/K head dimensions must match")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")
    value_dim = int(value.shape[-1])
    qk_dim = int(query.shape[-1])
    if value_dim <= 0 or value_dim > qk_dim:
        raise ValueError(
            f"compressed V dimension must be in [1, {qk_dim}], got {value_dim}"
        )
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Q/K/V dtypes must match")
    if query.device != key.device or query.device != value.device:
        raise ValueError("Q/K/V devices must match")
    return value_dim


def sage_attention_with_compressed_value(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    is_causal: bool,
    scaling: float,
    sageattn_auto: Callable[..., torch.Tensor] | None = None,
    sageattn_masked: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run SageAttention while retaining a lower-dimensional V representation.

    Q/K stay at their native head dimension.  V is zero-padded immediately
    before the SageAttention call, and the padded output channels are removed
    immediately afterwards.  The caller can inject API-compatible functions
    for CPU unit tests; production calls lazily import SageAttention.
    """

    value_dim = _validate_qkv(query, key, value)
    qk_dim = int(query.shape[-1])
    if value_dim < qk_dim:
        padded_value = F.pad(value, (0, qk_dim - value_dim))
    else:
        padded_value = value

    if sageattn_auto is None or sageattn_masked is None:
        loaded_auto, loaded_masked = _load_sageattention_apis()
        if sageattn_auto is None:
            sageattn_auto = loaded_auto
        if sageattn_masked is None:
            sageattn_masked = loaded_masked

    common: dict[str, Any] = {
        "tensor_layout": "HND",
        "sm_scale": float(scaling),
    }
    if attention_mask is None:
        output = sageattn_auto(
            query,
            key,
            padded_value,
            is_causal=bool(is_causal),
            **common,
        )
    else:
        if is_causal:
            raise ValueError(
                "SageAttention explicit-mask path expects the causal structure "
                "to be represented by attention_mask"
            )
        output = sageattn_masked(
            query,
            key,
            padded_value,
            is_causal=False,
            attn_mask=attention_mask,
            **common,
        )

    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"SageAttention returned unsupported output type {type(output)!r}")
    if tuple(output.shape[:-1]) != tuple(query.shape[:-1]) or output.shape[-1] != qk_dim:
        raise RuntimeError(
            "SageAttention returned an unexpected shape: "
            f"{tuple(output.shape)} vs {tuple(query.shape)}"
        )
    return output[..., :value_dim]
