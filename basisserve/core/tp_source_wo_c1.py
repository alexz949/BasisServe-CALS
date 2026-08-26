"""TP-source C1 factors at the post-attention ``W_o`` boundary.

This module deliberately knows nothing about the attention implementation or
its KV cache.  It partitions the dense ``W_o`` input into contiguous
tensor-parallel source blocks and represents each block product as

    y_s @ W_s = (y_s @ A_s) @ D_s.

The compressed coordinates are suitable for a private AllGather followed by
a replicated decoder.  The helpers below also materialize the equivalent
dense weight used by quality-only Hugging Face evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TPSourceWOLayout:
    """Validated geometry and ideal ring-collective accounting."""

    input_width: int
    output_width: int
    tp_size: int
    source_rank: int
    dtype_bytes: int = 2

    def __post_init__(self) -> None:
        if min(
            self.input_width,
            self.output_width,
            self.tp_size,
            self.source_rank,
            self.dtype_bytes,
        ) <= 0:
            raise ValueError("TP-source W_o dimensions must be positive")
        if self.tp_size <= 1 or self.input_width % self.tp_size:
            raise ValueError("W_o input width must be divisible by TP size")
        if self.source_rank > self.source_width:
            raise ValueError("source rank exceeds the dense TP source width")

    @property
    def source_width(self) -> int:
        return self.input_width // self.tp_size

    @property
    def gathered_width(self) -> int:
        return self.tp_size * self.source_rank

    @property
    def retained_ratio_vs_dense_allgather(self) -> float:
        return self.source_rank / self.source_width

    @property
    def reduction_vs_dense_allgather(self) -> float:
        return 1.0 - self.retained_ratio_vs_dense_allgather

    @property
    def dense_allgather_ring_bytes_per_rank(self) -> int:
        return (self.tp_size - 1) * self.source_width * self.dtype_bytes

    @property
    def compressed_allgather_ring_bytes_per_rank(self) -> int:
        return (self.tp_size - 1) * self.source_rank * self.dtype_bytes

    @property
    def dense_allreduce_ring_bytes_per_rank(self) -> int:
        return (
            2
            * (self.tp_size - 1)
            * self.output_width
            * self.dtype_bytes
            // self.tp_size
        )

    @property
    def reduction_vs_dense_allreduce(self) -> float:
        return 1.0 - (
            self.compressed_allgather_ring_bytes_per_rank
            / self.dense_allreduce_ring_bytes_per_rank
        )

    def accounting(self) -> dict[str, int | float | str]:
        return {
            "collective": "private_allgather",
            "input_width": self.input_width,
            "output_width": self.output_width,
            "tp_size": self.tp_size,
            "source_width": self.source_width,
            "source_rank": self.source_rank,
            "gathered_width": self.gathered_width,
            "dtype_bytes": self.dtype_bytes,
            "dense_allgather_ring_bytes_per_rank": (
                self.dense_allgather_ring_bytes_per_rank
            ),
            "compressed_allgather_ring_bytes_per_rank": (
                self.compressed_allgather_ring_bytes_per_rank
            ),
            "dense_allreduce_ring_bytes_per_rank": (
                self.dense_allreduce_ring_bytes_per_rank
            ),
            "reduction_vs_dense_allgather": (
                self.reduction_vs_dense_allgather
            ),
            "reduction_vs_dense_allreduce": self.reduction_vs_dense_allreduce,
            "kv_cache_compression": "none",
        }


def covariance_to_source_blocks(
    covariance: Tensor,
    layout: TPSourceWOLayout,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return a symmetric ``[S,S,d,d]`` covariance view."""

    if covariance.ndim != 2 or tuple(covariance.shape) != (
        layout.input_width,
        layout.input_width,
    ):
        raise ValueError("covariance does not match the W_o input width")
    work = covariance.to(
        device=covariance.device if device is None else device,
        dtype=covariance.dtype if dtype is None else dtype,
    )
    blocks = (
        work.reshape(
            layout.tp_size,
            layout.source_width,
            layout.tp_size,
            layout.source_width,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return 0.5 * (blocks + blocks.permute(1, 0, 3, 2))


def weight_to_source_targets(
    weight: Tensor,
    layout: TPSourceWOLayout,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return dense source products as ``[S,d,output_width]``."""

    if weight.ndim != 2 or tuple(weight.shape) != (
        layout.output_width,
        layout.input_width,
    ):
        raise ValueError("W_o weight does not match the TP-source layout")
    work = weight.to(
        device=weight.device if device is None else device,
        dtype=weight.dtype if dtype is None else dtype,
    )
    return (
        work.transpose(0, 1)
        .reshape(layout.tp_size, layout.source_width, layout.output_width)
        .contiguous()
    )


def validate_factors(
    encoders: Tensor,
    decoders: Tensor,
    layout: TPSourceWOLayout,
) -> None:
    if tuple(encoders.shape) != (
        layout.tp_size,
        layout.source_width,
        layout.source_rank,
    ):
        raise ValueError("TP-source encoders have incompatible geometry")
    if tuple(decoders.shape) != (
        layout.tp_size,
        layout.source_rank,
        layout.output_width,
    ):
        raise ValueError("TP-source decoders have incompatible geometry")
    if not bool(torch.isfinite(encoders).all()) or not bool(
        torch.isfinite(decoders).all()
    ):
        raise ValueError("TP-source factors contain non-finite values")


def fold_factors_to_dense_weight(
    encoders: Tensor,
    decoders: Tensor,
    layout: TPSourceWOLayout,
) -> Tensor:
    """Materialize the quality-equivalent dense ``W_o`` weight."""

    validate_factors(encoders, decoders, layout)
    products = torch.bmm(encoders, decoders)
    return (
        products.reshape(layout.input_width, layout.output_width)
        .transpose(0, 1)
        .contiguous()
    )


def identity_factors(
    weight: Tensor,
    layout: TPSourceWOLayout,
    *,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the analytic exact endpoint for ``source_rank=source_width``."""

    if layout.source_rank != layout.source_width:
        raise ValueError("identity factors require full source rank")
    targets = weight_to_source_targets(weight, layout)
    factor_dtype = weight.dtype if dtype is None else dtype
    encoders = (
        torch.eye(
            layout.source_width,
            device=weight.device,
            dtype=factor_dtype,
        )
        .unsqueeze(0)
        .repeat(layout.tp_size, 1, 1)
        .contiguous()
    )
    return encoders, targets.to(dtype=factor_dtype).contiguous()


__all__ = [
    "TPSourceWOLayout",
    "covariance_to_source_blocks",
    "fold_factors_to_dense_weight",
    "identity_factors",
    "validate_factors",
    "weight_to_source_targets",
]
