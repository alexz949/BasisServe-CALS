"""C1-Value-derived retrieval indexes for exact Key page selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class C1VToPreRoPEKFactors:
    """Per-physical-head least-squares maps from C1 Value to pre-RoPE Key."""

    weight: Tensor

    def validate(self, *, kv_heads: int, value_rank: int, head_dim: int) -> None:
        expected = (kv_heads, value_rank, head_dim)
        if tuple(self.weight.shape) != expected:
            raise ValueError(
                f"V-to-K weight must have shape {expected}, got "
                f"{tuple(self.weight.shape)}"
            )
        if not self.weight.is_floating_point():
            raise TypeError("V-to-K weight must be floating point")


def _validate_feature_pair(c1_value: Tensor, key: Tensor) -> tuple[int, int, int]:
    if c1_value.ndim != 4 or key.ndim != 4:
        raise ValueError("C1 Value and Key must be [batch, KV heads, sequence, dim]")
    if tuple(c1_value.shape[:3]) != tuple(key.shape[:3]):
        raise ValueError("C1 Value and Key must share batch/head/sequence geometry")
    if not c1_value.is_floating_point() or not key.is_floating_point():
        raise TypeError("C1 Value and Key must be floating point")
    return int(c1_value.shape[1]), int(c1_value.shape[-1]), int(key.shape[-1])


def fit_c1_v_to_pre_rope_k(
    c1_value: Tensor,
    pre_rope_key: Tensor,
) -> C1VToPreRoPEKFactors:
    """Fit independent physical-head maps with a parameter-free SVD least square.

    The fit intentionally has no ridge or damping hyperparameter.  ``gelsd``
    returns the minimum-norm least-squares solution when a head design is rank
    deficient.  Fitting runs in FP64 on CPU because this is an offline oracle.
    """

    kv_heads, value_rank, head_dim = _validate_feature_pair(
        c1_value, pre_rope_key
    )
    design = c1_value.detach().permute(1, 0, 2, 3).reshape(
        kv_heads, -1, value_rank
    ).to(device="cpu", dtype=torch.float64)
    target = pre_rope_key.detach().permute(1, 0, 2, 3).reshape(
        kv_heads, -1, head_dim
    ).to(device="cpu", dtype=torch.float64)
    solutions = []
    for head in range(kv_heads):
        solutions.append(
            torch.linalg.lstsq(
                design[head],
                target[head],
                driver="gelsd",
            ).solution
        )
    weight = torch.stack(solutions).to(
        device=c1_value.device,
        dtype=torch.float32,
    )
    factors = C1VToPreRoPEKFactors(weight)
    factors.validate(
        kv_heads=kv_heads,
        value_rank=value_rank,
        head_dim=head_dim,
    )
    return factors


def project_c1_v_to_pre_rope_k(
    c1_value: Tensor,
    factors: C1VToPreRoPEKFactors,
) -> Tensor:
    """Project resident C1 Values into approximate pre-RoPE Keys."""

    if c1_value.ndim != 4 or not c1_value.is_floating_point():
        raise ValueError("C1 Value must be floating [batch, KV heads, sequence, rank]")
    _, kv_heads, _, value_rank = map(int, c1_value.shape)
    head_dim = int(factors.weight.shape[-1])
    factors.validate(
        kv_heads=kv_heads,
        value_rank=value_rank,
        head_dim=head_dim,
    )
    return torch.einsum(
        "bhsv,hvd->bhsd",
        c1_value.float(),
        factors.weight.to(device=c1_value.device, dtype=torch.float32),
    )


def _rotate_half(value: Tensor) -> Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _broadcast_rotary(
    value: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    if value.ndim != 4:
        raise ValueError("rotary input must be [batch, heads, sequence, head dim]")
    if cos.ndim != 3 or sin.ndim != 3 or tuple(cos.shape) != tuple(sin.shape):
        raise ValueError("cos and sin must share [batch, sequence, head dim] geometry")
    if int(cos.shape[0]) not in (1, int(value.shape[0])):
        raise ValueError("rotary batch dimension does not broadcast to the input")
    assert cos.shape[1] == value.shape[2]
    assert 0 < cos.shape[2] <= value.shape[3] and cos.shape[2] % 2 == 0
    return (
        cos.to(device=value.device, dtype=torch.float32).unsqueeze(1),
        sin.to(device=value.device, dtype=torch.float32).unsqueeze(1),
    )


def apply_rotary(value: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply half-split RoPE to the configured rotary prefix."""

    cosine, sine = _broadcast_rotary(value, cos, sin)
    value_fp32 = value.float()
    width = cosine.shape[-1]
    prefix = value_fp32[..., :width]
    rotated = prefix * cosine + _rotate_half(prefix) * sine
    return torch.cat((rotated, value_fp32[..., width:]), dim=-1)


def invert_rotary(value: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Undo prefix RoPE up to the precision of the supplied post-RoPE tensor."""

    cosine, sine = _broadcast_rotary(value, cos, sin)
    value_fp32 = value.float()
    width = cosine.shape[-1]
    prefix = value_fp32[..., :width]
    restored = prefix * cosine - _rotate_half(prefix) * sine
    return torch.cat((restored, value_fp32[..., width:]), dim=-1)


def page_center_rotary_embeddings(
    cos: Tensor,
    sin: Tensor,
    *,
    page_size: int,
) -> tuple[Tensor, Tensor]:
    """Replace each token's RoPE angle with its physical page-center angle."""

    if page_size <= 0:
        raise ValueError("page size must be positive")
    if cos.ndim != 3 or tuple(cos.shape) != tuple(sin.shape):
        raise ValueError("cos and sin must share [batch, sequence, head dim] geometry")
    sequence = int(cos.shape[1])
    positions = torch.arange(sequence, device=cos.device)
    page_start = (positions // page_size) * page_size
    page_stop = torch.minimum(
        page_start + page_size,
        torch.full_like(page_start, sequence),
    )
    centers = page_start + (page_stop - page_start - 1) // 2
    return cos.index_select(1, centers), sin.index_select(1, centers)


def relative_squared_error(reference: Tensor, candidate: Tensor) -> float:
    """Return squared error divided by reference energy in FP64."""

    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError("reference and candidate must share geometry")
    difference = candidate.detach().double().cpu() - reference.detach().double().cpu()
    denominator = reference.detach().double().cpu().square().sum()
    return float(difference.square().sum() / denominator.clamp_min(torch.finfo(torch.float64).tiny))
