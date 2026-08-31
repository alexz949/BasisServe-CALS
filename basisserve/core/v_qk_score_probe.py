"""Head-specific linear Value-to-QK routing probes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class CenteredScoreProbeFit:
    """Factors and convergence diagnostics for a centered-score probe."""

    weight: Tensor
    iterations: int
    maximum_relative_residual: float
    relative_residual_history: tuple[float, ...]
    damping: Tensor


def _validate_statistics(
    query: Tensor,
    source_gram: Tensor,
    source_score_cross: Tensor,
) -> tuple[int, int, int, int, int]:
    if query.ndim != 4:
        raise ValueError("query statistics must be [KV heads, group, rows, head dim]")
    if source_gram.ndim != 4:
        raise ValueError("source Gram must be [KV heads, rows, rank, rank]")
    if source_score_cross.ndim != 4:
        raise ValueError("source-score cross must be [KV heads, group, rows, rank]")
    kv_heads, group, rows, head_dim = map(int, query.shape)
    if tuple(source_gram.shape[:2]) != (kv_heads, rows):
        raise ValueError("source Gram does not match query KV-head/row geometry")
    rank = int(source_gram.shape[-1])
    if int(source_gram.shape[-2]) != rank:
        raise ValueError("source Gram matrices must be square")
    if tuple(source_score_cross.shape) != (kv_heads, group, rows, rank):
        raise ValueError("source-score cross does not match query/source geometry")
    if not all(
        tensor.is_floating_point()
        for tensor in (query, source_gram, source_score_cross)
    ):
        raise TypeError("centered-score statistics must be floating point")
    return kv_heads, group, rows, head_dim, rank


def fit_centered_score_probe(
    query: Tensor,
    source_gram: Tensor,
    source_score_cross: Tensor,
    *,
    relative_damping: float = 1.0e-5,
    cg_iterations: int = 64,
    cg_relative_tolerance: float = 1.0e-6,
) -> CenteredScoreProbeFit:
    """Solve the centered-score normal equations with batched matrix-free CG.

    The supplied statistics represent valid causal rows.  For each row,
    ``source_gram`` is ``C_centered.T @ C_centered`` and
    ``source_score_cross`` is ``C_centered.T @ exact_centered_score``.
    """

    if relative_damping < 0 or cg_relative_tolerance < 0:
        raise ValueError("damping and CG tolerance must be nonnegative")
    if cg_iterations <= 0:
        raise ValueError("CG iterations must be positive")
    kv_heads, group, _, head_dim, rank = _validate_statistics(
        query, source_gram, source_score_cross
    )
    device = query.device
    dtype = query.dtype
    gram = source_gram.to(device=device, dtype=dtype)
    cross = source_score_cross.to(device=device, dtype=dtype)
    scale = math.sqrt(head_dim)
    right_hand_side = torch.einsum(
        "kgnd,kgnr->kgdr", query, cross
    ) / scale
    query_energy = query.square().sum(dim=-1)
    gram_trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    mean_hessian_diagonal = torch.einsum(
        "kgn,kn->kg", query_energy, gram_trace
    ) / float(head_dim * head_dim * rank)
    damping = relative_damping * mean_hessian_diagonal
    gram_diagonal = gram.diagonal(dim1=-2, dim2=-1)
    normal_diagonal = torch.einsum(
        "kgnd,knr->kgdr", query.square(), gram_diagonal
    ) / float(head_dim)
    normal_diagonal = normal_diagonal + damping[:, :, None, None]
    diagonal_floor = (
        torch.finfo(dtype).eps
        * normal_diagonal.amax(dim=(-2, -1), keepdim=True)
    )
    inverse_diagonal = torch.maximum(
        normal_diagonal, diagonal_floor
    ).reciprocal()

    def normal_operator(direction: Tensor) -> Tensor:
        transformed_query = torch.einsum(
            "kgnd,kgdr->kgnr", query, direction
        )
        transformed_scores = torch.einsum(
            "knrs,kgns->kgnr", gram, transformed_query
        )
        result = torch.einsum(
            "kgnd,kgnr->kgdr", query, transformed_scores
        ) / float(head_dim)
        return result + damping[:, :, None, None] * direction

    solution = torch.zeros(
        kv_heads,
        group,
        head_dim,
        rank,
        device=device,
        dtype=dtype,
    )
    residual = right_hand_side.clone()
    preconditioned_residual = residual * inverse_diagonal
    direction = preconditioned_residual.clone()
    residual_energy = residual.square().sum(dim=(-2, -1))
    residual_preconditioned_energy = (
        residual * preconditioned_residual
    ).sum(dim=(-2, -1))
    initial_energy = residual_energy.clone().clamp_min(torch.finfo(dtype).tiny)
    history = [float(torch.sqrt(residual_energy / initial_energy).max())]
    completed = 0
    for iteration in range(cg_iterations):
        image = normal_operator(direction)
        denominator = (direction * image).sum(dim=(-2, -1))
        active = residual_energy > (
            cg_relative_tolerance * cg_relative_tolerance * initial_energy
        )
        safe_denominator = torch.where(
            active,
            denominator,
            torch.ones_like(denominator),
        )
        alpha = torch.where(
            active,
            residual_preconditioned_energy / safe_denominator,
            torch.zeros_like(residual_preconditioned_energy),
        )
        solution = solution + alpha[:, :, None, None] * direction
        residual = residual - alpha[:, :, None, None] * image
        next_energy = residual.square().sum(dim=(-2, -1))
        relative = torch.sqrt(next_energy / initial_energy)
        history.append(float(relative.max()))
        completed = iteration + 1
        if bool((relative <= cg_relative_tolerance).all()):
            residual_energy = next_energy
            break
        next_preconditioned_residual = residual * inverse_diagonal
        next_preconditioned_energy = (
            residual * next_preconditioned_residual
        ).sum(dim=(-2, -1))
        beta = torch.where(
            active,
            next_preconditioned_energy
            / residual_preconditioned_energy.clamp_min(torch.finfo(dtype).tiny),
            torch.zeros_like(next_preconditioned_energy),
        )
        direction = (
            next_preconditioned_residual + beta[:, :, None, None] * direction
        )
        preconditioned_residual = next_preconditioned_residual
        residual_preconditioned_energy = next_preconditioned_energy
        residual_energy = next_energy
    weight = solution.reshape(kv_heads * group, head_dim, rank)
    return CenteredScoreProbeFit(
        weight=weight,
        iterations=completed,
        maximum_relative_residual=history[-1],
        relative_residual_history=tuple(history),
        damping=damping,
    )


def apply_qk_score_probe(
    query: Tensor,
    source: Tensor,
    weight: Tensor,
    *,
    heads_per_group: int,
) -> Tensor:
    """Apply ``(Q M) C.T / sqrt(head_dim)`` to one set of query rows."""

    if query.ndim != 2 or source.ndim != 3 or weight.ndim != 3:
        raise ValueError(
            "query, source, and weight must be [QH,D], [KVH,T,R], and [QH,D,R]"
        )
    query_heads, head_dim = map(int, query.shape)
    kv_heads, _, rank = map(int, source.shape)
    if (
        heads_per_group <= 0
        or query_heads != kv_heads * heads_per_group
        or tuple(weight.shape) != (query_heads, head_dim, rank)
    ):
        raise ValueError("query, source, and weight have incompatible GQA geometry")
    kv_index = torch.arange(query_heads, device=query.device) // heads_per_group
    transformed_query = torch.einsum(
        "hd,hdr->hr", query.float(), weight.to(query.device, torch.float32)
    ) / math.sqrt(head_dim)
    expanded_source = source.index_select(0, kv_index).float()
    return torch.einsum("hr,htr->ht", transformed_query, expanded_source)
