"""Output-aware post-RoPE K/Q closure for compressed-V GQA attention.

The routines in this module never construct an explicit Jacobian.  They
differentiate the causal softmax analytically and expose matrix-free JVP and
VJP operations suitable for Gauss--Newton conjugate gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class C1KFactorGradient:
    """Gradients for shared Key and head-specific Query projectors."""

    key: Tensor
    query: Tensor


@dataclass(frozen=True)
class C1KOutputMetrics:
    """Sufficient scalar metrics for one collection of attention windows."""

    output_squared_error: float
    target_output_energy: float
    causal_score_squared_error: float
    causal_dense_score_energy: float
    tokens: int

    @property
    def relative_output_mse(self) -> float:
        return self.output_squared_error / max(self.target_output_energy, 1.0e-300)

    @property
    def relative_causal_score_error(self) -> float:
        return self.causal_score_squared_error / max(
            self.causal_dense_score_energy,
            1.0e-300,
        )


def _validate_geometry(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decoder: Tensor,
    key_projector: Tensor,
    query_projector: Tensor,
) -> tuple[int, int, int, int, int, int, int]:
    if query.ndim != 5:
        raise ValueError("query must have shape [batch, groups, heads, sequence, head_dim]")
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError("key/value must have shape [batch, groups, sequence, width]")
    batch, groups, heads, sequence, head_dim = map(int, query.shape)
    if tuple(key.shape[:3]) != (batch, groups, sequence):
        raise ValueError("query and key batch/group/sequence geometry differs")
    if tuple(value.shape[:3]) != (batch, groups, sequence):
        raise ValueError("query and value batch/group/sequence geometry differs")
    value_rank = int(value.shape[-1])
    if decoder.ndim != 4 or tuple(decoder.shape[:3]) != (groups, heads, value_rank):
        raise ValueError("decoder must have shape [groups, heads, value_rank, output_dim]")
    if tuple(key_projector.shape[:2]) != (groups, head_dim):
        raise ValueError("Key projector must have shape [groups, head_dim, rank]")
    if tuple(query_projector.shape[:3]) != (groups, heads, head_dim):
        raise ValueError(
            "Query projector must have shape [groups, heads, head_dim, rank]"
        )
    rank = int(key_projector.shape[-1])
    if int(query_projector.shape[-1]) != rank:
        raise ValueError("Key and Query projector ranks differ")
    if key.shape[-1] != head_dim:
        raise ValueError("Query and Key head dimensions differ")
    return batch, groups, heads, sequence, head_dim, value_rank, rank


def _causal_probabilities(
    projected_query: Tensor,
    projected_key: Tensor,
    *,
    query_start: int,
    scale: float,
) -> tuple[Tensor, Tensor]:
    scores = torch.einsum(
        "bghqr,bgtr->bghqt",
        projected_query,
        projected_key,
    ).mul_(scale)
    query_positions = torch.arange(
        query_start,
        query_start + scores.shape[-2],
        device=scores.device,
    )
    key_positions = torch.arange(scores.shape[-1], device=scores.device)
    valid = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    probabilities = torch.softmax(
        scores.masked_fill(~valid.view(1, 1, 1, *valid.shape), -torch.inf),
        dim=-1,
        dtype=torch.float32,
    )
    return probabilities, valid


def _decoded_output(probabilities: Tensor, value: Tensor, decoder: Tensor) -> Tensor:
    attended = torch.einsum("bghqt,bgtv->bghqv", probabilities, value)
    return torch.einsum("bghqv,ghvo->bqo", attended, decoder)


def c1k_teacher_and_student_output(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decoder: Tensor,
    key_projector: Tensor,
    query_projector: Tensor,
    *,
    query_chunk_size: int,
    scale: float | None = None,
) -> tuple[Tensor, Tensor, C1KOutputMetrics]:
    """Return Dense-K/C1-V target, C1-K prediction, and causal metrics."""

    batch, _, _, sequence, head_dim, _, _ = _validate_geometry(
        query,
        key,
        value,
        decoder,
        key_projector,
        query_projector,
    )
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    scale = head_dim**-0.5 if scale is None else float(scale)
    query = query.float()
    key = key.float()
    value = value.float()
    decoder = decoder.float()
    projected_query = torch.einsum("bghqd,ghdr->bghqr", query, query_projector.float())
    projected_key = torch.einsum("bgtd,gdr->bgtr", key, key_projector.float())
    teacher_chunks = []
    student_chunks = []
    output_error = 0.0
    target_energy = 0.0
    score_error = 0.0
    score_energy = 0.0
    for start in range(0, sequence, query_chunk_size):
        stop = min(start + query_chunk_size, sequence)
        teacher_probability, valid = _causal_probabilities(
            query[:, :, :, start:stop],
            key,
            query_start=start,
            scale=scale,
        )
        student_probability, _ = _causal_probabilities(
            projected_query[:, :, :, start:stop],
            projected_key,
            query_start=start,
            scale=scale,
        )
        teacher = _decoded_output(teacher_probability, value, decoder)
        student = _decoded_output(student_probability, value, decoder)
        teacher_chunks.append(teacher)
        student_chunks.append(student)
        output_error += float((teacher - student).double().square().sum())
        target_energy += float(teacher.double().square().sum())

        dense_score = torch.einsum(
            "bghqd,bgtd->bghqt",
            query[:, :, :, start:stop],
            key,
        ).mul_(scale)
        compressed_score = torch.einsum(
            "bghqr,bgtr->bghqt",
            projected_query[:, :, :, start:stop],
            projected_key,
        ).mul_(scale)
        valid_scores = valid.view(1, 1, 1, *valid.shape)
        score_error += float(
            (dense_score - compressed_score)[valid_scores.expand_as(dense_score)]
            .double()
            .square()
            .sum()
        )
        score_energy += float(
            dense_score[valid_scores.expand_as(dense_score)].double().square().sum()
        )
    metrics = C1KOutputMetrics(
        output_squared_error=output_error,
        target_output_energy=target_energy,
        causal_score_squared_error=score_error,
        causal_dense_score_energy=score_energy,
        tokens=batch * sequence,
    )
    return torch.cat(teacher_chunks, dim=1), torch.cat(student_chunks, dim=1), metrics


def c1k_jvp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decoder: Tensor,
    key_projector: Tensor,
    query_projector: Tensor,
    *,
    delta_key_projector: Tensor | None = None,
    delta_query_projector: Tensor | None = None,
    query_chunk_size: int,
    scale: float | None = None,
) -> Tensor:
    """Apply the C1-K output Jacobian to one projector perturbation."""

    batch, _, _, sequence, head_dim, _, _ = _validate_geometry(
        query,
        key,
        value,
        decoder,
        key_projector,
        query_projector,
    )
    if delta_key_projector is None and delta_query_projector is None:
        raise ValueError("at least one C1-K perturbation is required")
    if delta_key_projector is not None and delta_key_projector.shape != key_projector.shape:
        raise ValueError("delta Key projector geometry differs")
    if delta_query_projector is not None and delta_query_projector.shape != query_projector.shape:
        raise ValueError("delta Query projector geometry differs")
    scale = head_dim**-0.5 if scale is None else float(scale)
    query = query.float()
    key = key.float()
    value = value.float()
    decoder = decoder.float()
    projected_query = torch.einsum("bghqd,ghdr->bghqr", query, query_projector.float())
    projected_key = torch.einsum("bgtd,gdr->bgtr", key, key_projector.float())
    delta_query = (
        torch.zeros_like(projected_query)
        if delta_query_projector is None
        else torch.einsum(
            "bghqd,ghdr->bghqr",
            query,
            delta_query_projector.float(),
        )
    )
    delta_key = (
        torch.zeros_like(projected_key)
        if delta_key_projector is None
        else torch.einsum("bgtd,gdr->bgtr", key, delta_key_projector.float())
    )
    chunks = []
    for start in range(0, sequence, query_chunk_size):
        stop = min(start + query_chunk_size, sequence)
        probability, _ = _causal_probabilities(
            projected_query[:, :, :, start:stop],
            projected_key,
            query_start=start,
            scale=scale,
        )
        delta_score = (
            torch.einsum(
                "bghqr,bgtr->bghqt",
                delta_query[:, :, :, start:stop],
                projected_key,
            )
            + torch.einsum(
                "bghqr,bgtr->bghqt",
                projected_query[:, :, :, start:stop],
                delta_key,
            )
        ).mul_(scale)
        delta_probability = probability * (
            delta_score - (probability * delta_score).sum(dim=-1, keepdim=True)
        )
        chunks.append(_decoded_output(delta_probability, value, decoder))
    return torch.cat(chunks, dim=1).reshape(batch, sequence, decoder.shape[-1])


def c1k_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decoder: Tensor,
    key_projector: Tensor,
    query_projector: Tensor,
    output_cotangent: Tensor,
    *,
    query_chunk_size: int,
    scale: float | None = None,
) -> C1KFactorGradient:
    """Apply the transpose C1-K output Jacobian without autograd."""

    batch, groups, heads, sequence, head_dim, _, rank = _validate_geometry(
        query,
        key,
        value,
        decoder,
        key_projector,
        query_projector,
    )
    expected_output = (batch, sequence, int(decoder.shape[-1]))
    if tuple(output_cotangent.shape) != expected_output:
        raise ValueError(f"output cotangent must have shape {expected_output}")
    scale = head_dim**-0.5 if scale is None else float(scale)
    query = query.float()
    key = key.float()
    value = value.float()
    decoder = decoder.float()
    projected_query = torch.einsum("bghqd,ghdr->bghqr", query, query_projector.float())
    projected_key = torch.einsum("bgtd,gdr->bgtr", key, key_projector.float())
    key_gradient = torch.zeros(
        groups,
        head_dim,
        rank,
        device=query.device,
        dtype=torch.float32,
    )
    query_gradient = torch.zeros(
        groups,
        heads,
        head_dim,
        rank,
        device=query.device,
        dtype=torch.float32,
    )
    for start in range(0, sequence, query_chunk_size):
        stop = min(start + query_chunk_size, sequence)
        projected_query_chunk = projected_query[:, :, :, start:stop]
        probability, _ = _causal_probabilities(
            projected_query_chunk,
            projected_key,
            query_start=start,
            scale=scale,
        )
        attended_cotangent = torch.einsum(
            "bqo,ghvo->bghqv",
            output_cotangent[:, start:stop].float(),
            decoder,
        )
        probability_cotangent = torch.einsum(
            "bghqv,bgtv->bghqt",
            attended_cotangent,
            value,
        )
        score_cotangent = probability * (
            probability_cotangent
            - (probability * probability_cotangent).sum(dim=-1, keepdim=True)
        )
        projected_query_cotangent = torch.einsum(
            "bghqt,bgtr->bghqr",
            score_cotangent,
            projected_key,
        ).mul_(scale)
        projected_key_cotangent = torch.einsum(
            "bghqt,bghqr->bgtr",
            score_cotangent,
            projected_query_chunk,
        ).mul_(scale)
        query_gradient.add_(
            torch.einsum(
                "bghqd,bghqr->ghdr",
                query[:, :, :, start:stop],
                projected_query_cotangent,
            )
        )
        key_gradient.add_(
            torch.einsum("bgtd,bgtr->gdr", key, projected_key_cotangent)
        )
    return C1KFactorGradient(key=key_gradient, query=query_gradient)


def canonicalize_c1k_factors(
    key_projector: Tensor,
    query_projector: Tensor,
) -> tuple[Tensor, Tensor]:
    """Orthonormalize shared K coordinates and preserve every QK score."""

    if key_projector.ndim != 3 or query_projector.ndim != 4:
        raise ValueError("C1-K factor ranks are invalid")
    if (
        key_projector.shape[0] != query_projector.shape[0]
        or key_projector.shape[1] != query_projector.shape[2]
        or key_projector.shape[2] != query_projector.shape[3]
    ):
        raise ValueError("C1-K factor geometry differs")
    orthonormal_key, triangular = torch.linalg.qr(key_projector.float(), mode="reduced")
    canonical_query = torch.einsum(
        "ghdr,grs->ghds",
        query_projector.float(),
        triangular.mT,
    )
    return orthonormal_key.contiguous(), canonical_query.contiguous()


def conjugate_gradient(
    operator,
    right_hand_side: Tensor,
    *,
    max_iterations: int,
    relative_tolerance: float,
) -> tuple[Tensor, dict[str, float | int | bool]]:
    """Solve one positive-semidefinite matrix-free normal equation."""

    if max_iterations <= 0 or relative_tolerance < 0:
        raise ValueError("invalid CG controls")
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    direction = residual.clone()
    rhs_norm = float(torch.linalg.vector_norm(right_hand_side.double()))
    residual_square = torch.sum(residual.double() * residual.double())
    converged = rhs_norm == 0.0
    negative_curvature = False
    iterations = 0
    for iteration in range(max_iterations):
        if converged:
            break
        image = operator(direction)
        curvature = torch.sum(direction.double() * image.double())
        if not torch.isfinite(curvature) or float(curvature) <= 0.0:
            negative_curvature = True
            break
        step = residual_square / curvature
        solution.add_(direction, alpha=float(step))
        residual.add_(image, alpha=-float(step))
        next_square = torch.sum(residual.double() * residual.double())
        iterations = iteration + 1
        relative_residual = math.sqrt(float(next_square)) / max(rhs_norm, 1.0e-300)
        if relative_residual <= relative_tolerance:
            residual_square = next_square
            converged = True
            break
        direction.mul_(float(next_square / residual_square)).add_(residual)
        residual_square = next_square
    return solution, {
        "iterations": iterations,
        "converged": converged,
        "negative_curvature": negative_curvature,
        "relative_residual": math.sqrt(float(residual_square)) / max(rhs_norm, 1.0e-300),
    }
