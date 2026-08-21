"""Balanced neuron repartitioning for row-parallel MLP output projections.

An MLP intermediate coordinate can be permuted consistently across the gate,
up, and down projections without changing the represented function.  This
module builds exact-capacity partitions of those coordinates.  It contains no
checkpoint mutation or distributed runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import torch
from torch import Tensor


def _finite_matrix(name: str, value: Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _normalize_rows(value: Tensor) -> Tensor:
    return value.float() / value.float().norm(dim=1, keepdim=True).clamp_min(1.0e-12)


def _canonicalize_unoriented_rows(value: Tensor) -> Tensor:
    """Choose one deterministic sign for unoriented one-dimensional subspaces."""

    work = _normalize_rows(value)
    pivots = work.abs().argmax(dim=1, keepdim=True)
    signs = work.gather(1, pivots).sign()
    signs[signs == 0] = 1
    return work * signs


def partition_index_sha256(groups: Sequence[Tensor]) -> str:
    permutation = torch.cat(
        [group.detach().to(device="cpu", dtype=torch.int64) for group in groups]
    ).contiguous()
    return hashlib.sha256(permutation.numpy().tobytes()).hexdigest()


def validate_balanced_partition(
    groups: Sequence[Tensor],
    *,
    width: int,
    tp_size: int,
) -> tuple[Tensor, ...]:
    if width <= 0 or tp_size <= 1 or width % tp_size:
        raise ValueError("width must be positively divisible by TP size")
    if len(groups) != tp_size:
        raise ValueError("partition group count differs from TP size")
    capacity = width // tp_size
    normalized = tuple(
        group.detach().to(device="cpu", dtype=torch.int64).contiguous()
        for group in groups
    )
    if any(group.ndim != 1 or int(group.numel()) != capacity for group in normalized):
        raise ValueError("partition groups do not have exact equal capacity")
    permutation = torch.cat(normalized)
    if int(permutation.min()) < 0 or int(permutation.max()) >= width:
        raise ValueError("partition contains an out-of-range neuron")
    if not torch.equal(permutation.sort().values, torch.arange(width)):
        raise ValueError("partition is not a permutation of all neurons")
    return normalized


def contiguous_balanced_partition(width: int, tp_size: int) -> tuple[Tensor, ...]:
    if width <= 0 or tp_size <= 1 or width % tp_size:
        raise ValueError("width must be positively divisible by TP size")
    capacity = width // tp_size
    return tuple(
        torch.arange(shard * capacity, (shard + 1) * capacity, dtype=torch.int64)
        for shard in range(tp_size)
    )


def random_balanced_partition(
    width: int,
    tp_size: int,
    *,
    seed: int,
) -> tuple[Tensor, ...]:
    if width <= 0 or tp_size <= 1 or width % tp_size:
        raise ValueError("width must be positively divisible by TP size")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(width, generator=generator)
    capacity = width // tp_size
    return tuple(
        permutation[shard * capacity : (shard + 1) * capacity].sort().values
        for shard in range(tp_size)
    )


@dataclass(frozen=True)
class NeuronSketchBank:
    """Train-only sketches used by the repartitioning controls."""

    weight_directions: Tensor
    contribution_directions: Tensor
    activation_energy: Tensor
    weight_energy: Tensor
    contribution_energy: Tensor
    diagnostics: dict[str, Any]


@torch.no_grad()
def build_neuron_sketch_bank(
    activation: Tensor,
    down_weight: Tensor,
    *,
    device: torch.device,
    weight_sketch_dim: int,
    activation_sketch_dim: int,
    contribution_output_sketch_dim: int,
    seed: int,
) -> NeuronSketchBank:
    """Sketch neuron directions and activation-weighted rank-one contributions.

    For neuron ``j``, the exact contribution is the outer product
    ``activation[:, j] outer down_weight[:, j]``.  Its low-dimensional sketch
    is the Kronecker product of independent activation and output sketches.
    Validation activations must not be passed to this function.
    """

    _finite_matrix("activation", activation)
    _finite_matrix("down_weight", down_weight)
    rows, neurons = map(int, activation.shape)
    hidden, weight_neurons = map(int, down_weight.shape)
    if neurons != weight_neurons:
        raise ValueError("activation and down weight have different neuron counts")
    if (
        min(weight_sketch_dim, activation_sketch_dim, contribution_output_sketch_dim)
        <= 0
        or contribution_output_sketch_dim > weight_sketch_dim
    ):
        raise ValueError("neuron sketch dimensions are invalid")
    if rows <= 0 or hidden <= 0:
        raise ValueError("activation/down-weight geometry is empty")

    work_activation = activation.to(device=device, dtype=torch.float32)
    work_weight = down_weight.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(seed)
    activation_projection = torch.randn(
        rows,
        activation_sketch_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) / math.sqrt(activation_sketch_dim)
    output_projection = torch.randn(
        hidden,
        weight_sketch_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) / math.sqrt(weight_sketch_dim)

    activation_sketch = work_activation.transpose(0, 1) @ activation_projection
    weight_sketch = work_weight.transpose(0, 1) @ output_projection
    activation_energy = work_activation.square().sum(dim=0)
    weight_energy = work_weight.square().sum(dim=0)
    contribution_energy = activation_energy * weight_energy

    normalized_activation = _normalize_rows(activation_sketch)
    normalized_output = _normalize_rows(
        weight_sketch[:, :contribution_output_sketch_dim]
    )
    contribution = (
        normalized_activation.unsqueeze(2) * normalized_output.unsqueeze(1)
    ).flatten(1)
    contribution = _normalize_rows(contribution)
    weight_directions = _canonicalize_unoriented_rows(weight_sketch)

    energies = {
        "activation": activation_energy,
        "weight": weight_energy,
        "contribution": contribution_energy,
    }
    energy_diagnostics = {
        name: {
            "minimum": float(value.min()),
            "maximum": float(value.max()),
            "mean": float(value.mean()),
            "sum": float(value.double().sum()),
        }
        for name, value in energies.items()
    }
    result = NeuronSketchBank(
        weight_directions=weight_directions.cpu().contiguous(),
        contribution_directions=contribution.cpu().contiguous(),
        activation_energy=activation_energy.cpu().contiguous(),
        weight_energy=weight_energy.cpu().contiguous(),
        contribution_energy=contribution_energy.cpu().contiguous(),
        diagnostics={
            "method": "independent_gaussian_two_sided_rank_one_sketch",
            "fit_split": "train",
            "validation_used": False,
            "seed": int(seed),
            "rows": rows,
            "neurons": neurons,
            "hidden_size": hidden,
            "weight_sketch_dim": int(weight_sketch_dim),
            "activation_sketch_dim": int(activation_sketch_dim),
            "contribution_output_sketch_dim": int(
                contribution_output_sketch_dim
            ),
            "contribution_feature_dim": int(
                activation_sketch_dim * contribution_output_sketch_dim
            ),
            "energy": energy_diagnostics,
        },
    )
    del work_activation, work_weight
    return result


def _balanced_greedy_assignment(
    scores: Tensor,
    sample_weights: Tensor,
    *,
    capacity: int,
) -> Tensor:
    neurons, clusters = map(int, scores.shape)
    if capacity <= 0 or capacity * clusters != neurons:
        raise ValueError("capacity does not exactly cover all samples")
    utilities = scores.double() * sample_weights.double().unsqueeze(1)
    order = torch.argsort(utilities.flatten(), descending=True, stable=True).tolist()
    assignments = [-1] * neurons
    remaining = [capacity] * clusters
    assigned = 0
    for flattened in order:
        neuron, cluster = divmod(flattened, clusters)
        if assignments[neuron] < 0 and remaining[cluster] > 0:
            assignments[neuron] = cluster
            remaining[cluster] -= 1
            assigned += 1
            if assigned == neurons:
                break
    if assigned != neurons or any(remaining):
        raise RuntimeError("balanced greedy assignment did not fill every cluster")
    return torch.tensor(assignments, dtype=torch.int64)


def _refine_balanced_swaps(
    scores: Tensor,
    assignments: Tensor,
    sample_weights: Tensor,
    *,
    rounds: int,
) -> tuple[Tensor, int]:
    result = assignments.clone()
    clusters = int(scores.shape[1])
    swaps = 0
    for _ in range(rounds):
        improved = False
        for first in range(clusters):
            for second in range(first + 1, clusters):
                # Earlier swaps in this round may have changed both groups.
                # Re-read their current membership for every pair so a swap
                # always exchanges one live member from each cluster.
                first_indices = torch.nonzero(
                    result == first, as_tuple=False
                ).flatten()
                second_indices = torch.nonzero(
                    result == second, as_tuple=False
                ).flatten()
                first_gain = sample_weights[first_indices] * (
                    scores[first_indices, second] - scores[first_indices, first]
                )
                second_gain = sample_weights[second_indices] * (
                    scores[second_indices, first] - scores[second_indices, second]
                )
                first_best = int(first_gain.argmax())
                second_best = int(second_gain.argmax())
                if float(first_gain[first_best] + second_gain[second_best]) > 1.0e-12:
                    first_neuron = int(first_indices[first_best])
                    second_neuron = int(second_indices[second_best])
                    result[first_neuron] = second
                    result[second_neuron] = first
                    swaps += 1
                    improved = True
        if not improved:
            break
    return result, swaps


@dataclass(frozen=True)
class BalancedClusteringResult:
    groups: tuple[Tensor, ...]
    assignments: Tensor
    centers: Tensor
    diagnostics: dict[str, Any]


@torch.no_grad()
def balanced_spherical_kmeans(
    features: Tensor,
    tp_size: int,
    *,
    sample_weights: Tensor | None,
    iterations: int,
    swap_rounds: int,
    seed: int,
) -> BalancedClusteringResult:
    """Cluster unit directions while enforcing exact equal TP capacities."""

    _finite_matrix("features", features)
    neurons, feature_dim = map(int, features.shape)
    if tp_size <= 1 or neurons % tp_size:
        raise ValueError("feature count must be divisible by TP size")
    if iterations <= 0 or swap_rounds < 0:
        raise ValueError("balanced k-means iteration controls are invalid")
    work = _normalize_rows(features.detach().to(device="cpu", dtype=torch.float32))
    if sample_weights is None:
        weights = torch.ones(neurons, dtype=torch.float32)
    else:
        weights = sample_weights.detach().to(device="cpu", dtype=torch.float32)
        if weights.ndim != 1 or int(weights.numel()) != neurons:
            raise ValueError("sample weights and feature rows differ")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("sample weights must be finite and positive")
        weights = weights / weights.mean()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.multinomial(weights, 1, generator=generator))
    center_indices = [first]
    closest = 1.0 - work @ work[first]
    for _ in range(1, tp_size):
        probabilities = weights * closest.clamp_min(0.0).square()
        probabilities[torch.tensor(center_indices)] = 0
        if float(probabilities.sum()) <= 0.0:
            remaining = torch.ones(neurons, dtype=torch.bool)
            remaining[torch.tensor(center_indices)] = False
            next_index = int(torch.nonzero(remaining, as_tuple=False)[0])
        else:
            next_index = int(
                torch.multinomial(probabilities, 1, generator=generator)
            )
        center_indices.append(next_index)
        closest = torch.minimum(closest, 1.0 - work @ work[next_index])
    centers = work[torch.tensor(center_indices)].clone()

    capacity = neurons // tp_size
    previous: Tensor | None = None
    history = []
    swap_counts = []
    converged = False
    for _ in range(iterations):
        scores = work @ centers.transpose(0, 1)
        assignments = _balanced_greedy_assignment(
            scores,
            weights,
            capacity=capacity,
        )
        assignments, swaps = _refine_balanced_swaps(
            scores,
            assignments,
            weights,
            rounds=swap_rounds,
        )
        objective = float(
            (
                weights
                * scores[torch.arange(neurons), assignments]
            ).sum()
            / weights.sum()
        )
        history.append(objective)
        swap_counts.append(swaps)
        if previous is not None and torch.equal(assignments, previous):
            converged = True
            break
        previous = assignments.clone()
        updated = []
        for cluster in range(tp_size):
            selected = assignments == cluster
            center = (work[selected] * weights[selected, None]).sum(dim=0)
            if float(center.norm()) <= 1.0e-12:
                center = work[torch.nonzero(selected, as_tuple=False)[0, 0]]
            updated.append(center / center.norm().clamp_min(1.0e-12))
        centers = torch.stack(updated)

    scores = work @ centers.transpose(0, 1)
    assignments = _balanced_greedy_assignment(scores, weights, capacity=capacity)
    assignments, final_swaps = _refine_balanced_swaps(
        scores,
        assignments,
        weights,
        rounds=swap_rounds,
    )
    final_objective = float(
        (weights * scores[torch.arange(neurons), assignments]).sum() / weights.sum()
    )
    groups = tuple(
        torch.nonzero(assignments == cluster, as_tuple=False).flatten().sort().values
        for cluster in range(tp_size)
    )
    order = sorted(range(tp_size), key=lambda cluster: int(groups[cluster][0]))
    groups = tuple(groups[cluster] for cluster in order)
    assignments = torch.empty(neurons, dtype=torch.int64)
    for cluster, group in enumerate(groups):
        assignments[group] = cluster
    centers = centers[torch.tensor(order)].contiguous()
    groups = validate_balanced_partition(groups, width=neurons, tp_size=tp_size)
    return BalancedClusteringResult(
        groups=groups,
        assignments=assignments,
        centers=centers,
        diagnostics={
            "method": "capacity_constrained_weighted_spherical_kmeans",
            "seed": int(seed),
            "neurons": neurons,
            "feature_dimension": feature_dim,
            "tp_size": int(tp_size),
            "capacity": capacity,
            "maximum_iterations": int(iterations),
            "iterations_run": len(history),
            "swap_rounds": int(swap_rounds),
            "swap_counts": swap_counts,
            "final_assignment_swaps": final_swaps,
            "converged_before_final_assignment": converged,
            "objective_history": history,
            "final_weighted_cosine_objective": final_objective,
            "partition_index_sha256": partition_index_sha256(groups),
        },
    )


__all__ = [
    "BalancedClusteringResult",
    "NeuronSketchBank",
    "balanced_spherical_kmeans",
    "build_neuron_sketch_bank",
    "contiguous_balanced_partition",
    "partition_index_sha256",
    "random_balanced_partition",
    "validate_balanced_partition",
]
