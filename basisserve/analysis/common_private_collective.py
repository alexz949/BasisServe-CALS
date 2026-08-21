"""Activation-aware common/private latent collective primitives.

The row-parallel operator is partitioned as ``W = [W_0 ... W_{P-1}]`` and
each TP shard owns matching activations ``X_p``.  A common/private decoder is

    W_p ~= U_s E_s,p + U_p E_p,

where the common code is reduced and the private codes are gathered.  This
module contains only offline fitting, exact collective simulation, accounting,
and quality metrics.  It does not implement a distributed runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence

import torch
from torch import Tensor

from basisserve.analysis.mlp_topk_chebyshev import polar_retract_columns
from basisserve.sketching.coordinate_selection import compute_uncentered_pod


def _matrix(name: str, value: Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _orthogonality_error(basis: Tensor) -> float:
    if basis.shape[1] == 0:
        return 0.0
    work = basis.double()
    gram = work.transpose(0, 1) @ work
    identity = torch.eye(int(work.shape[1]), dtype=torch.float64, device=work.device)
    return float((gram - identity).abs().max())


@dataclass(frozen=True)
class CommonPrivateBudget:
    baseline_rank: int
    shared_fraction: float
    shared_rank: int
    total_private_rank: int
    ideal_ring_units: int


def common_private_budgets(
    baseline_rank: int,
    shared_fractions: Sequence[float],
) -> tuple[CommonPrivateBudget, ...]:
    """Create integer plans while preserving ``2*rs + Rprivate = 2*r``."""

    if baseline_rank <= 0:
        raise ValueError("baseline rank must be positive")
    fractions = tuple(float(value) for value in shared_fractions)
    if not fractions or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions
    ):
        raise ValueError("shared fractions must be finite values in [0,1]")
    plans = []
    for fraction in fractions:
        shared = int(round(fraction * baseline_rank))
        private = 2 * baseline_rank - 2 * shared
        plans.append(
            CommonPrivateBudget(
                baseline_rank=int(baseline_rank),
                shared_fraction=fraction,
                shared_rank=shared,
                total_private_rank=private,
                ideal_ring_units=2 * shared + private,
            )
        )
    if len({(plan.shared_rank, plan.total_private_rank) for plan in plans}) != len(
        plans
    ):
        raise ValueError("integer rounding produced duplicate common/private plans")
    return tuple(plans)


def uniform_private_ranks(total_private_rank: int, tp_size: int) -> tuple[int, ...]:
    if total_private_rank < 0 or tp_size <= 0:
        raise ValueError("private rank and TP size are invalid")
    quotient, remainder = divmod(int(total_private_rank), int(tp_size))
    return tuple(quotient + (1 if shard < remainder else 0) for shard in range(tp_size))


def spectrum_aware_private_ranks(
    singular_values: Sequence[Tensor],
    total_private_rank: int,
) -> tuple[int, ...]:
    """Greedily select the largest available local residual singular values."""

    if total_private_rank < 0 or not singular_values:
        raise ValueError("spectrum allocation inputs are invalid")
    spectra = []
    for shard, values in enumerate(singular_values):
        if values.ndim != 1 or not bool(torch.isfinite(values).all()):
            raise ValueError(f"shard {shard} singular values are invalid")
        work = values.detach().to(device="cpu", dtype=torch.float64)
        if work.numel() and bool((work < 0).any()):
            raise ValueError("singular values must be nonnegative")
        if work.numel() > 1 and bool((work[:-1] < work[1:]).any()):
            raise ValueError("singular values must be nonincreasing")
        spectra.append(work)
    if sum(int(values.numel()) for values in spectra) < total_private_rank:
        raise ValueError("available spectra do not cover the private-rank budget")
    ranks = [0] * len(spectra)
    heap: list[tuple[float, int, int]] = []
    for shard, values in enumerate(spectra):
        if values.numel():
            heapq.heappush(heap, (-float(values[0]), shard, 0))
    for _ in range(total_private_rank):
        if not heap:
            raise RuntimeError("spectrum heap ended before the requested rank")
        _, shard, index = heapq.heappop(heap)
        ranks[shard] += 1
        next_index = index + 1
        if next_index < int(spectra[shard].numel()):
            heapq.heappush(
                heap,
                (-float(spectra[shard][next_index]), shard, next_index),
            )
    return tuple(ranks)


@dataclass(frozen=True)
class PrivatePODBank:
    bases: tuple[Tensor, ...]
    singular_values: tuple[Tensor, ...]
    probe_ranks: tuple[int, ...]
    spectrum_ranks: tuple[int, ...]
    diagnostics: tuple[dict[str, Any], ...]
    maximum_shared_private_orthogonality: float


@torch.no_grad()
def fit_private_pod_bank(
    local_outputs: Sequence[Tensor],
    shared_basis: Tensor,
    total_private_rank: int,
    *,
    maximum_private_ranks: Sequence[int] | None = None,
    device: torch.device,
    oversample: int,
    niter: int,
    seed: int,
    probe_multiplier: float = 1.5,
    probe_extra: int = 16,
) -> PrivatePODBank:
    """Fit certified adaptive local residual POD banks using training rows only.

    Probe ranks expand until the greedy allocation does not touch an
    unobserved spectral boundary.  The resulting allocation is therefore
    complete with respect to every computed approximate POD spectrum.
    """

    _matrix("shared_basis", shared_basis)
    if not local_outputs or total_private_rank <= 0:
        raise ValueError("private POD fitting requires a positive private budget")
    if oversample < 0 or niter < 0 or probe_multiplier < 1.0 or probe_extra <= 0:
        raise ValueError("private POD fitting controls are invalid")
    rows = int(local_outputs[0].shape[0])
    width = int(local_outputs[0].shape[1])
    for shard, output in enumerate(local_outputs):
        _matrix(f"local_outputs[{shard}]", output)
        if tuple(output.shape) != (rows, width):
            raise ValueError("local output matrices have different shapes")
    if int(shared_basis.shape[0]) != width:
        raise ValueError("shared basis and local outputs have different widths")
    shared_rank = int(shared_basis.shape[1])
    if _orthogonality_error(shared_basis) > 2.0e-5:
        raise ValueError("shared basis is not orthonormal")
    residual_cap = min(rows, width - shared_rank)
    if residual_cap <= 0:
        raise ValueError("shared basis leaves no private residual subspace")
    shard_count = len(local_outputs)
    if maximum_private_ranks is None:
        capacities = (residual_cap,) * shard_count
    else:
        if len(maximum_private_ranks) != shard_count:
            raise ValueError("private rank caps and shard count differ")
        capacities = tuple(
            min(residual_cap, int(value)) for value in maximum_private_ranks
        )
        if any(value <= 0 for value in capacities):
            raise ValueError("private rank caps must be positive")
    if total_private_rank > sum(capacities):
        raise ValueError("private budget exceeds the total residual rank")

    uniform = uniform_private_ranks(total_private_rank, shard_count)
    probes = [
        min(
            capacities[shard],
            max(
                rank,
                int(math.ceil(rank * probe_multiplier)),
                rank + probe_extra,
            ),
        )
        for shard, rank in enumerate(uniform)
    ]
    cached: list[tuple[Tensor, Tensor, dict[str, Any]] | None] = [None] * shard_count
    shared_device = shared_basis.to(device=device, dtype=torch.float32)

    def fit_shard(shard: int) -> None:
        output = local_outputs[shard].to(device=device, dtype=torch.float32)
        if shared_rank:
            output = output - (output @ shared_device) @ shared_device.transpose(0, 1)
        pod = compute_uncentered_pod(
            output,
            probes[shard],
            oversample=oversample,
            niter=niter,
            seed=seed + 104729 * shard + 17 * probes[shard],
        )
        cached[shard] = (
            pod.basis.detach().cpu().contiguous(),
            pod.singular_values[: probes[shard]].detach().cpu().contiguous(),
            pod.diagnostics(),
        )
        del output, pod

    for shard in range(shard_count):
        fit_shard(shard)
    expansion_rounds = 0
    while True:
        spectra = tuple(item[1] for item in cached if item is not None)
        if len(spectra) != shard_count:
            raise AssertionError("private POD cache is incomplete")
        allocation = spectrum_aware_private_ranks(spectra, total_private_rank)
        saturated = [
            shard
            for shard, rank in enumerate(allocation)
            if rank >= probes[shard] and probes[shard] < capacities[shard]
        ]
        if not saturated:
            break
        expansion_rounds += 1
        if expansion_rounds > 16:
            raise RuntimeError("private spectrum probes did not converge")
        for shard in saturated:
            probes[shard] = min(
                capacities[shard],
                max(
                    probes[shard] + probe_extra,
                    int(math.ceil(probes[shard] * probe_multiplier)),
                    allocation[shard] + probe_extra,
                ),
            )
            fit_shard(shard)

    bases = []
    singular_values = []
    diagnostics = []
    cross_errors = []
    for shard, item in enumerate(cached):
        if item is None:
            raise AssertionError("private POD cache is incomplete")
        basis, values, metadata = item
        basis_device = basis.to(device=device, dtype=torch.float32)
        if shared_rank:
            basis_device = basis_device - shared_device @ (
                shared_device.transpose(0, 1) @ basis_device
            )
        basis_device, retraction = polar_retract_columns(basis_device)
        cross = (
            float(
                (shared_device.double().transpose(0, 1) @ basis_device.double())
                .abs()
                .max()
            )
            if shared_rank
            else 0.0
        )
        if cross > 2.0e-5:
            raise RuntimeError("private POD basis is not orthogonal to shared basis")
        bases.append(basis_device.cpu().contiguous())
        singular_values.append(values)
        diagnostics.append(
            {
                **metadata,
                "shard": shard,
                "probe_rank": probes[shard],
                "shared_private_orthogonality_max_abs": cross,
                "polar_retraction": retraction,
            }
        )
        cross_errors.append(cross)
    return PrivatePODBank(
        bases=tuple(bases),
        singular_values=tuple(singular_values),
        probe_ranks=tuple(probes),
        spectrum_ranks=allocation,
        diagnostics=tuple(diagnostics),
        maximum_shared_private_orthogonality=max(cross_errors, default=0.0),
    )


@dataclass(frozen=True)
class CommonPrivateFactors:
    shared_basis: Tensor
    private_bases: tuple[Tensor, ...]
    baseline_rank: int
    allocation: str
    fit_split: str = "train"
    validation_used_for_fit: bool = False

    @property
    def shared_rank(self) -> int:
        return int(self.shared_basis.shape[1])

    @property
    def private_ranks(self) -> tuple[int, ...]:
        return tuple(int(value.shape[1]) for value in self.private_bases)

    @property
    def total_private_rank(self) -> int:
        return sum(self.private_ranks)


def factors_from_bank(
    shared_basis: Tensor,
    bank: PrivatePODBank,
    private_ranks: Sequence[int],
    *,
    baseline_rank: int,
    allocation: str,
) -> CommonPrivateFactors:
    ranks = tuple(map(int, private_ranks))
    if len(ranks) != len(bank.bases) or any(rank < 0 for rank in ranks):
        raise ValueError("private rank allocation and POD bank differ")
    if any(rank > int(basis.shape[1]) for rank, basis in zip(ranks, bank.bases)):
        raise ValueError("private rank allocation exceeds a POD probe")
    return CommonPrivateFactors(
        shared_basis=shared_basis.float().contiguous(),
        private_bases=tuple(
            basis[:, :rank].float().contiguous()
            for rank, basis in zip(ranks, bank.bases)
        ),
        baseline_rank=int(baseline_rank),
        allocation=str(allocation),
    )


def shared_only_factors(
    shared_basis: Tensor,
    *,
    baseline_rank: int,
    tp_size: int,
) -> CommonPrivateFactors:
    if tp_size <= 0:
        raise ValueError("TP size must be positive")
    return CommonPrivateFactors(
        shared_basis=shared_basis.float().contiguous(),
        private_bases=tuple(
            torch.empty(
                int(shared_basis.shape[0]),
                0,
                dtype=torch.float32,
                device=shared_basis.device,
            )
            for _ in range(tp_size)
        ),
        baseline_rank=int(baseline_rank),
        allocation="shared_only",
    )


@dataclass(frozen=True)
class CommonPrivateEncoders:
    shared: tuple[Tensor, ...]
    private: tuple[Tensor, ...]


@torch.no_grad()
def compile_common_private_encoders(
    weight_shards: Sequence[Tensor],
    factors: CommonPrivateFactors,
) -> CommonPrivateEncoders:
    if len(weight_shards) != len(factors.private_bases):
        raise ValueError("weight and private shard counts differ")
    shared = factors.shared_basis.float()
    width = int(shared.shape[0])
    shared_encoders = []
    private_encoders = []
    for shard, (weight, private) in enumerate(
        zip(weight_shards, factors.private_bases)
    ):
        _matrix(f"weight_shards[{shard}]", weight)
        if int(weight.shape[0]) != width:
            raise ValueError("weight shard and output basis widths differ")
        work = weight.float()
        shared_encoder = shared.transpose(0, 1) @ work
        residual = work - shared @ shared_encoder
        private_encoder = private.float().transpose(0, 1) @ residual
        shared_encoders.append(shared_encoder.contiguous())
        private_encoders.append(private_encoder.contiguous())
    return CommonPrivateEncoders(
        shared=tuple(shared_encoders),
        private=tuple(private_encoders),
    )


@torch.no_grad()
def reconstruct_weight_shards(
    factors: CommonPrivateFactors,
    encoders: CommonPrivateEncoders,
) -> tuple[Tensor, ...]:
    if not (
        len(factors.private_bases) == len(encoders.shared) == len(encoders.private)
    ):
        raise ValueError("factor and encoder shard counts differ")
    shared = factors.shared_basis.float()
    return tuple(
        (
            shared @ shared_encoder.float() + private.float() @ private_encoder.float()
        ).contiguous()
        for private, shared_encoder, private_encoder in zip(
            factors.private_bases,
            encoders.shared,
            encoders.private,
        )
    )


@torch.no_grad()
def simulate_latent_collectives(
    activation_shards: Sequence[Tensor],
    factors: CommonPrivateFactors,
    encoders: CommonPrivateEncoders,
) -> dict[str, Tensor]:
    """Simulate AllReduce plus ordered AllGather exactly on one device."""

    if not (
        len(activation_shards)
        == len(factors.private_bases)
        == len(encoders.shared)
        == len(encoders.private)
    ):
        raise ValueError("activation, factor, and encoder shard counts differ")
    shared_codes = []
    private_codes = []
    rows = int(activation_shards[0].shape[0])
    for shard, (activation, shared_encoder, private_encoder) in enumerate(
        zip(activation_shards, encoders.shared, encoders.private)
    ):
        _matrix(f"activation_shards[{shard}]", activation)
        if int(activation.shape[0]) != rows:
            raise ValueError("activation shards have different row counts")
        shared_codes.append(activation.float() @ shared_encoder.float().transpose(0, 1))
        private_codes.append(
            activation.float() @ private_encoder.float().transpose(0, 1)
        )
    shared_code = torch.stack(shared_codes, dim=0).sum(dim=0)
    shared_output = shared_code @ factors.shared_basis.float().transpose(0, 1)
    if factors.total_private_rank:
        gathered_code = torch.cat(private_codes, dim=1)
        gathered_decoder = torch.cat(factors.private_bases, dim=1).float()
        private_output = gathered_code @ gathered_decoder.transpose(0, 1)
    else:
        gathered_code = torch.empty(rows, 0, device=shared_output.device)
        private_output = torch.zeros_like(shared_output)
    return {
        "shared_code": shared_code,
        "gathered_private_code": gathered_code,
        "shared_output": shared_output,
        "private_output": private_output,
        "output": shared_output + private_output,
    }


def collective_accounting(
    factors: CommonPrivateFactors,
    *,
    input_shard_widths: Sequence[int],
    active_tokens: int,
    dtype_bytes: int,
) -> dict[str, Any]:
    tp_size = len(factors.private_bases)
    if (
        tp_size <= 0
        or len(input_shard_widths) != tp_size
        or any(int(width) <= 0 for width in input_shard_widths)
        or active_tokens <= 0
        or dtype_bytes <= 0
    ):
        raise ValueError("collective accounting inputs are invalid")
    shared_rank = factors.shared_rank
    private_ranks = factors.private_ranks
    private_total = sum(private_ranks)
    padded_private = tp_size * max(private_ranks, default=0)
    ring_factor = (tp_size - 1) / tp_size
    allreduce_bytes = 2.0 * ring_factor * active_tokens * shared_rank * dtype_bytes
    allgather_ideal_bytes = ring_factor * active_tokens * private_total * dtype_bytes
    allgather_padded_bytes = ring_factor * active_tokens * padded_private * dtype_bytes
    output_width = int(factors.shared_basis.shape[0])
    decoder_parameters = output_width * (shared_rank + private_total)
    local_encoder_parameters = tuple(
        (shared_rank + private_rank) * int(input_width)
        for private_rank, input_width in zip(private_ranks, input_shard_widths)
    )
    return {
        "tp_size": tp_size,
        "active_tokens": int(active_tokens),
        "dtype_bytes": int(dtype_bytes),
        "shared_rank": shared_rank,
        "private_ranks": list(private_ranks),
        "total_private_rank": private_total,
        "ideal_ring_units": 2 * shared_rank + private_total,
        "padded_ring_units": 2 * shared_rank + padded_private,
        "allreduce_ring_bytes_per_rank": allreduce_bytes,
        "allgather_ideal_ring_bytes_per_rank": allgather_ideal_bytes,
        "allgather_padded_ring_bytes_per_rank": allgather_padded_bytes,
        "total_ideal_ring_bytes_per_rank": allreduce_bytes + allgather_ideal_bytes,
        "total_padded_ring_bytes_per_rank": allreduce_bytes + allgather_padded_bytes,
        "collective_launches": int(shared_rank > 0) + int(private_total > 0),
        "decoder_parameters": decoder_parameters,
        "decoder_bytes_read": decoder_parameters * dtype_bytes,
        "local_encoder_parameters_per_rank": list(local_encoder_parameters),
        "total_local_encoder_parameters": sum(local_encoder_parameters),
        "maximum_local_encoder_parameters": max(local_encoder_parameters),
        "local_encoder_bytes_read_per_rank": [
            value * dtype_bytes for value in local_encoder_parameters
        ],
        "decoder_macs_per_token": output_width * (shared_rank + private_total),
        "local_encoder_macs_per_token_per_rank": list(local_encoder_parameters),
    }


def _distribution(values: Tensor) -> dict[str, float]:
    work = values.detach().to(device="cpu", dtype=torch.float64)
    if work.numel() == 0:
        return {name: 0.0 for name in ("mean", "median", "p90", "p95", "p99")}
    quantiles = torch.quantile(
        work,
        torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64),
    )
    return {
        "mean": float(work.mean()),
        "median": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
    }


@torch.no_grad()
def common_private_output_metrics(
    teacher_output: Tensor,
    local_outputs: Sequence[Tensor],
    factors: CommonPrivateFactors,
    *,
    device: torch.device,
    chunk_size: int,
) -> dict[str, Any]:
    """Evaluate exact simulated collective output without materializing it all."""

    _matrix("teacher_output", teacher_output)
    if not local_outputs or len(local_outputs) != len(factors.private_bases):
        raise ValueError("local output and private factor shard counts differ")
    rows, width = map(int, teacher_output.shape)
    for shard, output in enumerate(local_outputs):
        _matrix(f"local_outputs[{shard}]", output)
        if tuple(output.shape) != (rows, width):
            raise ValueError("local and teacher output shapes differ")
    if chunk_size <= 0:
        raise ValueError("metric chunk size must be positive")
    shared = factors.shared_basis.to(device=device, dtype=torch.float32)
    private_decoders = tuple(
        basis.to(device=device, dtype=torch.float32) for basis in factors.private_bases
    )
    private_encoders = tuple(
        (
            basis - shared @ (shared.transpose(0, 1) @ basis)
            if factors.shared_rank and basis.shape[1]
            else basis
        )
        for basis in private_decoders
    )
    totals = {name: 0.0 for name in ("teacher", "prediction", "cross", "error")}
    per_token = []
    near_zero = 0
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        teacher = teacher_output[start:stop].to(device=device, dtype=torch.float32)
        local = [
            output[start:stop].to(device=device, dtype=torch.float32)
            for output in local_outputs
        ]
        prediction = torch.zeros_like(teacher)
        if factors.shared_rank:
            dense = torch.stack(local, dim=0).sum(dim=0)
            prediction.add_((dense @ shared) @ shared.transpose(0, 1))
        for output, encoder, decoder in zip(
            local,
            private_encoders,
            private_decoders,
        ):
            if encoder.shape[1]:
                prediction.add_((output @ encoder) @ decoder.transpose(0, 1))
        teacher64 = teacher.double()
        prediction64 = prediction.double()
        error64 = teacher64 - prediction64
        teacher_per = teacher64.square().sum(dim=1)
        error_per = error64.square().sum(dim=1)
        totals["teacher"] += float(teacher_per.sum())
        totals["prediction"] += float(prediction64.square().sum())
        totals["cross"] += float((teacher64 * prediction64).sum())
        totals["error"] += float(error_per.sum())
        valid = teacher_per > 1.0e-24
        near_zero += int((~valid).sum())
        per_token.append((error_per[valid] / teacher_per[valid]).cpu())
    teacher_energy = totals["teacher"]
    prediction_energy = totals["prediction"]
    if teacher_energy <= 0.0:
        raise ValueError("teacher output energy is zero")
    normalized_cross = totals["cross"] / teacher_energy
    prediction_teacher_energy = prediction_energy / teacher_energy
    relative_mse = totals["error"] / teacher_energy
    cosine_denominator = math.sqrt(teacher_energy * prediction_energy)
    identity = 1.0 + prediction_teacher_energy - 2.0 * normalized_cross
    return {
        "relative_mse": relative_mse,
        "normalized_cross": normalized_cross,
        "prediction_teacher_energy": prediction_teacher_energy,
        "cosine_similarity": (
            totals["cross"] / cosine_denominator if cosine_denominator > 0.0 else 0.0
        ),
        "relative_mse_identity": identity,
        "relative_mse_identity_absolute_discrepancy": abs(relative_mse - identity),
        "per_token_relative_squared_error": _distribution(
            torch.cat(per_token) if per_token else torch.empty(0)
        ),
        "near_zero_teacher_tokens": near_zero,
        "teacher_energy": teacher_energy,
        "prediction_energy": prediction_energy,
    }


__all__ = [
    "CommonPrivateBudget",
    "CommonPrivateEncoders",
    "CommonPrivateFactors",
    "PrivatePODBank",
    "collective_accounting",
    "common_private_budgets",
    "common_private_output_metrics",
    "compile_common_private_encoders",
    "factors_from_bank",
    "fit_private_pod_bank",
    "reconstruct_weight_shards",
    "shared_only_factors",
    "simulate_latent_collectives",
    "spectrum_aware_private_ranks",
    "uniform_private_ranks",
]
