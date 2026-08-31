"""Canonical representation controls for rigorous C1 experiments.

The helpers in this module are deliberately offline and model-agnostic.  They
define the two representations that must be compared before introducing a
distributed runtime:

    LR-AllReduce:  (sum_p Z_p F_p) D
    C1-AllGather:   sum_p (Z_p E_p) D_p

The LR-AllReduce fitter solves the globally optimal activation-weighted
rank-constrained output projection.  It is therefore a strong shared-decoder
control rather than a deliberately weak baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class StrongLRAllReduceFactors:
    """One activation-aware shared-decoder solution.

    ``input_factor`` is the concatenation of the source-specific encoders and
    has shape ``[input_width, rank]``.  ``decoder`` has shape
    ``[rank, output_width]`` and is replicated by a TP runtime.
    """

    rank: int
    input_factor: Tensor
    decoder: Tensor
    singular_values: Tensor
    metrics: dict[str, float | int | str]

    def reconstructed_weight(self) -> Tensor:
        """Return the logical PyTorch weight ``[output_width, input_width]``."""

        return self.decoder.transpose(0, 1) @ self.input_factor.transpose(0, 1)


def _matrix(name: str, value: Tensor) -> tuple[int, int]:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return int(value.shape[0]), int(value.shape[1])


def _symmetric_covariance(name: str, value: Tensor, width: int) -> Tensor:
    rows, columns = _matrix(name, value)
    if (rows, columns) != (width, width):
        raise ValueError(f"{name} must have shape {(width, width)}")
    return 0.5 * (value + value.transpose(0, 1))


def _canonicalize_columns(value: Tensor) -> Tensor:
    pivots = value.abs().argmax(dim=0)
    columns = torch.arange(value.shape[1], device=value.device)
    signs = value[pivots, columns].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return value * signs.unsqueeze(0)


def trace_damped_covariance(
    covariance: Tensor,
    *,
    relative_damping: float,
) -> tuple[Tensor, float]:
    """Apply the same trace-relative ridge used by the current C1 fitter."""

    if relative_damping < 0:
        raise ValueError("relative damping must be nonnegative")
    width, columns = _matrix("covariance", covariance)
    if width != columns:
        raise ValueError("covariance must be square")
    work = 0.5 * (covariance + covariance.transpose(0, 1))
    absolute = relative_damping * float(torch.trace(work)) / width
    work = work.clone()
    if absolute:
        work.diagonal().add_(absolute)
    return work, absolute


def relative_output_mse(weight: Tensor, approximation: Tensor, covariance: Tensor) -> float:
    """Return covariance-weighted output MSE relative to the dense operator."""

    output_width, input_width = _matrix("weight", weight)
    if tuple(approximation.shape) != (output_width, input_width):
        raise ValueError("approximation and weight have different shapes")
    covariance = _symmetric_covariance("covariance", covariance, input_width)
    residual = weight - approximation
    numerator = ((residual @ covariance) * residual).sum().clamp_min(0)
    denominator = ((weight @ covariance) * weight).sum().clamp_min(
        torch.finfo(weight.dtype).tiny
    )
    return float(numerator / denominator)


@torch.no_grad()
def fit_strong_lr_allreduce_rank_bank(
    weight: Tensor,
    fit_covariance: Tensor,
    heldout_covariance: Tensor,
    *,
    ranks: Sequence[int],
    covariance_damping: float = 1.0e-5,
    work_dtype: torch.dtype = torch.float32,
    factor_dtype: torch.dtype = torch.bfloat16,
) -> tuple[StrongLRAllReduceFactors, ...]:
    """Fit globally optimal activation-aware LR-AllReduce factors.

    The leading eigenspace of ``W C W^T`` is the global minimizer of

    ``||W - D^T F^T||_C^2``

    for the requested rank.  One eigendecomposition is shared across all
    ranks, so rank sweeps do not accidentally give one method more solver
    effort than another.
    """

    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("work dtype must be float32 or float64")
    if factor_dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError("unsupported factor dtype")
    output_width, input_width = _matrix("weight", weight)
    selected_ranks = tuple(dict.fromkeys(int(rank) for rank in ranks))
    if not selected_ranks:
        raise ValueError("at least one LR-AllReduce rank is required")
    maximum_rank = min(output_width, input_width)
    if any(rank <= 0 or rank > maximum_rank for rank in selected_ranks):
        raise ValueError(f"ranks must lie in [1, {maximum_rank}]")

    device = weight.device
    work_weight = weight.detach().to(device=device, dtype=work_dtype)
    fit_raw = _symmetric_covariance(
        "fit covariance",
        fit_covariance.to(device=device, dtype=work_dtype),
        input_width,
    )
    heldout = _symmetric_covariance(
        "heldout covariance",
        heldout_covariance.to(device=device, dtype=work_dtype),
        input_width,
    )
    fit_damped, absolute_damping = trace_damped_covariance(
        fit_raw,
        relative_damping=covariance_damping,
    )

    started = time.perf_counter()
    output_gram = work_weight @ fit_damped @ work_weight.transpose(0, 1)
    output_gram = 0.5 * (output_gram + output_gram.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(output_gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0)
    eigenvectors = _canonicalize_columns(eigenvectors.index_select(1, order))
    decomposition_seconds = time.perf_counter() - started

    results = []
    tiny = torch.finfo(work_dtype).tiny
    for rank in selected_ranks:
        rank_started = time.perf_counter()
        output_subspace = eigenvectors[:, :rank]
        retained = eigenvalues[:rank]
        singular_values = retained.sqrt()
        balance = singular_values.sqrt().clamp_min(tiny)
        output_basis = output_subspace * balance.unsqueeze(0)
        input_factor = (
            work_weight.transpose(0, 1) @ output_subspace
        ) / balance.unsqueeze(0)
        decoder = output_basis.transpose(0, 1).contiguous()
        approximation = output_basis @ input_factor.transpose(0, 1)

        stored_input = input_factor.to(dtype=factor_dtype)
        stored_decoder = decoder.to(dtype=factor_dtype)
        quantized = (
            stored_decoder.to(dtype=work_dtype).transpose(0, 1)
            @ stored_input.to(dtype=work_dtype).transpose(0, 1)
        )
        eigen_residual = output_gram @ output_subspace - (
            output_subspace * retained.unsqueeze(0)
        )
        residual_denominator = torch.linalg.vector_norm(
            output_gram @ output_subspace
        ).clamp_min(tiny)
        relative_eigen_residual = float(
            torch.linalg.vector_norm(eigen_residual) / residual_denominator
        )
        results.append(
            StrongLRAllReduceFactors(
                rank=rank,
                input_factor=stored_input.detach().cpu().contiguous(),
                decoder=stored_decoder.detach().cpu().contiguous(),
                singular_values=singular_values.float().detach().cpu().contiguous(),
                metrics={
                    "method": "global_activation_aware_svd",
                    "solver_status": "global_closed_form_optimum",
                    "rank": rank,
                    "input_width": input_width,
                    "output_width": output_width,
                    "covariance_damping": float(covariance_damping),
                    "absolute_covariance_damping": float(absolute_damping),
                    "fit_damped_relative_mse": relative_output_mse(
                        work_weight, approximation, fit_damped
                    ),
                    "fit_raw_relative_mse": relative_output_mse(
                        work_weight, approximation, fit_raw
                    ),
                    "heldout_relative_mse": relative_output_mse(
                        work_weight, approximation, heldout
                    ),
                    "factor_dtype_fit_raw_relative_mse": relative_output_mse(
                        work_weight, quantized, fit_raw
                    ),
                    "factor_dtype_heldout_relative_mse": relative_output_mse(
                        work_weight, quantized, heldout
                    ),
                    "relative_eigen_residual": relative_eigen_residual,
                    "shared_decomposition_seconds": decomposition_seconds,
                    "rank_materialization_seconds": time.perf_counter()
                    - rank_started,
                    "work_dtype": str(work_dtype).removeprefix("torch."),
                    "factor_dtype": str(factor_dtype).removeprefix("torch."),
                },
            )
        )
        metrics = results[-1].metrics
        for name, value in tuple(metrics.items()):
            if name.endswith("_relative_mse"):
                metrics[name.removesuffix("_mse") + "_l2"] = math.sqrt(
                    max(float(value), 0.0)
                )
    return tuple(results)


def private_products_weight(
    source_encoders: Sequence[Tensor],
    source_decoders: Sequence[Tensor],
) -> Tensor:
    """Materialize the C1 logical weight ``[output_width, input_width]``."""

    if not source_encoders or len(source_encoders) != len(source_decoders):
        raise ValueError("C1 source factor lists must be nonempty and aligned")
    products = []
    output_width: int | None = None
    for source, (encoder, decoder) in enumerate(
        zip(source_encoders, source_decoders)
    ):
        input_width, rank = _matrix(f"source_encoders[{source}]", encoder)
        decoder_rank, current_output = _matrix(
            f"source_decoders[{source}]", decoder
        )
        if input_width <= 0 or rank != decoder_rank:
            raise ValueError("C1 encoder and decoder ranks differ")
        if output_width is None:
            output_width = current_output
        elif current_output != output_width:
            raise ValueError("C1 source decoders have different output widths")
        products.append(encoder @ decoder)
    return torch.cat(products, dim=0).transpose(0, 1).contiguous()


def permuted_private_products_weight(
    source_encoders: Sequence[Tensor],
    source_decoders: Sequence[Tensor],
    permutation: Sequence[int],
) -> Tensor:
    """Assign paired private factors to source blocks in ``permutation`` order.

    Entry ``permutation[p]`` selects the fitted ``(E, D)`` pair applied to
    logical input source ``p``.  The factors remain paired and are never
    refitted, making this a source-specialization sanity control.
    """

    source_count = len(source_encoders)
    order = tuple(int(source) for source in permutation)
    if source_count == 0 or len(source_decoders) != source_count:
        raise ValueError("C1 source factor lists must be nonempty and aligned")
    if len(order) != source_count or sorted(order) != list(range(source_count)):
        raise ValueError("permutation must be a bijection over all sources")
    products = []
    output_width: int | None = None
    source_width: int | None = None
    for logical_source, fitted_source in enumerate(order):
        encoder = source_encoders[fitted_source]
        decoder = source_decoders[fitted_source]
        current_width, rank = _matrix(
            f"source_encoders[{fitted_source}]", encoder
        )
        decoder_rank, current_output = _matrix(
            f"source_decoders[{fitted_source}]", decoder
        )
        if rank != decoder_rank:
            raise ValueError("C1 encoder and decoder ranks differ")
        if source_width is None:
            source_width = current_width
        elif current_width != source_width:
            raise ValueError(
                "paired-factor permutations require equal source widths"
            )
        if output_width is None:
            output_width = current_output
        elif current_output != output_width:
            raise ValueError("C1 source decoders have different output widths")
        products.append(encoder @ decoder)
    return torch.cat(products, dim=0).transpose(0, 1).contiguous()


def rotate_private_factors(
    source_encoders: Sequence[Tensor],
    source_decoders: Sequence[Tensor],
    rotations: Sequence[Tensor],
    *,
    orthogonality_tolerance: float | None = None,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Apply source-private orthogonal changes of latent basis.

    Each product is preserved algebraically as
    ``(E_p Q_p) (Q_p.T D_p) = E_p D_p``.
    """

    source_count = len(source_encoders)
    if not (
        source_count
        and len(source_decoders) == source_count
        and len(rotations) == source_count
    ):
        raise ValueError("encoders, decoders, and rotations must be aligned")
    rotated_encoders = []
    rotated_decoders = []
    for source, (encoder, decoder, rotation) in enumerate(
        zip(source_encoders, source_decoders, rotations)
    ):
        _, rank = _matrix(f"source_encoders[{source}]", encoder)
        decoder_rank, _ = _matrix(f"source_decoders[{source}]", decoder)
        rotation_rows, rotation_columns = _matrix(
            f"rotations[{source}]", rotation
        )
        if rank != decoder_rank or (rotation_rows, rotation_columns) != (
            rank,
            rank,
        ):
            raise ValueError("latent rotation does not match the source rank")
        if rotation.device != encoder.device or rotation.dtype != encoder.dtype:
            raise ValueError("latent rotations must match encoder device and dtype")
        if decoder.device != encoder.device or decoder.dtype != encoder.dtype:
            raise ValueError("source encoder and decoder device/dtype differ")
        identity = torch.eye(rank, device=rotation.device, dtype=rotation.dtype)
        residual = rotation.transpose(0, 1) @ rotation - identity
        tolerance = (
            64.0 * rank * torch.finfo(rotation.dtype).eps
            if orthogonality_tolerance is None
            else float(orthogonality_tolerance)
        )
        if tolerance < 0 or float(residual.abs().max()) > tolerance:
            raise ValueError(f"rotation {source} is not orthogonal")
        rotated_encoders.append((encoder @ rotation).contiguous())
        rotated_decoders.append(
            (rotation.transpose(0, 1) @ decoder).contiguous()
        )
    return tuple(rotated_encoders), tuple(rotated_decoders)


def simulate_c1_allgather(
    source_activations: Sequence[Tensor],
    source_encoders: Sequence[Tensor],
    source_decoders: Sequence[Tensor],
) -> Tensor:
    """Evaluate ordered private latents followed by the joint decoder."""

    if not (
        source_activations
        and len(source_activations)
        == len(source_encoders)
        == len(source_decoders)
    ):
        raise ValueError("C1 activation and factor lists must be aligned")
    rows = int(source_activations[0].shape[0])
    latents = []
    for source, (activation, encoder, decoder) in enumerate(
        zip(source_activations, source_encoders, source_decoders)
    ):
        if activation.ndim != 2 or int(activation.shape[0]) != rows:
            raise ValueError(f"source activation {source} has incompatible rows")
        if int(activation.shape[1]) != int(encoder.shape[0]):
            raise ValueError(f"source activation {source} has incompatible width")
        if int(encoder.shape[1]) != int(decoder.shape[0]):
            raise ValueError(f"source factors {source} have incompatible ranks")
        latents.append(activation @ encoder)
    gathered = torch.cat(tuple(latents), dim=1)
    joint_decoder = torch.cat(tuple(source_decoders), dim=0)
    return gathered @ joint_decoder


def simulate_c1_local_decode_allreduce(
    source_activations: Sequence[Tensor],
    source_encoders: Sequence[Tensor],
    source_decoders: Sequence[Tensor],
) -> Tensor:
    """Decode private latents locally and sum the hidden-width outputs."""

    if not (
        source_activations
        and len(source_activations)
        == len(source_encoders)
        == len(source_decoders)
    ):
        raise ValueError("C1 activation and factor lists must be aligned")
    output = None
    rows = int(source_activations[0].shape[0])
    for source, (activation, encoder, decoder) in enumerate(
        zip(source_activations, source_encoders, source_decoders)
    ):
        if activation.ndim != 2 or int(activation.shape[0]) != rows:
            raise ValueError(f"source activation {source} has incompatible rows")
        if tuple(encoder.shape) != (
            int(activation.shape[1]),
            int(decoder.shape[0]),
        ):
            raise ValueError(f"source factors {source} have incompatible shapes")
        contribution = (activation @ encoder) @ decoder
        output = contribution if output is None else output + contribution
    assert output is not None
    return output


def embed_c1_in_lr_allreduce(
    source_encoders: Sequence[Tensor],
    source_decoders: Sequence[Tensor],
) -> tuple[tuple[Tensor, ...], Tensor]:
    """Embed a private C1 solution into disjoint shared coordinates exactly."""

    if not source_encoders or len(source_encoders) != len(source_decoders):
        raise ValueError("C1 source factor lists must be nonempty and aligned")
    ranks = []
    output_width: int | None = None
    for source, (encoder, decoder) in enumerate(
        zip(source_encoders, source_decoders)
    ):
        _, rank = _matrix(f"source_encoders[{source}]", encoder)
        decoder_rank, current_output = _matrix(
            f"source_decoders[{source}]", decoder
        )
        if rank != decoder_rank:
            raise ValueError("C1 encoder and decoder ranks differ")
        if output_width is None:
            output_width = current_output
        elif current_output != output_width:
            raise ValueError("C1 source decoders have different output widths")
        ranks.append(rank)
    total_rank = sum(ranks)
    embedded = []
    offset = 0
    for encoder, rank in zip(source_encoders, ranks):
        value = encoder.new_zeros((int(encoder.shape[0]), total_rank))
        value[:, offset : offset + rank] = encoder
        embedded.append(value)
        offset += rank
    return tuple(embedded), torch.cat(tuple(source_decoders), dim=0).contiguous()


def simulate_lr_allreduce(
    source_activations: Sequence[Tensor],
    source_encoders: Sequence[Tensor],
    shared_decoder: Tensor,
) -> Tensor:
    """Evaluate source-specific encoders, latent sum, and shared decoder."""

    if not source_activations or len(source_activations) != len(source_encoders):
        raise ValueError("LR-AllReduce activations and encoders must be aligned")
    rank, _ = _matrix("shared_decoder", shared_decoder)
    rows = int(source_activations[0].shape[0])
    reduced = None
    for source, (activation, encoder) in enumerate(
        zip(source_activations, source_encoders)
    ):
        if activation.ndim != 2 or int(activation.shape[0]) != rows:
            raise ValueError(f"source activation {source} has incompatible rows")
        if tuple(encoder.shape) != (int(activation.shape[1]), rank):
            raise ValueError(f"source encoder {source} has incompatible shape")
        latent = activation @ encoder
        reduced = latent if reduced is None else reduced + latent
    assert reduced is not None
    return reduced @ shared_decoder


def ring_allreduce_bytes_per_rank(
    *,
    rows: int,
    rank: int,
    tp_size: int,
    dtype_bytes: int,
) -> float:
    """Ideal ring AllReduce traffic per rank."""

    if min(rows, rank, dtype_bytes) <= 0 or tp_size <= 1:
        raise ValueError("invalid AllReduce accounting inputs")
    return 2.0 * (tp_size - 1) / tp_size * rows * rank * dtype_bytes


def ring_allgather_bytes_per_rank(
    *,
    rows: int,
    source_ranks: Sequence[int],
    dtype_bytes: int,
) -> float:
    """Ideal average ring AllGather traffic per rank.

    The value is exact for uniform source widths.  Ragged implementations must
    additionally report their actual padded or packed runtime traffic.
    """

    ranks = tuple(int(rank) for rank in source_ranks)
    if rows <= 0 or dtype_bytes <= 0 or len(ranks) <= 1 or any(rank <= 0 for rank in ranks):
        raise ValueError("invalid AllGather accounting inputs")
    tp_size = len(ranks)
    return (tp_size - 1) / tp_size * rows * sum(ranks) * dtype_bytes


def wire_matched_allreduce_rank(source_ranks: Sequence[int]) -> int:
    """Return the integer ideal-ring AllReduce rank matching an AllGather."""

    ranks = tuple(int(rank) for rank in source_ranks)
    if len(ranks) <= 1 or any(rank <= 0 for rank in ranks):
        raise ValueError("source ranks must describe multiple positive TP sources")
    total = sum(ranks)
    if total % 2:
        raise ValueError("ideal equal-wire rank is non-integral")
    return total // 2


def decoder_row_basis(decoder: Tensor) -> Tensor:
    """Return an orthonormal column basis for a decoder's row space."""

    rows, columns = _matrix("decoder", decoder)
    if rows > columns:
        raise ValueError("decoder row rank cannot exceed its output width")
    basis, upper = torch.linalg.qr(decoder.transpose(0, 1), mode="reduced")
    singular_values = torch.linalg.svdvals(upper)
    tolerance = max(rows, columns) * torch.finfo(decoder.dtype).eps * singular_values[0]
    rank = int((singular_values > tolerance).sum())
    if rank != rows:
        raise ValueError(f"decoder is rank deficient: expected {rows}, got {rank}")
    return basis.contiguous()


def pairwise_subspace_metrics(
    bases: Sequence[Tensor],
) -> tuple[dict[str, float | int], ...]:
    """Measure principal angles, chordal distance, and projection overlap."""

    if len(bases) <= 1:
        raise ValueError("at least two source subspaces are required")
    checked = []
    for source, basis in enumerate(bases):
        output_width, rank = _matrix(f"bases[{source}]", basis)
        if output_width < rank:
            raise ValueError("subspace basis has more columns than rows")
        identity = torch.eye(rank, dtype=basis.dtype, device=basis.device)
        residual = basis.transpose(0, 1) @ basis - identity
        tolerance = 32.0 * output_width * torch.finfo(basis.dtype).eps
        if float(residual.abs().max()) > tolerance:
            raise ValueError(f"basis {source} is not column-orthonormal")
        checked.append(basis)
    bases = tuple(checked)
    rank = int(bases[0].shape[1])
    if any(int(basis.shape[1]) != rank for basis in bases):
        raise ValueError("pairwise source metrics require equal decoder ranks")
    results = []
    for left in range(len(bases)):
        for right in range(left + 1, len(bases)):
            cosines = torch.linalg.svdvals(
                bases[left].transpose(0, 1) @ bases[right]
            ).clamp(0.0, 1.0)
            angles = torch.rad2deg(torch.acos(cosines))
            overlap = float(cosines.square().sum() / rank)
            results.append(
                {
                    "left_source": left,
                    "right_source": right,
                    "rank": rank,
                    "minimum_angle_degrees": float(angles.min()),
                    "median_angle_degrees": float(angles.median()),
                    "mean_angle_degrees": float(angles.mean()),
                    "maximum_angle_degrees": float(angles.max()),
                    "mean_cosine_squared": overlap,
                    "projection_overlap": overlap,
                    "chordal_distance_squared": float(
                        rank - cosines.square().sum()
                    ),
                    "normalized_chordal_distance_squared": 1.0 - overlap,
                }
            )
    return tuple(results)


def pairwise_decoder_subspace_metrics(
    decoders: Sequence[Tensor],
) -> tuple[dict[str, float | int], ...]:
    """Measure pairwise row-space structure directly from source decoders."""

    return pairwise_subspace_metrics(
        tuple(decoder_row_basis(decoder) for decoder in decoders)
    )


def subspace_union_energy_ranks(
    bases: Sequence[Tensor],
    *,
    thresholds: Sequence[float] = (0.9, 0.95, 0.99, 0.999),
) -> dict[str, object]:
    """Return the energy ranks of the union of gauge-invariant row bases."""

    if not bases:
        raise ValueError("at least one source subspace is required")
    selected = tuple(float(threshold) for threshold in thresholds)
    if not selected or any(not 0.0 < threshold <= 1.0 for threshold in selected):
        raise ValueError("energy thresholds must lie in (0, 1]")
    output_width = int(bases[0].shape[0])
    if any(
        basis.ndim != 2
        or int(basis.shape[0]) != output_width
        or not bool(torch.isfinite(basis).all())
        for basis in bases
    ):
        raise ValueError("source subspaces have incompatible output widths")
    stacked = torch.cat(tuple(basis.transpose(0, 1) for basis in bases), dim=0)
    gram = stacked @ stacked.transpose(0, 1)
    gram = 0.5 * (gram + gram.transpose(0, 1))
    energy = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0)
    singular_values = energy.sqrt()
    cumulative = energy.cumsum(0) / energy.sum()
    energy_ranks = {
        f"{100.0 * threshold:g}%": int(
            torch.searchsorted(
                cumulative,
                cumulative.new_tensor(threshold),
                right=False,
            )
        )
        + 1
        for threshold in selected
    }
    tolerance = (
        max(stacked.shape)
        * torch.finfo(stacked.dtype).eps
        * singular_values[0]
    )
    return {
        "algebraic_rank": int((singular_values > tolerance).sum()),
        "energy_ranks": energy_ranks,
        "singular_values": [float(value) for value in singular_values],
    }


def decoder_union_energy_ranks(
    decoders: Sequence[Tensor],
    *,
    thresholds: Sequence[float] = (0.9, 0.95, 0.99, 0.999),
) -> dict[str, object]:
    """Measure union energy ranks directly from source decoder row spaces."""

    return subspace_union_energy_ranks(
        tuple(decoder_row_basis(decoder) for decoder in decoders),
        thresholds=thresholds,
    )


__all__ = [
    "StrongLRAllReduceFactors",
    "decoder_row_basis",
    "decoder_union_energy_ranks",
    "embed_c1_in_lr_allreduce",
    "fit_strong_lr_allreduce_rank_bank",
    "pairwise_decoder_subspace_metrics",
    "pairwise_subspace_metrics",
    "permuted_private_products_weight",
    "private_products_weight",
    "relative_output_mse",
    "ring_allgather_bytes_per_rank",
    "ring_allreduce_bytes_per_rank",
    "simulate_c1_allgather",
    "simulate_c1_local_decode_allreduce",
    "simulate_lr_allreduce",
    "subspace_union_energy_ranks",
    "trace_damped_covariance",
    "rotate_private_factors",
    "wire_matched_allreduce_rank",
]
