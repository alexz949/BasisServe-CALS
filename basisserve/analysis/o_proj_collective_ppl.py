"""Quality-equivalent output projections for row-parallel ``o_proj`` collectives.

The helpers in this module do not launch collectives.  They fit and apply the
linear maps that an activation-aware shared AllReduce or private AllGather
factorization implements, so a normal Hugging Face model can be used for
end-to-end quality evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from basisserve.analysis.mlp_topk_chebyshev import polar_retract_columns
from basisserve.sketching.coordinate_selection import compute_uncentered_pod


@dataclass(frozen=True)
class EqualBytePlan:
    hidden_size: int
    tp_size: int
    baseline_rank: int
    private_total_rank: int
    private_rank_per_tp: int
    dtype_bytes: int

    @property
    def input_width_per_tp(self) -> int:
        return self.hidden_size // self.tp_size

    @property
    def ideal_ring_bytes_per_rank(self) -> int:
        return (
            2
            * (self.tp_size - 1)
            * self.baseline_rank
            * self.dtype_bytes
            // self.tp_size
        )

    def accounting(self, method: str, *, num_attention_heads: int) -> dict[str, Any]:
        if method not in {"ar", "tp_ag", "head_ag"}:
            raise ValueError(f"unknown collective method: {method}")
        if num_attention_heads <= 0 or self.hidden_size % num_attention_heads:
            raise ValueError("attention-head geometry is invalid")
        if method == "ar":
            latent_per_rank = self.baseline_rank
            encoder_parameters_per_rank = (
                self.input_width_per_tp * self.baseline_rank
            )
            decoder_parameters_per_rank = self.hidden_size * self.baseline_rank
            collective = "allreduce"
            logical_groups = 1
        elif method == "tp_ag":
            latent_per_rank = self.private_rank_per_tp
            encoder_parameters_per_rank = (
                self.input_width_per_tp * self.private_rank_per_tp
            )
            decoder_parameters_per_rank = (
                self.hidden_size * self.private_total_rank
            )
            collective = "allgather"
            logical_groups = self.tp_size
        else:
            if num_attention_heads % self.tp_size:
                raise ValueError("attention heads are not divisible by TP")
            head_width = self.hidden_size // num_attention_heads
            heads_per_rank = num_attention_heads // self.tp_size
            if self.private_rank_per_tp % heads_per_rank:
                raise ValueError("private TP rank is not divisible across heads")
            rank_per_head = self.private_rank_per_tp // heads_per_rank
            latent_per_rank = heads_per_rank * rank_per_head
            encoder_parameters_per_rank = heads_per_rank * head_width * rank_per_head
            decoder_parameters_per_rank = (
                self.hidden_size * num_attention_heads * rank_per_head
            )
            collective = "allgather"
            logical_groups = num_attention_heads
        if collective == "allreduce":
            ring_bytes = (
                2
                * (self.tp_size - 1)
                * latent_per_rank
                * self.dtype_bytes
                // self.tp_size
            )
        else:
            ring_bytes = (
                (self.tp_size - 1) * latent_per_rank * self.dtype_bytes
            )
        return {
            "collective": collective,
            "logical_groups": logical_groups,
            "latent_elements_per_rank": latent_per_rank,
            "ideal_ring_bytes_per_rank": ring_bytes,
            "encoder_parameters_per_rank": encoder_parameters_per_rank,
            "decoder_parameters_per_rank": decoder_parameters_per_rank,
            "encoder_hbm_bytes_per_rank": encoder_parameters_per_rank
            * self.dtype_bytes,
            "decoder_hbm_bytes_per_rank": decoder_parameters_per_rank
            * self.dtype_bytes,
        }


def equal_byte_plan(
    *, hidden_size: int, tp_size: int, baseline_rank: int, dtype_bytes: int = 2
) -> EqualBytePlan:
    if hidden_size <= 0 or tp_size <= 1 or hidden_size % tp_size:
        raise ValueError("hidden size must be positively divisible by TP")
    if baseline_rank <= 0 or baseline_rank > hidden_size or dtype_bytes <= 0:
        raise ValueError("baseline rank or dtype size is invalid")
    private_total_rank = 2 * baseline_rank
    if private_total_rank % tp_size:
        raise ValueError("equal-byte private rank is not divisible by TP")
    private_rank_per_tp = private_total_rank // tp_size
    if private_rank_per_tp > hidden_size // tp_size:
        raise ValueError("private rank exceeds a physical TP input shard")
    return EqualBytePlan(
        hidden_size=int(hidden_size),
        tp_size=int(tp_size),
        baseline_rank=int(baseline_rank),
        private_total_rank=int(private_total_rank),
        private_rank_per_tp=int(private_rank_per_tp),
        dtype_bytes=int(dtype_bytes),
    )


@torch.no_grad()
def fit_shared_output_basis_from_teacher(
    teacher_outputs: Tensor,
    rank: int,
    *,
    device: torch.device,
    oversample: int,
    niter: int,
    seed: int,
) -> tuple[Tensor, dict[str, Any]]:
    """Fit a shared AllReduce output basis to materialized teacher rows."""

    if teacher_outputs.ndim != 2:
        raise ValueError("teacher outputs must be a matrix")
    rows, output_width = map(int, teacher_outputs.shape)
    if not 0 < rank <= min(rows, output_width):
        raise ValueError("shared output-POD geometry is invalid")
    if not bool(torch.isfinite(teacher_outputs).all()):
        raise ValueError("teacher outputs contain non-finite values")
    pod = compute_uncentered_pod(
        teacher_outputs.to(device=device),
        rank,
        oversample=oversample,
        niter=niter,
        seed=seed,
    )
    basis, retraction = polar_retract_columns(pod.basis)
    diagnostics = {
        "fit": "materialized_teacher_output_pod",
        "pod": pod.diagnostics(),
        "polar_retraction": retraction,
    }
    return basis.detach().cpu().contiguous(), diagnostics


@torch.no_grad()
def fit_shared_output_basis(
    activations: Tensor,
    weight: Tensor,
    rank: int,
    *,
    device: torch.device,
    oversample: int,
    niter: int,
    seed: int,
    output_chunk_size: int = 512,
) -> tuple[Tensor, dict[str, Any]]:
    """Fit a POD basis to the dense teacher output ``X W^T``."""

    if activations.ndim != 2 or weight.ndim != 2:
        raise ValueError("activations and weight must be matrices")
    rows, input_width = map(int, activations.shape)
    output_width, weight_input_width = map(int, weight.shape)
    if input_width != weight_input_width or not 0 < rank <= min(rows, output_width):
        raise ValueError("shared output-POD geometry is invalid")
    if output_chunk_size <= 0:
        raise ValueError("output chunk size must be positive")
    compute_dtype = (
        weight.dtype
        if weight.dtype in {torch.bfloat16, torch.float16}
        else torch.bfloat16
    )
    teacher = torch.empty(rows, output_width, dtype=compute_dtype)
    work_weight = weight.to(device=device, dtype=compute_dtype)
    for start in range(0, rows, output_chunk_size):
        stop = min(start + output_chunk_size, rows)
        teacher[start:stop].copy_(
            torch.nn.functional.linear(
                activations[start:stop].to(device=device, dtype=compute_dtype),
                work_weight,
            ).cpu()
        )
    basis, diagnostics = fit_shared_output_basis_from_teacher(
        teacher,
        rank,
        device=device,
        oversample=oversample,
        niter=niter,
        seed=seed,
    )
    diagnostics["fit"] = "dense_teacher_output_pod"
    return basis, diagnostics


@torch.no_grad()
def fit_private_output_bases(
    activations: Tensor,
    weight: Tensor,
    *,
    group_count: int,
    rank_per_group: int,
    device: torch.device,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Fit exact activation-aware output PODs for contiguous input groups.

    For a local weight ``W_g`` we use ``W_g = Q R`` and diagonalize the much
    smaller covariance of ``X_g R^T``.  Its right singular vectors lifted by
    ``Q`` are exactly the output POD vectors of ``X_g W_g^T``.
    """

    if activations.ndim != 2 or weight.ndim != 2:
        raise ValueError("activations and weight must be matrices")
    rows, input_width = map(int, activations.shape)
    output_width, weight_input_width = map(int, weight.shape)
    if input_width != weight_input_width or group_count <= 0:
        raise ValueError("private output-POD geometry is invalid")
    if input_width % group_count:
        raise ValueError("input width is not divisible by the logical groups")
    group_width = input_width // group_count
    if not 0 < rank_per_group <= min(rows, group_width, output_width):
        raise ValueError("private rank exceeds a logical group capacity")

    bases: list[Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    for group in range(group_count):
        start = group * group_width
        stop = start + group_width
        local_weight = weight[:, start:stop].to(device=device, dtype=torch.float32)
        local_activation = activations[:, start:stop].to(
            device=device, dtype=torch.float32
        )
        q_basis, triangular = torch.linalg.qr(local_weight, mode="reduced")
        reduced_output = local_activation @ triangular.transpose(0, 1)
        covariance = reduced_output.transpose(0, 1) @ reduced_output
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.arange(
            group_width - 1,
            group_width - rank_per_group - 1,
            -1,
            device=device,
        )
        selected_values = eigenvalues.index_select(0, order).clamp_min(0.0)
        reduced_basis = eigenvectors.index_select(1, order)
        basis = q_basis @ reduced_basis
        basis, retraction = polar_retract_columns(basis)
        total_energy = eigenvalues.clamp_min(0.0).sum(dtype=torch.float64)
        retained = selected_values.sum(dtype=torch.float64)
        bases.append(basis.detach().cpu().contiguous())
        diagnostics.append(
            {
                "group": group,
                "input_start": start,
                "input_stop": stop,
                "rank": rank_per_group,
                "retained_energy_fraction": float(
                    retained / total_energy.clamp_min(1.0e-300)
                ),
                "polar_retraction": retraction,
                "solver": "thin_qr_reduced_covariance_eigh",
            }
        )
    return torch.stack(bases).contiguous(), diagnostics


def _validate_basis(basis: Tensor, *, output_width: int) -> None:
    if basis.ndim != 2 or int(basis.shape[0]) != output_width:
        raise ValueError("basis has the wrong output width")
    if not bool(torch.isfinite(basis).all()):
        raise ValueError("basis contains non-finite values")


@torch.no_grad()
def project_weight_shared(weight: Tensor, basis: Tensor) -> Tensor:
    """Return ``U U^T W`` in float32 on the weight device."""

    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    _validate_basis(basis, output_width=int(weight.shape[0]))
    work_weight = weight.float()
    work_basis = basis.to(device=weight.device, dtype=torch.float32)
    return (work_basis @ (work_basis.transpose(0, 1) @ work_weight)).contiguous()


@torch.no_grad()
def project_weight_private(weight: Tensor, bases: Tensor) -> Tensor:
    """Return the concatenated maps ``[U_g U_g^T W_g]_g`` in float32."""

    if weight.ndim != 2 or bases.ndim != 3:
        raise ValueError("weight must be 2-D and private bases must be 3-D")
    groups = int(bases.shape[0])
    output_width, input_width = map(int, weight.shape)
    if groups <= 0 or input_width % groups:
        raise ValueError("private groups do not divide the input width")
    group_width = input_width // groups
    result = torch.empty(
        output_width, input_width, device=weight.device, dtype=torch.float32
    )
    for group in range(groups):
        start = group * group_width
        stop = start + group_width
        basis = bases[group]
        _validate_basis(basis, output_width=output_width)
        work_basis = basis.to(device=weight.device, dtype=torch.float32)
        local_weight = weight[:, start:stop].float()
        result[:, start:stop] = work_basis @ (
            work_basis.transpose(0, 1) @ local_weight
        )
    return result.contiguous()


def validate_equal_ring_bytes(
    plan: EqualBytePlan, *, num_attention_heads: int
) -> dict[str, dict[str, Any]]:
    methods = {
        method: plan.accounting(method, num_attention_heads=num_attention_heads)
        for method in ("ar", "tp_ag", "head_ag")
    }
    expected = plan.ideal_ring_bytes_per_rank
    if any(row["ideal_ring_bytes_per_rank"] != expected for row in methods.values()):
        raise AssertionError("AR and AG ring-byte accounting is not equal")
    return methods
