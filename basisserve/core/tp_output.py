"""Low-rank communication for tensor-parallel output projections.

This module implements a correctness-first prototype for the communication
pattern discussed in BASISServe:

    dense row-parallel output:
        y = AllReduce_i(x_i @ W_i^T)

    low-rank communication output:
        z = AllReduce_i(x_i @ A_i)
        y = z @ R^T

where the logical PyTorch weight ``W`` has shape ``[d_out, d_in]`` and is
sharded across its input dimension.  The low-rank factors satisfy

    W ~= R @ A^T,

with ``A`` sharded across its first (input) dimension and ``R`` replicated
across tensor-parallel ranks.  Only the activation ``z`` is communicated.

The factorization is an *offline/checkpoint-conversion* operation.  The forward
path never runs SVD or eigendecomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA fast path
    triton = None
    tl = None

FactorizationMethod = Literal["svd", "svd_lowrank", "gram_eigh"]
HadamardReconstructionMethod = Literal["fwht", "dense_basis", "triton_fwht"]


class RuntimeBreakdownRecorder:
    """Optional per-forward CUDA/CPU timing recorder for serving breakdowns."""

    def __init__(self) -> None:
        self.enabled = False
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.tokens = 0
        self.dense_payload_bytes = 0
        self.compressed_payload_bytes = 0
        self._cpu_segment_ms: dict[str, float] = {}
        self._cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def record_call(self, tokens: int) -> None:
        if not self.enabled:
            return
        self.calls += 1
        self.tokens += int(tokens)

    def record_payload(
        self,
        *,
        dense_payload_elements: int,
        compressed_payload_elements: int,
        element_size_bytes: int,
    ) -> None:
        if not self.enabled:
            return
        self.dense_payload_bytes += int(dense_payload_elements) * int(element_size_bytes)
        self.compressed_payload_bytes += int(compressed_payload_elements) * int(element_size_bytes)

    def record_segment(self, name: str, fn, *, device: torch.device):
        if not self.enabled:
            return fn()
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            self._cuda_events.setdefault(name, []).append((start, end))
            return result

        t0 = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._cpu_segment_ms[name] = self._cpu_segment_ms.get(name, 0.0) + elapsed_ms
        return result

    def summary(self) -> dict[str, object]:
        if self._cuda_events and torch.cuda.is_available():
            torch.cuda.synchronize()
        segment_ms = dict(self._cpu_segment_ms)
        for name, events in self._cuda_events.items():
            segment_ms[name] = segment_ms.get(name, 0.0) + sum(
                float(start.elapsed_time(end)) for start, end in events
            )
        total_ms = float(segment_ms.pop("total", sum(segment_ms.values())))
        return {
            "calls": self.calls,
            "tokens": self.tokens,
            "segment_ms": segment_ms,
            "total_ms": total_ms,
            "dense_payload_bytes": self.dense_payload_bytes,
            "compressed_payload_bytes": self.compressed_payload_bytes,
        }


@dataclass(frozen=True)
class LowRankOutputFactors:
    """Factors for a full logical output-projection weight.

    Attributes:
        input_factor:
            Tensor with shape ``[d_in, rank]``.  Tensor-parallel ranks shard
            this tensor across ``d_in``.
        output_basis:
            Tensor with shape ``[d_out, rank]``.  This is replicated and can
            be used directly as the weight of ``nn.Linear(rank, d_out)``.
        singular_values:
            The retained singular values, in descending order.
        relative_frobenius_error:
            ``||W - W_hat||_F / ||W||_F`` computed in FP32.
    """

    input_factor: torch.Tensor
    output_basis: torch.Tensor
    singular_values: torch.Tensor
    relative_frobenius_error: float

    @property
    def rank(self) -> int:
        return int(self.input_factor.shape[1])

    @property
    def in_features(self) -> int:
        return int(self.input_factor.shape[0])

    @property
    def out_features(self) -> int:
        return int(self.output_basis.shape[0])

    def reconstruct_weight(self) -> torch.Tensor:
        """Return the approximated PyTorch linear weight ``[d_out, d_in]``."""

        return self.output_basis @ self.input_factor.transpose(0, 1)


@dataclass(frozen=True)
class DistributedLowRankOutputFactors:
    """Factors owned by one tensor-parallel rank."""

    local_input_factor: torch.Tensor  # [local_d_in, rank]
    output_basis: torch.Tensor  # [d_out, rank], replicated
    singular_values: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.local_input_factor.shape[1])


@dataclass(frozen=True)
class CommunicationEstimate:
    """Logical and ring-all-reduce communication estimates per decode step."""

    world_size: int
    tokens: int
    full_width: int
    low_rank_width: int
    element_size_bytes: int

    @property
    def full_payload_bytes(self) -> int:
        return self.tokens * self.full_width * self.element_size_bytes

    @property
    def low_rank_payload_bytes(self) -> int:
        return self.tokens * self.low_rank_width * self.element_size_bytes

    @property
    def payload_reduction(self) -> float:
        return self.full_payload_bytes / max(self.low_rank_payload_bytes, 1)

    @property
    def estimated_full_ring_bytes_per_rank(self) -> float:
        if self.world_size <= 1:
            return 0.0
        factor = 2.0 * (self.world_size - 1) / self.world_size
        return factor * self.full_payload_bytes

    @property
    def estimated_low_rank_ring_bytes_per_rank(self) -> float:
        if self.world_size <= 1:
            return 0.0
        factor = 2.0 * (self.world_size - 1) / self.world_size
        return factor * self.low_rank_payload_bytes


def resolve_rank(
    out_features: int,
    in_features: int,
    *,
    rank: int | None = None,
    rank_ratio: float | None = None,
    multiple: int = 1,
) -> int:
    """Resolve a rank from either an absolute value or communication ratio.

    ``rank_ratio`` is relative to ``out_features`` because the communicated
    tensor shrinks from ``d_out`` to ``rank``.
    """

    if (rank is None) == (rank_ratio is None):
        raise ValueError("specify exactly one of rank or rank_ratio")
    if out_features <= 0 or in_features <= 0:
        raise ValueError("in_features and out_features must be positive")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")

    max_rank = min(out_features, in_features)
    if rank_ratio is not None:
        if not 0.0 < rank_ratio <= 1.0:
            raise ValueError(f"rank_ratio must be in (0, 1], got {rank_ratio}")
        raw_rank = int(round(out_features * rank_ratio))
    else:
        assert rank is not None
        raw_rank = int(rank)

    raw_rank = max(1, min(raw_rank, max_rank))
    if multiple > 1:
        rounded = int(math.ceil(raw_rank / multiple) * multiple)
        raw_rank = min(max_rank, rounded)
    return raw_rank


def _relative_weight_error(weight: torch.Tensor, approximation: torch.Tensor) -> float:
    weight_fp32 = weight.detach().to(torch.float32)
    approximation_fp32 = approximation.detach().to(torch.float32)
    denominator = torch.linalg.vector_norm(weight_fp32).clamp_min(1e-12)
    numerator = torch.linalg.vector_norm(weight_fp32 - approximation_fp32)
    return float((numerator / denominator).item())


@torch.no_grad()
def factorize_output_weight(
    weight: torch.Tensor,
    rank: int,
    *,
    method: FactorizationMethod = "svd",
    niter: int = 4,
    oversample: int = 16,
    factor_dtype: torch.dtype | None = None,
) -> LowRankOutputFactors:
    """Factor a full logical PyTorch output-projection weight.

    Args:
        weight: PyTorch linear weight with shape ``[d_out, d_in]``.
        rank: Retained rank.
        method: ``svd`` is deterministic and exact; ``svd_lowrank`` is useful
            for large exploratory checkpoints; ``gram_eigh`` matches the
            distributed factorization path.
        factor_dtype: Dtype used to store returned factors.  By default it
            matches ``weight.dtype``.
    """

    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    out_features, in_features = map(int, weight.shape)
    max_rank = min(out_features, in_features)
    if not 0 < rank <= max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")
    if niter < 0:
        raise ValueError(f"niter must be non-negative, got {niter}")
    if oversample < 0:
        raise ValueError(f"oversample must be non-negative, got {oversample}")

    target_dtype = factor_dtype or weight.dtype
    math_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    math_weight = weight.detach().to(math_dtype).transpose(0, 1).contiguous()

    if method == "svd":
        left, singular_values, right_t = torch.linalg.svd(math_weight, full_matrices=False)
        left = left[:, :rank]
        singular_values = singular_values[:rank]
        output_basis = right_t[:rank, :].transpose(0, 1).contiguous()
        input_factor = left * singular_values.unsqueeze(0)
    elif method == "svd_lowrank":
        q = min(max_rank, rank + oversample)
        left, singular_values, right = torch.svd_lowrank(math_weight, q=q, niter=niter)
        order = torch.argsort(singular_values, descending=True)[:rank]
        singular_values = singular_values[order]
        left = left[:, order]
        output_basis = right[:, order].contiguous()
        input_factor = left * singular_values.unsqueeze(0)
    elif method == "gram_eigh":
        # M = W^T and M^T M = W W^T.  The dominant eigenvectors are the
        # shared output basis; M @ basis contains U Sigma.
        weight_for_gram = weight.detach().to(math_dtype)
        gram = weight_for_gram @ weight_for_gram.transpose(0, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)[:rank]
        eigenvalues = eigenvalues[order].clamp_min(0.0)
        output_basis = eigenvectors[:, order].contiguous()
        singular_values = torch.sqrt(eigenvalues)
        input_factor = math_weight @ output_basis
    else:
        raise ValueError(f"unsupported factorization method: {method}")

    input_factor = input_factor.to(device=weight.device, dtype=target_dtype).contiguous()
    output_basis = output_basis.to(device=weight.device, dtype=target_dtype).contiguous()
    singular_values = singular_values.to(device=weight.device, dtype=torch.float32).contiguous()

    approximation = output_basis.to(torch.float32) @ input_factor.to(torch.float32).transpose(0, 1)
    error = _relative_weight_error(weight, approximation)
    return LowRankOutputFactors(
        input_factor=input_factor,
        output_basis=output_basis,
        singular_values=singular_values,
        relative_frobenius_error=error,
    )


def _group_world_size(process_group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(process_group)


def _group_rank(process_group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank(process_group)


def _global_rank_from_group_rank(process_group: dist.ProcessGroup | None, group_rank: int) -> int:
    if process_group is None:
        return group_rank
    return dist.get_global_rank(process_group, group_rank)


@torch.no_grad()
def factorize_row_parallel_weight_distributed(
    local_weight: torch.Tensor,
    rank: int,
    *,
    process_group: dist.ProcessGroup | None = None,
    root_group_rank: int = 0,
    factor_dtype: torch.dtype | None = None,
    synchronize_collectives: bool = False,
) -> DistributedLowRankOutputFactors:
    """Offline factorization from already row-parallel weight shards.

    Every rank contributes only ``W_i W_i^T``.  The root rank performs the
    eigendecomposition and broadcasts the shared output basis.  No rank needs
    to gather the full logical ``W``.

    This routine is intentionally separate from the serving forward path.  It
    should be run during checkpoint conversion or one-time model startup, not
    per request or per decode step.
    """

    if local_weight.ndim != 2:
        raise ValueError(
            f"local_weight must be [d_out, local_d_in], got {tuple(local_weight.shape)}"
        )
    if not dist.is_available() or not dist.is_initialized():
        full = factorize_output_weight(
            local_weight,
            rank,
            method="gram_eigh",
            factor_dtype=factor_dtype,
        )
        return DistributedLowRankOutputFactors(
            local_input_factor=full.input_factor,
            output_basis=full.output_basis,
            singular_values=full.singular_values,
        )

    world_size = dist.get_world_size(process_group)
    group_rank = dist.get_rank(process_group)
    if not 0 <= root_group_rank < world_size:
        raise ValueError(
            f"root_group_rank must be in [0, {world_size}), got {root_group_rank}"
        )

    out_features, local_in_features = map(int, local_weight.shape)
    max_rank = min(out_features, local_in_features * world_size)
    if not 0 < rank <= max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")

    target_dtype = factor_dtype or local_weight.dtype
    local_weight_fp32 = local_weight.detach().to(torch.float32)
    gram = local_weight_fp32 @ local_weight_fp32.transpose(0, 1)

    root_global_rank = _global_rank_from_group_rank(process_group, root_group_rank)
    dist.reduce(gram, dst=root_global_rank, op=dist.ReduceOp.SUM, group=process_group)
    if synchronize_collectives and gram.is_cuda:
        torch.cuda.synchronize(gram.device)

    output_basis_fp32 = torch.empty(
        (out_features, rank), device=local_weight.device, dtype=torch.float32
    )
    singular_values = torch.empty((rank,), device=local_weight.device, dtype=torch.float32)

    if group_rank == root_group_rank:
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            order = torch.argsort(eigenvalues, descending=True)[:rank]
            top_eigenvalues = eigenvalues[order].clamp_min(0.0)
            output_basis_fp32.copy_(eigenvectors[:, order])
            singular_values.copy_(torch.sqrt(top_eigenvalues))
        except torch.OutOfMemoryError:
            if not gram.is_cuda:
                raise
            print(
                "[basisserve] CUDA OOM during PCA gram eigendecomposition; "
                "falling back to CPU eigh for setup.",
                flush=True,
            )
            gram_cpu = gram.detach().cpu()
            del gram
            torch.cuda.empty_cache()
            eigenvalues_cpu, eigenvectors_cpu = torch.linalg.eigh(gram_cpu)
            order_cpu = torch.argsort(eigenvalues_cpu, descending=True)[:rank]
            top_eigenvalues_cpu = eigenvalues_cpu[order_cpu].clamp_min(0.0)
            output_basis_fp32.copy_(
                eigenvectors_cpu[:, order_cpu].to(device=output_basis_fp32.device)
            )
            singular_values.copy_(
                torch.sqrt(top_eigenvalues_cpu).to(device=singular_values.device)
            )
            del gram_cpu, eigenvalues_cpu, eigenvectors_cpu
            torch.cuda.empty_cache()

    dist.broadcast(output_basis_fp32, src=root_global_rank, group=process_group)
    dist.broadcast(singular_values, src=root_global_rank, group=process_group)
    if synchronize_collectives and local_weight.is_cuda:
        torch.cuda.synchronize(local_weight.device)

    local_input_factor = local_weight_fp32.transpose(0, 1) @ output_basis_fp32
    return DistributedLowRankOutputFactors(
        local_input_factor=local_input_factor.to(target_dtype).contiguous(),
        output_basis=output_basis_fp32.to(target_dtype).contiguous(),
        singular_values=singular_values.contiguous(),
    )


def _all_reduce_sum_(tensor: torch.Tensor, process_group: dist.ProcessGroup | None) -> torch.Tensor:
    if _group_world_size(process_group) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)
    return tensor


def _all_gather_last_dim(
    tensor: torch.Tensor,
    process_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Gather equal-width rank-local tensors and concatenate their last dims."""

    world_size = _group_world_size(process_group)
    if world_size <= 1:
        return tensor
    local_width = int(tensor.shape[-1])
    flat = tensor.contiguous().reshape(-1, local_width)
    rank_major = torch.empty(
        (world_size * flat.shape[0], local_width),
        device=flat.device,
        dtype=flat.dtype,
    )
    dist.all_gather_into_tensor(rank_major, flat, group=process_group)
    gathered = (
        rank_major.reshape(world_size, flat.shape[0], local_width)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(flat.shape[0], world_size * local_width)
    )
    return gathered.reshape(*tensor.shape[:-1], world_size * local_width)


class DenseRowParallelOutput(nn.Module):
    """Reference row-parallel output projection with full-width all-reduce."""

    def __init__(
        self,
        local_weight: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        process_group: dist.ProcessGroup | None = None,
        communication_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if local_weight.ndim != 2:
            raise ValueError("local_weight must be [d_out, local_d_in]")
        self.local_weight = nn.Parameter(local_weight.detach().contiguous(), requires_grad=False)
        self.process_group = process_group
        self.communication_dtype = communication_dtype
        if bias is None:
            self.register_parameter("bias", None)
        else:
            if bias.shape != (local_weight.shape[0],):
                raise ValueError(
                    f"bias must have shape ({local_weight.shape[0]},), got {tuple(bias.shape)}"
                )
            self.bias = nn.Parameter(bias.detach().contiguous(), requires_grad=False)
        self.runtime_breakdown = RuntimeBreakdownRecorder()

    @property
    def local_in_features(self) -> int:
        return int(self.local_weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.local_weight.shape[0])

    @property
    def world_size(self) -> int:
        return _group_world_size(self.process_group)

    def enable_runtime_breakdown(self, enabled: bool = True) -> None:
        self.runtime_breakdown.enabled = enabled

    def reset_runtime_breakdown(self) -> None:
        self.runtime_breakdown.reset()

    def runtime_breakdown_summary(self) -> dict[str, object]:
        summary = self.runtime_breakdown.summary()
        summary.update(
            {
                "module_type": "dense",
                "local_in_features": self.local_in_features,
                "out_features": self.out_features,
                "rank": None,
                "world_size": self.world_size,
            }
        )
        return summary

    def forward(self, local_hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and local_hidden_states.requires_grad:
            raise RuntimeError("DenseRowParallelOutput is an inference-only prototype")
        recorder = self.runtime_breakdown
        if not recorder.enabled:
            partial = F.linear(local_hidden_states, self.local_weight, bias=None)
            original_dtype = partial.dtype
            if self.communication_dtype is not None and partial.dtype != self.communication_dtype:
                partial = partial.to(self.communication_dtype)
            _all_reduce_sum_(partial, self.process_group)
            if partial.dtype != original_dtype:
                partial = partial.to(original_dtype)
            if self.bias is not None:
                partial = partial + self.bias
            return partial

        device = local_hidden_states.device

        def body() -> torch.Tensor:
            partial = recorder.record_segment(
                "local_matmul",
                lambda: F.linear(local_hidden_states, self.local_weight, bias=None),
                device=device,
            )
            original_dtype = partial.dtype
            tokens = int(partial.numel() // max(self.out_features, 1))
            recorder.record_call(tokens)
            recorder.record_payload(
                dense_payload_elements=int(partial.numel()),
                compressed_payload_elements=0,
                element_size_bytes=partial.element_size(),
            )
            if self.communication_dtype is not None and partial.dtype != self.communication_dtype:
                partial = recorder.record_segment(
                    "cast_to_comm",
                    lambda: partial.to(self.communication_dtype),
                    device=device,
                )
            partial = recorder.record_segment(
                "all_reduce",
                lambda: _all_reduce_sum_(partial, self.process_group),
                device=device,
            )
            if partial.dtype != original_dtype:
                partial = recorder.record_segment(
                    "cast_from_comm",
                    lambda: partial.to(original_dtype),
                    device=device,
                )
            if self.bias is not None:
                partial = recorder.record_segment(
                    "bias",
                    lambda: partial + self.bias,
                    device=device,
                )
            return partial

        return recorder.record_segment("total", body, device=device)


class LowRankAllReduceOutput(nn.Module):
    """Row-parallel output projection whose all-reduce happens at low rank.

    ``local_input_factor`` has shape ``[local_d_in, rank]`` and
    ``output_basis`` has shape ``[d_out, rank]``.  Both are checkpoint weights;
    no factorization occurs in :meth:`forward`.
    """

    def __init__(
        self,
        local_input_factor: torch.Tensor,
        output_basis: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        process_group: dist.ProcessGroup | None = None,
        communication_dtype: torch.dtype | None = None,
        debug_sync: bool = False,
    ) -> None:
        super().__init__()
        if local_input_factor.ndim != 2:
            raise ValueError("local_input_factor must be [local_d_in, rank]")
        if output_basis.ndim != 2:
            raise ValueError("output_basis must be [d_out, rank]")
        if local_input_factor.shape[1] != output_basis.shape[1]:
            raise ValueError(
                "local_input_factor and output_basis must use the same rank "
                f"({local_input_factor.shape[1]} != {output_basis.shape[1]})"
            )

        # F.linear(x, weight) expects [out_features, in_features].
        self.local_projection_weight = nn.Parameter(
            local_input_factor.detach().transpose(0, 1).contiguous(),
            requires_grad=False,
        )
        self.output_basis_weight = nn.Parameter(
            output_basis.detach().contiguous(),
            requires_grad=False,
        )
        self.process_group = process_group
        self.communication_dtype = communication_dtype
        self.debug_sync = bool(debug_sync)

        if bias is None:
            self.register_parameter("bias", None)
        else:
            if bias.shape != (output_basis.shape[0],):
                raise ValueError(
                    f"bias must have shape ({output_basis.shape[0]},), got {tuple(bias.shape)}"
                )
            self.bias = nn.Parameter(bias.detach().contiguous(), requires_grad=False)
        self.runtime_breakdown = RuntimeBreakdownRecorder()

    @property
    def rank(self) -> int:
        return int(self.local_projection_weight.shape[0])

    @property
    def local_in_features(self) -> int:
        return int(self.local_projection_weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.output_basis_weight.shape[0])

    @property
    def world_size(self) -> int:
        return _group_world_size(self.process_group)

    def communication_estimate(
        self,
        tokens: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> CommunicationEstimate:
        if tokens <= 0:
            raise ValueError(f"tokens must be positive, got {tokens}")
        tensor_dtype = dtype or self.communication_dtype or self.local_projection_weight.dtype
        element_size = torch.empty((), dtype=tensor_dtype).element_size()
        return CommunicationEstimate(
            world_size=self.world_size,
            tokens=tokens,
            full_width=self.out_features,
            low_rank_width=self.rank,
            element_size_bytes=element_size,
        )

    def enable_runtime_breakdown(self, enabled: bool = True) -> None:
        self.runtime_breakdown.enabled = enabled

    def reset_runtime_breakdown(self) -> None:
        self.runtime_breakdown.reset()

    def runtime_breakdown_summary(self) -> dict[str, object]:
        summary = self.runtime_breakdown.summary()
        summary.update(
            {
                "module_type": "low_rank",
                "local_in_features": self.local_in_features,
                "out_features": self.out_features,
                "rank": self.rank,
                "world_size": self.world_size,
                "debug_sync": self.debug_sync,
            }
        )
        return summary

    def _debug_synchronize(self, tensor: torch.Tensor) -> None:
        if self.debug_sync and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def forward(
        self,
        local_hidden_states: torch.Tensor,
        *,
        return_reduced_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled() and local_hidden_states.requires_grad:
            raise RuntimeError("LowRankAllReduceOutput is an inference-only prototype")
        if local_hidden_states.shape[-1] != self.local_in_features:
            raise ValueError(
                f"expected local hidden width {self.local_in_features}, "
                f"got {local_hidden_states.shape[-1]}"
            )

        recorder = self.runtime_breakdown
        if not recorder.enabled:
            latent = F.linear(local_hidden_states, self.local_projection_weight, bias=None)
            self._debug_synchronize(latent)
            compute_dtype = latent.dtype
            if self.communication_dtype is not None and latent.dtype != self.communication_dtype:
                latent = latent.to(self.communication_dtype)
                self._debug_synchronize(latent)
            _all_reduce_sum_(latent, self.process_group)
            self._debug_synchronize(latent)
            if latent.dtype != compute_dtype:
                latent = latent.to(compute_dtype)
                self._debug_synchronize(latent)

            output = F.linear(latent, self.output_basis_weight, self.bias)
            self._debug_synchronize(output)
            if return_reduced_latent:
                return output, latent
            return output

        device = local_hidden_states.device

        def body() -> tuple[torch.Tensor, torch.Tensor]:
            latent = recorder.record_segment(
                "encode",
                lambda: F.linear(local_hidden_states, self.local_projection_weight, bias=None),
                device=device,
            )
            self._debug_synchronize(latent)
            compute_dtype = latent.dtype
            tokens = int(latent.numel() // max(self.rank, 1))
            recorder.record_call(tokens)
            recorder.record_payload(
                dense_payload_elements=tokens * self.out_features,
                compressed_payload_elements=int(latent.numel()),
                element_size_bytes=latent.element_size(),
            )
            if self.communication_dtype is not None and latent.dtype != self.communication_dtype:
                latent = recorder.record_segment(
                    "cast_to_comm",
                    lambda: latent.to(self.communication_dtype),
                    device=device,
                )
                self._debug_synchronize(latent)
            latent = recorder.record_segment(
                "all_reduce",
                lambda: _all_reduce_sum_(latent, self.process_group),
                device=device,
            )
            self._debug_synchronize(latent)
            if latent.dtype != compute_dtype:
                latent = recorder.record_segment(
                    "cast_from_comm",
                    lambda: latent.to(compute_dtype),
                    device=device,
                )
                self._debug_synchronize(latent)
            output = recorder.record_segment(
                "decode",
                lambda: F.linear(latent, self.output_basis_weight, self.bias),
                device=device,
            )
            self._debug_synchronize(output)
            return output, latent

        output, latent = recorder.record_segment("total", body, device=device)
        if return_reduced_latent:
            return output, latent
        return output

    @classmethod
    @torch.no_grad()
    def from_full_weight(
        cls,
        full_weight: torch.Tensor,
        rank: int,
        *,
        tp_rank: int,
        tp_world_size: int,
        bias: torch.Tensor | None = None,
        process_group: dist.ProcessGroup | None = None,
        communication_dtype: torch.dtype | None = None,
        method: FactorizationMethod = "svd",
        factor_dtype: torch.dtype | None = None,
    ) -> "LowRankAllReduceOutput":
        """Factor the logical weight first, then shard the input factor."""

        if not 0 <= tp_rank < tp_world_size:
            raise ValueError(f"tp_rank must be in [0, {tp_world_size}), got {tp_rank}")
        out_features, in_features = map(int, full_weight.shape)
        if in_features % tp_world_size != 0:
            raise ValueError(
                f"input width {in_features} must be divisible by TP size {tp_world_size}"
            )
        factors = factorize_output_weight(
            full_weight,
            rank,
            method=method,
            factor_dtype=factor_dtype,
        )
        local_width = in_features // tp_world_size
        start = tp_rank * local_width
        local_input_factor = factors.input_factor.narrow(0, start, local_width).contiguous()
        return cls(
            local_input_factor,
            factors.output_basis,
            bias=bias,
            process_group=process_group,
            communication_dtype=communication_dtype,
        )

    @classmethod
    @torch.no_grad()
    def from_local_weight_distributed(
        cls,
        local_weight: torch.Tensor,
        rank: int,
        *,
        bias: torch.Tensor | None = None,
        process_group: dist.ProcessGroup | None = None,
        communication_dtype: torch.dtype | None = None,
        root_group_rank: int = 0,
        factor_dtype: torch.dtype | None = None,
    ) -> "LowRankAllReduceOutput":
        factors = factorize_row_parallel_weight_distributed(
            local_weight,
            rank,
            process_group=process_group,
            root_group_rank=root_group_rank,
            factor_dtype=factor_dtype,
        )
        return cls(
            factors.local_input_factor,
            factors.output_basis,
            bias=bias,
            process_group=process_group,
            communication_dtype=communication_dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"local_in_features={self.local_in_features}, "
            f"out_features={self.out_features}, rank={self.rank}, "
            f"world_size={self.world_size}"
        )


class PrivateAllGatherOutput(nn.Module):
    """Private-basis row-parallel output using a low-rank all-gather.

    Rank ``i`` encodes its local input shard with a private basis of width
    ``local_rank``.  The rank-local latents are all-gathered and concatenated
    in process-group rank order, then decoded with the corresponding
    concatenated output bases.  ``output_basis`` therefore has shape
    ``[d_out, world_size * local_rank]``.
    """

    def __init__(
        self,
        local_input_factor: torch.Tensor,
        output_basis: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        process_group: dist.ProcessGroup | None = None,
        communication_dtype: torch.dtype | None = None,
        debug_sync: bool = False,
    ) -> None:
        super().__init__()
        if local_input_factor.ndim != 2:
            raise ValueError("local_input_factor must be [local_d_in, local_rank]")
        if output_basis.ndim != 2:
            raise ValueError("output_basis must be [d_out, total_rank]")
        world_size = _group_world_size(process_group)
        expected_total_rank = int(local_input_factor.shape[1]) * world_size
        if int(output_basis.shape[1]) != expected_total_rank:
            raise ValueError(
                "output_basis width must equal world_size * local_rank "
                f"({output_basis.shape[1]} != {world_size} * {local_input_factor.shape[1]})"
            )

        self.local_projection_weight = nn.Parameter(
            local_input_factor.detach().transpose(0, 1).contiguous(),
            requires_grad=False,
        )
        self.output_basis_weight = nn.Parameter(
            output_basis.detach().contiguous(),
            requires_grad=False,
        )
        self.process_group = process_group
        self.communication_dtype = communication_dtype
        self.debug_sync = bool(debug_sync)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            if bias.shape != (output_basis.shape[0],):
                raise ValueError(
                    f"bias must have shape ({output_basis.shape[0]},), got {tuple(bias.shape)}"
                )
            self.bias = nn.Parameter(bias.detach().contiguous(), requires_grad=False)
        self.runtime_breakdown = RuntimeBreakdownRecorder()

    @property
    def local_rank(self) -> int:
        return int(self.local_projection_weight.shape[0])

    @property
    def rank(self) -> int:
        """Return the total gathered rank for reporting compatibility."""

        return int(self.output_basis_weight.shape[1])

    @property
    def local_in_features(self) -> int:
        return int(self.local_projection_weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.output_basis_weight.shape[0])

    @property
    def world_size(self) -> int:
        return _group_world_size(self.process_group)

    def enable_runtime_breakdown(self, enabled: bool = True) -> None:
        self.runtime_breakdown.enabled = enabled

    def reset_runtime_breakdown(self) -> None:
        self.runtime_breakdown.reset()

    def runtime_breakdown_summary(self) -> dict[str, object]:
        summary = self.runtime_breakdown.summary()
        summary.update(
            {
                "module_type": "private_all_gather",
                "collective": "all_gather",
                "local_in_features": self.local_in_features,
                "out_features": self.out_features,
                "rank": self.rank,
                "local_rank": self.local_rank,
                "world_size": self.world_size,
                "debug_sync": self.debug_sync,
            }
        )
        return summary

    def _debug_synchronize(self, tensor: torch.Tensor) -> None:
        if self.debug_sync and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def forward(
        self,
        local_hidden_states: torch.Tensor,
        *,
        return_gathered_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled() and local_hidden_states.requires_grad:
            raise RuntimeError("PrivateAllGatherOutput is an inference-only prototype")
        if local_hidden_states.shape[-1] != self.local_in_features:
            raise ValueError(
                f"expected local hidden width {self.local_in_features}, "
                f"got {local_hidden_states.shape[-1]}"
            )

        recorder = self.runtime_breakdown
        if not recorder.enabled:
            latent = F.linear(local_hidden_states, self.local_projection_weight, bias=None)
            self._debug_synchronize(latent)
            compute_dtype = latent.dtype
            if self.communication_dtype is not None and latent.dtype != self.communication_dtype:
                latent = latent.to(self.communication_dtype)
                self._debug_synchronize(latent)
            gathered = _all_gather_last_dim(latent, self.process_group)
            self._debug_synchronize(gathered)
            if gathered.dtype != compute_dtype:
                gathered = gathered.to(compute_dtype)
                self._debug_synchronize(gathered)
            output = F.linear(gathered, self.output_basis_weight, self.bias)
            self._debug_synchronize(output)
            if return_gathered_latent:
                return output, gathered
            return output

        device = local_hidden_states.device

        def body() -> tuple[torch.Tensor, torch.Tensor]:
            latent = recorder.record_segment(
                "encode",
                lambda: F.linear(local_hidden_states, self.local_projection_weight, bias=None),
                device=device,
            )
            self._debug_synchronize(latent)
            compute_dtype = latent.dtype
            tokens = int(latent.numel() // max(self.local_rank, 1))
            recorder.record_call(tokens)
            recorder.record_payload(
                dense_payload_elements=tokens * self.out_features,
                compressed_payload_elements=int(latent.numel()),
                element_size_bytes=latent.element_size(),
            )
            if self.communication_dtype is not None and latent.dtype != self.communication_dtype:
                latent = recorder.record_segment(
                    "cast_to_comm",
                    lambda: latent.to(self.communication_dtype),
                    device=device,
                )
                self._debug_synchronize(latent)
            gathered = recorder.record_segment(
                "all_gather",
                lambda: _all_gather_last_dim(latent, self.process_group),
                device=device,
            )
            self._debug_synchronize(gathered)
            if gathered.dtype != compute_dtype:
                gathered = recorder.record_segment(
                    "cast_from_comm",
                    lambda: gathered.to(compute_dtype),
                    device=device,
                )
                self._debug_synchronize(gathered)
            output = recorder.record_segment(
                "decode",
                lambda: F.linear(gathered, self.output_basis_weight, self.bias),
                device=device,
            )
            self._debug_synchronize(output)
            return output, gathered

        output, gathered = recorder.record_segment("total", body, device=device)
        if return_gathered_latent:
            return output, gathered
        return output

    def extra_repr(self) -> str:
        return (
            f"local_in_features={self.local_in_features}, "
            f"out_features={self.out_features}, local_rank={self.local_rank}, "
            f"total_rank={self.rank}, world_size={self.world_size}"
        )


def _require_power_of_two(value: int, *, name: str) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def hadamard_transform_features(out_features: int) -> int:
    """Return the FWHT width needed to cover ``out_features`` output dims."""

    if out_features <= 0:
        raise ValueError(f"out_features must be positive, got {out_features}")
    return 1 << (out_features - 1).bit_length()


def _fwht_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Unnormalized fast Walsh-Hadamard transform over the last dimension."""

    width = int(tensor.shape[-1])
    _require_power_of_two(width, name="Hadamard width")
    prefix = tensor.shape[:-1]
    output = tensor
    step = 1
    while step < width:
        output = output.reshape(*prefix, width // (2 * step), 2, step)
        left = output[..., 0, :]
        right = output[..., 1, :]
        output = torch.cat((left + right, left - right), dim=-1)
        step *= 2
    return output.reshape(*prefix, width)


if triton is not None and tl is not None:

    @triton.jit
    def _sparse_fwht_kernel(
        latent_ptr,
        position_to_rank_ptr,
        output_ptr,
        rank: tl.constexpr,
        scale: tl.constexpr,
        BLOCK: tl.constexpr,
        LOG2_BLOCK: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        rank_positions = tl.load(position_to_rank_ptr + offsets)
        valid = rank_positions >= 0
        safe_rank_positions = tl.maximum(rank_positions, 0)
        values = tl.load(
            latent_ptr + row * rank + safe_rank_positions,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        state = tl.where(valid, values, 0.0)

        for stage in tl.static_range(0, LOG2_BLOCK):
            pairs = tl.reshape(state, (BLOCK // (2 * (1 << stage)), 2, 1 << stage))
            left, right = tl.split(tl.trans(pairs, (0, 2, 1)))
            joined = tl.join(left + right, left - right)
            state = tl.reshape(tl.trans(joined, (0, 2, 1)), (BLOCK,))

        tl.store(output_ptr + row * BLOCK + offsets, state * scale)

else:
    _sparse_fwht_kernel = None


def _position_to_rank(indices: torch.Tensor, out_features: int) -> torch.Tensor:
    position_to_rank = torch.full(
        (out_features,),
        -1,
        device=indices.device,
        dtype=torch.int32,
    )
    position_to_rank[indices.to(dtype=torch.long)] = torch.arange(
        int(indices.numel()),
        device=indices.device,
        dtype=torch.int32,
    )
    return position_to_rank


def _triton_sparse_fwht_last_dim(
    latent: torch.Tensor,
    position_to_rank: torch.Tensor,
    transform_features: int,
    *,
    out_features: int | None = None,
) -> torch.Tensor:
    """Apply a sparse-input FWHT with a Triton fast path and PyTorch fallback."""

    _require_power_of_two(transform_features, name="transform_features")
    logical_out_features = int(out_features if out_features is not None else transform_features)
    if logical_out_features <= 0 or logical_out_features > transform_features:
        raise ValueError(
            "out_features must be within the Hadamard transform width "
            f"(got {logical_out_features}, transform_features={transform_features})"
        )
    if (
        _sparse_fwht_kernel is None
        or not latent.is_cuda
        or not position_to_rank.is_cuda
        or transform_features > 8192
    ):
        padded = torch.zeros(
            *latent.shape[:-1],
            transform_features,
            device=latent.device,
            dtype=latent.dtype,
        )
        valid = position_to_rank >= 0
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        rank_positions = position_to_rank.index_select(0, indices).to(dtype=torch.long)
        source = latent.index_select(-1, rank_positions.to(device=latent.device))
        padded.index_copy_(-1, indices.to(device=latent.device), source)
        output = _fwht_last_dim(padded) * (1.0 / math.sqrt(transform_features))
        return output[..., :logical_out_features]

    rank = int(latent.shape[-1])
    flat_latent = latent.contiguous().reshape(-1, rank)
    output = torch.empty(
        flat_latent.shape[0],
        transform_features,
        device=latent.device,
        dtype=latent.dtype,
    )
    log2_out_features = int(math.log2(transform_features))
    _sparse_fwht_kernel[(flat_latent.shape[0],)](
        flat_latent,
        position_to_rank.contiguous(),
        output,
        rank,
        1.0 / math.sqrt(transform_features),
        BLOCK=transform_features,
        LOG2_BLOCK=log2_out_features,
        num_warps=8,
    )
    output = output.reshape(*latent.shape[:-1], transform_features)
    return output[..., :logical_out_features]


def hadamard_basis_columns(
    out_features: int,
    indices: torch.Tensor,
    *,
    transform_features: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return normalized Sylvester-Hadamard columns ``H[:, indices]``.

    This helper is intended for setup/factorization.  Runtime reconstruction
    should use :class:`HadamardLowRankAllReduceOutput`, which applies FWHT
    instead of materializing these columns.
    """

    transform_width = int(transform_features or hadamard_transform_features(out_features))
    _require_power_of_two(transform_width, name="transform_features")
    if out_features <= 0 or out_features > transform_width:
        raise ValueError(
            "out_features must be within the Hadamard transform width "
            f"(got {out_features}, transform_features={transform_width})"
        )
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1-D, got shape {tuple(indices.shape)}")
    if indices.numel() == 0:
        raise ValueError("indices must not be empty")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= transform_width:
        raise ValueError(f"indices must be within [0, {transform_width})")

    basis = torch.zeros(
        transform_width,
        int(indices.numel()),
        device=device,
        dtype=torch.float32,
    )
    basis[indices.to(device=device, dtype=torch.long), torch.arange(indices.numel(), device=device)] = 1.0
    basis = _fwht_last_dim(basis.transpose(0, 1)).transpose(0, 1)
    basis = basis * (1.0 / math.sqrt(transform_width))
    basis = basis[:out_features, :]
    return basis.to(dtype=dtype)


@torch.no_grad()
def factorize_output_weight_with_hadamard_basis(
    weight: torch.Tensor,
    indices: torch.Tensor,
    *,
    factor_dtype: torch.dtype | None = None,
) -> LowRankOutputFactors:
    """Project an output projection onto fixed Hadamard output columns."""

    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    out_features, _ = map(int, weight.shape)
    output_basis = hadamard_basis_columns(
        out_features,
        indices.to(device=weight.device),
        device=weight.device,
        dtype=factor_dtype or weight.dtype,
    )
    math_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    input_factor = weight.detach().to(math_dtype).transpose(0, 1) @ output_basis.to(math_dtype)
    input_factor = input_factor.to(dtype=factor_dtype or weight.dtype).contiguous()
    approximation = output_basis.to(torch.float32) @ input_factor.to(torch.float32).transpose(0, 1)
    error = _relative_weight_error(weight, approximation)
    singular_values = torch.empty(
        int(indices.numel()),
        device=weight.device,
        dtype=torch.float32,
    )
    singular_values.fill_(float("nan"))
    return LowRankOutputFactors(
        input_factor=input_factor,
        output_basis=output_basis,
        singular_values=singular_values,
        relative_frobenius_error=error,
    )


class HadamardLowRankAllReduceOutput(nn.Module):
    """Low-rank row-parallel output using FWHT reconstruction.

    The fixed output basis is a subset of normalized Hadamard columns.  Forward
    communicates the selected coefficients, scatters them into a full-width
    buffer, and applies an FWHT instead of a dense ``latent @ R.T`` GEMM.
    """

    def __init__(
        self,
        local_input_factor: torch.Tensor,
        indices: torch.Tensor,
        *,
        out_features: int,
        bias: torch.Tensor | None = None,
        process_group: dist.ProcessGroup | None = None,
        communication_dtype: torch.dtype | None = None,
        reconstruction: HadamardReconstructionMethod = "fwht",
        debug_sync: bool = False,
    ) -> None:
        super().__init__()
        if local_input_factor.ndim != 2:
            raise ValueError("local_input_factor must be [local_d_in, rank]")
        if reconstruction not in {"fwht", "dense_basis", "triton_fwht"}:
            raise ValueError(f"unsupported Hadamard reconstruction: {reconstruction}")
        transform_features = hadamard_transform_features(out_features)
        if indices.ndim != 1:
            raise ValueError(f"indices must be 1-D, got shape {tuple(indices.shape)}")
        if int(indices.numel()) != int(local_input_factor.shape[1]):
            raise ValueError(
                f"indices length must match rank ({indices.numel()} != {local_input_factor.shape[1]})"
            )
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= transform_features:
            raise ValueError(f"indices must be within [0, {transform_features})")

        self.local_projection_weight = nn.Parameter(
            local_input_factor.detach().transpose(0, 1).contiguous(),
            requires_grad=False,
        )
        self.register_buffer("indices", indices.detach().to(dtype=torch.long).contiguous())
        self.out_features_value = int(out_features)
        self.transform_features_value = int(transform_features)
        self.process_group = process_group
        self.communication_dtype = communication_dtype
        self.reconstruction = reconstruction
        self.debug_sync = bool(debug_sync)

        if reconstruction == "dense_basis":
            output_basis = hadamard_basis_columns(
                out_features,
                indices.to(device=local_input_factor.device),
                transform_features=transform_features,
                device=local_input_factor.device,
                dtype=local_input_factor.dtype,
            )
            self.output_basis_weight = nn.Parameter(
                output_basis.detach().contiguous(),
                requires_grad=False,
            )
        else:
            self.register_parameter("output_basis_weight", None)

        if reconstruction == "triton_fwht":
            self.register_buffer(
                "position_to_rank",
                _position_to_rank(
                    indices.to(device=local_input_factor.device),
                    transform_features,
                ),
            )
        else:
            self.register_buffer("position_to_rank", None)

        if bias is None:
            self.register_parameter("bias", None)
        else:
            if bias.shape != (out_features,):
                raise ValueError(f"bias must have shape ({out_features},), got {tuple(bias.shape)}")
            self.bias = nn.Parameter(bias.detach().contiguous(), requires_grad=False)
        self.runtime_breakdown = RuntimeBreakdownRecorder()

    @property
    def rank(self) -> int:
        return int(self.local_projection_weight.shape[0])

    @property
    def local_in_features(self) -> int:
        return int(self.local_projection_weight.shape[1])

    @property
    def out_features(self) -> int:
        return self.out_features_value

    @property
    def transform_features(self) -> int:
        return self.transform_features_value

    @property
    def world_size(self) -> int:
        return _group_world_size(self.process_group)

    def communication_estimate(
        self,
        tokens: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> CommunicationEstimate:
        if tokens <= 0:
            raise ValueError(f"tokens must be positive, got {tokens}")
        tensor_dtype = dtype or self.communication_dtype or self.local_projection_weight.dtype
        element_size = torch.empty((), dtype=tensor_dtype).element_size()
        return CommunicationEstimate(
            world_size=self.world_size,
            tokens=tokens,
            full_width=self.out_features,
            low_rank_width=self.rank,
            element_size_bytes=element_size,
        )

    def enable_runtime_breakdown(self, enabled: bool = True) -> None:
        self.runtime_breakdown.enabled = enabled

    def reset_runtime_breakdown(self) -> None:
        self.runtime_breakdown.reset()

    def runtime_breakdown_summary(self) -> dict[str, object]:
        summary = self.runtime_breakdown.summary()
        summary.update(
            {
                "module_type": "hadamard_low_rank",
                "local_in_features": self.local_in_features,
                "out_features": self.out_features,
                "transform_features": self.transform_features,
                "rank": self.rank,
                "world_size": self.world_size,
                "reconstruction": self.reconstruction,
                "debug_sync": self.debug_sync,
            }
        )
        return summary

    def _debug_synchronize(self, tensor: torch.Tensor) -> None:
        if self.debug_sync and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def forward(
        self,
        local_hidden_states: torch.Tensor,
        *,
        return_reduced_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled() and local_hidden_states.requires_grad:
            raise RuntimeError("HadamardLowRankAllReduceOutput is an inference-only prototype")
        if local_hidden_states.shape[-1] != self.local_in_features:
            raise ValueError(
                f"expected local hidden width {self.local_in_features}, "
                f"got {local_hidden_states.shape[-1]}"
            )

        recorder = self.runtime_breakdown
        if not recorder.enabled:
            latent = F.linear(local_hidden_states, self.local_projection_weight, bias=None)
            self._debug_synchronize(latent)
            compute_dtype = latent.dtype
            if self.communication_dtype is not None and latent.dtype != self.communication_dtype:
                latent = latent.to(self.communication_dtype)
                self._debug_synchronize(latent)
            _all_reduce_sum_(latent, self.process_group)
            self._debug_synchronize(latent)
            if latent.dtype != compute_dtype:
                latent = latent.to(compute_dtype)
                self._debug_synchronize(latent)

            if self.reconstruction == "dense_basis":
                output = F.linear(latent, self.output_basis_weight, self.bias)
            elif self.reconstruction == "triton_fwht":
                output = _triton_sparse_fwht_last_dim(
                    latent,
                    self.position_to_rank,
                    self.transform_features,
                    out_features=self.out_features,
                )
                if self.bias is not None:
                    output = output + self.bias
            else:
                padded = torch.zeros(
                    *latent.shape[:-1],
                    self.transform_features,
                    device=latent.device,
                    dtype=latent.dtype,
                )
                padded.index_copy_(-1, self.indices.to(device=latent.device), latent)
                output = _fwht_last_dim(padded) * (1.0 / math.sqrt(self.transform_features))
                output = output[..., : self.out_features]
                if self.bias is not None:
                    output = output + self.bias
            self._debug_synchronize(output)
            if return_reduced_latent:
                return output, latent
            return output

        device = local_hidden_states.device

        def decode(latent: torch.Tensor) -> torch.Tensor:
            if self.reconstruction == "dense_basis":
                return F.linear(latent, self.output_basis_weight, self.bias)
            if self.reconstruction == "triton_fwht":
                output = _triton_sparse_fwht_last_dim(
                    latent,
                    self.position_to_rank,
                    self.transform_features,
                    out_features=self.out_features,
                )
                if self.bias is not None:
                    output = output + self.bias
                return output
            padded = torch.zeros(
                *latent.shape[:-1],
                self.transform_features,
                device=latent.device,
                dtype=latent.dtype,
            )
            padded.index_copy_(-1, self.indices.to(device=latent.device), latent)
            output = _fwht_last_dim(padded) * (1.0 / math.sqrt(self.transform_features))
            output = output[..., : self.out_features]
            if self.bias is not None:
                output = output + self.bias
            return output

        def body() -> tuple[torch.Tensor, torch.Tensor]:
            latent = recorder.record_segment(
                "encode",
                lambda: F.linear(local_hidden_states, self.local_projection_weight, bias=None),
                device=device,
            )
            self._debug_synchronize(latent)
            compute_dtype = latent.dtype
            tokens = int(latent.numel() // max(self.rank, 1))
            recorder.record_call(tokens)
            recorder.record_payload(
                dense_payload_elements=tokens * self.out_features,
                compressed_payload_elements=int(latent.numel()),
                element_size_bytes=latent.element_size(),
            )
            if self.communication_dtype is not None and latent.dtype != self.communication_dtype:
                latent = recorder.record_segment(
                    "cast_to_comm",
                    lambda: latent.to(self.communication_dtype),
                    device=device,
                )
                self._debug_synchronize(latent)
            latent = recorder.record_segment(
                "all_reduce",
                lambda: _all_reduce_sum_(latent, self.process_group),
                device=device,
            )
            self._debug_synchronize(latent)
            if latent.dtype != compute_dtype:
                latent = recorder.record_segment(
                    "cast_from_comm",
                    lambda: latent.to(compute_dtype),
                    device=device,
                )
                self._debug_synchronize(latent)
            output = recorder.record_segment(
                "decode",
                lambda: decode(latent),
                device=device,
            )
            self._debug_synchronize(output)
            return output, latent

        output, latent = recorder.record_segment("total", body, device=device)
        if return_reduced_latent:
            return output, latent
        return output

    def extra_repr(self) -> str:
        return (
            f"local_in_features={self.local_in_features}, "
            f"out_features={self.out_features}, rank={self.rank}, "
            f"world_size={self.world_size}, reconstruction={self.reconstruction}"
        )
