"""Utilities for decoder-closed, budget-preserving GQA rank candidates.

The functions in this module deliberately separate two operations:

* selecting an already-generated Value encoder at a different rank; and
* analytically closing the complete layer decoder for the resulting ragged
  eight-group rank vector.

No encoder optimization is performed here.  Folded rank-bank payloads are
recovered into the dense Value-head coordinate system only so the existing
routed quadratic decoder solver can be reused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Callable, Mapping, Sequence

import torch

from basisserve.core.gqa_routed_ov_joint import (
    LinearSolveDiagnostics,
    RoutedOVQuadratic,
    gauge_canonicalize_ragged,
    solve_free_decoder_ragged,
)


DEFAULT_ALLOWED_RANKS = (32, 48, 64, 80, 96, 112, 128)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash the exact tensor dtype, shape, and contiguous byte payload."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DecoderClosedRankSwap:
    """One directed, within-layer, budget-preserving adjacent rank transfer."""

    layer_index: int
    receiver_group: int
    donor_group: int
    receiver_rank_before: int
    receiver_rank_after: int
    donor_rank_before: int
    donor_rank_after: int
    ranks_before: tuple[int, ...]
    ranks_after: tuple[int, ...]

    @property
    def candidate_id(self) -> str:
        return (
            f"layer{self.layer_index:02d}__recv{self.receiver_group}_"
            f"r{self.receiver_rank_before:03d}to{self.receiver_rank_after:03d}__"
            f"donor{self.donor_group}_"
            f"r{self.donor_rank_before:03d}to{self.donor_rank_after:03d}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecoderClosedRankSwap":
        return cls(
            layer_index=int(payload["layer_index"]),
            receiver_group=int(payload["receiver_group"]),
            donor_group=int(payload["donor_group"]),
            receiver_rank_before=int(payload["receiver_rank_before"]),
            receiver_rank_after=int(payload["receiver_rank_after"]),
            donor_rank_before=int(payload["donor_rank_before"]),
            donor_rank_after=int(payload["donor_rank_after"]),
            ranks_before=tuple(int(item) for item in payload["ranks_before"]),
            ranks_after=tuple(int(item) for item in payload["ranks_after"]),
        )


def validate_rank_swap(
    swap: DecoderClosedRankSwap,
    *,
    allowed_ranks: Sequence[int] = DEFAULT_ALLOWED_RANKS,
    rank_step: int = 16,
) -> None:
    """Assert all local/global serving invariants implied by one rank swap."""

    allowed = {int(rank) for rank in allowed_ranks}
    before = tuple(int(rank) for rank in swap.ranks_before)
    after = tuple(int(rank) for rank in swap.ranks_after)
    if not before or len(before) != len(after):
        raise ValueError("rank vectors must have the same non-zero length")
    if not 0 <= swap.receiver_group < len(before):
        raise ValueError("receiver group is out of range")
    if not 0 <= swap.donor_group < len(before):
        raise ValueError("donor group is out of range")
    if swap.receiver_group == swap.donor_group:
        raise ValueError("receiver and donor groups must differ")
    if any(rank not in allowed for rank in (*before, *after)):
        raise ValueError("rank vector contains a rank outside the allowed set")
    if swap.receiver_rank_before != before[swap.receiver_group]:
        raise ValueError("receiver before-rank does not match the rank vector")
    if swap.receiver_rank_after != after[swap.receiver_group]:
        raise ValueError("receiver after-rank does not match the rank vector")
    if swap.donor_rank_before != before[swap.donor_group]:
        raise ValueError("donor before-rank does not match the rank vector")
    if swap.donor_rank_after != after[swap.donor_group]:
        raise ValueError("donor after-rank does not match the rank vector")
    if swap.receiver_rank_after - swap.receiver_rank_before != rank_step:
        raise ValueError("receiver rank must increase by exactly one rank step")
    if swap.donor_rank_before - swap.donor_rank_after != rank_step:
        raise ValueError("donor rank must decrease by exactly one rank step")
    changed = [index for index, pair in enumerate(zip(before, after)) if pair[0] != pair[1]]
    if changed != sorted((swap.receiver_group, swap.donor_group)):
        raise ValueError("a rank swap must change exactly the receiver and donor")
    if sum(before) != sum(after):
        raise ValueError("a rank swap must preserve the layer rank sum")


def enumerate_decoder_closed_rank_swaps(
    schedule: Sequence[Sequence[int]],
    *,
    allowed_ranks: Sequence[int] = DEFAULT_ALLOWED_RANKS,
    rank_step: int = 16,
    layers: Sequence[int] | None = None,
) -> tuple[DecoderClosedRankSwap, ...]:
    """Enumerate and deterministically deduplicate all directed layer swaps."""

    allowed = tuple(sorted({int(rank) for rank in allowed_ranks}))
    allowed_set = set(allowed)
    selected_layers = (
        tuple(range(len(schedule)))
        if layers is None
        else tuple(sorted({int(layer) for layer in layers}))
    )
    candidates: dict[tuple[int, tuple[int, ...]], DecoderClosedRankSwap] = {}
    for layer_index in selected_layers:
        if not 0 <= layer_index < len(schedule):
            raise IndexError(f"layer index out of range: {layer_index}")
        before = tuple(int(rank) for rank in schedule[layer_index])
        if not before or any(rank not in allowed_set for rank in before):
            raise ValueError(f"layer {layer_index} has an invalid rank vector")
        for receiver_group, receiver_rank in enumerate(before):
            receiver_after = receiver_rank + rank_step
            if receiver_after not in allowed_set:
                continue
            for donor_group, donor_rank in enumerate(before):
                if donor_group == receiver_group:
                    continue
                donor_after = donor_rank - rank_step
                if donor_after not in allowed_set:
                    continue
                after = list(before)
                after[receiver_group] = receiver_after
                after[donor_group] = donor_after
                swap = DecoderClosedRankSwap(
                    layer_index=layer_index,
                    receiver_group=receiver_group,
                    donor_group=donor_group,
                    receiver_rank_before=receiver_rank,
                    receiver_rank_after=receiver_after,
                    donor_rank_before=donor_rank,
                    donor_rank_after=donor_after,
                    ranks_before=before,
                    ranks_after=tuple(after),
                )
                validate_rank_swap(
                    swap,
                    allowed_ranks=allowed,
                    rank_step=rank_step,
                )
                candidates[(layer_index, swap.ranks_after)] = swap
    return tuple(
        candidates[key]
        for key in sorted(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
    )


def enumerate_layer_rank_swaps(
    *,
    layer_index: int,
    ranks: Sequence[int],
    allowed_ranks: Sequence[int] = DEFAULT_ALLOWED_RANKS,
    rank_step: int = 16,
) -> tuple[DecoderClosedRankSwap, ...]:
    """Compatibility wrapper for enumerating one complete layer vector."""

    schedule = [tuple(int(item) for item in ranks)]
    candidates = enumerate_decoder_closed_rank_swaps(
        schedule,
        allowed_ranks=allowed_ranks,
        rank_step=rank_step,
        layers=(0,),
    )
    return tuple(
        DecoderClosedRankSwap(
            layer_index=int(layer_index),
            receiver_group=item.receiver_group,
            donor_group=item.donor_group,
            receiver_rank_before=item.receiver_rank_before,
            receiver_rank_after=item.receiver_rank_after,
            donor_rank_before=item.donor_rank_before,
            donor_rank_after=item.donor_rank_after,
            ranks_before=item.ranks_before,
            ranks_after=item.ranks_after,
        )
        for item in candidates
    )


def enumerate_schedule_rank_swaps(
    schedule: Sequence[Sequence[int]],
    *,
    allowed_ranks: Sequence[int] = DEFAULT_ALLOWED_RANKS,
    rank_step: int = 16,
) -> tuple[DecoderClosedRankSwap, ...]:
    return enumerate_decoder_closed_rank_swaps(
        schedule,
        allowed_ranks=allowed_ranks,
        rank_step=rank_step,
    )


def marginal_topk_proposal_ids(
    schedule: Sequence[Sequence[int]],
    costs: Mapping[tuple[int, int], Mapping[int, float]],
    *,
    allowed_ranks: Sequence[int] = DEFAULT_ALLOWED_RANKS,
    rank_step: int = 16,
    receiver_top_k: int = 4,
    donor_top_k: int = 4,
) -> frozenset[str]:
    """Return old-additive-cost proposal IDs for retrospective coverage only."""

    if receiver_top_k <= 0 or donor_top_k <= 0:
        raise ValueError("proposal top-K values must be positive")
    allowed = {int(rank) for rank in allowed_ranks}
    proposed: set[str] = set()
    for layer_index, raw_ranks in enumerate(schedule):
        ranks = tuple(int(rank) for rank in raw_ranks)
        receivers: list[tuple[float, int]] = []
        donors: list[tuple[float, int]] = []
        for group, rank in enumerate(ranks):
            curve = costs[(layer_index, group)]
            if rank + rank_step in allowed:
                benefit = float(curve[rank]) - float(curve[rank + rank_step])
                receivers.append((benefit, group))
            if rank - rank_step in allowed:
                harm = float(curve[rank - rank_step]) - float(curve[rank])
                donors.append((harm, group))
        receivers = sorted(receivers, key=lambda item: (-item[0], item[1]))[
            :receiver_top_k
        ]
        donors = sorted(donors, key=lambda item: (item[0], item[1]))[:donor_top_k]
        for _, receiver in receivers:
            for _, donor in donors:
                if receiver == donor:
                    continue
                after = list(ranks)
                after[receiver] += rank_step
                after[donor] -= rank_step
                swap = DecoderClosedRankSwap(
                    layer_index=layer_index,
                    receiver_group=receiver,
                    donor_group=donor,
                    receiver_rank_before=ranks[receiver],
                    receiver_rank_after=after[receiver],
                    donor_rank_before=ranks[donor],
                    donor_rank_after=after[donor],
                    ranks_before=ranks,
                    ranks_after=tuple(after),
                )
                validate_rank_swap(
                    swap,
                    allowed_ranks=tuple(sorted(allowed)),
                    rank_step=rank_step,
                )
                proposed.add(swap.candidate_id)
    return frozenset(proposed)


@dataclass(frozen=True)
class RecoveredRaggedAnchor:
    A_unique: torch.Tensor
    D_heads: torch.Tensor
    source_group_checksums: tuple[dict[str, Any], ...]
    maximum_value_encoder_error: float
    maximum_head_product_error: float
    value_encoder_errors: tuple[float, ...]
    head_product_errors: tuple[float, ...]

    @property
    def source_group_v_sha256(self) -> tuple[str, ...]:
        return tuple(str(item["v_sha256"]) for item in self.source_group_checksums)

    @property
    def source_group_o_sha256(self) -> tuple[str, ...]:
        return tuple(str(item["o_sha256"]) for item in self.source_group_checksums)


def recover_ragged_anchor_from_folded(
    *,
    dense_v: torch.Tensor,
    dense_o: torch.Tensor,
    group_ranks: Sequence[int],
    num_heads: int,
    num_groups: int,
    head_dim: int,
    hidden_size: int,
    work_dtype: torch.dtype,
    device: torch.device | str,
    folded_payloads: Mapping[int, Mapping[str, Any]] | None = None,
    payload_for_rank: Callable[[int], Mapping[str, Any]] | None = None,
    use_full_rank_payloads: bool = False,
) -> RecoveredRaggedAnchor:
    """Recover one padded ragged A/D anchor from folded rank-bank payloads.

    Full-rank groups default to the historical dense passthrough semantics.
    Set ``use_full_rank_payloads`` only for banks that explicitly advertise
    full-rank factor overrides.
    """

    if folded_payloads is not None and payload_for_rank is not None:
        raise ValueError("pass either folded payloads or a payload loader, not both")
    if folded_payloads is None:
        if payload_for_rank is None:
            raise ValueError("folded payloads or a payload loader are required")
        folded_payloads = {
            int(rank): payload_for_rank(int(rank))
            for rank in sorted(set(map(int, group_ranks)))
            if int(rank) < int(head_dim)
            or (use_full_rank_payloads and int(rank) == int(head_dim))
        }
    if num_heads % num_groups:
        raise ValueError("query-head count must be divisible by KV-group count")
    ranks = tuple(int(rank) for rank in group_ranks)
    if len(ranks) != num_groups or any(not 0 < rank <= head_dim for rank in ranks):
        raise ValueError("group ranks do not match the GQA geometry")
    target_device = torch.device(device)
    A = torch.zeros(
        num_groups,
        head_dim,
        head_dim,
        device=target_device,
        dtype=work_dtype,
    )
    D = torch.zeros(
        num_heads,
        head_dim,
        hidden_size,
        device=target_device,
        dtype=work_dtype,
    )
    dense_math = dense_v.detach().to(device=target_device, dtype=work_dtype)
    dense_o_math = dense_o.detach().to(device=target_device, dtype=work_dtype)
    if tuple(dense_math.shape) != (num_groups * head_dim, hidden_size):
        raise ValueError("dense V tensor does not match the declared geometry")
    if tuple(dense_o_math.shape) != (num_heads, head_dim, hidden_size):
        raise ValueError("dense O tensor does not match the declared geometry")
    heads_per_group = num_heads // num_groups
    value_errors: list[float] = []
    product_errors: list[float] = []
    checksums: list[dict[str, Any]] = []
    for group, rank in enumerate(ranks):
        v_rows = slice(group * head_dim, (group + 1) * head_dim)
        first_head = group * heads_per_group
        heads = slice(first_head, first_head + heads_per_group)
        if rank == head_dim and not use_full_rank_payloads:
            A[group] = torch.eye(
                head_dim,
                device=target_device,
                dtype=work_dtype,
            )
            D[heads] = dense_o_math[heads]
            value_errors.append(0.0)
            product_errors.extend([0.0] * heads_per_group)
            checksums.append(
                {
                    "group_index": group,
                    "rank": rank,
                    "dense_passthrough": True,
                    "v_sha256": tensor_sha256(dense_v[v_rows]),
                    "o_sha256": tensor_sha256(dense_o[heads]),
                }
            )
            continue
        if rank not in folded_payloads:
            raise KeyError(f"missing folded rank-{rank} payload")
        payload = folded_payloads[rank]
        if payload.get("v_proj_compressed_bias") is not None:
            raise ValueError("decoder closure does not support V bias")
        compressed_v = payload["v_proj_compressed_weight"]
        compressed_o_weight = payload["o_decoder_weight"]
        expected_v = (num_groups * rank, hidden_size)
        expected_o = (hidden_size, num_heads * rank)
        if tuple(compressed_v.shape) != expected_v:
            raise ValueError(f"rank-{rank} folded V shape is invalid")
        if tuple(compressed_o_weight.shape) != expected_o:
            raise ValueError(f"rank-{rank} folded O shape is invalid")
        v_group = compressed_v[group * rank : (group + 1) * rank]
        o_start = first_head * rank
        o_group = compressed_o_weight[
            :, o_start : (first_head + heads_per_group) * rank
        ]
        compressed_o = (
            compressed_o_weight.to(device=target_device, dtype=work_dtype)
            .transpose(0, 1)
            .reshape(num_heads, rank, hidden_size)
        )
        left = dense_math[v_rows].transpose(0, 1)
        target = v_group.to(device=target_device, dtype=work_dtype).transpose(0, 1)
        recovered = torch.linalg.lstsq(left, target).solution
        A[group, :, :rank] = recovered
        D[heads, :rank] = compressed_o[heads]
        denominator = torch.linalg.vector_norm(target).clamp_min(
            torch.finfo(work_dtype).tiny
        )
        value_errors.append(
            float(torch.linalg.vector_norm(left @ recovered - target) / denominator)
        )
        # Compute the exact Frobenius product error without materializing the
        # hidden_size x hidden_size head product.  For matrices X and D,
        # ||X D||_F^2 = tr((X^T X)(D D^T)); both Gram factors are only r x r.
        residual = left @ recovered - target
        target_gram = target.transpose(0, 1) @ target
        residual_gram = residual.transpose(0, 1) @ residual
        for head in range(first_head, first_head + heads_per_group):
            decoder_gram = compressed_o[head] @ compressed_o[head].transpose(0, 1)
            installed_squared = torch.sum(target_gram * decoder_gram)
            error_squared = torch.sum(residual_gram * decoder_gram)
            product_errors.append(
                float(
                    error_squared.clamp_min(0).sqrt()
                    / installed_squared.clamp_min(torch.finfo(work_dtype).tiny).sqrt()
                )
            )
        checksums.append(
            {
                "group_index": group,
                "rank": rank,
                "dense_passthrough": False,
                "v_sha256": tensor_sha256(v_group),
                "o_sha256": tensor_sha256(o_group),
            }
        )
    return RecoveredRaggedAnchor(
        A_unique=A,
        D_heads=D,
        source_group_checksums=tuple(checksums),
        maximum_value_encoder_error=max(value_errors, default=0.0),
        maximum_head_product_error=max(product_errors, default=0.0),
        value_encoder_errors=tuple(value_errors),
        head_product_errors=tuple(product_errors),
    )


@dataclass(frozen=True)
class DecoderClosureResult:
    A_unique: torch.Tensor
    D_heads: torch.Tensor
    gauge_product_error: float
    encoder_sha256_loaded: str
    encoder_sha256_before_solve: str
    encoder_sha256_after_solve: str
    decoder_diagnostics: LinearSolveDiagnostics

    @property
    def decoder(self) -> LinearSolveDiagnostics:
        return self.decoder_diagnostics


def close_ragged_decoder_with_fixed_encoders(
    *,
    objective: RoutedOVQuadratic,
    initial_A: torch.Tensor,
    initial_D: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    group_ranks: Sequence[int],
    relative_jitter: float = 0.0,
) -> DecoderClosureResult:
    """Canonicalize the fixed encoder gauge, then solve only the decoder."""

    mapping = torch.as_tensor(
        head_to_kv_group,
        device=initial_A.device,
        dtype=torch.long,
    )
    loaded = tensor_sha256(initial_A)
    A, gauge_D, gauge_error = gauge_canonicalize_ragged(
        initial_A.detach().clone(),
        initial_D.detach().clone(),
        mapping,
        group_ranks,
    )
    before = tensor_sha256(A)
    D, diagnostics = solve_free_decoder_ragged(
        objective=objective,
        A_unique=A,
        head_to_kv_group=mapping,
        group_ranks=group_ranks,
        relative_jitter=relative_jitter,
    )
    after = tensor_sha256(A)
    if before != after:
        raise RuntimeError("decoder solve mutated a fixed Value encoder")
    del gauge_D
    return DecoderClosureResult(
        A_unique=A,
        D_heads=D,
        gauge_product_error=float(gauge_error),
        encoder_sha256_loaded=loaded,
        encoder_sha256_before_solve=before,
        encoder_sha256_after_solve=after,
        decoder_diagnostics=diagnostics,
    )


def close_fixed_ragged_decoder(
    *,
    objective: RoutedOVQuadratic,
    A_unique: torch.Tensor,
    source_D_heads: torch.Tensor,
    head_to_kv_group: torch.Tensor,
    group_ranks: Sequence[int],
    relative_jitter: float = 0.0,
) -> DecoderClosureResult:
    """Compatibility spelling for fixed-encoder decoder closure."""

    return close_ragged_decoder_with_fixed_encoders(
        objective=objective,
        initial_A=A_unique,
        initial_D=source_D_heads,
        head_to_kv_group=head_to_kv_group,
        group_ranks=group_ranks,
        relative_jitter=relative_jitter,
    )
