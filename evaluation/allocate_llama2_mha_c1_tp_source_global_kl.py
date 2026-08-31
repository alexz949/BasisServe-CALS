#!/usr/bin/env python3
"""Allocate Llama-2 C1 ranks over TP sources with exact terminal KL.

The 32 MHA Value heads are partitioned into eight equal four-head sources owned
by TP=8.  The default is the physical contiguous layout; an optional CKA audit
artifact supplies a train-only, per-layer balanced head permutation.  All
heads inside one source share a rank, while ranks may differ across sources and
layers.  Around the uniform C1 anchor, this program changes one source at a
time, performs a full-layer closed-form joint decoder refit, and measures
paired dense-teacher terminal KL.

Mean and one-standard-error costs are allocated by exact dynamic programming
under the variable-size collective rank budget.  Candidate schedules are
rebuilt with full-layer decoder closure, selected on disjoint C4 contexts, and
only then evaluated on WikiText-2.  Standard rectangular padded-AllGather cost
is reported but is not the optimization constraint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import save_file
import torch
import torch.distributed as dist
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.gqa_routed_ov_joint import (  # noqa: E402
    covariance_with_trace_damping,
    evaluate_quadratic,
    fit_routed_ov_joint,
    quadratic_from_target,
)
from basisserve.core.metric_rank_allocation import (  # noqa: E402
    MetricRankOption,
    allocate_metric_rank_exact,
)
from basisserve.core.decoder_closed_rank_candidates import (  # noqa: E402
    close_ragged_decoder_with_fixed_encoders,
    tensor_sha256,
)
from evaluation.allocate_llama2_mha_c1_global_kl import (  # noqa: E402
    _distributed_teacher_metrics,
    _install_banks,
    _load_factor_results,
    _parse_factor_dirs,
    _select_windows,
    _sha256,
)
from evaluation.allocate_palu_llama2_global_kl import (  # noqa: E402
    _capture_teacher,
    _evaluate_teacher_metrics,
    _paired_delta,
)
from evaluation.fit_llama2_mha_c1_joint import (  # noqa: E402
    _activation_covariance_blocks,
    _load_snapshot_layer,
    _snapshot_manifest,
)
from evaluation.pilot_llama2_mha_c1_per_head_fisher_kl import (  # noqa: E402
    HEAD_DIM,
    HEADS_PER_SOURCE,
    HIDDEN_SIZE,
    NUM_HEADS,
    TP_SIZE,
    _load_layer_factors,
    _padded_candidate_factors,
    _ragged_module,
)
from evaluation.reproduce_palu_paper_llama2_distributed import (  # noqa: E402
    _all_reduce_sum,
    _model_layers,
    _wikitext,
    distributed_official_ppl,
)


FORMAT = "basisserve.llama2_7b.mha_c1.tp_source_global_kl_allocation.v2"
CKA_AUDIT_FORMAT = "basisserve.llama2_7b.mha_c1.cka_tp_head_allocation_audit.v1"
NUM_LAYERS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument(
        "--head-groups",
        type=Path,
        help="optional completed CKA audit result.json containing per-layer groups",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--factor-dir",
        action="append",
        required=True,
        metavar="RANK=PATH",
    )
    parser.add_argument("--anchor-rank", type=int, default=64)
    parser.add_argument("--candidate-ranks", default="48,64,80")
    parser.add_argument("--profile-windows", type=int, default=8)
    parser.add_argument("--confirmation-windows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fit-windows", type=int, default=128)
    parser.add_argument("--covariance-damping", type=float, default=1.0e-7)
    parser.add_argument("--covariance-row-chunk-size", type=int, default=8192)
    parser.add_argument("--decoder-relative-jitter", type=float, default=0.0)
    parser.add_argument(
        "--post-allocation-encoder-sweeps",
        type=int,
        default=0,
        help=(
            "After Global-KL freezes the TP-source schedule, run this many "
            "source-ordered ragged encoder BCD sweeps and close the full-layer "
            "decoder after every sweep; zero preserves the allocation-only path"
        ),
    )
    parser.add_argument(
        "--post-allocation-minimum-encoder-sweeps", type=int, default=2
    )
    parser.add_argument(
        "--post-allocation-encoder-relative-tolerance", type=float, default=1.0e-6
    )
    parser.add_argument("--post-allocation-encoder-patience", type=int, default=2)
    parser.add_argument(
        "--post-allocation-encoder-relative-damping", type=float, default=1.0e-8
    )
    parser.add_argument(
        "--post-allocation-encoder-cg-relative-tolerance",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--post-allocation-encoder-cg-iterations", type=int, default=16
    )
    parser.add_argument(
        "--post-allocation-maximum-backtracks", type=int, default=10
    )
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-max-chunks", type=int, default=146)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _log(message: str, *, all_ranks: bool = False) -> None:
    if all_ranks or _rank() == 0:
        print(f"[rank {_rank()}] {message}", flush=True)


def _parse_ranks(raw: str) -> tuple[int, ...]:
    ranks = tuple(sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()}))
    if not ranks or any(rank <= 0 or rank > HEAD_DIM for rank in ranks):
        raise ValueError("candidate ranks must lie inside the Value-head width")
    return ranks


def _contiguous_head_groups() -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(range(source * HEADS_PER_SOURCE, (source + 1) * HEADS_PER_SOURCE))
        for source in range(TP_SIZE)
    )


def _validate_layer_head_groups(
    groups: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    selected = tuple(tuple(int(head) for head in group) for group in groups)
    if len(selected) != TP_SIZE or any(
        len(group) != HEADS_PER_SOURCE for group in selected
    ):
        raise ValueError("head groups must contain eight equal four-head sources")
    flattened = [head for group in selected for head in group]
    if sorted(flattened) != list(range(NUM_HEADS)):
        raise ValueError("head groups must partition all 32 logical heads exactly once")
    return selected


def _load_head_groups(
    path: Path | None,
) -> tuple[tuple[tuple[int, ...], ...], dict[str, Any]]:
    if path is None:
        groups = _contiguous_head_groups()
        return (groups,) * NUM_LAYERS, {
            "method": "contiguous_physical_tp_layout",
            "path": None,
            "sha256": None,
        }
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("format") != CKA_AUDIT_FORMAT or payload.get("status") != "complete":
        raise ValueError("head-groups artifact is not a completed CKA audit")
    configuration = payload.get("configuration", {})
    if int(configuration.get("tp_size", -1)) != TP_SIZE:
        raise ValueError("CKA audit TP size differs from the allocator")
    indexed = {int(row["layer"]): row for row in payload.get("records", ())}
    if set(indexed) != set(range(NUM_LAYERS)):
        raise ValueError("CKA audit does not cover all 32 layers")
    groups = tuple(
        _validate_layer_head_groups(indexed[layer]["cka_groups"])
        for layer in range(NUM_LAYERS)
    )
    return groups, {
        "method": "train_activation_cka_balanced_tp_groups",
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "audit_format": payload["format"],
        "audit_configuration": configuration,
        "audit_aggregate": payload.get("aggregate"),
        "groups_selected_from": "train_snapshot_only",
        "audit_validation_used_for_group_selection": False,
    }


def _head_ranks_from_source_ranks(
    source_ranks: Sequence[int],
    head_groups: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    selected = tuple(int(rank) for rank in source_ranks)
    groups = _validate_layer_head_groups(head_groups)
    if len(selected) != TP_SIZE:
        raise ValueError("source-rank schedule must contain eight ranks")
    result = [-1] * NUM_HEADS
    for source, group in enumerate(groups):
        for head in group:
            result[head] = selected[source]
    if min(result) <= 0:
        raise ValueError("source ranks must be positive")
    return tuple(result)


def _grouped_collective_cost(
    head_ranks: Sequence[int],
    head_groups: Sequence[Sequence[int]],
) -> dict[str, Any]:
    ranks = tuple(int(rank) for rank in head_ranks)
    groups = _validate_layer_head_groups(head_groups)
    if len(ranks) != NUM_HEADS or min(ranks) <= 0:
        raise ValueError("head ranks must contain 32 positive values")
    source_widths = tuple(sum(ranks[head] for head in group) for group in groups)
    return {
        "tp_size": TP_SIZE,
        "heads_per_source": HEADS_PER_SOURCE,
        "source_widths": list(source_widths),
        "ideal_allgather_width": sum(source_widths),
        "padded_allgather_width": TP_SIZE * max(source_widths),
    }


def _is_contiguous_layout(
    head_groups_by_layer: Sequence[Sequence[Sequence[int]]],
) -> bool:
    contiguous = _contiguous_head_groups()
    return all(tuple(tuple(group) for group in layer) == contiguous for layer in head_groups_by_layer)


@dataclass(frozen=True)
class CachedLayerFactors:
    A: Tensor
    D: Tensor
    ranks: tuple[int, ...]


DECODER_CLOSED_BOUNDARIES = frozenset(("decoder_only", "after_redecoder"))


class _DecoderClosedHeldoutSelector:
    """Keep the earliest held-out optimum whose decoder is fully closed."""

    def __init__(self, validation_objective: Any, mapping: Tensor) -> None:
        self.validation_objective = validation_objective
        self.mapping = mapping
        self.records: list[dict[str, Any]] = []
        self.best_loss = float("inf")
        self.best_A: Tensor | None = None
        self.best_D: Tensor | None = None
        self.best_boundary: str | None = None
        self.best_sweep: int | None = None

    def __call__(self, checkpoint: Any, A: Tensor, D: Tensor) -> None:
        validation_loss = evaluate_quadratic(
            self.validation_objective,
            A,
            D,
            self.mapping,
        )
        boundary = str(checkpoint.boundary)
        eligible = boundary in DECODER_CLOSED_BOUNDARIES
        relative = float(validation_loss) / max(
            abs(float(self.validation_objective.constant)),
            1.0e-300,
        )
        self.records.append(
            {
                "boundary": boundary,
                "sweep": int(checkpoint.sweep),
                "fit_loss": float(checkpoint.loss),
                "validation_loss": float(validation_loss),
                "validation_relative_mse": relative,
                "selection_eligible": eligible,
            }
        )
        if eligible and float(validation_loss) < self.best_loss:
            self.best_loss = float(validation_loss)
            self.best_A = A.detach().clone()
            self.best_D = D.detach().clone()
            self.best_boundary = boundary
            self.best_sweep = int(checkpoint.sweep)

    def selected(self) -> tuple[Tensor, Tensor]:
        if self.best_A is None or self.best_D is None:
            raise RuntimeError("post-allocation sweep produced no decoder-closed checkpoint")
        return self.best_A, self.best_D


def _source_ordered_encoder_groups(
    head_groups: Sequence[Sequence[int]],
    *,
    num_heads: int,
) -> tuple[int, ...]:
    order = tuple(int(head) for group in head_groups for head in group)
    if len(order) != num_heads or sorted(order) != list(range(num_heads)):
        raise ValueError("TP-source groups must partition every encoder group exactly once")
    return order


def _require_zero_sweep_factor_banks(
    factor_results: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and summarize the cold-start banks required by this pipeline."""

    if not factor_results:
        raise ValueError("post-allocation sweep requires factor banks")
    per_rank = {}
    reference: dict[str, Any] | None = None
    invariant_fields = (
        "snapshot_dir",
        "fit_windows",
        "validation_snapshot_dir",
        "validation_window_start",
        "validation_windows",
        "validation_row_start",
        "validation_rows",
    )
    for rank, payload in sorted(factor_results.items()):
        config = payload.get("fit_config")
        if not isinstance(config, Mapping):
            raise ValueError(f"rank {rank} factor bank has no fit_config")
        sweeps = int(config.get("encoder_sweeps", -1))
        minimum = int(config.get("minimum_encoder_sweeps", -1))
        if sweeps != 0 or minimum != 0:
            raise ValueError(
                "post-allocation sweep requires zero-sweep factor banks; "
                f"rank {rank} records encoder_sweeps={sweeps}, minimum={minimum}"
            )
        current = {field: config.get(field) for field in invariant_fields}
        missing = [field for field, value in current.items() if value is None]
        if missing:
            raise ValueError(
                f"rank {rank} zero-sweep factor bank lacks fields: {missing}"
            )
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError("zero-sweep factor banks disagree on fit/held-out snapshots")
        per_rank[str(rank)] = {
            "encoder_sweeps": sweeps,
            "minimum_encoder_sweeps": minimum,
        }
    assert reference is not None
    return {
        "verified_zero_sweep": True,
        "per_rank": per_rank,
        **reference,
    }


@torch.no_grad()
def _fit_fixed_ragged_schedule(
    *,
    fit_objective: Any,
    validation_objective: Any,
    initial_A: Tensor,
    initial_D: Tensor,
    head_ranks: Sequence[int],
    head_groups: Sequence[Sequence[int]],
    maximum_sweeps: int,
    minimum_sweeps: int,
    relative_objective_tolerance: float,
    patience: int,
    decoder_relative_jitter: float,
    encoder_relative_damping: float,
    cg_relative_tolerance: float,
    cg_iterations: int,
    maximum_backtracks: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Sweep a frozen heterogeneous schedule under the full-layer objective."""

    if initial_A.ndim != 3 or initial_D.ndim != 3:
        raise ValueError("post-allocation factors must be rank-three tensors")
    heads = int(initial_D.shape[0])
    ranks = tuple(int(rank) for rank in head_ranks)
    if len(ranks) != heads or min(ranks) <= 0:
        raise ValueError("post-allocation head ranks are invalid")
    if not 0 < minimum_sweeps <= maximum_sweeps:
        raise ValueError("post-allocation sweep limits are invalid")
    if patience <= 0 or relative_objective_tolerance < 0:
        raise ValueError("post-allocation convergence controls are invalid")
    if encoder_relative_damping < 0 or cg_relative_tolerance <= 0:
        raise ValueError("post-allocation encoder solver controls are invalid")
    if cg_iterations <= 0 or maximum_backtracks < 0:
        raise ValueError("post-allocation iteration controls are invalid")

    device = initial_A.device
    mapping = torch.arange(heads, dtype=torch.long, device=device)
    update_order = _source_ordered_encoder_groups(
        head_groups,
        num_heads=heads,
    )
    selector = _DecoderClosedHeldoutSelector(validation_objective, mapping)
    result = fit_routed_ov_joint(
        objective=fit_objective,
        initial_A=initial_A,
        initial_D=initial_D,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
        maximum_sweeps=maximum_sweeps,
        minimum_sweeps=minimum_sweeps,
        relative_objective_tolerance=relative_objective_tolerance,
        patience=patience,
        decoder_relative_jitter=decoder_relative_jitter,
        encoder_relative_damping=encoder_relative_damping,
        cg_relative_tolerance=cg_relative_tolerance,
        cg_max_iterations=cg_iterations,
        cg_fixed_iterations=True,
        maximum_backtracks=maximum_backtracks,
        encoder_group_indices=update_order,
        final_decoder_solve=True,
        verify_encoder_step_objective=True,
        group_ranks=ranks,
        checkpoint_callback=selector,
        decoder_stationarity_override=lambda *_: 0.0,
        work_dtype=torch.float64,
        work_device=device,
    )
    selected_A, selected_D = selector.selected()
    selected_fit_loss = evaluate_quadratic(
        fit_objective,
        selected_A,
        selected_D,
        mapping,
    )
    selected_validation_loss = evaluate_quadratic(
        validation_objective,
        selected_A,
        selected_D,
        mapping,
    )
    diagnostics = {
        "method": "fixed_allocation_source_ordered_encoder_bcd",
        "decoder_closure": "full_layer_after_every_complete_encoder_sweep",
        "source_update_order": [list(map(int, group)) for group in head_groups],
        "encoder_group_update_order": list(update_order),
        "maximum_sweeps": maximum_sweeps,
        "minimum_sweeps": minimum_sweeps,
        "executed_sweeps": len(result.sweeps),
        "selection": {
            "criterion": (
                "earliest minimum held-out attention-output MSE among "
                "decoder-closed checkpoints"
            ),
            "boundary": selector.best_boundary,
            "sweep": selector.best_sweep,
            "checkpoints": selector.records,
        },
        "fit": {
            "initial_loss": result.initial_loss,
            "decoder_only_loss": result.decoder_only_loss,
            "solver_endpoint_loss": result.final_loss,
            "selected_loss": float(selected_fit_loss),
            "selected_relative_mse": float(selected_fit_loss)
            / max(abs(float(fit_objective.constant)), 1.0e-300),
        },
        "heldout": {
            "selected_loss": float(selected_validation_loss),
            "selected_relative_mse": float(selected_validation_loss)
            / max(abs(float(validation_objective.constant)), 1.0e-300),
        },
        "sweeps": [
            {
                "sweep": int(item.sweep),
                "loss_before_decoder": float(item.loss_before_decoder),
                "loss_after_decoder": float(item.loss_after_decoder),
                "loss_after_encoders": float(item.loss_after_encoders),
                "relative_improvement": float(item.relative_improvement),
                "encoder_group_order": [
                    int(step.group_index) for step in item.encoder_steps
                ],
            }
            for item in result.sweeps
        ],
    }
    return selected_A, selected_D, diagnostics


def _broadcast_fit_covariance(
    *,
    snapshot_dir: Path,
    layer: int,
    fit_windows: int,
    row_chunk_size: int,
    expected_o_weight: Tensor,
    device: torch.device,
) -> Tensor:
    shape = (NUM_HEADS, NUM_HEADS, HEAD_DIM, HEAD_DIM)
    covariance = torch.empty(shape, dtype=torch.float64, device=device)
    if _rank() == 0:
        manifest = _snapshot_manifest(snapshot_dir)
        activation, weight, _ = _load_snapshot_layer(snapshot_dir, manifest, layer)
        positions = int(manifest["calibration"]["positions_per_window"])
        rows = fit_windows * positions
        if rows <= 0 or rows > len(activation):
            raise ValueError("fit windows exceed the stored activation snapshot")
        if not torch.equal(weight, expected_o_weight.to(dtype=weight.dtype)):
            raise ValueError(f"snapshot O weight differs at layer {layer}")
        covariance.copy_(
            _activation_covariance_blocks(
                activation[:rows],
                device=device,
                dtype=torch.float64,
                row_chunk_size=row_chunk_size,
            )
        )
        del activation, weight
    dist.broadcast(covariance, src=0)
    return covariance


def _snapshot_covariance_rows(
    *,
    snapshot_dir: Path,
    layer: int,
    row_start: int,
    rows: int,
    row_chunk_size: int,
    expected_o_weight: Tensor,
    device: torch.device,
) -> Tensor:
    """Load one deterministic row range without replicating it across ranks."""

    if row_start < 0 or rows <= 0:
        raise ValueError("snapshot covariance row range is invalid")
    manifest = _snapshot_manifest(snapshot_dir)
    activation, weight, _ = _load_snapshot_layer(snapshot_dir, manifest, layer)
    row_stop = row_start + rows
    if row_stop > len(activation):
        raise ValueError("snapshot covariance row range exceeds stored activations")
    if not torch.equal(weight, expected_o_weight.to(dtype=weight.dtype)):
        raise ValueError(f"snapshot O weight differs at layer {layer}")
    covariance = _activation_covariance_blocks(
        activation[row_start:row_stop],
        device=device,
        dtype=torch.float64,
        row_chunk_size=row_chunk_size,
    )
    del activation, weight
    return covariance


def _objective(
    covariance: Tensor,
    dense_o_weight: Tensor,
    *,
    layer: int,
    device: torch.device,
) -> Any:
    target = (
        dense_o_weight.to(device=device, dtype=torch.float64)
        .transpose(0, 1)
        .reshape(NUM_HEADS, HEAD_DIM, HIDDEN_SIZE)
        .contiguous()
    )
    return quadratic_from_target(
        covariance=covariance,
        target=target,
        name=f"llama2_c1_tp_source_global_kl_layer_{layer:03d}",
        trace_normalize=False,
    )


def _schedule_factors(
    source: Mapping[int, tuple[Tensor, Tensor]],
    *,
    source_ranks: Sequence[int],
    head_groups: Sequence[Sequence[int]] | None = None,
    maximum_rank: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    selected = tuple(int(rank) for rank in source_ranks)
    if len(selected) != TP_SIZE or any(rank not in source for rank in selected):
        raise ValueError("source schedule does not match the factor bank")
    head_ranks = _head_ranks_from_source_ranks(
        selected,
        _contiguous_head_groups() if head_groups is None else head_groups,
    )
    A = torch.zeros(
        NUM_HEADS,
        HEAD_DIM,
        maximum_rank,
        device=device,
        dtype=torch.float64,
    )
    D = torch.zeros(
        NUM_HEADS,
        maximum_rank,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.float64,
    )
    for head, rank in enumerate(head_ranks):
        source_A, source_D = source[rank]
        A[head, :, :rank].copy_(source_A[head].to(device=device, dtype=torch.float64))
        D[head, :rank].copy_(source_D[head].to(device=device, dtype=torch.float64))
    return A, D, head_ranks


def _schedule_accounting(
    schedule: Sequence[Sequence[int]],
    *,
    anchor_rank: int = 64,
) -> dict[str, Any]:
    selected = tuple(tuple(int(rank) for rank in layer) for layer in schedule)
    if len(selected) != NUM_LAYERS or any(len(layer) != TP_SIZE for layer in selected):
        raise ValueError("schedule must contain 32 layers by eight TP sources")
    if anchor_rank <= 0:
        raise ValueError("anchor rank must be positive")
    ideal_by_layer = [HEADS_PER_SOURCE * sum(layer) for layer in selected]
    padded_by_layer = [
        TP_SIZE * HEADS_PER_SOURCE * max(layer) for layer in selected
    ]
    uniform_width = NUM_HEADS * anchor_rank
    dense_width = NUM_HEADS * HEAD_DIM
    flat = [rank for layer in selected for rank in layer]
    return {
        "source_rank_sum": sum(flat),
        "total_value_head_rank": HEADS_PER_SOURCE * sum(flat),
        "source_rank_histogram": {
            str(rank): flat.count(rank) for rank in sorted(set(flat))
        },
        "anchor_rank": anchor_rank,
        "changed_sources_from_anchor": sum(rank != anchor_rank for rank in flat),
        "ideal_variable_allgather_width_by_layer": ideal_by_layer,
        "padded_rectangular_allgather_width_by_layer": padded_by_layer,
        "ideal_variable_allgather_total_width": sum(ideal_by_layer),
        "padded_rectangular_allgather_total_width": sum(padded_by_layer),
        "uniform_anchor_total_width": NUM_LAYERS * uniform_width,
        "dense_value_total_width": NUM_LAYERS * dense_width,
        "ideal_dense_reduction": (NUM_LAYERS * dense_width) / sum(ideal_by_layer),
        "padded_overhead_vs_uniform_anchor": (
            sum(padded_by_layer) / (NUM_LAYERS * uniform_width) - 1.0
        ),
        "layers_with_padded_overhead": sum(
            width > uniform_width for width in padded_by_layer
        ),
        **(
            {
                "changed_sources_from_uniform_64": sum(rank != 64 for rank in flat),
                "uniform_v64_total_width": NUM_LAYERS * uniform_width,
                "padded_overhead_vs_uniform_v64": (
                    sum(padded_by_layer) / (NUM_LAYERS * uniform_width) - 1.0
                ),
            }
            if anchor_rank == 64
            else {}
        ),
    }


def _allocate(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_ranks: Sequence[int],
    anchor_rank: int,
    cost_key: str,
) -> tuple[list[list[int]], float, list[dict[str, Any]]]:
    indexed = {
        (int(row["layer"]), int(row["source"]), int(row["candidate_rank"])): row
        for row in records
    }
    options = []
    coordinates = []
    for layer in range(NUM_LAYERS):
        for source in range(TP_SIZE):
            coordinate = []
            for rank in candidate_ranks:
                if rank == anchor_rank:
                    cost = 0.0
                else:
                    cost = float(
                        indexed[(layer, source, rank)]["terminal_kl_delta"][cost_key]
                    )
                coordinate.append(
                    MetricRankOption(
                        option_id=f"layer_{layer:03d}.source_{source}.r{rank}.{cost_key}",
                        source_family="exact_terminal_kl",
                        rank=rank,
                        scalar_cost=cost,
                        is_anchor=rank == anchor_rank,
                    )
                )
            options.append(tuple(coordinate))
            coordinates.append((layer, source))
    target_budget = NUM_LAYERS * TP_SIZE * anchor_rank
    allocation = allocate_metric_rank_exact(
        options,
        total_rank_budget=target_budget,
        anchor_rank=anchor_rank,
    )
    schedule = [[anchor_rank] * TP_SIZE for _ in range(NUM_LAYERS)]
    contributions = []
    for (layer, source), option in zip(
        coordinates,
        allocation.selected_options,
        strict=True,
    ):
        schedule[layer][source] = int(option.rank)
        contributions.append(
            {
                "layer": layer,
                "source": source,
                "rank": int(option.rank),
                "cost": float(option.scalar_cost),
            }
        )
    return schedule, float(allocation.total_cost), contributions


def _restore_folded(model: nn.Module, folded_modules: Sequence[nn.Module]) -> None:
    for layer, module in zip(_model_layers(model), folded_modules, strict=True):
        layer.self_attn = module


def _install_cached(
    model: nn.Module,
    *,
    folded_modules: Sequence[nn.Module],
    banks: Sequence[Any],
    cache: Sequence[CachedLayerFactors],
    runtime_source_group_size: int = HEADS_PER_SOURCE,
) -> None:
    if len(cache) != NUM_LAYERS:
        raise ValueError("factor cache does not cover every layer")
    for layer_index, factors in enumerate(cache):
        _model_layers(model)[layer_index].self_attn = _ragged_module(
            folded_modules[layer_index],
            dense_v_weight=banks[layer_index].dense_v,
            A=factors.A,
            D=factors.D,
            ranks=factors.ranks,
            source_group_size=runtime_source_group_size,
        )


def _build_schedule(
    model: nn.Module,
    *,
    schedule: Sequence[Sequence[int]],
    folded_modules: Sequence[nn.Module],
    banks: Sequence[Any],
    factor_dirs: Mapping[int, Path],
    factor_results: Mapping[int, Mapping[str, Any]],
    fit_covariances: Sequence[Tensor],
    head_groups_by_layer: Sequence[Sequence[Sequence[int]]],
    candidate_ranks: Sequence[int],
    anchor_rank: int,
    decoder_relative_jitter: float,
    runtime_source_group_size: int,
    device: torch.device,
) -> tuple[tuple[CachedLayerFactors, ...], list[dict[str, Any]]]:
    cache = []
    diagnostics = []
    mapping = torch.arange(NUM_HEADS, dtype=torch.long, device=device)
    maximum_rank = max(candidate_ranks)
    for layer_index, source_ranks in enumerate(schedule):
        source = _load_layer_factors(
            factor_dirs,
            factor_results,
            layer=layer_index,
            ranks=candidate_ranks,
        )
        initial_A, initial_D, head_ranks = _schedule_factors(
            source,
            source_ranks=source_ranks,
            head_groups=head_groups_by_layer[layer_index],
            maximum_rank=maximum_rank,
            device=device,
        )
        if all(rank == anchor_rank for rank in source_ranks):
            A, D = initial_A, initial_D
            diagnostic = {
                "layer": layer_index,
                "closure": "source_uniform_anchor",
            }
        else:
            covariance = fit_covariances[layer_index].to(device=device)
            objective = _objective(
                covariance,
                banks[layer_index].dense_o,
                layer=layer_index,
                device=device,
            )
            closure = close_ragged_decoder_with_fixed_encoders(
                objective=objective,
                initial_A=initial_A,
                initial_D=initial_D,
                head_to_kv_group=mapping,
                group_ranks=head_ranks,
                relative_jitter=decoder_relative_jitter,
            )
            A, D = closure.A_unique, closure.D_heads
            loss = evaluate_quadratic(objective, A, D, mapping)
            diagnostic = {
                "layer": layer_index,
                "closure": "full_layer_closed_form_decoder_refit",
                "fit_relative_mse": float(loss / objective.constant),
                "absolute_jitter": closure.decoder.absolute_jitters[0],
                "condition_estimate": closure.decoder.condition_estimates[0],
                "matrix_dimension": closure.decoder.matrix_dimensions[0],
                "relative_residual": closure.decoder.relative_residuals[0],
                "wall_time_seconds": closure.decoder.wall_times_seconds[0],
                "encoder_sha256": closure.encoder_sha256_after_solve,
            }
            del closure, objective, covariance
        cached = CachedLayerFactors(
            A=A.detach().to(device="cpu", dtype=torch.float16).contiguous(),
            D=D.detach().to(device="cpu", dtype=torch.float16).contiguous(),
            ranks=tuple(head_ranks),
        )
        cache.append(cached)
        diagnostics.append(diagnostic)
        _model_layers(model)[layer_index].self_attn = _ragged_module(
            folded_modules[layer_index],
            dense_v_weight=banks[layer_index].dense_v,
            A=cached.A,
            D=cached.D,
            ranks=cached.ranks,
            source_group_size=runtime_source_group_size,
        )
        _log(f"schedule closure layer {layer_index + 1}/{NUM_LAYERS}")
        del source, initial_A, initial_D, A, D
        torch.cuda.empty_cache()
    return tuple(cache), diagnostics


@torch.no_grad()
def _post_allocation_sweep_cache(
    *,
    initial_cache: Sequence[CachedLayerFactors],
    fit_covariances: Sequence[Tensor],
    banks: Sequence[Any],
    head_groups_by_layer: Sequence[Sequence[Sequence[int]]],
    validation_snapshot_dir: Path,
    validation_row_start: int,
    validation_rows: int,
    row_chunk_size: int,
    maximum_sweeps: int,
    minimum_sweeps: int,
    relative_objective_tolerance: float,
    patience: int,
    decoder_relative_jitter: float,
    encoder_relative_damping: float,
    cg_relative_tolerance: float,
    cg_iterations: int,
    maximum_backtracks: int,
    device: torch.device,
) -> tuple[tuple[CachedLayerFactors, ...], list[dict[str, Any]]]:
    """Fit owned layers in parallel, then broadcast the complete swept cache."""

    if not (
        len(initial_cache)
        == len(fit_covariances)
        == len(banks)
        == len(head_groups_by_layer)
        == NUM_LAYERS
    ):
        raise ValueError("post-allocation sweep inputs do not cover every layer")
    local: dict[int, tuple[CachedLayerFactors, dict[str, Any]]] = {}
    for layer_index in range(_rank(), NUM_LAYERS, _world_size()):
        started = time.perf_counter()
        initial = initial_cache[layer_index]
        fit_covariance = fit_covariances[layer_index].to(
            device=device,
            dtype=torch.float64,
        )
        validation_covariance = _snapshot_covariance_rows(
            snapshot_dir=validation_snapshot_dir,
            layer=layer_index,
            row_start=validation_row_start,
            rows=validation_rows,
            row_chunk_size=row_chunk_size,
            expected_o_weight=banks[layer_index].dense_o,
            device=device,
        )
        fit_objective = _objective(
            fit_covariance,
            banks[layer_index].dense_o,
            layer=layer_index,
            device=device,
        )
        validation_objective = _objective(
            validation_covariance,
            banks[layer_index].dense_o,
            layer=layer_index,
            device=device,
        )
        selected_A, selected_D, diagnostics = _fit_fixed_ragged_schedule(
            fit_objective=fit_objective,
            validation_objective=validation_objective,
            initial_A=initial.A.to(device=device, dtype=torch.float64),
            initial_D=initial.D.to(device=device, dtype=torch.float64),
            head_ranks=initial.ranks,
            head_groups=head_groups_by_layer[layer_index],
            maximum_sweeps=maximum_sweeps,
            minimum_sweeps=minimum_sweeps,
            relative_objective_tolerance=relative_objective_tolerance,
            patience=patience,
            decoder_relative_jitter=decoder_relative_jitter,
            encoder_relative_damping=encoder_relative_damping,
            cg_relative_tolerance=cg_relative_tolerance,
            cg_iterations=cg_iterations,
            maximum_backtracks=maximum_backtracks,
        )
        diagnostics = {
            "layer": layer_index,
            "closure": (
                "fixed_allocation_source_ordered_encoder_bcd_plus_"
                "full_layer_decoder_closure"
            ),
            "wall_time_seconds": time.perf_counter() - started,
            **diagnostics,
        }
        local[layer_index] = (
            CachedLayerFactors(
                A=selected_A.to(device="cpu", dtype=torch.float16).contiguous(),
                D=selected_D.to(device="cpu", dtype=torch.float16).contiguous(),
                ranks=initial.ranks,
            ),
            diagnostics,
        )
        _log(
            f"post-allocation sweep layer {layer_index + 1}/{NUM_LAYERS} "
            f"selected={diagnostics['selection']['boundary']}@"
            f"{diagnostics['selection']['sweep']}",
            all_ranks=True,
        )
        del (
            fit_covariance,
            validation_covariance,
            fit_objective,
            validation_objective,
            selected_A,
            selected_D,
        )
        torch.cuda.empty_cache()
    dist.barrier()

    complete_cache: list[CachedLayerFactors] = []
    complete_diagnostics: list[dict[str, Any]] = []
    for layer_index in range(NUM_LAYERS):
        owner = layer_index % _world_size()
        template = initial_cache[layer_index]
        if _rank() == owner:
            owned, diagnostic = local[layer_index]
            A = owned.A.to(device=device)
            D = owned.D.to(device=device)
            payload: list[Any] = [diagnostic]
        else:
            A = torch.empty_like(template.A, device=device)
            D = torch.empty_like(template.D, device=device)
            payload = [None]
        dist.broadcast(A, src=owner)
        dist.broadcast(D, src=owner)
        dist.broadcast_object_list(payload, src=owner)
        if not isinstance(payload[0], dict):
            raise RuntimeError("failed to broadcast post-allocation diagnostics")
        complete_cache.append(
            CachedLayerFactors(
                A=A.cpu().contiguous(),
                D=D.cpu().contiguous(),
                ranks=template.ranks,
            )
        )
        complete_diagnostics.append(payload[0])
        del A, D
    return tuple(complete_cache), complete_diagnostics


def _summary(result: Mapping[str, Any]) -> str:
    selected_name = result["selection"]["selected_candidate"]
    allocation_name = result["selection"].get(
        "global_kl_selected_candidate",
        selected_name,
    )
    lines = [
        "# Llama-2-7B C1 per-TP-source Global-KL allocation",
        "",
        "## Outcome",
        "",
        "| Schedule | Confirmation KL | WikiText-2 PPL | Changed sources | Padded overhead |",
        "|:---|---:|---:|---:|---:|",
    ]
    for label, row in result["schedules"].items():
        test = row.get("test")
        ppl = f"{test['ppl']:.9f}" if test is not None else "—"
        lines.append(
            f"| {label} | {row['confirmation']['terminal_kl']['mean']:.9g} | "
            f"{ppl} | {row['accounting']['changed_sources_from_anchor']} | "
            f"{100.0 * row['accounting']['padded_overhead_vs_uniform_anchor']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Selected: **{selected_name}**.",
            f"Global-KL allocation: **{allocation_name}**.",
            "",
            "## Selected TP-source ranks",
            "",
            "| Layer | Source ranks 0–7 |",
            "|---:|:---|",
        ]
    )
    schedule = result["selection"]["selected_schedule"]
    for layer, ranks in enumerate(schedule):
        lines.append(f"| {layer} | {ranks} |")
    post_allocation = result.get("post_allocation_refit")
    if post_allocation is not None:
        lines.extend(
            [
                "",
                "The allocation was frozen before refitting. Encoder blocks were "
                "visited in TP-source order, and the full-layer decoder was closed "
                "after each complete encoder sweep.",
            ]
        )
    lines.extend(
        [
            "",
            "The DP constrains ideal variable-size collective width. Padded rectangular "
            "AllGather cost is diagnostic only.",
            f"Head allocation: **{result['head_allocation']['method']}**.",
            "",
            "## Command",
            "",
            f"`{result['command']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch this allocator with torchrun")
    torch.set_num_threads(args.torch_num_threads)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    candidate_ranks = _parse_ranks(args.candidate_ranks)
    if args.anchor_rank not in candidate_ranks:
        raise ValueError("candidate ranks must include the anchor")
    supported_rank_grids = {
        (64, (48, 64, 80)),
        (64, (32, 48, 64, 80, 96)),
        (96, (64, 80, 96, 112)),
    }
    if (args.anchor_rank, candidate_ranks) not in supported_rank_grids:
        raise ValueError(
            "supported grids are anchor64 with 48/64/80 or 32/48/64/80/96, "
            "and anchor96 with 64/80/96/112"
        )
    head_groups_by_layer, head_allocation = _load_head_groups(args.head_groups)
    contiguous_layout = _is_contiguous_layout(head_groups_by_layer)
    runtime_source_group_size = HEADS_PER_SOURCE if contiguous_layout else 1
    positive = (
        args.profile_windows,
        args.confirmation_windows,
        args.sequence_length,
        args.batch_size,
        args.fit_windows,
        args.covariance_row_chunk_size,
        args.vocab_chunk_size,
        args.eval_seqlen,
        args.eval_max_chunks,
        args.torch_num_threads,
    )
    if min(positive) <= 0:
        raise ValueError("sample and compute arguments must be positive")
    if args.post_allocation_encoder_sweeps < 0:
        raise ValueError("post-allocation encoder sweeps must be non-negative")
    post_allocation_enabled = args.post_allocation_encoder_sweeps > 0
    if post_allocation_enabled:
        if not (
            0
            < args.post_allocation_minimum_encoder_sweeps
            <= args.post_allocation_encoder_sweeps
        ):
            raise ValueError("post-allocation minimum sweeps are invalid")
        if (
            args.post_allocation_encoder_relative_tolerance < 0
            or args.post_allocation_encoder_patience <= 0
            or args.post_allocation_encoder_relative_damping < 0
            or args.post_allocation_encoder_cg_relative_tolerance <= 0
            or args.post_allocation_encoder_cg_iterations <= 0
            or args.post_allocation_maximum_backtracks < 0
        ):
            raise ValueError("post-allocation encoder solver controls are invalid")

    output_dir = args.output_dir.expanduser().resolve()
    exists = torch.tensor(int(output_dir.exists()), dtype=torch.int32, device=device)
    _all_reduce_sum(exists)
    if int(exists.item()):
        raise FileExistsError(output_dir)

    model_path = Path(args.model).expanduser().resolve()
    factor_dirs = _parse_factor_dirs(args.factor_dir)
    if set(factor_dirs) != set(candidate_ranks):
        raise ValueError("factor directories must exactly match the candidate ranks")
    factor_results = _load_factor_results(
        factor_dirs,
        model_config_sha256=_sha256(model_path / "config.json"),
        layer_count=NUM_LAYERS,
    )
    zero_sweep_sources = None
    if post_allocation_enabled:
        zero_sweep_sources = _require_zero_sweep_factor_banks(factor_results)
        requested_snapshot = args.snapshot_dir.expanduser().resolve()
        recorded_snapshot = Path(
            str(zero_sweep_sources["snapshot_dir"])
        ).expanduser().resolve()
        if requested_snapshot != recorded_snapshot:
            raise ValueError(
                "post-allocation fit snapshot differs from the zero-sweep banks"
            )
        if int(zero_sweep_sources["fit_windows"]) != args.fit_windows:
            raise ValueError(
                "post-allocation fit windows differ from the zero-sweep banks"
            )
    profile_sequences, confirmation_sequences, windows_provenance = _select_windows(
        args.windows,
        profile_windows=args.profile_windows,
        confirmation_windows=args.confirmation_windows,
    )
    if args.sequence_length > profile_sequences.shape[1]:
        raise ValueError("requested sequence length exceeds stored C4 windows")
    profile_sequences = profile_sequences[:, : args.sequence_length].contiguous()
    confirmation_sequences = confirmation_sequences[:, : args.sequence_length].contiguous()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.config.use_cache = False
    if (
        model.config.model_type != "llama"
        or int(model.config.num_hidden_layers) != NUM_LAYERS
        or int(model.config.num_attention_heads) != NUM_HEADS
        or int(model.config.num_key_value_heads) != NUM_HEADS
        or int(model.config.hidden_size) != HIDDEN_SIZE
    ):
        raise ValueError("allocator requires Llama-2-7B MHA geometry")

    profile_teacher = _capture_teacher(
        model,
        profile_sequences,
        batch_size=args.batch_size,
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
        label="TP-source profile",
    )
    local_confirmation_sequences = confirmation_sequences[
        _rank() :: _world_size()
    ].contiguous()
    confirmation_teacher = _capture_teacher(
        model,
        local_confirmation_sequences,
        batch_size=args.batch_size,
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
        label="local TP-source confirmation",
    )
    banks = _install_banks(
        model,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        anchor_rank=args.anchor_rank,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )
    folded_modules = tuple(layer.self_attn for layer in _model_layers(model))
    folded_profile = _evaluate_teacher_metrics(
        model,
        profile_teacher,
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
    )

    fit_covariances = []
    damping_by_layer = []
    anchor_controls = []
    local_records = []
    mapping = torch.arange(NUM_HEADS, dtype=torch.long, device=device)
    maximum_rank = max(candidate_ranks)
    interventions = [
        (source, rank)
        for source in range(TP_SIZE)
        for rank in candidate_ranks
        if rank != args.anchor_rank
    ]
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    for layer_index in range(NUM_LAYERS):
        source = _load_layer_factors(
            factor_dirs,
            factor_results,
            layer=layer_index,
            ranks=candidate_ranks,
        )
        covariance = _broadcast_fit_covariance(
            snapshot_dir=snapshot_dir,
            layer=layer_index,
            fit_windows=args.fit_windows,
            row_chunk_size=args.covariance_row_chunk_size,
            expected_o_weight=banks[layer_index].dense_o,
            device=device,
        )
        covariance, absolute_damping = covariance_with_trace_damping(
            covariance,
            relative_damping=args.covariance_damping,
        )
        fit_covariances.append(covariance.detach().cpu().contiguous())
        damping_by_layer.append(absolute_damping)
        objective = _objective(
            covariance,
            banks[layer_index].dense_o,
            layer=layer_index,
            device=device,
        )

        anchor_A, anchor_D = source[args.anchor_rank]
        anchor_head_ranks = (args.anchor_rank,) * NUM_HEADS
        ragged_anchor = _ragged_module(
            folded_modules[layer_index],
            dense_v_weight=banks[layer_index].dense_v,
            A=anchor_A,
            D=anchor_D,
            ranks=anchor_head_ranks,
            source_group_size=runtime_source_group_size,
        )
        _model_layers(model)[layer_index].self_attn = ragged_anchor
        anchor_metrics = _evaluate_teacher_metrics(
            model,
            profile_teacher,
            device=device,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        if _rank() == 0:
            anchor_controls.append(
                {
                    "layer": layer_index,
                    "ragged_anchor": anchor_metrics,
                    "ragged_minus_folded": _paired_delta(
                        anchor_metrics,
                        folded_profile,
                    ),
                }
            )

        local_interventions = interventions[_rank() :: _world_size()]
        for local_index, (source_index, candidate_rank) in enumerate(
            local_interventions
        ):
            candidate_heads = head_groups_by_layer[layer_index][source_index]
            initial_A, initial_D, head_ranks = _padded_candidate_factors(
                source,
                candidate_heads=candidate_heads,
                candidate_rank=candidate_rank,
                anchor_rank=args.anchor_rank,
                maximum_rank=maximum_rank,
                device=device,
            )
            closure = close_ragged_decoder_with_fixed_encoders(
                objective=objective,
                initial_A=initial_A,
                initial_D=initial_D,
                head_to_kv_group=mapping,
                group_ranks=head_ranks,
                relative_jitter=args.decoder_relative_jitter,
            )
            fit_loss = evaluate_quadratic(
                objective,
                closure.A_unique,
                closure.D_heads,
                mapping,
            )
            candidate_module = _ragged_module(
                ragged_anchor,
                dense_v_weight=banks[layer_index].dense_v,
                A=closure.A_unique,
                D=closure.D_heads,
                ranks=head_ranks,
                source_group_size=runtime_source_group_size,
            )
            _model_layers(model)[layer_index].self_attn = candidate_module
            metrics = _evaluate_teacher_metrics(
                model,
                profile_teacher,
                device=device,
                vocab_chunk_size=args.vocab_chunk_size,
            )
            _model_layers(model)[layer_index].self_attn = ragged_anchor
            delta = _paired_delta(metrics, anchor_metrics)
            cost = _grouped_collective_cost(
                head_ranks,
                head_groups_by_layer[layer_index],
            )
            local_records.append(
                {
                    "layer": layer_index,
                    "source": source_index,
                    "heads": list(candidate_heads),
                    "anchor_rank": args.anchor_rank,
                    "candidate_rank": candidate_rank,
                    "terminal_kl": metrics["terminal_kl"],
                    "nll": metrics["nll"],
                    "terminal_kl_delta": delta["terminal_kl"],
                    "nll_delta": delta["nll"],
                    "fit_relative_mse": float(fit_loss / objective.constant),
                    "ideal_width_delta": (
                        cost["ideal_allgather_width"]
                        - NUM_HEADS * args.anchor_rank
                    ),
                    "padded_width_delta": (
                        cost["padded_allgather_width"]
                        - NUM_HEADS * args.anchor_rank
                    ),
                    "decoder_solve": {
                        "relative_residual": closure.decoder.relative_residuals[0],
                        "condition_estimate": closure.decoder.condition_estimates[0],
                        "matrix_dimension": closure.decoder.matrix_dimensions[0],
                        "wall_time_seconds": closure.decoder.wall_times_seconds[0],
                        "encoder_sha256": closure.encoder_sha256_after_solve,
                    },
                }
            )
            _log(
                f"profile layer={layer_index} candidate "
                f"{local_index + 1}/{len(local_interventions)} "
                f"source={source_index} rank={candidate_rank} "
                f"dKL={delta['terminal_kl']['mean']:.6g}",
                all_ranks=True,
            )
            del initial_A, initial_D, closure, candidate_module
            torch.cuda.empty_cache()
        _model_layers(model)[layer_index].self_attn = folded_modules[layer_index]
        del source, covariance, objective, ragged_anchor
        torch.cuda.empty_cache()
        _log(f"profile layer {layer_index + 1}/{NUM_LAYERS} complete")

    gathered: list[Any] | None = [None] * _world_size() if _rank() == 0 else None
    dist.gather_object(local_records, gathered, dst=0)
    payload: list[Any] = [None]
    if _rank() == 0:
        assert gathered is not None
        records = sorted(
            [record for shard in gathered for record in shard],
            key=lambda row: (row["layer"], row["source"], row["candidate_rank"]),
        )
        expected_records = NUM_LAYERS * TP_SIZE * (len(candidate_ranks) - 1)
        if len(records) != expected_records:
            raise RuntimeError("distributed profile lost candidate records")
        candidates = {}
        predicted = {}
        contributions = {}
        for label, cost_key in (
            ("mean_dp", "mean"),
            ("ucb_dp", "one_standard_error_ucb"),
        ):
            schedule, cost, rows = _allocate(
                records,
                candidate_ranks=candidate_ranks,
                anchor_rank=args.anchor_rank,
                cost_key=cost_key,
            )
            candidates[label] = schedule
            predicted[label] = cost
            contributions[label] = rows
        payload[0] = {
            "records": records,
            "candidates": candidates,
            "predicted": predicted,
            "contributions": contributions,
        }
    dist.broadcast_object_list(payload, src=0)
    profile_result = payload[0]
    if not isinstance(profile_result, dict):
        raise RuntimeError("failed to broadcast allocation profile")

    uniform_schedule = [[args.anchor_rank] * TP_SIZE for _ in range(NUM_LAYERS)]
    schedules = {"uniform_anchor": uniform_schedule, **profile_result["candidates"]}
    confirmation = {}
    caches = {}
    closure_diagnostics = {}
    for label, schedule in schedules.items():
        _restore_folded(model, folded_modules)
        cache, diagnostics = _build_schedule(
            model,
            schedule=schedule,
            folded_modules=folded_modules,
            banks=banks,
            factor_dirs=factor_dirs,
            factor_results=factor_results,
            fit_covariances=fit_covariances,
            head_groups_by_layer=head_groups_by_layer,
            candidate_ranks=candidate_ranks,
            anchor_rank=args.anchor_rank,
            decoder_relative_jitter=args.decoder_relative_jitter,
            runtime_source_group_size=runtime_source_group_size,
            device=device,
        )
        caches[label] = cache
        closure_diagnostics[label] = diagnostics
        confirmation[label] = _distributed_teacher_metrics(
            model,
            confirmation_teacher,
            device=device,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        _log(
            f"confirmation {label}: "
            f"KL={confirmation[label]['terminal_kl']['mean']:.9g}"
        )

    allocation_selected_name = min(
        schedules,
        key=lambda label: (confirmation[label]["terminal_kl"]["mean"], label),
    )
    selected_schedule = schedules[allocation_selected_name]
    deployment_name = allocation_selected_name
    post_allocation_refit = None
    if post_allocation_enabled:
        assert zero_sweep_sources is not None
        swept_cache, swept_diagnostics = _post_allocation_sweep_cache(
            initial_cache=caches[allocation_selected_name],
            fit_covariances=fit_covariances,
            banks=banks,
            head_groups_by_layer=head_groups_by_layer,
            validation_snapshot_dir=Path(
                str(zero_sweep_sources["validation_snapshot_dir"])
            ).expanduser().resolve(),
            validation_row_start=int(zero_sweep_sources["validation_row_start"]),
            validation_rows=int(zero_sweep_sources["validation_rows"]),
            row_chunk_size=args.covariance_row_chunk_size,
            maximum_sweeps=args.post_allocation_encoder_sweeps,
            minimum_sweeps=args.post_allocation_minimum_encoder_sweeps,
            relative_objective_tolerance=(
                args.post_allocation_encoder_relative_tolerance
            ),
            patience=args.post_allocation_encoder_patience,
            decoder_relative_jitter=args.decoder_relative_jitter,
            encoder_relative_damping=(
                args.post_allocation_encoder_relative_damping
            ),
            cg_relative_tolerance=(
                args.post_allocation_encoder_cg_relative_tolerance
            ),
            cg_iterations=args.post_allocation_encoder_cg_iterations,
            maximum_backtracks=args.post_allocation_maximum_backtracks,
            device=device,
        )
        deployment_name = f"{allocation_selected_name}_post_sweep"
        schedules[deployment_name] = selected_schedule
        caches[deployment_name] = swept_cache
        closure_diagnostics[deployment_name] = swept_diagnostics
        _restore_folded(model, folded_modules)
        _install_cached(
            model,
            folded_modules=folded_modules,
            banks=banks,
            cache=swept_cache,
            runtime_source_group_size=runtime_source_group_size,
        )
        confirmation[deployment_name] = _distributed_teacher_metrics(
            model,
            confirmation_teacher,
            device=device,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        _log(
            f"confirmation {deployment_name}: "
            f"KL={confirmation[deployment_name]['terminal_kl']['mean']:.9g}"
        )
        post_allocation_refit = {
            "enabled": True,
            "allocation_frozen": True,
            "allocation_candidate": allocation_selected_name,
            "deployment_candidate": deployment_name,
            "factor_bank_requirement": zero_sweep_sources,
            "objective": "full-layer attention-output MSE with cross-head covariance",
            "encoder_update_unit": (
                "head-private encoder blocks visited in recorded TP-source order"
            ),
            "decoder_closure": "full-layer after every complete encoder sweep",
            "maximum_encoder_sweeps": args.post_allocation_encoder_sweeps,
            "minimum_encoder_sweeps": (
                args.post_allocation_minimum_encoder_sweeps
            ),
            "encoder_relative_tolerance": (
                args.post_allocation_encoder_relative_tolerance
            ),
            "encoder_patience": args.post_allocation_encoder_patience,
            "encoder_relative_damping": (
                args.post_allocation_encoder_relative_damping
            ),
            "encoder_cg_relative_tolerance": (
                args.post_allocation_encoder_cg_relative_tolerance
            ),
            "encoder_cg_iterations": args.post_allocation_encoder_cg_iterations,
            "encoder_cg_fixed_iterations": True,
            "maximum_backtracks": args.post_allocation_maximum_backtracks,
            "heldout_selection": (
                "earliest minimum attention-output MSE among decoder-closed "
                "checkpoints"
            ),
            "layers": swept_diagnostics,
        }

    if _rank() == 0:
        test_text = _wikitext("test")
    else:
        test_text = ""
    texts = [test_text]
    dist.broadcast_object_list(texts, src=0)
    test_text = texts[0]
    test_metrics = {}
    for label in dict.fromkeys(
        ("uniform_anchor", allocation_selected_name, deployment_name)
    ):
        _restore_folded(model, folded_modules)
        _install_cached(
            model,
            folded_modules=folded_modules,
            banks=banks,
            cache=caches[label],
            runtime_source_group_size=runtime_source_group_size,
        )
        test_metrics[label] = distributed_official_ppl(
            model,
            tokenizer,
            text=test_text,
            seqlen=args.eval_seqlen,
            max_chunks=args.eval_max_chunks,
            device=device,
        )
        _log(f"test {label}: PPL={test_metrics[label]['ppl']:.9f}")

    if _rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
        factor_dir = output_dir / "selected_factors"
        factor_dir.mkdir()
        artifacts = {}
        for layer_index, factors in enumerate(caches[deployment_name]):
            path = factor_dir / f"layer_{layer_index:03d}.safetensors"
            save_file(
                {
                    "value_coordinate_encoders": factors.A,
                    "head_output_decoders": factors.D,
                    "head_ranks": torch.tensor(factors.ranks, dtype=torch.int32),
                },
                str(path),
            )
            artifacts[str(layer_index)] = {
                "file": str(path.relative_to(output_dir)),
                "sha256": _sha256(path),
                "encoder_sha256": tensor_sha256(factors.A),
                "decoder_sha256": tensor_sha256(factors.D),
            }
        schedule_rows = {}
        for label, schedule in schedules.items():
            row = {
                "schedule": schedule,
                "accounting": _schedule_accounting(
                    schedule,
                    anchor_rank=args.anchor_rank,
                ),
                "confirmation": confirmation[label],
                "closure_diagnostics": closure_diagnostics[label],
            }
            if label in test_metrics:
                row["test"] = test_metrics[label]
            schedule_rows[label] = row
        result = {
            "format": FORMAT,
            "status": "complete",
            "command": shlex.join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "world_size": _world_size(),
            "model": str(model_path),
            "model_config_sha256": _sha256(model_path / "config.json"),
            "geometry": {
                "layers": NUM_LAYERS,
                "attention_heads": NUM_HEADS,
                "tp_sources": TP_SIZE,
                "heads_per_source": HEADS_PER_SOURCE,
                "head_dim": HEAD_DIM,
            },
            "head_allocation": {
                **head_allocation,
                "contiguous_layout": contiguous_layout,
                "groups_by_layer": [
                    [list(group) for group in layer_groups]
                    for layer_groups in head_groups_by_layer
                ],
                "quality_reference_runtime_source_group_size": (
                    runtime_source_group_size
                ),
                "deployment_interpretation": (
                    "permute Q/K/V head blocks and matching output-decoder blocks "
                    "into the recorded per-layer TP ownership order"
                ),
            },
            "profile": {
                "dataset": "c4_select",
                "windows": args.profile_windows,
                "sequence_length": args.sequence_length,
                "batch_size": args.batch_size,
                "folded_anchor": folded_profile,
                "per_layer_ragged_anchor_controls": anchor_controls,
                "records": profile_result["records"],
                "absolute_covariance_damping_by_layer": damping_by_layer,
            },
            "confirmation": {
                "dataset": "c4_select",
                "windows": args.confirmation_windows,
                "sequence_length": args.sequence_length,
                "batch_size": args.batch_size,
                "disjoint_from_profile": True,
                "windows_provenance": windows_provenance,
            },
            "selection": {
                "constraint": "exact ideal variable-size collective source-rank budget",
                "target_source_rank_sum": NUM_LAYERS * TP_SIZE * args.anchor_rank,
                "predicted_additive_costs": profile_result["predicted"],
                "contributions": profile_result["contributions"],
                "global_kl_selected_candidate": allocation_selected_name,
                "selected_candidate": deployment_name,
                "selected_schedule": selected_schedule,
                "selected_accounting": _schedule_accounting(
                    selected_schedule,
                    anchor_rank=args.anchor_rank,
                ),
                "uniform_is_eligible": True,
                "selection_metric": "lowest disjoint-confirmation mean terminal KL",
            },
            "schedules": schedule_rows,
            "selected_artifacts": artifacts,
            "post_allocation_refit": post_allocation_refit,
            "factor_sources": {
                str(rank): {
                    "path": str(factor_dirs[rank]),
                    "results_sha256": _sha256(factor_dirs[rank] / "results.json"),
                }
                for rank in candidate_ranks
            },
            "test_protocol": {
                "dataset": "wikitext2_test",
                "seqlen": args.eval_seqlen,
                "max_chunks": args.eval_max_chunks,
                "schedule_frozen_before_test": True,
            },
            "numerics": {
                "model_and_runtime_factors": "float16",
                "decoder_closure": "float64",
                "terminal_kl_probability": "float32",
                "terminal_kl_accumulation": "float64",
                "ragged_runtime": (
                    "logical-head quality reference; noncontiguous CKA ownership uses "
                    "source_group_size=1 while preserving the exact attention function"
                ),
            },
            "environment": {
                "python_executable": sys.executable,
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(device),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            },
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "summary.md").write_text(
            _summary(result),
            encoding="utf-8",
        )
        _log(f"wrote {output_dir}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
