#!/usr/bin/env python3
"""Allocate Qwen3-32B C1 ranks with whole-layer Global-KL interventions.

Each intervention assigns one candidate rank to all eight physical KV sources
in a decoder layer.  This measures the real layer-level interaction at the
terminal logits instead of adding eight independently measured source costs.
The exact-budget DP selects one static rank per layer, so every TP rank has an
equal-size collective message within a layer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.decoder_closed_rank_candidates import tensor_sha256  # noqa: E402
from basisserve.core.metric_rank_allocation import (  # noqa: E402
    MetricRankOption,
    allocate_metric_rank_exact,
)
from evaluation import allocate_qwen3_32b_c1_tp_source_global_kl as common  # noqa: E402
from evaluation import run_qwen3_32b_c1_tp_source_global_kl_sharded as runtime  # noqa: E402


FORMAT = "basisserve.qwen3_32b.gqa_c1.layer_global_kl_allocation.v1"
PROFILE_FORMAT = "basisserve.qwen3_32b.gqa_c1.layer_global_kl_profile_shard.v1"
MODEL_LABEL = "Qwen3-32B"


def activate_model_profile(name: str) -> None:
    global FORMAT, PROFILE_FORMAT, MODEL_LABEL
    runtime.activate_model_profile(name)
    if name == "qwen3_32b":
        slug = "qwen3_32b"
        MODEL_LABEL = "Qwen3-32B"
    elif name == "qwen3_8b":
        slug = "qwen3_8b"
        MODEL_LABEL = "Qwen3-8B-Base"
    else:
        raise ValueError(f"unknown Qwen3 C1 model profile: {name}")
    FORMAT = f"basisserve.{slug}.gqa_c1.layer_global_kl_allocation.v1"
    PROFILE_FORMAT = (
        f"basisserve.{slug}.gqa_c1.layer_global_kl_profile_shard.v1"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    profile = subparsers.add_parser("profile")
    runtime._add_shared_args(profile)
    profile.add_argument("--profile-shard-index", type=int, required=True)
    profile.add_argument("--profile-shard-count", type=int, default=2)

    finalize = subparsers.add_parser("finalize")
    runtime._add_shared_args(finalize)
    finalize.add_argument("--profile-shard-count", type=int, default=2)
    finalize.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _log(message: str, *, shard: int | None = None) -> None:
    prefix = "[Qwen3 layer Global-KL]"
    if shard is not None:
        prefix = f"[Qwen3 layer Global-KL shard {shard}]"
    print(f"{prefix} {message}", flush=True)


def _configuration(
    args: argparse.Namespace,
    *,
    model_path: Path,
    snapshot_dir: Path,
    candidate_ranks: Sequence[int],
    factor_dirs: Mapping[int, Path],
    factor_results: Mapping[int, Mapping[str, Any]],
    windows_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    configuration = runtime._configuration(
        args,
        model_path=model_path,
        snapshot_dir=snapshot_dir,
        candidate_ranks=candidate_ranks,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        windows_provenance=windows_provenance,
    )
    configuration.update(
        {
            "allocation_unit": "decoder_layer",
            "intervention": "all_eight_physical_kv_sources_at_one_rank",
            "rank_128_endpoint": "exact_identity_dense_o_without_decoder_closure",
        }
    )
    return configuration


def _profile_payload(
    *,
    status: str,
    args: argparse.Namespace,
    configuration: Mapping[str, Any],
    assigned_layers: Sequence[int],
    completed_layers: Sequence[int],
    anchor_metrics: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    damping_by_layer: Mapping[int, float],
    started: float,
) -> dict[str, Any]:
    return {
        "format": PROFILE_FORMAT,
        "status": status,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile_shard_index": args.profile_shard_index,
        "profile_shard_count": args.profile_shard_count,
        "assigned_layers": list(map(int, assigned_layers)),
        "completed_layers": list(map(int, completed_layers)),
        "configuration": dict(configuration),
        "uniform_anchor": dict(anchor_metrics),
        "records": list(records),
        "absolute_covariance_damping_by_layer": {
            str(layer): float(value) for layer, value in damping_by_layer.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "visible_cuda_devices": len(runtime._cuda_device_indices()),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in runtime._cuda_device_indices()
            ],
        },
    }


@torch.inference_mode()
def _profile(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    assigned_layers = runtime._shard_layers(
        args.profile_shard_index,
        args.profile_shard_count,
    )
    (
        model_path,
        snapshot_dir,
        candidate_ranks,
        factor_dirs,
        factor_results,
        profile_sequences,
        _,
        windows_provenance,
    ) = runtime._validate_shared(args)
    configuration = _configuration(
        args,
        model_path=model_path,
        snapshot_dir=snapshot_dir,
        candidate_ranks=candidate_ranks,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        windows_provenance=windows_provenance,
    )
    profile_dir = args.profile_dir.expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    shard_path = profile_dir / f"shard_{args.profile_shard_index:02d}.json"
    records: list[dict[str, Any]] = []
    completed_layers: list[int] = []
    damping_by_layer: dict[int, float] = {}
    prior_anchor: dict[str, Any] | None = None
    if shard_path.is_file():
        prior = json.loads(shard_path.read_text(encoding="utf-8"))
        if prior.get("format") != PROFILE_FORMAT:
            raise ValueError(f"incompatible profile checkpoint: {shard_path}")
        if prior.get("configuration") != configuration:
            raise ValueError(f"profile checkpoint configuration changed: {shard_path}")
        if tuple(map(int, prior.get("assigned_layers", ()))) != assigned_layers:
            raise ValueError(f"profile checkpoint layer assignment changed: {shard_path}")
        if prior.get("status") == "complete":
            _log(f"completed checkpoint already exists: {shard_path}", shard=args.profile_shard_index)
            return
        records = [dict(row) for row in prior.get("records", ())]
        completed_layers = list(map(int, prior.get("completed_layers", ())))
        damping_by_layer = {
            int(layer): float(value)
            for layer, value in prior.get(
                "absolute_covariance_damping_by_layer", {}
            ).items()
        }
        prior_anchor = dict(prior["uniform_anchor"])
    if any(layer not in assigned_layers for layer in completed_layers):
        raise ValueError("profile checkpoint completed an unassigned layer")
    expected_prior_records = len(completed_layers) * (len(candidate_ranks) - 1)
    if len(records) != expected_prior_records:
        raise ValueError("profile checkpoint record count is incomplete")

    started = time.perf_counter()
    model = runtime._load_model(args, model_path)
    teacher = common._capture_teacher(
        model,
        profile_sequences,
        batch_size=args.batch_size,
        vocab_chunk_size=args.vocab_chunk_size,
        label=f"layer profile shard {args.profile_shard_index}",
    )
    snapshot_cache, _ = common._load_snapshot_cache(
        snapshot_dir,
        model_path=model_path,
    )
    factor_cache = common._load_factor_cache(factor_dirs, factor_results)
    layers = common._decoder_layers(model)
    dense_v_weights = tuple(
        layer.self_attn.v_proj.weight.detach().cpu().clone() for layer in layers
    )
    uniform_schedule = [
        [args.anchor_rank] * common.NUM_KV_HEADS
        for _ in range(common.NUM_LAYERS)
    ]
    common._install_schedule(
        model,
        schedule=uniform_schedule,
        dense_v_weights=dense_v_weights,
        factor_cache=factor_cache,
        snapshot_cache=snapshot_cache,
        anchor_rank=args.anchor_rank,
        covariance_damping=args.covariance_damping,
        decoder_relative_jitter=args.decoder_relative_jitter,
        keep_factors=False,
    )
    anchor_metrics = common._evaluate_teacher_metrics(
        model,
        teacher,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    if prior_anchor is not None:
        previous = prior_anchor["terminal_kl"]["values"]
        current = anchor_metrics["terminal_kl"]["values"]
        if len(previous) != len(current) or max(
            abs(float(left) - float(right))
            for left, right in zip(previous, current, strict=True)
        ) > 1.0e-7:
            raise RuntimeError("resumed profile anchor changed numerically")
    _log(
        f"anchor KL={anchor_metrics['terminal_kl']['mean']:.9g}",
        shard=args.profile_shard_index,
    )

    intervention_ranks = [
        rank for rank in candidate_ranks if rank != args.anchor_rank
    ]
    completed = set(completed_layers)
    for layer_index in assigned_layers:
        if layer_index in completed:
            _log(f"resume skips layer {layer_index}", shard=args.profile_shard_index)
            continue
        layer = layers[layer_index]
        device = layer.self_attn.v_proj.weight.device
        anchor_v = layer.self_attn.v_proj.weight.detach().clone()
        anchor_o = layer.self_attn.o_proj.weight.detach().clone()
        objective, absolute_damping = common._objective(
            snapshot_cache[layer_index],
            layer=layer_index,
            device=device,
            covariance_damping=args.covariance_damping,
        )
        layer_records = []
        for intervention_index, candidate_rank in enumerate(intervention_ranks):
            source_ranks = [candidate_rank] * common.NUM_KV_HEADS
            factors, closure = common._closed_factors(
                bank=factor_cache[layer_index],
                snapshot=snapshot_cache[layer_index],
                objective=objective,
                source_ranks=source_ranks,
                anchor_rank=args.anchor_rank,
                decoder_relative_jitter=args.decoder_relative_jitter,
                device=device,
            )
            common._install_factors(
                layer,
                dense_v_weight=dense_v_weights[layer_index],
                factors=factors,
            )
            metrics = common._evaluate_teacher_metrics(
                model,
                teacher,
                vocab_chunk_size=args.vocab_chunk_size,
            )
            layer.self_attn.v_proj.weight.copy_(anchor_v)
            layer.self_attn.o_proj.weight.copy_(anchor_o)
            delta = common._paired_delta(metrics, anchor_metrics)
            layer_records.append(
                {
                    "layer": layer_index,
                    "anchor_rank": args.anchor_rank,
                    "candidate_rank": candidate_rank,
                    "physical_kv_sources_changed": common.NUM_KV_HEADS,
                    "terminal_kl": metrics["terminal_kl"],
                    "nll": metrics["nll"],
                    "terminal_kl_delta": delta["terminal_kl"],
                    "nll_delta": delta["nll"],
                    "layer_rank_delta": candidate_rank - args.anchor_rank,
                    "collective_width": common.NUM_KV_HEADS * candidate_rank,
                    "decoder_solve": closure,
                }
            )
            _log(
                f"layer={layer_index} candidate={intervention_index + 1}/"
                f"{len(intervention_ranks)} rank={candidate_rank} "
                f"dKL={delta['terminal_kl']['mean']:.6g}",
                shard=args.profile_shard_index,
            )
            del factors, metrics
            torch.cuda.empty_cache()
        records.extend(layer_records)
        completed_layers.append(layer_index)
        damping_by_layer[layer_index] = absolute_damping
        common._atomic_json(
            shard_path,
            _profile_payload(
                status="running",
                args=args,
                configuration=configuration,
                assigned_layers=assigned_layers,
                completed_layers=completed_layers,
                anchor_metrics=anchor_metrics,
                records=records,
                damping_by_layer=damping_by_layer,
                started=started,
            ),
        )
        del objective, anchor_v, anchor_o, layer_records
        torch.cuda.empty_cache()
        _log(
            f"checkpointed {len(completed_layers)}/{len(assigned_layers)} layers",
            shard=args.profile_shard_index,
        )

    common._atomic_json(
        shard_path,
        _profile_payload(
            status="complete",
            args=args,
            configuration=configuration,
            assigned_layers=assigned_layers,
            completed_layers=completed_layers,
            anchor_metrics=anchor_metrics,
            records=records,
            damping_by_layer=damping_by_layer,
            started=started,
        ),
    )
    _log(f"completed {shard_path}", shard=args.profile_shard_index)


def _merge_profile_shards(
    profile_dir: Path,
    *,
    shard_count: int,
    configuration: Mapping[str, Any],
    candidate_ranks: Sequence[int],
) -> dict[str, Any]:
    shards = []
    for shard_index in range(shard_count):
        path = profile_dir / f"shard_{shard_index:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != PROFILE_FORMAT or payload.get("status") != "complete":
            raise ValueError(f"profile shard is incomplete: {path}")
        if int(payload.get("profile_shard_index", -1)) != shard_index:
            raise ValueError(f"profile shard index mismatch: {path}")
        if int(payload.get("profile_shard_count", -1)) != shard_count:
            raise ValueError(f"profile shard count mismatch: {path}")
        if payload.get("configuration") != configuration:
            raise ValueError(f"profile shard configuration changed: {path}")
        expected_layers = runtime._shard_layers(shard_index, shard_count)
        if tuple(map(int, payload.get("assigned_layers", ()))) != expected_layers:
            raise ValueError(f"profile shard assignment mismatch: {path}")
        if tuple(sorted(map(int, payload.get("completed_layers", ())))) != expected_layers:
            raise ValueError(f"profile shard missed assigned layers: {path}")
        shards.append((path, payload))
    records = sorted(
        [dict(row) for _, shard in shards for row in shard["records"]],
        key=lambda row: (row["layer"], row["candidate_rank"]),
    )
    expected_records = common.NUM_LAYERS * (len(candidate_ranks) - 1)
    if len(records) != expected_records:
        raise ValueError("merged profile does not contain every layer intervention")
    keys = {
        (int(row["layer"]), int(row["candidate_rank"])) for row in records
    }
    if len(keys) != expected_records:
        raise ValueError("merged profile contains duplicate layer interventions")
    damping = {}
    for _, shard in shards:
        for layer, value in shard["absolute_covariance_damping_by_layer"].items():
            layer_index = int(layer)
            if layer_index in damping:
                raise ValueError("profile shards duplicate covariance damping")
            damping[layer_index] = float(value)
    if set(damping) != set(range(common.NUM_LAYERS)):
        raise ValueError("profile shards lack per-layer covariance damping")
    return {
        "records": records,
        "absolute_covariance_damping_by_layer": [
            damping[layer] for layer in range(common.NUM_LAYERS)
        ],
        "uniform_anchor_by_shard": [
            shard["uniform_anchor"] for _, shard in shards
        ],
        "shards": [
            {
                "path": str(path),
                "sha256": common._sha256(path),
                "assigned_layers": shard["assigned_layers"],
                "elapsed_seconds": shard["elapsed_seconds"],
            }
            for path, shard in shards
        ],
    }


def _allocate_layer_ranks(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_ranks: Sequence[int],
    anchor_rank: int,
    cost_key: str,
) -> tuple[list[list[int]], float, list[dict[str, Any]]]:
    indexed = {
        (int(row["layer"]), int(row["candidate_rank"])): row
        for row in records
    }
    options = []
    for layer in range(common.NUM_LAYERS):
        coordinate = []
        for rank in candidate_ranks:
            cost = (
                0.0
                if rank == anchor_rank
                else float(indexed[(layer, rank)]["terminal_kl_delta"][cost_key])
            )
            coordinate.append(
                MetricRankOption(
                    option_id=f"layer_{layer:03d}.r{rank}.{cost_key}",
                    source_family="whole_layer_terminal_kl",
                    rank=rank,
                    scalar_cost=cost,
                    is_anchor=rank == anchor_rank,
                )
            )
        options.append(tuple(coordinate))
    allocation = allocate_metric_rank_exact(
        options,
        total_rank_budget=common.NUM_LAYERS * anchor_rank,
        anchor_rank=anchor_rank,
    )
    schedule = []
    contributions = []
    for layer, option in enumerate(allocation.selected_options):
        rank = int(option.rank)
        schedule.append([rank] * common.NUM_KV_HEADS)
        contributions.append(
            {
                "layer": layer,
                "rank": rank,
                "cost": float(option.scalar_cost),
            }
        )
    return schedule, float(allocation.total_cost), contributions


def _layer_schedule_accounting(
    schedule: Sequence[Sequence[int]],
    *,
    anchor_rank: int,
) -> dict[str, Any]:
    selected = tuple(tuple(map(int, layer)) for layer in schedule)
    if len(selected) != common.NUM_LAYERS or any(
        len(layer) != common.NUM_KV_HEADS for layer in selected
    ):
        raise ValueError(
            f"schedule must contain {common.NUM_LAYERS} layers by "
            f"{common.NUM_KV_HEADS} TP sources"
        )
    if any(len(set(layer)) != 1 for layer in selected):
        raise ValueError("per-layer schedule must use one rank for all TP sources")
    layer_ranks = [layer[0] for layer in selected]
    source_rank_sum = common.NUM_KV_HEADS * sum(layer_ranks)
    return {
        "layer_ranks": layer_ranks,
        "layer_rank_sum": sum(layer_ranks),
        "layer_rank_histogram": {
            str(rank): layer_ranks.count(rank) for rank in sorted(set(layer_ranks))
        },
        "source_rank_sum": source_rank_sum,
        "source_rank_histogram": {
            str(rank): common.NUM_KV_HEADS * layer_ranks.count(rank)
            for rank in sorted(set(layer_ranks))
        },
        "anchor_rank": anchor_rank,
        "changed_layers_from_anchor": sum(rank != anchor_rank for rank in layer_ranks),
        "changed_sources_from_anchor": common.NUM_KV_HEADS
        * sum(rank != anchor_rank for rank in layer_ranks),
        "rectangular_collective_width_by_layer": [
            common.NUM_KV_HEADS * rank for rank in layer_ranks
        ],
        "rectangular_collective_total_width": source_rank_sum,
        "uniform_anchor_total_width": (
            common.NUM_LAYERS * common.NUM_KV_HEADS * anchor_rank
        ),
        "dense_value_total_width": (
            common.NUM_LAYERS * common.NUM_KV_HEADS * common.HEAD_DIM
        ),
        "dense_reduction": (
            common.NUM_LAYERS * common.NUM_KV_HEADS * common.HEAD_DIM
        )
        / source_rank_sum,
        "ragged_padding_overhead": 0.0,
    }


def _summary(result: Mapping[str, Any]) -> str:
    lines = [
        f"# {MODEL_LABEL} C1 per-layer Global-KL allocation",
        "",
        "## Outcome",
        "",
        "| Schedule | Confirmation KL | Changed layers | Ragged padding |",
        "|:---|---:|---:|---:|",
    ]
    for label, row in result["schedules"].items():
        lines.append(
            f"| {label} | {row['confirmation']['terminal_kl']['mean']:.9g} | "
            f"{row['accounting']['changed_layers_from_anchor']} | 0.00% |"
        )
    lines.extend(
        [
            "",
            f"Selected: **{result['selection']['selected_candidate']}**.",
            "",
            "## Selected layer ranks",
            "",
            "| Layer | Rank for all eight physical KV sources | Collective width |",
            "|---:|---:|---:|",
        ]
    )
    accounting = result["selection"]["selected_accounting"]
    for layer, rank in enumerate(accounting["layer_ranks"]):
        lines.append(f"| {layer} | {rank} | {common.NUM_KV_HEADS * rank} |")
    lines.extend(
        [
            "",
            "Every layer uses an equal message width on all eight TP ranks; no ragged "
            "collective or padding is required.",
            "",
            "## Command",
            "",
            f"`{result['command']}`",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def _finalize(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    started = time.perf_counter()
    (
        model_path,
        snapshot_dir,
        candidate_ranks,
        factor_dirs,
        factor_results,
        _,
        confirmation_sequences,
        windows_provenance,
    ) = runtime._validate_shared(args)
    configuration = _configuration(
        args,
        model_path=model_path,
        snapshot_dir=snapshot_dir,
        candidate_ranks=candidate_ranks,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        windows_provenance=windows_provenance,
    )
    profile_dir = args.profile_dir.expanduser().resolve()
    merged = _merge_profile_shards(
        profile_dir,
        shard_count=args.profile_shard_count,
        configuration=configuration,
        candidate_ranks=candidate_ranks,
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    model = runtime._load_model(args, model_path)
    teacher = common._capture_teacher(
        model,
        confirmation_sequences,
        batch_size=args.batch_size,
        vocab_chunk_size=args.vocab_chunk_size,
        label="layer confirmation",
    )
    snapshot_cache, snapshot_manifest = common._load_snapshot_cache(
        snapshot_dir,
        model_path=model_path,
    )
    factor_cache = common._load_factor_cache(factor_dirs, factor_results)
    layers = common._decoder_layers(model)
    dense_v_weights = tuple(
        layer.self_attn.v_proj.weight.detach().cpu().clone() for layer in layers
    )
    uniform_schedule = [
        [args.anchor_rank] * common.NUM_KV_HEADS
        for _ in range(common.NUM_LAYERS)
    ]
    _, anchor_install = common._install_schedule(
        model,
        schedule=uniform_schedule,
        dense_v_weights=dense_v_weights,
        factor_cache=factor_cache,
        snapshot_cache=snapshot_cache,
        anchor_rank=args.anchor_rank,
        covariance_damping=args.covariance_damping,
        decoder_relative_jitter=args.decoder_relative_jitter,
        keep_factors=False,
    )

    candidates = {}
    predicted = {}
    contributions = {}
    for label, cost_key in (
        ("mean_dp", "mean"),
        ("ucb_dp", "one_standard_error_ucb"),
    ):
        schedule, cost, rows = _allocate_layer_ranks(
            merged["records"],
            candidate_ranks=candidate_ranks,
            anchor_rank=args.anchor_rank,
            cost_key=cost_key,
        )
        candidates[label] = schedule
        predicted[label] = cost
        contributions[label] = rows

    schedules = {"uniform_anchor": uniform_schedule, **candidates}
    confirmation = {}
    closure_diagnostics = {"uniform_anchor": anchor_install}
    for label, schedule in schedules.items():
        if label != "uniform_anchor":
            _, diagnostics = common._install_schedule(
                model,
                schedule=schedule,
                dense_v_weights=dense_v_weights,
                factor_cache=factor_cache,
                snapshot_cache=snapshot_cache,
                anchor_rank=args.anchor_rank,
                covariance_damping=args.covariance_damping,
                decoder_relative_jitter=args.decoder_relative_jitter,
                keep_factors=False,
            )
            closure_diagnostics[label] = diagnostics
        confirmation[label] = common._evaluate_teacher_metrics(
            model,
            teacher,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        _log(
            f"confirmation {label}: "
            f"KL={confirmation[label]['terminal_kl']['mean']:.9g}"
        )

    selected_name = min(
        schedules,
        key=lambda label: (confirmation[label]["terminal_kl"]["mean"], label),
    )
    selected_schedule = schedules[selected_name]
    selected_factors, selected_install = common._install_schedule(
        model,
        schedule=selected_schedule,
        dense_v_weights=dense_v_weights,
        factor_cache=factor_cache,
        snapshot_cache=snapshot_cache,
        anchor_rank=args.anchor_rank,
        covariance_damping=args.covariance_damping,
        decoder_relative_jitter=args.decoder_relative_jitter,
        keep_factors=True,
    )
    assert selected_factors is not None
    closure_diagnostics[selected_name] = selected_install

    output_dir.mkdir(parents=True, exist_ok=False)
    selected_dir = output_dir / "selected_factors"
    selected_dir.mkdir()
    artifacts = {}
    for layer_index, factors in enumerate(selected_factors):
        path = selected_dir / f"layer_{layer_index:03d}.safetensors"
        save_file(
            {
                "value_coordinate_encoders": factors.A,
                "head_output_decoders": factors.D,
                "source_ranks": torch.tensor(factors.ranks, dtype=torch.int32),
            },
            str(path),
        )
        artifacts[str(layer_index)] = {
            "file": str(path.relative_to(output_dir)),
            "sha256": common._sha256(path),
            "encoder_sha256": tensor_sha256(factors.A),
            "decoder_sha256": tensor_sha256(factors.D),
        }

    schedule_rows = {
        label: {
            "schedule": schedule,
            "accounting": _layer_schedule_accounting(
                schedule,
                anchor_rank=args.anchor_rank,
            ),
            "confirmation": confirmation[label],
            "closure_diagnostics": closure_diagnostics[label],
        }
        for label, schedule in schedules.items()
    }
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    selected_accounting = _layer_schedule_accounting(
        selected_schedule,
        anchor_rank=args.anchor_rank,
    )
    result = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model": str(model_path),
        "model_config_sha256": common._sha256(model_path / "config.json"),
        "geometry": {
            "layers": common.NUM_LAYERS,
            "query_heads": common.NUM_QUERY_HEADS,
            "physical_kv_heads": common.NUM_KV_HEADS,
            "tp_sources": common.NUM_KV_HEADS,
            "query_heads_per_source": common.HEADS_PER_SOURCE,
            "head_dim": common.HEAD_DIM,
        },
        "profile": {
            "execution": "independent sharded model replicas",
            "intervention_unit": "whole decoder layer",
            "sources_changed_per_intervention": common.NUM_KV_HEADS,
            "factor_stage": configuration["factor_stage"],
            "profile_shard_count": args.profile_shard_count,
            "dataset": "c4_train_fresh_documents",
            "windows": args.profile_windows,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "forward_batches_per_candidate": math.ceil(
                args.profile_windows / args.batch_size
            ),
            "uniform_anchor_by_shard": merged["uniform_anchor_by_shard"],
            "records": merged["records"],
            "absolute_covariance_damping_by_layer": merged[
                "absolute_covariance_damping_by_layer"
            ],
            "shards": merged["shards"],
        },
        "confirmation": {
            "dataset": "c4_train_fresh_documents",
            "windows": args.confirmation_windows,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "forward_batches_per_schedule": math.ceil(
                args.confirmation_windows / args.batch_size
            ),
            "disjoint_from_profile": True,
            "disjoint_from_als_fit_and_heldout": True,
            "windows_provenance": windows_provenance,
        },
        "selection": {
            "constraint": "exact per-layer rank budget with one rank per layer",
            "target_layer_rank_sum": common.NUM_LAYERS * args.anchor_rank,
            "target_source_rank_sum": (
                common.NUM_LAYERS * common.NUM_KV_HEADS * args.anchor_rank
            ),
            "candidate_ranks": list(candidate_ranks),
            "rank_128_endpoint": "exact identity Value encoder plus dense O initialization",
            "predicted_additive_costs": predicted,
            "contributions": contributions,
            "selected_candidate": selected_name,
            "selected_schedule": selected_schedule,
            "selected_accounting": selected_accounting,
            "uniform_is_eligible": True,
            "selection_metric": "lowest disjoint-confirmation mean terminal KL",
        },
        "schedules": schedule_rows,
        "selected_artifacts": artifacts,
        "factor_sources": {
            str(rank): {
                "path": str(factor_dirs[rank]),
                "results_sha256": common._sha256(factor_dirs[rank] / "results.json"),
            }
            for rank in factor_dirs
        },
        "snapshot": {
            "path": str(snapshot_dir),
            "manifest_sha256": common._sha256(snapshot_dir / "manifest.json"),
            "fit_windows": snapshot_manifest["calibration"]["fit_windows"],
            "fit_rows": snapshot_manifest["calibration"]["fit_rows"],
            "heldout_not_used_for_global_kl_selection": True,
        },
        "numerics": {
            "model_dtype": str(dtype),
            "deployed_factor_dtype": "torch.bfloat16",
            "decoder_closure": "float32",
            "terminal_kl_probability": "float32",
            "terminal_kl_accumulation": "float64",
            "collective_shape": "static equal rank across all eight TP sources per layer",
            "ragged_collective_required": False,
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in runtime._cuda_device_indices()
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in runtime._cuda_device_indices()
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    common._atomic_json(output_dir / "result.json", result)
    (output_dir / "summary.md").write_text(_summary(result), encoding="utf-8")
    _log(f"selected {selected_name}; wrote {output_dir}")


def main() -> None:
    args = parse_args()
    if args.stage == "profile":
        _profile(args)
    else:
        _finalize(args)


if __name__ == "__main__":
    main()
