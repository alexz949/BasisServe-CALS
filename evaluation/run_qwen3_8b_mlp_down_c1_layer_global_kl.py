#!/usr/bin/env python3
"""Allocate Qwen3-8B MLP AASVD ranks by per-layer terminal-logits KL.

The rank candidates are prefixes of one maximum-rank independent AASVD bank.
Every profile intervention changes one complete MLP ``down_proj`` layer while
the other 35 layers remain at the uniform anchor rank.  Exact-budget dynamic
programming then selects one rank per layer at the same average communication
width as the uniform anchor.  A disjoint C4 split confirms the complete ragged
schedules before compatible factor directories are exported.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.metric_rank_allocation import (  # noqa: E402
    MetricRankOption,
    allocate_metric_rank_exact,
)
from basisserve.core.mlp_down_c1 import (  # noqa: E402
    FACTOR_FORMAT,
    MLPDownC1Linear,
    load_mlp_down_c1_manifest,
)
from evaluation import allocate_qwen3_32b_c1_tp_source_global_kl as common  # noqa: E402
from evaluation.capture_attention_o_proj_ppl_snapshots import (  # noqa: E402
    _git_commit,
    _sha256,
)


PROFILE_FORMAT = "basisserve.qwen3_8b.mlp_down_c1.layer_global_kl_profile.v1"
RESULT_FORMAT = "basisserve.qwen3_8b.mlp_down_c1.layer_global_kl_allocation.v1"
NUM_LAYERS = 36
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12288
TP_SIZE = 4
ANCHOR_RANK = 2560
CANDIDATE_RANKS = (1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584)
FRESH_WINDOW_START = 320


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--anchor-factor-dir", type=Path, required=True)
    parser.add_argument("--max-factor-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-ranks", default=",".join(map(str, CANDIDATE_RANKS))
    )
    parser.add_argument("--anchor-rank", type=int, default=ANCHOR_RANK)
    parser.add_argument("--window-start", type=int, default=FRESH_WINDOW_START)
    parser.add_argument("--profile-windows", type=int, default=8)
    parser.add_argument("--confirmation-windows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    profile = subparsers.add_parser("profile")
    _add_common_args(profile)
    profile.add_argument("--profile-shard-index", type=int, required=True)
    profile.add_argument("--profile-shard-count", type=int, default=4)
    finalize = subparsers.add_parser("finalize")
    _add_common_args(finalize)
    finalize.add_argument("--profile-shard-count", type=int, default=4)
    finalize.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _parse_ranks(raw: str, *, anchor_rank: int) -> tuple[int, ...]:
    ranks = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
    if ranks != tuple(sorted(set(ranks))):
        raise ValueError("candidate ranks must be distinct and increasing")
    if anchor_rank not in ranks:
        raise ValueError("candidate ranks must include the anchor rank")
    if any(rank <= 0 or rank > HIDDEN_SIZE or rank % 256 for rank in ranks):
        raise ValueError(
            "candidate ranks must be positive multiples of 256 not exceeding 4096"
        )
    if not min(ranks) < anchor_rank < max(ranks):
        raise ValueError("candidate grid must span both sides of the anchor")
    return ranks


def _assigned_layers(index: int, count: int) -> tuple[int, ...]:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("profile shard index/count are invalid")
    layers = tuple(range(index, NUM_LAYERS, count))
    if not layers:
        raise ValueError("profile shard has no assigned layers")
    return layers


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _validate_factor_manifests(
    model_path: Path,
    anchor_dir: Path,
    maximum_dir: Path,
    *,
    anchor_rank: int,
    candidate_ranks: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = model_path / "config.json"
    anchor = load_mlp_down_c1_manifest(anchor_dir, model_config_path=config_path)
    maximum = load_mlp_down_c1_manifest(maximum_dir, model_config_path=config_path)
    for label, manifest, expected_rank in (
        ("anchor", anchor, anchor_rank),
        ("maximum", maximum, max(candidate_ranks)),
    ):
        geometry = manifest.get("model", {})
        observed = (
            geometry.get("model_type"),
            int(geometry.get("num_hidden_layers", -1)),
            int(geometry.get("hidden_size", -1)),
            int(geometry.get("intermediate_size", -1)),
        )
        if observed != ("qwen3", NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE):
            raise ValueError(f"{label} factor bank has unexpected geometry: {observed}")
        config = manifest.get("fit_config", {})
        if (
            config.get("algorithm") != "teacher_output_pca_exact_shared_decoder"
            or config.get("objective")
            != "post_swiglu_complete_mlp_output_mse_after_tp_sum"
            or int(config.get("rank", -1)) != expected_rank
            or int(config.get("tp_size", -1)) != TP_SIZE
            or float(config.get("damping", math.nan)) != 0.0
            or int(config.get("als_sweeps", -1)) != 0
        ):
            raise ValueError(f"{label} factor bank is not the required independent AASVD")
        if set(map(int, manifest.get("artifacts", {}))) != set(range(NUM_LAYERS)):
            raise ValueError(f"{label} factor bank does not cover all layers")
    if anchor.get("covariance", {}).get("manifest_sha256") != maximum.get(
        "covariance", {}
    ).get("manifest_sha256"):
        raise ValueError("anchor and maximum AASVD banks use different covariances")
    return anchor, maximum


def _load_layer_factors(
    root: Path,
    manifest: Mapping[str, Any],
    layer: int,
    *,
    selected_rank: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    artifact = manifest["artifacts"][str(layer)]
    path = root / artifact["file"]
    if _sha256(path) != artifact["sha256"]:
        raise ValueError(f"factor hash mismatch at layer {layer}: {path}")
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {"input_factor", "output_basis"}:
        raise ValueError(f"unexpected factor tensors in {path}")
    input_factor = tensors["input_factor"]
    output_basis = tensors["output_basis"]
    source_rank = int(output_basis.shape[1])
    if tuple(input_factor.shape) != (INTERMEDIATE_SIZE, source_rank) or tuple(
        output_basis.shape
    ) != (HIDDEN_SIZE, source_rank):
        raise ValueError(f"malformed factor geometry in {path}")
    if not 0 < selected_rank <= source_rank:
        raise ValueError(f"rank-{source_rank} source cannot provide rank {selected_rank}")
    return (
        input_factor[:, :selected_rank].contiguous(),
        output_basis[:, :selected_rank].contiguous(),
        {
            "path": str(path),
            "sha256": artifact["sha256"],
            "source_rank": source_rank,
            "selected_rank": selected_rank,
        },
    )


def _new_module(
    input_factor: Tensor,
    output_basis: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    bias: Tensor | None,
) -> MLPDownC1Linear:
    selected_bias = None if bias is None else bias.detach().to(device=device, dtype=dtype)
    module = MLPDownC1Linear(
        input_factor.to(device=device, dtype=dtype),
        output_basis.to(device=device, dtype=dtype),
        bias=selected_bias,
    )
    return module.eval()


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) != NUM_LAYERS:
        raise TypeError("could not locate the 36 Qwen3 decoder layers")
    return layers


def _model_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    weight = model.get_input_embeddings().weight
    return weight.device, weight.dtype


def _install_schedule(
    model: nn.Module,
    *,
    schedule: Sequence[int],
    anchor_dir: Path,
    maximum_dir: Path,
    anchor_manifest: Mapping[str, Any],
    maximum_manifest: Mapping[str, Any],
    anchor_rank: int,
) -> None:
    if len(schedule) != NUM_LAYERS:
        raise ValueError("rank schedule must cover every MLP layer")
    layers = _decoder_layers(model)
    device, dtype = _model_device_dtype(model)
    for layer, rank in enumerate(map(int, schedule)):
        if rank == anchor_rank:
            root, manifest = anchor_dir, anchor_manifest
        else:
            root, manifest = maximum_dir, maximum_manifest
        input_factor, output_basis, _ = _load_layer_factors(
            root, manifest, layer, selected_rank=rank
        )
        current = layers[layer].mlp.down_proj
        bias = getattr(current, "bias", None)
        layers[layer].mlp.down_proj = _new_module(
            input_factor,
            output_basis,
            device=device,
            dtype=dtype,
            bias=bias,
        )
        del current, input_factor, output_basis
    gc.collect()
    torch.cuda.empty_cache()


def _load_model(args: argparse.Namespace, model_path: Path) -> nn.Module:
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        device_map={"": args.device},
    ).eval()
    model.config.use_cache = False
    observed = (
        str(model.config.model_type),
        int(model.config.num_hidden_layers),
        int(model.config.hidden_size),
        int(model.config.intermediate_size),
    )
    if observed != ("qwen3", NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE):
        raise ValueError(f"unexpected model geometry: {observed}")
    return model


def _shared_configuration(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path,
    Path,
    tuple[int, ...],
    dict[str, Any],
    dict[str, Any],
    Tensor,
    Tensor,
    dict[str, Any],
    dict[str, Any],
]:
    positive = (
        args.profile_windows,
        args.confirmation_windows,
        args.sequence_length,
        args.batch_size,
        args.vocab_chunk_size,
        args.torch_num_threads,
    )
    if min(positive) <= 0:
        raise ValueError("sample and compute arguments must be positive")
    if args.batch_size < max(args.profile_windows, args.confirmation_windows):
        raise ValueError("batch size must cover each Global-KL split in one forward")
    if args.window_start < FRESH_WINDOW_START:
        raise ValueError("Global-KL windows overlap AASVD fit or heldout documents")
    ranks = _parse_ranks(args.candidate_ranks, anchor_rank=args.anchor_rank)
    model_path = args.model_path.expanduser().resolve()
    anchor_dir = args.anchor_factor_dir.expanduser().resolve()
    maximum_dir = args.max_factor_dir.expanduser().resolve()
    anchor_manifest, maximum_manifest = _validate_factor_manifests(
        model_path,
        anchor_dir,
        maximum_dir,
        anchor_rank=args.anchor_rank,
        candidate_ranks=ranks,
    )
    profile, confirmation, windows = common._select_fresh_windows(
        args.windows,
        window_start=args.window_start,
        profile_windows=args.profile_windows,
        confirmation_windows=args.confirmation_windows,
        sequence_length=args.sequence_length,
    )
    configuration = {
        "model_path": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "anchor_factor_dir": str(anchor_dir),
        "anchor_factor_manifest_sha256": _sha256(anchor_dir / "manifest.json"),
        "maximum_factor_dir": str(maximum_dir),
        "maximum_factor_manifest_sha256": _sha256(maximum_dir / "manifest.json"),
        "candidate_ranks": list(ranks),
        "anchor_rank": int(args.anchor_rank),
        "target_layer_rank_sum": NUM_LAYERS * int(args.anchor_rank),
        "windows": windows,
        "profile_windows": int(args.profile_windows),
        "confirmation_windows": int(args.confirmation_windows),
        "sequence_length": int(args.sequence_length),
        "batch_size": int(args.batch_size),
        "vocab_chunk_size": int(args.vocab_chunk_size),
        "model_dtype": args.model_dtype,
        "attn_implementation": args.attn_implementation,
        "factor_protocol": "independent_aasvd_nested_prefixes",
        "allocation_unit": "complete_mlp_down_proj_layer",
    }
    return (
        model_path,
        anchor_dir,
        maximum_dir,
        ranks,
        anchor_manifest,
        maximum_manifest,
        profile,
        confirmation,
        windows,
        configuration,
    )


def _profile_payload(
    *,
    status: str,
    args: argparse.Namespace,
    configuration: Mapping[str, Any],
    assigned_layers: Sequence[int],
    completed_layers: Sequence[int],
    anchor_metrics: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    started: float,
) -> dict[str, Any]:
    return {
        "format": PROFILE_FORMAT,
        "status": status,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile_shard_index": int(args.profile_shard_index),
        "profile_shard_count": int(args.profile_shard_count),
        "assigned_layers": list(map(int, assigned_layers)),
        "completed_layers": list(map(int, completed_layers)),
        "configuration": dict(configuration),
        "uniform_anchor": dict(anchor_metrics),
        "records": list(records),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
    }


@torch.inference_mode()
def _profile(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("MLP layer Global-KL profiling requires CUDA")
    assigned = _assigned_layers(args.profile_shard_index, args.profile_shard_count)
    (
        model_path,
        anchor_dir,
        maximum_dir,
        ranks,
        anchor_manifest,
        maximum_manifest,
        profile_sequences,
        _,
        _,
        configuration,
    ) = _shared_configuration(args)
    profile_dir = args.profile_dir.expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    output_path = profile_dir / f"shard_{args.profile_shard_index:02d}.json"
    records: list[dict[str, Any]] = []
    completed: list[int] = []
    if output_path.is_file():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if prior.get("format") != PROFILE_FORMAT:
            raise ValueError(f"incompatible profile checkpoint: {output_path}")
        if prior.get("configuration") != configuration:
            raise ValueError(f"profile configuration changed: {output_path}")
        if prior.get("assigned_layers") != list(assigned):
            raise ValueError(f"profile layer assignment changed: {output_path}")
        if prior.get("status") == "complete":
            print(f"[MLP layer Global-KL] already complete: {output_path}", flush=True)
            return
        records = [dict(row) for row in prior.get("records", ())]
        completed = list(map(int, prior.get("completed_layers", ())))
        expected = len(completed) * (len(ranks) - 1)
        if len(records) != expected or any(layer not in assigned for layer in completed):
            raise ValueError(f"profile checkpoint is internally incomplete: {output_path}")

    started = time.perf_counter()
    model = _load_model(args, model_path)
    teacher = common._capture_teacher(
        model,
        profile_sequences,
        batch_size=args.batch_size,
        vocab_chunk_size=args.vocab_chunk_size,
        label=f"mlp_profile_shard_{args.profile_shard_index}",
    )
    _install_schedule(
        model,
        schedule=[args.anchor_rank] * NUM_LAYERS,
        anchor_dir=anchor_dir,
        maximum_dir=maximum_dir,
        anchor_manifest=anchor_manifest,
        maximum_manifest=maximum_manifest,
        anchor_rank=args.anchor_rank,
    )
    anchor_metrics = common._evaluate_teacher_metrics(
        model, teacher, vocab_chunk_size=args.vocab_chunk_size
    )
    layers = _decoder_layers(model)
    device, dtype = _model_device_dtype(model)
    intervention_ranks = tuple(rank for rank in ranks if rank != args.anchor_rank)
    for layer in assigned:
        if layer in completed:
            continue
        anchor_module = layers[layer].mlp.down_proj
        bias = getattr(anchor_module, "bias", None)
        maximum_input, maximum_output, source = _load_layer_factors(
            maximum_dir,
            maximum_manifest,
            layer,
            selected_rank=max(ranks),
        )
        for rank in intervention_ranks:
            candidate = _new_module(
                maximum_input[:, :rank].contiguous(),
                maximum_output[:, :rank].contiguous(),
                device=device,
                dtype=dtype,
                bias=bias,
            )
            layers[layer].mlp.down_proj = candidate
            metrics = common._evaluate_teacher_metrics(
                model, teacher, vocab_chunk_size=args.vocab_chunk_size
            )
            layers[layer].mlp.down_proj = anchor_module
            delta = common._paired_delta(metrics, anchor_metrics)
            records.append(
                {
                    "layer": layer,
                    "anchor_rank": int(args.anchor_rank),
                    "candidate_rank": rank,
                    "terminal_kl": metrics["terminal_kl"],
                    "nll": metrics["nll"],
                    "terminal_kl_delta": delta["terminal_kl"],
                    "nll_delta": delta["nll"],
                    "factor_source": source,
                }
            )
            print(
                f"[MLP layer Global-KL shard {args.profile_shard_index}] "
                f"layer={layer} rank={rank} "
                f"dKL={delta['terminal_kl']['mean']:.8g} "
                f"ucb={delta['terminal_kl']['one_standard_error_ucb']:.8g}",
                flush=True,
            )
            del candidate, metrics, delta
            torch.cuda.empty_cache()
        completed.append(layer)
        payload = _profile_payload(
            status="in_progress",
            args=args,
            configuration=configuration,
            assigned_layers=assigned,
            completed_layers=completed,
            anchor_metrics=anchor_metrics,
            records=records,
            started=started,
        )
        _atomic_json(output_path, payload)
        del maximum_input, maximum_output
    payload = _profile_payload(
        status="complete",
        args=args,
        configuration=configuration,
        assigned_layers=assigned,
        completed_layers=completed,
        anchor_metrics=anchor_metrics,
        records=records,
        started=started,
    )
    _atomic_json(output_path, payload)
    print(f"[MLP layer Global-KL] saved {output_path}", flush=True)


def _merge_profiles(
    profile_dir: Path,
    *,
    shard_count: int,
    ranks: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    shards = []
    configuration: dict[str, Any] | None = None
    for index in range(shard_count):
        path = profile_dir / f"shard_{index:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != PROFILE_FORMAT or payload.get("status") != "complete":
            raise ValueError(f"incomplete profile shard: {path}")
        expected_layers = list(_assigned_layers(index, shard_count))
        if payload.get("assigned_layers") != expected_layers or payload.get(
            "completed_layers"
        ) != expected_layers:
            raise ValueError(f"profile shard does not cover its assigned layers: {path}")
        if configuration is None:
            configuration = dict(payload["configuration"])
        elif payload.get("configuration") != configuration:
            raise ValueError("profile shard configurations differ")
        shards.append(payload)
    records = sorted(
        [dict(row) for shard in shards for row in shard["records"]],
        key=lambda row: (int(row["layer"]), int(row["candidate_rank"])),
    )
    expected = NUM_LAYERS * (len(ranks) - 1)
    keys = {(int(row["layer"]), int(row["candidate_rank"])) for row in records}
    if len(records) != expected or len(keys) != expected:
        raise ValueError("profile shards do not contain every layer/rank intervention")
    assert configuration is not None
    return records, shards, configuration


def _allocate(
    records: Sequence[Mapping[str, Any]],
    *,
    ranks: Sequence[int],
    anchor_rank: int,
    cost_key: str,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    indexed = {
        (int(row["layer"]), int(row["candidate_rank"])): row for row in records
    }
    options = []
    for layer in range(NUM_LAYERS):
        layer_options = []
        for rank in ranks:
            cost = (
                0.0
                if rank == anchor_rank
                else float(indexed[(layer, rank)]["terminal_kl_delta"][cost_key])
            )
            layer_options.append(
                MetricRankOption(
                    option_id=f"layer_{layer:02d}.r{rank}.{cost_key}",
                    source_family="mlp_aasvd_layer_terminal_kl",
                    rank=rank,
                    scalar_cost=cost,
                    is_anchor=rank == anchor_rank,
                )
            )
        options.append(tuple(layer_options))
    allocation = allocate_metric_rank_exact(
        options,
        total_rank_budget=NUM_LAYERS * anchor_rank,
        anchor_rank=anchor_rank,
    )
    schedule = tuple(int(option.rank) for option in allocation.selected_options)
    return schedule, {
        "cost_key": cost_key,
        "predicted_additive_kl_delta": float(allocation.total_cost),
        "changed_layers": int(allocation.changed_coordinates),
        "total_absolute_rank_deviation": int(
            allocation.total_absolute_rank_deviation
        ),
        "contributions": [
            {
                "layer": layer,
                "rank": int(option.rank),
                "predicted_kl_delta": float(option.scalar_cost),
            }
            for layer, option in enumerate(allocation.selected_options)
        ],
    }


def _accounting(schedule: Sequence[int], *, anchor_rank: int) -> dict[str, Any]:
    layer_ranks = tuple(map(int, schedule))
    total = sum(layer_ranks)
    if len(layer_ranks) != NUM_LAYERS or total != NUM_LAYERS * anchor_rank:
        raise ValueError("rank schedule violates the exact average-rank budget")
    histogram = {
        str(rank): layer_ranks.count(rank) for rank in sorted(set(layer_ranks))
    }
    return {
        "layer_ranks": list(layer_ranks),
        "rank_histogram": histogram,
        "layer_rank_sum": total,
        "average_rank": total / NUM_LAYERS,
        "communication_fraction_of_dense_allreduce": total
        / (NUM_LAYERS * HIDDEN_SIZE),
        "communication_reduction_fraction": 1.0
        - total / (NUM_LAYERS * HIDDEN_SIZE),
        "tp_size": TP_SIZE,
    }


def _paired_delta_summary(
    candidate: Mapping[str, Any], anchor: Mapping[str, Any]
) -> dict[str, Any]:
    result = common._paired_delta(candidate, anchor)
    for metric in ("terminal_kl", "nll"):
        values = result[metric]["values"]
        result[metric]["improved_windows"] = sum(value < 0 for value in values)
        result[metric]["worsened_windows"] = sum(value > 0 for value in values)
    return result


def _materialize_factor_dir(
    output_dir: Path,
    *,
    schedule: Sequence[int],
    label: str,
    source_command: str,
    anchor_dir: Path,
    maximum_dir: Path,
    anchor_manifest: Mapping[str, Any],
    maximum_manifest: Mapping[str, Any],
    anchor_rank: int,
    result_path: Path,
) -> None:
    output_dir.mkdir(parents=True)
    artifacts: dict[str, Any] = {}
    for layer, rank in enumerate(map(int, schedule)):
        if rank == anchor_rank:
            root, manifest, selection = anchor_dir, anchor_manifest, "exact_anchor"
        else:
            root, manifest, selection = maximum_dir, maximum_manifest, "nested_prefix"
        input_factor, output_basis, source = _load_layer_factors(
            root, manifest, layer, selected_rank=rank
        )
        path = output_dir / f"layer_{layer:03d}.safetensors"
        save_file(
            {"input_factor": input_factor, "output_basis": output_basis}, str(path)
        )
        artifacts[str(layer)] = {
            "file": path.name,
            "sha256": _sha256(path),
            "input_factor_shape": list(input_factor.shape),
            "output_basis_shape": list(output_basis.shape),
            "factor_dtype": str(input_factor.dtype),
            "metrics": {
                "algorithm": "teacher_output_pca_exact_shared_decoder",
                "objective": "post_swiglu_complete_mlp_output_mse_after_tp_sum",
                "collective": "latent_allreduce",
                "rank": rank,
                "selection": selection,
                "source": source,
            },
        }
    accounting = _accounting(schedule, anchor_rank=anchor_rank)
    manifest = {
        "format": FACTOR_FORMAT,
        "schema_version": 1,
        "status": "complete",
        "command": source_command,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": dict(anchor_manifest["model"]),
        "covariance": dict(anchor_manifest["covariance"]),
        "fit_config": {
            "rank": None,
            "rank_schedule": list(map(int, schedule)),
            "tp_size": TP_SIZE,
            "factor_dtype": anchor_manifest["fit_config"]["factor_dtype"],
            "objective": "post_swiglu_complete_mlp_output_mse_after_tp_sum",
            "algorithm": "teacher_output_pca_exact_shared_decoder",
            "allocation": "per_layer_terminal_global_kl",
            "allocation_label": label,
            "allocation_result": str(result_path),
            "damping": 0.0,
            "als_sweeps": 0,
        },
        "communication": {
            "collective": "latent_allreduce",
            "dense_width": HIDDEN_SIZE,
            "layer_latent_widths": list(map(int, schedule)),
            **accounting,
        },
        "layers": list(range(NUM_LAYERS)),
        "artifacts": artifacts,
    }
    _atomic_json(output_dir / "manifest.json", manifest)


def _summary(result: Mapping[str, Any]) -> str:
    selections = result["selection"]
    confirmation = result["confirmation"]
    lines = [
        "# Qwen3-8B MLP AASVD per-layer Global-KL",
        "",
        f"- Average rank: `{result['accounting']['average_rank']}`",
        f"- Communication reduction: `{100 * result['accounting']['communication_reduction_fraction']:.2f}%`",
        f"- Candidate ranks: `{result['configuration']['candidate_ranks']}`",
        f"- Profile: `{result['configuration']['profile_windows']} x {result['configuration']['sequence_length']}` tokens",
        f"- Confirmation: `{result['configuration']['confirmation_windows']} x {result['configuration']['sequence_length']}` tokens; not used for DP selection.",
        "",
        "## Independent confirmation",
        "",
        "| Schedule | KL | PPL | delta-KL vs uniform | delta-NLL vs uniform |",
        "|:---|---:|---:|---:|---:|",
    ]
    uniform = confirmation["uniform_r2560"]
    lines.append(
        f"| uniform-r2560 | {uniform['terminal_kl']['mean']:.8g} | "
        f"{math.exp(uniform['nll']['mean']):.8g} | 0 | 0 |"
    )
    for name in ("mean_dp", "ucb_dp"):
        metrics = confirmation[name]
        delta = confirmation[f"{name}_minus_uniform"]
        lines.append(
            f"| {name} | {metrics['terminal_kl']['mean']:.8g} | "
            f"{math.exp(metrics['nll']['mean']):.8g} | "
            f"{delta['terminal_kl']['mean']:.8g} | {delta['nll']['mean']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Layer schedules",
            "",
            "| Layer | Mean-DP rank | UCB-DP rank |",
            "|---:|---:|---:|",
        ]
    )
    mean_schedule = selections["mean_dp"]["schedule"]
    ucb_schedule = selections["ucb_dp"]["schedule"]
    for layer in range(NUM_LAYERS):
        lines.append(f"| {layer} | {mean_schedule[layer]} | {ucb_schedule[layer]} |")
    lines.extend(["", "## Command", "", f"`{result['command']}`", ""])
    return "\n".join(lines)


@torch.inference_mode()
def _finalize(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("MLP layer Global-KL finalization requires CUDA")
    (
        model_path,
        anchor_dir,
        maximum_dir,
        ranks,
        anchor_manifest,
        maximum_manifest,
        _,
        confirmation_sequences,
        windows_provenance,
        configuration,
    ) = _shared_configuration(args)
    profile_dir = args.profile_dir.expanduser().resolve()
    records, shards, profile_configuration = _merge_profiles(
        profile_dir, shard_count=args.profile_shard_count, ranks=ranks
    )
    if profile_configuration != configuration:
        raise ValueError("finalize configuration differs from profile shards")
    mean_schedule, mean_selection = _allocate(
        records,
        ranks=ranks,
        anchor_rank=args.anchor_rank,
        cost_key="mean",
    )
    ucb_schedule, ucb_selection = _allocate(
        records,
        ranks=ranks,
        anchor_rank=args.anchor_rank,
        cost_key="one_standard_error_ucb",
    )
    output_dir = args.output_dir.expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    started = time.perf_counter()
    model = _load_model(args, model_path)
    teacher = common._capture_teacher(
        model,
        confirmation_sequences,
        batch_size=args.batch_size,
        vocab_chunk_size=args.vocab_chunk_size,
        label="mlp_confirmation",
    )
    confirmation: dict[str, Any] = {}
    for label, schedule in (
        ("uniform_r2560", (args.anchor_rank,) * NUM_LAYERS),
        ("mean_dp", mean_schedule),
        ("ucb_dp", ucb_schedule),
    ):
        _install_schedule(
            model,
            schedule=schedule,
            anchor_dir=anchor_dir,
            maximum_dir=maximum_dir,
            anchor_manifest=anchor_manifest,
            maximum_manifest=maximum_manifest,
            anchor_rank=args.anchor_rank,
        )
        confirmation[label] = common._evaluate_teacher_metrics(
            model, teacher, vocab_chunk_size=args.vocab_chunk_size
        )
        print(
            f"[MLP layer Global-KL] confirmation={label} "
            f"KL={confirmation[label]['terminal_kl']['mean']:.8g} "
            f"PPL={math.exp(confirmation[label]['nll']['mean']):.8g}",
            flush=True,
        )
    confirmation["mean_dp_minus_uniform"] = _paired_delta_summary(
        confirmation["mean_dp"], confirmation["uniform_r2560"]
    )
    confirmation["ucb_dp_minus_uniform"] = _paired_delta_summary(
        confirmation["ucb_dp"], confirmation["uniform_r2560"]
    )

    partial_dir.mkdir(parents=True)
    final_result_path = output_dir / "result.json"
    for label, schedule in (("mean_dp", mean_schedule), ("ucb_dp", ucb_schedule)):
        _materialize_factor_dir(
            partial_dir / f"{label}_factors",
            schedule=schedule,
            label=label,
            source_command=shlex.join(sys.argv),
            anchor_dir=anchor_dir,
            maximum_dir=maximum_dir,
            anchor_manifest=anchor_manifest,
            maximum_manifest=maximum_manifest,
            anchor_rank=args.anchor_rank,
            result_path=final_result_path,
        )
    accounting = _accounting(mean_schedule, anchor_rank=args.anchor_rank)
    result = {
        "format": RESULT_FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "configuration": configuration,
        "profile": {
            "records": records,
            "shards": [
                {
                    "index": int(shard["profile_shard_index"]),
                    "assigned_layers": shard["assigned_layers"],
                    "elapsed_seconds": float(shard["elapsed_seconds"]),
                }
                for shard in shards
            ],
        },
        "selection": {
            "objective": "minimum additive per-layer terminal KL delta",
            "constraint": f"exact sum of {NUM_LAYERS} layer ranks",
            "mean_dp": {"schedule": list(mean_schedule), **mean_selection},
            "ucb_dp": {"schedule": list(ucb_schedule), **ucb_selection},
        },
        "accounting": accounting,
        "confirmation": {
            "not_used_for_schedule_selection": True,
            "window_indices": windows_provenance["confirmation_indices"],
            **confirmation,
        },
        "selected_factor_dirs": {
            "mean_dp": str(output_dir / "mean_dp_factors"),
            "ucb_dp": str(output_dir / "ucb_dp_factors"),
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
    }
    _atomic_json(partial_dir / "result.json", result)
    (partial_dir / "summary.md").write_text(_summary(result), encoding="utf-8")
    os.replace(partial_dir, output_dir)
    print(f"[MLP layer Global-KL] wrote {output_dir}", flush=True)


def main() -> None:
    common.activate_model_profile("qwen3_8b")
    args = parse_args()
    if args.stage == "profile":
        _profile(args)
    else:
        _finalize(args)


if __name__ == "__main__":
    main()
