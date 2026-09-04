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
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.decoder_closed_rank_candidates import tensor_sha256  # noqa: E402
from basisserve.core.qwen_suffix_replay import (  # noqa: E402
    capture_qwen_anchor_replay,
    replay_qwen_suffix,
)
from basisserve.core.sampled_terminal_kl import (  # noqa: E402
    TeacherTerminalStatistics,
    gather_prediction_hidden,
    nested_prediction_positions,
    paired_terminal_kl_delta,
    parse_position_counts,
    selected_token_nll,
    streaming_output_logsumexp,
    teacher_terminal_statistics,
    terminal_kl_from_statistics,
)
from basisserve.core.metric_rank_allocation import (  # noqa: E402
    MetricRankOption,
    allocate_metric_rank_exact,
)
from evaluation import allocate_qwen3_32b_c1_tp_source_global_kl as common  # noqa: E402
from evaluation import run_qwen3_32b_c1_tp_source_global_kl_sharded as runtime  # noqa: E402
from evaluation.build_qwen3_8b_c1_two_sided_factorized_kl_schedule import (  # noqa: E402
    allocate_layer_schedule as allocate_factorized_layer_schedule,
    local_error_curves_from_factor_results,
    predict_factorized_costs,
    predict_two_sided_factorized_costs,
)


FORMAT = "basisserve.qwen3_32b.gqa_c1.layer_global_kl_allocation.v1"
PROFILE_FORMAT = "basisserve.qwen3_32b.gqa_c1.layer_global_kl_profile_shard.v1"
MODEL_LABEL = "Qwen3-32B"


def activate_model_profile(name: str) -> None:
    global FORMAT, PROFILE_FORMAT, MODEL_LABEL
    runtime.activate_model_profile(name)
    if name == "qwen3_32b":
        slug = "qwen3_32b"
        MODEL_LABEL = "Qwen3-32B"
        attention = "gqa"
    elif name == "qwen3_8b":
        slug = "qwen3_8b"
        MODEL_LABEL = "Qwen3-8B-Base"
        attention = "gqa"
    elif name == "llama31_8b":
        slug = "llama31_8b"
        MODEL_LABEL = "Llama-3.1-8B"
        attention = "gqa"
    elif name == "llama31_70b":
        slug = "llama31_70b"
        MODEL_LABEL = "Llama-3.1-70B"
        attention = "gqa"
    elif name == "llama2_7b":
        slug = "llama2_7b"
        MODEL_LABEL = "Llama-2-7B"
        attention = "mha"
    else:
        raise ValueError(f"unknown Qwen3 C1 model profile: {name}")
    FORMAT = f"basisserve.{slug}.{attention}_c1.layer_global_kl_allocation.v1"
    PROFILE_FORMAT = (
        f"basisserve.{slug}.{attention}_c1.layer_global_kl_profile_shard.v1"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    profile = subparsers.add_parser("profile")
    runtime._add_shared_args(profile)
    profile.add_argument("--profile-shard-index", type=int, required=True)
    profile.add_argument("--profile-shard-count", type=int, default=2)
    _add_factorized_args(profile)

    finalize = subparsers.add_parser("finalize")
    runtime._add_shared_args(finalize)
    finalize.add_argument("--profile-shard-count", type=int, default=2)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--target-average-rank", type=int, required=True)
    finalize.add_argument(
        "--force-selected-candidate",
        choices=(
            "uniform_anchor",
            "factorized_mean_dp",
            "two_sided_factorized_kl",
        ),
        help=(
            "Export this schedule after reporting every confirmation metric. "
            "By default the lowest confirmation-KL schedule is exported"
        ),
    )
    finalize.add_argument(
        "--allocation-factorized-exponent",
        type=float,
        help=(
            "Override the factorized interpolation exponent during allocation "
            "without changing or rerunning the measured layer-probe profile"
        ),
    )
    _add_factorized_args(finalize)
    return parser.parse_args()


def _add_factorized_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--factorized-probe-rank",
        type=int,
        help=(
            "Profile only this non-anchor rank, then infer every candidate-rank "
            "cost from the post-ALS reconstruction-error curves"
        ),
    )
    parser.add_argument(
        "--factorized-compression-probe-rank",
        type=int,
        help=(
            "Optional below-anchor probe. When set, compression and expansion "
            "use separate measured sensitivity slopes"
        ),
    )
    parser.add_argument(
        "--local-error-split",
        choices=("heldout", "fit"),
        default="heldout",
        help="Reconstruction-error split used by factorized cost prediction",
    )
    parser.add_argument("--factorized-exponent", type=float, default=1.0)
    parser.add_argument(
        "--secondary-windows",
        type=Path,
        help=(
            "Optional second-domain input_ids bank. The same profile and "
            "confirmation counts are appended to the primary C4 windows"
        ),
    )
    parser.add_argument("--secondary-window-start", type=int, default=0)
    parser.add_argument("--secondary-domain-name", default="wikitext2")
    parser.add_argument(
        "--secondary-window-mode",
        choices=("append", "replace"),
        default="append",
        help=(
            "Append an equal-sized secondary-domain bank to C4, or replace "
            "the C4 profile/confirmation sequences with that bank"
        ),
    )
    parser.add_argument(
        "--profile-backend",
        choices=("full_forward", "sampled_suffix"),
        default="sampled_suffix",
        help=(
            "Use the legacy full-position/full-forward profiler, or exact "
            "full-vocabulary KL on sampled positions with batch-local suffix replay"
        ),
    )
    parser.add_argument(
        "--terminal-position-counts",
        default="64,128,256,512,1024",
        help=(
            "Nested per-window prediction-position counts recorded by the "
            "sampled_suffix backend"
        ),
    )
    parser.add_argument("--terminal-position-seed", type=int, default=20260903)
    parser.add_argument(
        "--terminal-selection-position-count",
        type=int,
        help="Position count used by the primary DP; defaults to the largest count",
    )


def _intervention_ranks(
    args: argparse.Namespace,
    candidate_ranks: Sequence[int],
) -> tuple[int, ...]:
    probe_rank = args.factorized_probe_rank
    if probe_rank is None:
        assert args.factorized_compression_probe_rank is None
        return tuple(rank for rank in candidate_ranks if rank != args.anchor_rank)
    assert probe_rank != args.anchor_rank and probe_rank in candidate_ranks
    exponent = float(args.factorized_exponent)
    assert math.isfinite(exponent) and exponent > 0
    compression_probe_rank = args.factorized_compression_probe_rank
    if compression_probe_rank is not None:
        assert compression_probe_rank in candidate_ranks
        assert compression_probe_rank < args.anchor_rank < probe_rank
        return (int(compression_probe_rank), int(probe_rank))
    return (int(probe_rank),)


def _allocation_factorized_exponent(args: argparse.Namespace) -> float:
    override = getattr(args, "allocation_factorized_exponent", None)
    exponent = (
        float(args.factorized_exponent)
        if override is None
        else float(override)
    )
    assert math.isfinite(exponent) and exponent > 0
    return exponent


def _load_secondary_windows(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    resolved = args.secondary_windows.expanduser().resolve()
    manifest_path = resolved.parent / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert common._sha256(resolved) == manifest.get("artifact", {}).get("sha256")
    payload = load_file(str(resolved), device="cpu")
    assert set(payload) == {"input_ids"}
    input_ids = payload["input_ids"].to(torch.long)
    needed = args.profile_windows + args.confirmation_windows
    start = args.secondary_window_start
    stop = start + needed
    assert 0 <= start and stop <= len(input_ids)
    assert 1 < args.sequence_length <= input_ids.shape[1]
    records = manifest.get("records", ())
    assert len(records) == len(input_ids)
    selected = input_ids[start:stop, : args.sequence_length].contiguous()
    provenance = {
        "name": args.secondary_domain_name,
        "path": str(resolved),
        "sha256": common._sha256(resolved),
        "manifest": str(manifest_path),
        "manifest_sha256": common._sha256(manifest_path),
        "window_start": start,
        "window_stop_exclusive": stop,
        "profile_indices": list(range(start, start + args.profile_windows)),
        "confirmation_indices": list(range(start + args.profile_windows, stop)),
        "stored_sequence_length": int(input_ids.shape[1]),
        "used_sequence_length": args.sequence_length,
    }
    return selected[: args.profile_windows], selected[args.profile_windows :], provenance


def _validate_shared(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path,
    tuple[int, ...],
    dict[int, Path],
    dict[int, dict[str, Any]],
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    validated = list(runtime._validate_shared(args))
    if args.secondary_windows is None:
        return tuple(validated)
    assert args.factorized_probe_rank is not None
    assert args.secondary_domain_name
    secondary_profile, secondary_confirmation, secondary = (
        _load_secondary_windows(args)
    )
    if args.secondary_window_mode == "replace":
        validated[5] = secondary_profile
        validated[6] = secondary_confirmation
        validated[7] = {
            "domain_swap": "replace_primary_c4",
            "name": args.secondary_domain_name,
            "provenance": secondary,
        }
        return tuple(validated)
    assert args.batch_size >= 2 * max(
        args.profile_windows,
        args.confirmation_windows,
    )
    primary_profile = validated[5]
    primary_confirmation = validated[6]
    validated[5] = torch.cat((primary_profile, secondary_profile), dim=0)
    validated[6] = torch.cat((primary_confirmation, secondary_confirmation), dim=0)
    validated[7] = {
        "mixture": "equal_windows_per_domain",
        "domains": [
            {
                "name": "c4",
                "profile_offset": 0,
                "profile_windows": args.profile_windows,
                "confirmation_offset": 0,
                "confirmation_windows": args.confirmation_windows,
                "provenance": validated[7],
            },
            {
                "name": args.secondary_domain_name,
                "profile_offset": args.profile_windows,
                "profile_windows": args.profile_windows,
                "confirmation_offset": args.confirmation_windows,
                "confirmation_windows": args.confirmation_windows,
                "provenance": secondary,
            },
        ],
    }
    return tuple(validated)


def _profile_dataset_name(args: argparse.Namespace) -> str:
    if args.secondary_windows is None:
        return "c4_train_fresh_documents"
    if args.secondary_window_mode == "replace":
        return str(args.secondary_domain_name)
    return f"c4+{args.secondary_domain_name}"


def _domain_window_multiplier(args: argparse.Namespace) -> int:
    if args.secondary_windows is None or args.secondary_window_mode == "replace":
        return 1
    return 2


def _terminal_position_configuration(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], int]:
    counts = parse_position_counts(args.terminal_position_counts)
    assert max(counts) <= args.sequence_length - 1
    selection_count = (
        max(counts)
        if args.terminal_selection_position_count is None
        else int(args.terminal_selection_position_count)
    )
    assert selection_count in counts
    return counts, selection_count


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
            "rank_128_endpoint": (
                "exact_identity_dense_o_without_decoder_closure"
                if common.HEAD_DIM in candidate_ranks
                else "not_in_candidate_set"
            ),
            "allocation_mode": (
                "full_layer_global_kl"
                if args.factorized_probe_rank is None
                else "factorized_two_probe"
                if args.factorized_compression_probe_rank is not None
                else "factorized_one_probe"
            ),
            "intervention_ranks": list(_intervention_ranks(args, candidate_ranks)),
        }
    )
    if args.factorized_probe_rank is not None:
        configuration["factorized"] = {
            "probe_rank": int(args.factorized_probe_rank),
            "compression_probe_rank": args.factorized_compression_probe_rank,
            "local_error_split": args.local_error_split,
            "exponent": float(args.factorized_exponent),
            "secondary_window_mode": args.secondary_window_mode,
        }
    if args.profile_backend == "sampled_suffix":
        counts, selection_count = _terminal_position_configuration(args)
        configuration["profile_backend"] = args.profile_backend
        configuration["sampled_terminal_kl"] = {
            "position_counts": list(counts),
            "selection_position_count": selection_count,
            "position_seed": int(args.terminal_position_seed),
            "position_sampling": "uniform_without_replacement_per_window_nested",
            "valid_prediction_positions": [0, args.sequence_length - 2],
            "teacher_statistics_dtype": "float32",
            "output_head_arithmetic": "float32",
            "vocabulary": "exact_full_streaming",
            "backbone": "batch_local_anchor_suffix_replay",
        }
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
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in runtime._cuda_device_indices()
            },
        },
    }


@dataclass(frozen=True)
class _SampledTeacherBatch:
    start: int
    stop: int
    statistics: TeacherTerminalStatistics


def _empty_sampled_metric_state(
    assigned_layers: Sequence[int],
    intervention_ranks: Sequence[int],
    position_counts: Sequence[int],
) -> dict[str, Any]:
    empty_metrics = lambda: {
        str(count): {"terminal_kl": [], "nll": []}
        for count in position_counts
    }
    return {
        "completed_windows": 0,
        "uniform_anchor": empty_metrics(),
        "interventions": {
            f"{layer}:{rank}": empty_metrics()
            for layer in assigned_layers
            for rank in intervention_ranks
        },
    }


def _append_nested_window_means(
    destination: Mapping[str, Any],
    *,
    terminal_kl: torch.Tensor,
    nll: torch.Tensor,
    position_counts: Sequence[int],
) -> None:
    assert terminal_kl.ndim == 2
    assert tuple(terminal_kl.shape) == tuple(nll.shape)
    for count in position_counts:
        destination[str(count)]["terminal_kl"].extend(
            terminal_kl[:, :count].double().mean(dim=1).cpu().tolist()
        )
        destination[str(count)]["nll"].extend(
            nll[:, :count].double().mean(dim=1).cpu().tolist()
        )


def _sampled_metric_summary(raw: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    return {
        "terminal_kl": common._paired(raw["terminal_kl"]),
        "nll": common._paired(raw["nll"]),
    }


def _sampled_metric_sweep(
    raw: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    position_counts: Sequence[int],
) -> dict[str, Any]:
    return {
        str(count): _sampled_metric_summary(raw[str(count)])
        for count in position_counts
    }


def _sampled_delta_sweep(
    candidate: Mapping[str, Mapping[str, Sequence[float]]],
    anchor: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    position_counts: Sequence[int],
) -> dict[str, Any]:
    return {
        str(count): common._paired_delta(
            _sampled_metric_summary(candidate[str(count)]),
            _sampled_metric_summary(anchor[str(count)]),
        )
        for count in position_counts
    }


@torch.inference_mode()
def _capture_sampled_teacher_batches(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    positions: torch.Tensor,
    *,
    output_weight: torch.Tensor,
    start_window: int,
    batch_size: int,
    vocab_chunk_size: int,
    shard: int | None,
) -> list[_SampledTeacherBatch]:
    input_device = common._input_device(model)
    batches = []
    for start in range(start_window, len(sequences), batch_size):
        stop = min(start + batch_size, len(sequences))
        input_ids = sequences[start:stop].to(input_device)
        hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        selected = gather_prediction_hidden(hidden, positions[start:stop])
        statistics = teacher_terminal_statistics(
            selected,
            output_weight,
            vocab_chunk_size=vocab_chunk_size,
        )
        batches.append(
            _SampledTeacherBatch(
                start=start,
                stop=stop,
                statistics=TeacherTerminalStatistics(
                    logsumexp=statistics.logsumexp.cpu(),
                    expected_output_weight=statistics.expected_output_weight.cpu(),
                    expected_log_probability=(
                        statistics.expected_log_probability.cpu()
                    ),
                ),
            )
        )
        _log(
            f"sampled dense teacher windows={stop}/{len(sequences)} "
            f"positions={positions.shape[1]}",
            shard=shard,
        )
        del input_ids, hidden, selected, statistics
        torch.cuda.empty_cache()
    return batches


def _statistics_to_device(
    statistics: TeacherTerminalStatistics,
    device: torch.device,
) -> TeacherTerminalStatistics:
    return TeacherTerminalStatistics(
        logsumexp=statistics.logsumexp.to(device),
        expected_output_weight=statistics.expected_output_weight.to(device),
        expected_log_probability=statistics.expected_log_probability.to(device),
    )


@torch.inference_mode()
def _evaluate_sampled_teacher_metrics(
    model: torch.nn.Module,
    teacher: Sequence[_SampledTeacherBatch],
    sequences: torch.Tensor,
    positions: torch.Tensor,
    *,
    output_weight: torch.Tensor,
    position_counts: Sequence[int],
    selection_count: int,
    vocab_chunk_size: int,
) -> dict[str, Any]:
    """Evaluate exact-vocabulary KL on fixed sampled prediction positions."""

    raw = {
        str(count): {"terminal_kl": [], "nll": []}
        for count in position_counts
    }
    input_device = common._input_device(model)
    for batch in teacher:
        input_ids = sequences[batch.start : batch.stop].to(input_device)
        hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        batch_positions = positions[batch.start : batch.stop]
        selected = gather_prediction_hidden(hidden, batch_positions)
        logsumexp = streaming_output_logsumexp(
            selected,
            output_weight,
            vocab_chunk_size=vocab_chunk_size,
        )
        statistics = _statistics_to_device(batch.statistics, selected.device)
        terminal_kl = terminal_kl_from_statistics(
            statistics,
            selected,
            logsumexp,
        ).clamp_min(0.0)
        labels = input_ids[:, 1:].gather(
            1,
            batch_positions.to(input_ids.device),
        )
        nll = selected_token_nll(
            selected,
            logsumexp,
            output_weight,
            labels,
        )
        _append_nested_window_means(
            raw,
            terminal_kl=terminal_kl,
            nll=nll,
            position_counts=position_counts,
        )
        del input_ids, hidden, selected, logsumexp, statistics
        del terminal_kl, labels, nll
        torch.cuda.empty_cache()
    sweep = _sampled_metric_sweep(raw, position_counts=position_counts)
    return {
        **sweep[str(selection_count)],
        "position_count_sweep": sweep,
    }


def _completed_sampled_profile(
    *,
    args: argparse.Namespace,
    assigned_layers: Sequence[int],
    intervention_ranks: Sequence[int],
    position_counts: Sequence[int],
    selection_count: int,
    state: Mapping[str, Any],
    closures: Mapping[tuple[int, int], Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    anchor_raw = state["uniform_anchor"]
    anchor_sweep = _sampled_metric_sweep(
        anchor_raw,
        position_counts=position_counts,
    )
    anchor = {
        **anchor_sweep[str(selection_count)],
        "position_count_sweep": anchor_sweep,
    }
    records = []
    for layer in assigned_layers:
        for rank in intervention_ranks:
            raw = state["interventions"][f"{layer}:{rank}"]
            metric_sweep = _sampled_metric_sweep(
                raw,
                position_counts=position_counts,
            )
            delta_sweep = _sampled_delta_sweep(
                raw,
                anchor_raw,
                position_counts=position_counts,
            )
            selected_metrics = metric_sweep[str(selection_count)]
            selected_delta = delta_sweep[str(selection_count)]
            records.append(
                {
                    "layer": layer,
                    "anchor_rank": args.anchor_rank,
                    "candidate_rank": rank,
                    "physical_kv_sources_changed": common.NUM_KV_HEADS,
                    "terminal_kl": selected_metrics["terminal_kl"],
                    "nll": selected_metrics["nll"],
                    "terminal_kl_delta": selected_delta["terminal_kl"],
                    "nll_delta": selected_delta["nll"],
                    "position_count_sweep": {
                        str(count): {
                            **metric_sweep[str(count)],
                            "terminal_kl_delta": delta_sweep[str(count)][
                                "terminal_kl"
                            ],
                            "nll_delta": delta_sweep[str(count)]["nll"],
                        }
                        for count in position_counts
                    },
                    "layer_rank_delta": rank - args.anchor_rank,
                    "collective_width": common.NUM_KV_HEADS * rank,
                    "decoder_solve": dict(closures[(layer, rank)]),
                    "measurement": {
                        "prediction_positions_per_window": selection_count,
                        "position_sampling": "nested_uniform_without_replacement",
                        "vocabulary": "exact_full_streaming",
                        "teacher_representation": "fp32_sufficient_statistics",
                        "backbone": "batch_local_anchor_suffix_replay",
                    },
                }
            )
    return anchor, records


@torch.inference_mode()
def _profile_sampled_suffix(args: argparse.Namespace) -> None:
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
    ) = _validate_shared(args)
    configuration = _configuration(
        args,
        model_path=model_path,
        snapshot_dir=snapshot_dir,
        candidate_ranks=candidate_ranks,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        windows_provenance=windows_provenance,
    )
    position_counts, selection_count = _terminal_position_configuration(args)
    intervention_ranks = _intervention_ranks(args, candidate_ranks)
    positions = nested_prediction_positions(
        num_windows=len(profile_sequences),
        sequence_length=args.sequence_length,
        position_counts=position_counts,
        seed=args.terminal_position_seed,
    )
    profile_dir = args.profile_dir.expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    shard_path = profile_dir / f"shard_{args.profile_shard_index:02d}.json"
    state = _empty_sampled_metric_state(
        assigned_layers,
        intervention_ranks,
        position_counts,
    )
    if shard_path.is_file():
        prior = json.loads(shard_path.read_text(encoding="utf-8"))
        assert prior.get("format") == PROFILE_FORMAT
        assert prior.get("configuration") == configuration
        assert tuple(map(int, prior.get("assigned_layers", ()))) == assigned_layers
        if prior.get("status") == "complete":
            _log(
                f"completed checkpoint already exists: {shard_path}",
                shard=args.profile_shard_index,
            )
            return
        state = prior["sampled_profile_state"]
    completed_windows = int(state["completed_windows"])
    assert 0 <= completed_windows <= len(profile_sequences)
    for count in position_counts:
        expected = completed_windows
        assert len(state["uniform_anchor"][str(count)]["terminal_kl"]) == expected
        for values in state["interventions"].values():
            assert len(values[str(count)]["terminal_kl"]) == expected

    started = time.perf_counter()
    model = runtime._load_model(args, model_path)
    output_weight = model.lm_head.weight.detach().float()
    teacher_batches = _capture_sampled_teacher_batches(
        model,
        profile_sequences,
        positions,
        output_weight=output_weight,
        start_window=completed_windows,
        batch_size=args.batch_size,
        vocab_chunk_size=args.vocab_chunk_size,
        shard=args.profile_shard_index,
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
        decoder_relative_jitter=0.0,
        keep_factors=False,
    )

    interventions: dict[tuple[int, int], common.CachedFactors] = {}
    closures: dict[tuple[int, int], dict[str, Any]] = {}
    damping_by_layer: dict[int, float] = {}
    for layer_index in assigned_layers:
        layer = layers[layer_index]
        objective, absolute_damping = common._objective(
            snapshot_cache[layer_index],
            layer=layer_index,
            device=layer.self_attn.v_proj.weight.device,
            covariance_damping=args.covariance_damping,
        )
        damping_by_layer[layer_index] = absolute_damping
        for rank in intervention_ranks:
            factors, closure = common._closed_factors(
                bank=factor_cache[layer_index],
                snapshot=snapshot_cache[layer_index],
                objective=objective,
                source_ranks=[rank] * common.NUM_KV_HEADS,
                anchor_rank=args.anchor_rank,
                decoder_relative_jitter=0.0,
                device=layer.self_attn.v_proj.weight.device,
            )
            interventions[(layer_index, rank)] = factors
            closures[(layer_index, rank)] = closure
        del objective
        torch.cuda.empty_cache()

    input_device = common._input_device(model)
    for teacher_batch in teacher_batches:
        start, stop = teacher_batch.start, teacher_batch.stop
        input_ids = profile_sequences[start:stop].to(input_device)
        batch_positions = positions[start:stop]
        anchor_replay = capture_qwen_anchor_replay(model, input_ids=input_ids)
        anchor_hidden = gather_prediction_hidden(
            anchor_replay.final_hidden,
            batch_positions,
        )
        statistics = _statistics_to_device(
            teacher_batch.statistics,
            anchor_hidden.device,
        )
        anchor_lse = streaming_output_logsumexp(
            anchor_hidden,
            output_weight,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        anchor_kl_unclamped = terminal_kl_from_statistics(
            statistics,
            anchor_hidden,
            anchor_lse,
        )
        anchor_kl = anchor_kl_unclamped.clamp_min(0.0)
        labels = input_ids[:, 1:].gather(
            1,
            batch_positions.to(input_ids.device),
        )
        anchor_nll = selected_token_nll(
            anchor_hidden,
            anchor_lse,
            output_weight,
            labels,
        )
        _append_nested_window_means(
            state["uniform_anchor"],
            terminal_kl=anchor_kl,
            nll=anchor_nll,
            position_counts=position_counts,
        )

        for layer_index in assigned_layers:
            layer = layers[layer_index]
            anchor_v = layer.self_attn.v_proj.weight.detach().clone()
            anchor_o = layer.self_attn.o_proj.weight.detach().clone()
            for rank in intervention_ranks:
                try:
                    common._install_factors(
                        layer,
                        dense_v_weight=dense_v_weights[layer_index],
                        factors=interventions[(layer_index, rank)],
                    )
                    candidate_final = replay_qwen_suffix(
                        model,
                        anchor_replay,
                        intervention_layer=layer_index,
                    )
                finally:
                    layer.self_attn.v_proj.weight.copy_(anchor_v)
                    layer.self_attn.o_proj.weight.copy_(anchor_o)
                candidate_hidden = gather_prediction_hidden(
                    candidate_final,
                    batch_positions,
                )
                candidate_lse = streaming_output_logsumexp(
                    candidate_hidden,
                    output_weight,
                    vocab_chunk_size=args.vocab_chunk_size,
                )
                delta = paired_terminal_kl_delta(
                    statistics,
                    anchor_hidden=anchor_hidden,
                    candidate_hidden=candidate_hidden,
                    anchor_logsumexp=anchor_lse,
                    candidate_logsumexp=candidate_lse,
                )
                candidate_kl = (anchor_kl_unclamped + delta).clamp_min(0.0)
                candidate_nll = selected_token_nll(
                    candidate_hidden,
                    candidate_lse,
                    output_weight,
                    labels,
                )
                _append_nested_window_means(
                    state["interventions"][f"{layer_index}:{rank}"],
                    terminal_kl=candidate_kl,
                    nll=candidate_nll,
                    position_counts=position_counts,
                )
                _log(
                    f"windows={stop}/{len(profile_sequences)} layer={layer_index} "
                    f"rank={rank} dKL={delta.double().mean().item():.6g}",
                    shard=args.profile_shard_index,
                )
                del candidate_final, candidate_hidden, candidate_lse, delta
                del candidate_kl, candidate_nll
            del anchor_v, anchor_o
            torch.cuda.empty_cache()

        state["completed_windows"] = stop
        running = _profile_payload(
            status="running",
            args=args,
            configuration=configuration,
            assigned_layers=assigned_layers,
            completed_layers=(),
            anchor_metrics={},
            records=(),
            damping_by_layer=damping_by_layer,
            started=started,
        )
        running["sampled_profile_state"] = state
        common._atomic_json(shard_path, running)
        del input_ids, statistics, anchor_replay, anchor_hidden, anchor_lse
        del anchor_kl_unclamped, anchor_kl, labels, anchor_nll
        torch.cuda.empty_cache()
        _log(
            f"checkpointed windows={stop}/{len(profile_sequences)}",
            shard=args.profile_shard_index,
        )

    anchor_metrics, records = _completed_sampled_profile(
        args=args,
        assigned_layers=assigned_layers,
        intervention_ranks=intervention_ranks,
        position_counts=position_counts,
        selection_count=selection_count,
        state=state,
        closures=closures,
    )
    completed = _profile_payload(
        status="complete",
        args=args,
        configuration=configuration,
        assigned_layers=assigned_layers,
        completed_layers=assigned_layers,
        anchor_metrics=anchor_metrics,
        records=records,
        damping_by_layer=damping_by_layer,
        started=started,
    )
    completed["sampled_profile_state"] = {
        "completed_windows": int(state["completed_windows"]),
        "position_counts": list(position_counts),
    }
    common._atomic_json(shard_path, completed)
    _log(f"completed {shard_path}", shard=args.profile_shard_index)


@torch.inference_mode()
def _profile(args: argparse.Namespace) -> None:
    if args.profile_backend == "sampled_suffix":
        _profile_sampled_suffix(args)
        return
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
    ) = _validate_shared(args)
    configuration = _configuration(
        args,
        model_path=model_path,
        snapshot_dir=snapshot_dir,
        candidate_ranks=candidate_ranks,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        windows_provenance=windows_provenance,
    )
    intervention_ranks = _intervention_ranks(args, candidate_ranks)
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
    expected_prior_records = len(completed_layers) * len(intervention_ranks)
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
        decoder_relative_jitter=0.0,
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
                decoder_relative_jitter=0.0,
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
    intervention_ranks: Sequence[int] | None = None,
) -> dict[str, Any]:
    expected_ranks = (
        None
        if intervention_ranks is None
        else tuple(map(int, intervention_ranks))
    )
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
    ranks_per_layer = (
        len(candidate_ranks) - 1
        if expected_ranks is None
        else len(expected_ranks)
    )
    expected_records = common.NUM_LAYERS * ranks_per_layer
    if len(records) != expected_records:
        raise ValueError("merged profile does not contain every layer intervention")
    keys = {
        (int(row["layer"]), int(row["candidate_rank"])) for row in records
    }
    if expected_ranks is None:
        assert len(keys) == expected_records
    else:
        expected_keys = {
            (layer, rank)
            for layer in range(common.NUM_LAYERS)
            for rank in expected_ranks
        }
        assert keys == expected_keys
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
    target_average_rank: int,
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
        total_rank_budget=common.NUM_LAYERS * target_average_rank,
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


def _allocate_factorized_layer_ranks(
    records: Sequence[Mapping[str, Any]],
    factor_results: Mapping[int, Mapping[str, Any]],
    *,
    candidate_ranks: Sequence[int],
    anchor_rank: int,
    target_average_rank: int,
    probe_rank: int,
    local_error_split: str,
    exponent: float,
) -> tuple[list[list[int]], float, list[dict[str, Any]], dict[str, Any]]:
    indexed = {int(row["layer"]): row for row in records}
    assert len(indexed) == common.NUM_LAYERS
    assert all(
        int(row["candidate_rank"]) == probe_rank for row in indexed.values()
    )
    local_errors = local_error_curves_from_factor_results(
        factor_results,
        candidate_ranks=candidate_ranks,
        error_split=local_error_split,
        num_layers=common.NUM_LAYERS,
        head_dim=common.HEAD_DIM,
    )
    probe_deltas = tuple(
        float(indexed[layer]["terminal_kl_delta"]["mean"])
        for layer in range(common.NUM_LAYERS)
    )
    costs, sensitivities = predict_factorized_costs(
        local_errors,
        probe_deltas,
        candidate_ranks=candidate_ranks,
        anchor_rank=anchor_rank,
        probe_rank=probe_rank,
        exponent=exponent,
    )
    layer_ranks, total_cost = allocate_factorized_layer_schedule(
        costs,
        candidate_ranks=candidate_ranks,
        anchor_rank=anchor_rank,
        target_average_rank=target_average_rank,
    )
    schedule = [[rank] * common.NUM_KV_HEADS for rank in layer_ranks]
    contributions = [
        {
            "layer": layer,
            "rank": rank,
            "cost": float(costs[layer][rank]),
        }
        for layer, rank in enumerate(layer_ranks)
    ]
    method = {
        "name": "one_probe_factorized_terminal_kl",
        "formula": "delta_K_lr = s_l * (e_lr^alpha - e_l_anchor^alpha)",
        "anchor_rank": anchor_rank,
        "probe_rank": probe_rank,
        "terminal_interventions_per_layer": 1,
        "local_error": (
            f"post-ALS factor-dtype {local_error_split} relative MSE"
        ),
        "local_error_split": local_error_split,
        "exponent": exponent,
        "sensitivities": list(sensitivities),
        "zero_sensitivity_layers": [
            layer for layer, sensitivity in enumerate(sensitivities)
            if sensitivity == 0.0
        ],
        "predicted_costs": [
            {str(rank): float(curve[rank]) for rank in candidate_ranks}
            for curve in costs
        ],
    }
    return schedule, float(total_cost), contributions, method


def _allocate_two_sided_factorized_layer_ranks(
    records: Sequence[Mapping[str, Any]],
    factor_results: Mapping[int, Mapping[str, Any]],
    *,
    candidate_ranks: Sequence[int],
    anchor_rank: int,
    target_average_rank: int,
    compression_probe_rank: int,
    expansion_probe_rank: int,
    local_error_split: str,
    exponent: float,
) -> tuple[list[list[int]], float, list[dict[str, Any]], dict[str, Any]]:
    indexed = {
        (int(row["layer"]), int(row["candidate_rank"])): row
        for row in records
    }
    expected = {
        (layer, rank)
        for layer in range(common.NUM_LAYERS)
        for rank in (compression_probe_rank, expansion_probe_rank)
    }
    assert set(indexed) == expected
    local_errors = local_error_curves_from_factor_results(
        factor_results,
        candidate_ranks=candidate_ranks,
        error_split=local_error_split,
        num_layers=common.NUM_LAYERS,
        head_dim=common.HEAD_DIM,
    )
    compression_deltas = tuple(
        float(indexed[(layer, compression_probe_rank)]["terminal_kl_delta"]["mean"])
        for layer in range(common.NUM_LAYERS)
    )
    expansion_deltas = tuple(
        float(indexed[(layer, expansion_probe_rank)]["terminal_kl_delta"]["mean"])
        for layer in range(common.NUM_LAYERS)
    )
    (
        costs,
        compression_sensitivities,
        expansion_sensitivities,
    ) = predict_two_sided_factorized_costs(
        local_errors,
        compression_deltas,
        expansion_deltas,
        candidate_ranks=candidate_ranks,
        anchor_rank=anchor_rank,
        compression_probe_rank=compression_probe_rank,
        expansion_probe_rank=expansion_probe_rank,
        exponent=exponent,
    )
    layer_ranks, total_cost = allocate_factorized_layer_schedule(
        costs,
        candidate_ranks=candidate_ranks,
        anchor_rank=anchor_rank,
        target_average_rank=target_average_rank,
    )
    schedule = [[rank] * common.NUM_KV_HEADS for rank in layer_ranks]
    contributions = [
        {
            "layer": layer,
            "rank": rank,
            "cost": float(costs[layer][rank]),
        }
        for layer, rank in enumerate(layer_ranks)
    ]
    method = {
        "name": "two_sided_factorized_terminal_kl",
        "formula": (
            "delta_K_lr = s_l^- * delta_e_lr below anchor; "
            "s_l^+ * delta_e_lr above anchor"
        ),
        "anchor_rank": anchor_rank,
        "compression_probe_rank": compression_probe_rank,
        "expansion_probe_rank": expansion_probe_rank,
        "terminal_interventions_per_layer": 2,
        "local_error": (
            f"post-ALS factor-dtype {local_error_split} relative MSE"
        ),
        "local_error_split": local_error_split,
        "exponent": exponent,
        "compression_sensitivities": list(compression_sensitivities),
        "expansion_sensitivities": list(expansion_sensitivities),
        "zero_compression_sensitivity_layers": [
            layer
            for layer, sensitivity in enumerate(compression_sensitivities)
            if sensitivity == 0.0
        ],
        "zero_expansion_sensitivity_layers": [
            layer
            for layer, sensitivity in enumerate(expansion_sensitivities)
            if sensitivity == 0.0
        ],
        "predicted_costs": [
            {str(rank): float(curve[rank]) for rank in candidate_ranks}
            for curve in costs
        ],
    }
    return schedule, float(total_cost), contributions, method


def _records_at_position_count(
    records: Sequence[Mapping[str, Any]],
    position_count: int,
) -> list[dict[str, Any]]:
    """Project sampled-profile records onto one nested position count."""

    key = str(int(position_count))
    projected = []
    for record in records:
        sweep = record.get("position_count_sweep", {})
        assert key in sweep
        metrics = sweep[key]
        row = dict(record)
        row["terminal_kl"] = metrics["terminal_kl"]
        row["nll"] = metrics["nll"]
        row["terminal_kl_delta"] = metrics["terminal_kl_delta"]
        row["nll_delta"] = metrics["nll_delta"]
        projected.append(row)
    return projected


def _two_sided_position_count_sweep(
    records: Sequence[Mapping[str, Any]],
    factor_results: Mapping[int, Mapping[str, Any]],
    *,
    position_counts: Sequence[int],
    reference_schedule: Sequence[Sequence[int]],
    candidate_ranks: Sequence[int],
    anchor_rank: int,
    target_average_rank: int,
    compression_probe_rank: int,
    expansion_probe_rank: int,
    local_error_split: str,
    exponent: float,
) -> dict[str, Any]:
    reference_ranks = [int(layer[0]) for layer in reference_schedule]
    result = {}
    for count in position_counts:
        schedule, cost, _, method = _allocate_two_sided_factorized_layer_ranks(
            _records_at_position_count(records, count),
            factor_results,
            candidate_ranks=candidate_ranks,
            anchor_rank=anchor_rank,
            target_average_rank=target_average_rank,
            compression_probe_rank=compression_probe_rank,
            expansion_probe_rank=expansion_probe_rank,
            local_error_split=local_error_split,
            exponent=exponent,
        )
        ranks = [int(layer[0]) for layer in schedule]
        result[str(count)] = {
            "schedule": schedule,
            "predicted_additive_cost": cost,
            "accounting": _layer_schedule_accounting(
                schedule,
                anchor_rank=anchor_rank,
            ),
            "exact_layer_matches_to_selection_count": sum(
                left == right
                for left, right in zip(ranks, reference_ranks, strict=True)
            ),
            "rank_mae_to_selection_count": sum(
                abs(left - right)
                for left, right in zip(ranks, reference_ranks, strict=True)
            )
            / len(ranks),
            "compression_sensitivities": method[
                "compression_sensitivities"
            ],
            "expansion_sensitivities": method["expansion_sensitivities"],
        }
    return result


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
    ) = _validate_shared(args)
    configuration = _configuration(
        args,
        model_path=model_path,
        snapshot_dir=snapshot_dir,
        candidate_ranks=candidate_ranks,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        windows_provenance=windows_provenance,
    )
    intervention_ranks = _intervention_ranks(args, candidate_ranks)
    profile_dir = args.profile_dir.expanduser().resolve()
    merged = _merge_profile_shards(
        profile_dir,
        shard_count=args.profile_shard_count,
        configuration=configuration,
        candidate_ranks=candidate_ranks,
        intervention_ranks=intervention_ranks,
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    model = runtime._load_model(args, model_path)
    sampled_confirmation = args.profile_backend == "sampled_suffix"
    confirmation_positions: torch.Tensor | None = None
    output_weight: torch.Tensor | None = None
    if sampled_confirmation:
        position_counts, selection_count = _terminal_position_configuration(args)
        confirmation_positions = nested_prediction_positions(
            num_windows=len(confirmation_sequences),
            sequence_length=args.sequence_length,
            position_counts=position_counts,
            seed=args.terminal_position_seed + 1,
        )
        output_weight = model.lm_head.weight.detach().float()
        teacher = _capture_sampled_teacher_batches(
            model,
            confirmation_sequences,
            confirmation_positions,
            output_weight=output_weight,
            start_window=0,
            batch_size=args.batch_size,
            vocab_chunk_size=args.vocab_chunk_size,
            shard=None,
        )
    else:
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
        decoder_relative_jitter=0.0,
        keep_factors=False,
    )

    candidates = {}
    predicted = {}
    contributions = {}
    factorized_method = None
    allocation_exponent = _allocation_factorized_exponent(args)
    if args.factorized_probe_rank is None:
        for label, cost_key in (
            ("mean_dp", "mean"),
            ("ucb_dp", "one_standard_error_ucb"),
        ):
            schedule, cost, rows = _allocate_layer_ranks(
                merged["records"],
                candidate_ranks=candidate_ranks,
                anchor_rank=args.anchor_rank,
                target_average_rank=args.target_average_rank,
                cost_key=cost_key,
            )
            candidates[label] = schedule
            predicted[label] = cost
            contributions[label] = rows
    else:
        compression_probe_rank = args.factorized_compression_probe_rank
        if compression_probe_rank is None:
            schedule, cost, rows, factorized_method = _allocate_factorized_layer_ranks(
                merged["records"],
                factor_results,
                candidate_ranks=candidate_ranks,
                anchor_rank=args.anchor_rank,
                target_average_rank=args.target_average_rank,
                probe_rank=args.factorized_probe_rank,
                local_error_split=args.local_error_split,
                exponent=allocation_exponent,
            )
            candidate_label = "factorized_mean_dp"
        else:
            schedule, cost, rows, factorized_method = (
                _allocate_two_sided_factorized_layer_ranks(
                    merged["records"],
                    factor_results,
                    candidate_ranks=candidate_ranks,
                    anchor_rank=args.anchor_rank,
                    target_average_rank=args.target_average_rank,
                    compression_probe_rank=compression_probe_rank,
                    expansion_probe_rank=args.factorized_probe_rank,
                    local_error_split=args.local_error_split,
                    exponent=allocation_exponent,
                )
            )
            candidate_label = "two_sided_factorized_kl"
        candidates[candidate_label] = schedule
        predicted[candidate_label] = cost
        contributions[candidate_label] = rows
        if (
            args.profile_backend == "sampled_suffix"
            and compression_probe_rank is not None
        ):
            position_counts, selection_count = _terminal_position_configuration(args)
            assert factorized_method is not None
            factorized_method["terminal_position_count_sweep"] = (
                _two_sided_position_count_sweep(
                    merged["records"],
                    factor_results,
                    position_counts=position_counts,
                    reference_schedule=schedule,
                    candidate_ranks=candidate_ranks,
                    anchor_rank=args.anchor_rank,
                    target_average_rank=args.target_average_rank,
                    compression_probe_rank=compression_probe_rank,
                    expansion_probe_rank=args.factorized_probe_rank,
                    local_error_split=args.local_error_split,
                    exponent=allocation_exponent,
                )
            )
            factorized_method["selection_position_count"] = selection_count

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
                decoder_relative_jitter=0.0,
                keep_factors=False,
            )
            closure_diagnostics[label] = diagnostics
        if sampled_confirmation:
            assert confirmation_positions is not None and output_weight is not None
            confirmation[label] = _evaluate_sampled_teacher_metrics(
                model,
                teacher,
                confirmation_sequences,
                confirmation_positions,
                output_weight=output_weight,
                position_counts=position_counts,
                selection_count=selection_count,
                vocab_chunk_size=args.vocab_chunk_size,
            )
        else:
            confirmation[label] = common._evaluate_teacher_metrics(
                model,
                teacher,
                vocab_chunk_size=args.vocab_chunk_size,
            )
        _log(
            f"confirmation {label}: "
            f"KL={confirmation[label]['terminal_kl']['mean']:.9g}"
        )

    eligible_candidates = set(candidates)
    if args.target_average_rank == args.anchor_rank:
        eligible_candidates.add("uniform_anchor")
    if args.force_selected_candidate is None:
        selected_name = min(
            eligible_candidates,
            key=lambda label: (confirmation[label]["terminal_kl"]["mean"], label),
        )
        selection_metric = "lowest disjoint-confirmation mean terminal KL"
    else:
        selected_name = args.force_selected_candidate
        assert selected_name in eligible_candidates
        selection_metric = (
            "explicit forced export after disjoint-confirmation evaluation"
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
        decoder_relative_jitter=0.0,
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
            "backend": args.profile_backend,
            "sampled_terminal_kl": configuration.get("sampled_terminal_kl"),
            "intervention_unit": "whole decoder layer",
            "sources_changed_per_intervention": common.NUM_KV_HEADS,
            "factor_stage": configuration["factor_stage"],
            "allocation_mode": configuration["allocation_mode"],
            "intervention_ranks": list(intervention_ranks),
            "terminal_interventions_per_layer": len(intervention_ranks),
            "profile_shard_count": args.profile_shard_count,
            "dataset": _profile_dataset_name(args),
            "windows": args.profile_windows * _domain_window_multiplier(args),
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "forward_batches_per_candidate": math.ceil(
                (
                    args.profile_windows
                    * _domain_window_multiplier(args)
                )
                / args.batch_size
            ),
            "uniform_anchor_by_shard": merged["uniform_anchor_by_shard"],
            "records": merged["records"],
            "absolute_covariance_damping_by_layer": merged[
                "absolute_covariance_damping_by_layer"
            ],
            "shards": merged["shards"],
        },
        "confirmation": {
            "dataset": _profile_dataset_name(args),
            "backend": args.profile_backend,
            "sampled_terminal_kl": (
                {
                    **configuration["sampled_terminal_kl"],
                    "position_seed": args.terminal_position_seed + 1,
                }
                if sampled_confirmation
                else None
            ),
            "windows": (
                args.confirmation_windows * _domain_window_multiplier(args)
            ),
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "forward_batches_per_schedule": math.ceil(
                (
                    args.confirmation_windows
                    * _domain_window_multiplier(args)
                )
                / args.batch_size
            ),
            "disjoint_from_profile": True,
            "disjoint_from_als_fit_and_heldout": True,
            "windows_provenance": windows_provenance,
        },
        "selection": {
            "constraint": "exact per-layer rank budget with one rank per layer",
            "profiling_anchor_rank": args.anchor_rank,
            "target_average_rank": args.target_average_rank,
            "target_layer_rank_sum": (
                common.NUM_LAYERS * args.target_average_rank
            ),
            "target_source_rank_sum": (
                common.NUM_LAYERS
                * common.NUM_KV_HEADS
                * args.target_average_rank
            ),
            "candidate_ranks": list(candidate_ranks),
            "rank_128_endpoint": configuration["rank_128_endpoint"],
            "predicted_additive_costs": predicted,
            "contributions": contributions,
            "factorized_method": factorized_method,
            "selected_candidate": selected_name,
            "selected_schedule": selected_schedule,
            "selected_accounting": selected_accounting,
            "uniform_is_eligible": args.target_average_rank == args.anchor_rank,
            "selection_metric": selection_metric,
            "forced_selected_candidate": args.force_selected_candidate,
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
            "heldout_not_used_for_global_kl_selection": (
                args.factorized_probe_rank is None
            ),
            "heldout_used_for_factorized_local_error": (
                args.factorized_probe_rank is not None
                and args.local_error_split == "heldout"
            ),
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
