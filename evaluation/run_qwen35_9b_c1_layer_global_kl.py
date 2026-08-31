#!/usr/bin/env python3
"""Profile and allocate Qwen3.5-9B layer-ragged C1 ranks by terminal KL."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.global_rank_sensitivity import (  # noqa: E402
    logits_logsumexp,
    teacher_kl_sum,
)
from basisserve.core.metric_rank_allocation import (  # noqa: E402
    MetricRankOption,
    allocate_metric_rank_exact,
    paired_standard_error,
)
from basisserve.core.qwen35_full_attention_private_ag_runtime import (  # noqa: E402
    FACTOR_FORMAT as FULL_FACTOR_FORMAT,
)
from basisserve.core.qwen35_gdn_private_ag_runtime import (  # noqa: E402
    FACTOR_FORMAT as GDN_FACTOR_FORMAT,
    Qwen35PrivateAGOutput,
)
from basisserve.core.qwen35_gdn_runtime import qwen35_decoder_layers  # noqa: E402
from scripts.collect_qwen35_gdn_wo_activations import (  # noqa: E402
    _load_window_split,
)
from scripts.eval_qwen35_projected_gdn_nll import (  # noqa: E402
    _chunked_cross_entropy,
)


PROFILE_FORMAT = "basisserve.qwen35_9b.c1.layer_global_kl_profile_shard.v1"
RESULT_FORMAT = "basisserve.qwen35_9b.c1.layer_global_kl_allocation.v1"
NUM_LAYERS = 32
TP_SIZE = 8
LOCAL_WIDTH = 512
ANCHOR_RANK = 192
CANDIDATE_RANKS = (128, 192, 256, 320, 384, 448, 512)
GDN_LAYERS = tuple(layer for layer in range(NUM_LAYERS) if layer % 4 != 3)
FULL_LAYERS = tuple(layer for layer in range(NUM_LAYERS) if layer % 4 == 3)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--factor-root", required=True)
    parser.add_argument("--anchor-gdn-factors", required=True)
    parser.add_argument("--anchor-full-factors", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument(
        "--candidate-ranks",
        default=",".join(map(str, CANDIDATE_RANKS)),
    )
    parser.add_argument("--anchor-rank", type=int, default=ANCHOR_RANK)
    parser.add_argument("--profile-offset", type=int, default=320)
    parser.add_argument("--profile-windows", type=int, default=16)
    parser.add_argument("--confirmation-offset", type=int, default=336)
    parser.add_argument("--confirmation-windows", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=2)
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
    finalize.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _parse_ranks(raw: str, *, anchor_rank: int) -> tuple[int, ...]:
    ranks = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
    if (
        ranks != tuple(sorted(set(ranks)))
        or anchor_rank not in ranks
        or any(rank <= 0 or rank > LOCAL_WIDTH or rank % 64 for rank in ranks)
    ):
        raise ValueError(
            "candidate ranks must be distinct increasing multiples of 64, "
            "include the anchor, and not exceed 512"
        )
    return ranks


def _factor_paths(args: argparse.Namespace, ranks: Sequence[int]) -> dict[int, dict[str, Path]]:
    root = Path(args.factor_root).expanduser().resolve()
    result = {}
    for rank in ranks:
        if rank == args.anchor_rank:
            gdn = Path(args.anchor_gdn_factors).expanduser().resolve()
            full = Path(args.anchor_full_factors).expanduser().resolve()
        else:
            directory = root / f"r{rank}"
            gdn = directory / "gdn_private_ag_als10_all.pt"
            full = directory / "full_private_ag_als10_all.pt"
        if not gdn.is_file() or not full.is_file():
            raise FileNotFoundError(f"missing rank-{rank} factor pair: {gdn}, {full}")
        result[rank] = {"gdn": gdn, "full": full}
    return result


def _expected_layers(kind: str) -> tuple[int, ...]:
    return GDN_LAYERS if kind == "gdn" else FULL_LAYERS


def _load_factor_layers(
    path: Path,
    *,
    kind: str,
    rank: int,
    required_layers: Sequence[int],
    model_path: Path,
) -> dict[int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_format = GDN_FACTOR_FORMAT if kind == "gdn" else FULL_FACTOR_FORMAT
    if (
        payload.get("format") != expected_format
        or int(payload.get("schema_version", -1)) != 1
    ):
        raise ValueError(f"incompatible {kind} factor artifact: {path}")
    if Path(payload["model_path"]).resolve() != model_path:
        raise ValueError(f"factor artifact belongs to another model: {path}")
    if int(payload.get("tp_size", -1)) != TP_SIZE or int(payload.get("local_rank", -1)) != rank:
        raise ValueError(f"factor geometry disagrees with rank-{rank}: {path}")
    by_layer = {int(layer["layer_index"]): layer for layer in payload["layers"]}
    if tuple(sorted(by_layer)) != _expected_layers(kind):
        raise ValueError(f"factor artifact has incomplete {kind} layer coverage: {path}")
    selected = {}
    for layer_index in required_layers:
        if layer_index not in by_layer:
            continue
        layer = by_layer[layer_index]
        encoders = layer["private_encoders"]
        decoder = layer["joint_decoder_weight"]
        if tuple(encoders.shape) != (TP_SIZE, LOCAL_WIDTH, rank):
            raise ValueError(f"layer {layer_index} rank-{rank} encoders are malformed")
        if tuple(decoder.shape) != (4096, TP_SIZE * rank):
            raise ValueError(f"layer {layer_index} rank-{rank} decoder is malformed")
        if not torch.isfinite(encoders).all() or not torch.isfinite(decoder).all():
            raise ValueError(f"layer {layer_index} rank-{rank} factors are non-finite")
        selected[layer_index] = layer
    return selected


def _load_banks(
    args: argparse.Namespace,
    *,
    ranks: Sequence[int],
    required_layers: Sequence[int],
    include_anchor_all_layers: bool,
) -> tuple[dict[int, dict[int, dict[str, Any]]], dict[int, dict[str, Path]]]:
    model_path = Path(args.model_path).expanduser().resolve()
    paths = _factor_paths(args, ranks)
    banks = {}
    all_layers = tuple(range(NUM_LAYERS))
    for rank in ranks:
        requested = all_layers if rank == args.anchor_rank and include_anchor_all_layers else required_layers
        combined = {}
        for kind in ("gdn", "full"):
            combined.update(
                _load_factor_layers(
                    paths[rank][kind],
                    kind=kind,
                    rank=rank,
                    required_layers=requested,
                    model_path=model_path,
                )
            )
        expected = set(requested)
        if set(combined) != expected:
            raise ValueError(f"rank-{rank} factor bank misses requested layers")
        banks[rank] = combined
    return banks, paths


def _projection_parent(layers: Sequence[nn.Module], layer_index: int) -> tuple[nn.Module, str]:
    layer = layers[layer_index]
    if layer_index in GDN_LAYERS:
        return layer.linear_attn, "out_proj"
    return layer.self_attn, "o_proj"


def _module_from_factor(
    factor: Mapping[str, Any],
    *,
    dense_module: nn.Module,
) -> Qwen35PrivateAGOutput:
    weight = dense_module.weight
    bias = getattr(dense_module, "bias", None)
    return Qwen35PrivateAGOutput(
        factor["private_encoders"].to(device=weight.device, dtype=weight.dtype),
        factor["joint_decoder_weight"].to(device=weight.device, dtype=weight.dtype),
        bias=None if bias is None else bias.detach().to(weight.device, weight.dtype),
    )


def _install_schedule(
    layers: Sequence[nn.Module],
    *,
    dense_modules: Mapping[int, nn.Module],
    banks: Mapping[int, Mapping[int, Mapping[str, Any]]],
    schedule: Sequence[int],
) -> None:
    if len(schedule) != NUM_LAYERS:
        raise ValueError("rank schedule must cover 32 layers")
    replacements = {}
    for layer_index, rank in enumerate(schedule):
        replacements[layer_index] = _module_from_factor(
            banks[int(rank)][layer_index],
            dense_module=dense_modules[layer_index],
        )
    for layer_index, replacement in replacements.items():
        parent, attribute = _projection_parent(layers, layer_index)
        setattr(parent, attribute, replacement)
    gc.collect()
    torch.cuda.empty_cache()


@dataclass(frozen=True)
class TeacherBatch:
    input_ids: Tensor
    logits: Tensor
    logsumexp: Tensor
    nll: float


@torch.inference_mode()
def _capture_teacher(
    model: nn.Module,
    samples: Tensor,
    *,
    device: str,
    vocab_chunk_size: int,
) -> tuple[TeacherBatch, ...]:
    result = []
    for index, input_ids in enumerate(samples, start=1):
        batch = input_ids.unsqueeze(0).to(device=device, dtype=torch.long)
        logits = model(
            input_ids=batch,
            attention_mask=torch.ones_like(batch),
            use_cache=False,
        ).logits[:, :-1]
        targets = batch[:, 1:]
        nll = float(_chunked_cross_entropy(logits, targets).mean())
        lse = logits_logsumexp(logits, vocab_chunk_size=vocab_chunk_size)
        result.append(
            TeacherBatch(
                input_ids=batch.cpu(),
                logits=logits.cpu(),
                logsumexp=lse.cpu(),
                nll=nll,
            )
        )
        print(f"[Q35 Global-KL] teacher={index}/{len(samples)} nll={nll:.8f}", flush=True)
        del logits, lse
    return tuple(result)


def _value_summary(values: Sequence[float]) -> dict[str, Any]:
    checked = tuple(float(value) for value in values)
    if not checked or any(not math.isfinite(value) for value in checked):
        raise ValueError("metric values must be non-empty and finite")
    return {
        "mean": statistics.fmean(checked),
        "paired_standard_error": paired_standard_error(checked),
        "values": list(checked),
    }


@torch.inference_mode()
def _evaluate_teacher(
    model: nn.Module,
    teacher: Sequence[TeacherBatch],
    *,
    device: str,
    vocab_chunk_size: int,
    label: str,
) -> dict[str, Any]:
    kl_values = []
    nll_values = []
    for index, batch in enumerate(teacher, start=1):
        input_ids = batch.input_ids.to(device)
        logits = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        ).logits[:, :-1]
        kl_sum, tokens = teacher_kl_sum(
            logits,
            batch.logits,
            teacher_logsumexp=batch.logsumexp,
            vocab_chunk_size=vocab_chunk_size,
        )
        kl_values.append(kl_sum / tokens)
        nll_values.append(
            float(_chunked_cross_entropy(logits, input_ids[:, 1:]).mean())
        )
        print(
            f"[Q35 Global-KL] variant={label} window={index}/{len(teacher)} "
            f"kl={kl_values[-1]:.8g}",
            flush=True,
        )
        del logits
    nll = _value_summary(nll_values)
    nll["perplexity"] = math.exp(nll["mean"])
    return {"terminal_kl": _value_summary(kl_values), "nll": nll}


def _paired_delta(candidate: Mapping[str, Any], anchor: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for metric in ("terminal_kl", "nll"):
        values = [
            float(left) - float(right)
            for left, right in zip(
                candidate[metric]["values"],
                anchor[metric]["values"],
                strict=True,
            )
        ]
        summary = _value_summary(values)
        summary["one_standard_error_ucb"] = (
            summary["mean"] + summary["paired_standard_error"]
        )
        result[metric] = summary
    return result


def _assigned_layers(index: int, count: int) -> tuple[int, ...]:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("profile shard index/count are invalid")
    return tuple(range(index, NUM_LAYERS, count))


def _configuration(
    args: argparse.Namespace,
    *,
    ranks: Sequence[int],
    paths: Mapping[int, Mapping[str, Path]],
    windows_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "model_commit": Path(args.model_path).expanduser().resolve().name,
        "windows_sha256": windows_manifest["artifact"]["sha256"],
        "profile_offset": args.profile_offset,
        "profile_windows": args.profile_windows,
        "confirmation_offset": args.confirmation_offset,
        "confirmation_windows": args.confirmation_windows,
        "candidate_ranks": list(map(int, ranks)),
        "anchor_rank": args.anchor_rank,
        "target_layer_rank_sum": NUM_LAYERS * args.anchor_rank,
        "factor_paths": {
            str(rank): {kind: str(path) for kind, path in pair.items()}
            for rank, pair in paths.items()
        },
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "vocab_chunk_size": args.vocab_chunk_size,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_model(args: argparse.Namespace) -> nn.Module:
    from transformers import AutoModelForMultimodalLM

    return AutoModelForMultimodalLM.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        dtype=torch.float16,
        device_map={"": args.device},
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).eval()


@torch.inference_mode()
def _profile(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("Qwen3.5 Global-KL currently requires batch size 1")
    torch.set_num_threads(args.torch_num_threads)
    ranks = _parse_ranks(args.candidate_ranks, anchor_rank=args.anchor_rank)
    assigned = _assigned_layers(args.profile_shard_index, args.profile_shard_count)
    banks, paths = _load_banks(
        args,
        ranks=ranks,
        required_layers=assigned,
        include_anchor_all_layers=True,
    )
    samples, _, windows_manifest = _load_window_split(
        Path(args.windows).expanduser().resolve(),
        sample_offset=args.profile_offset,
        num_samples=args.profile_windows,
    )
    configuration = _configuration(
        args,
        ranks=ranks,
        paths=paths,
        windows_manifest=windows_manifest,
    )
    profile_dir = Path(args.profile_dir).expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    output_path = profile_dir / f"shard_{args.profile_shard_index:02d}.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.perf_counter()
    model = _load_model(args)
    layers = qwen35_decoder_layers(model)
    dense_modules = {
        layer_index: getattr(*_projection_parent(layers, layer_index))
        for layer_index in range(NUM_LAYERS)
    }
    teacher = _capture_teacher(
        model,
        samples,
        device=args.device,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    anchor_schedule = [args.anchor_rank] * NUM_LAYERS
    _install_schedule(
        layers,
        dense_modules=dense_modules,
        banks=banks,
        schedule=anchor_schedule,
    )
    anchor_metrics = _evaluate_teacher(
        model,
        teacher,
        device=args.device,
        vocab_chunk_size=args.vocab_chunk_size,
        label="uniform_r192_anchor",
    )
    records = []
    intervention_ranks = [rank for rank in ranks if rank != args.anchor_rank]
    for layer_ordinal, layer_index in enumerate(assigned, start=1):
        parent, attribute = _projection_parent(layers, layer_index)
        anchor_module = getattr(parent, attribute)
        for rank_ordinal, rank in enumerate(intervention_ranks, start=1):
            candidate = _module_from_factor(
                banks[rank][layer_index],
                dense_module=dense_modules[layer_index],
            )
            setattr(parent, attribute, candidate)
            metrics = _evaluate_teacher(
                model,
                teacher,
                device=args.device,
                vocab_chunk_size=args.vocab_chunk_size,
                label=f"l{layer_index}_r{rank}",
            )
            setattr(parent, attribute, anchor_module)
            delta = _paired_delta(metrics, anchor_metrics)
            records.append(
                {
                    "layer": layer_index,
                    "block_type": (
                        "linear_attention" if layer_index in GDN_LAYERS else "full_attention"
                    ),
                    "anchor_rank": args.anchor_rank,
                    "candidate_rank": rank,
                    "terminal_kl": metrics["terminal_kl"],
                    "nll": metrics["nll"],
                    "terminal_kl_delta": delta["terminal_kl"],
                    "nll_delta": delta["nll"],
                }
            )
            print(
                f"[Q35 Global-KL shard {args.profile_shard_index}] "
                f"layer={layer_index} {layer_ordinal}/{len(assigned)} "
                f"rank={rank} {rank_ordinal}/{len(intervention_ranks)} "
                f"dKL={delta['terminal_kl']['mean']:.8g}",
                flush=True,
            )
            del candidate, metrics
            torch.cuda.empty_cache()
    payload = {
        "format": PROFILE_FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile_shard_index": args.profile_shard_index,
        "profile_shard_count": args.profile_shard_count,
        "assigned_layers": list(assigned),
        "configuration": configuration,
        "uniform_anchor": anchor_metrics,
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
    }
    _atomic_json(output_path, payload)
    print(f"[Q35 Global-KL] saved {output_path}", flush=True)


def _merge_shards(
    profile_dir: Path,
    *,
    count: int,
    ranks: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    shards = []
    reference = None
    for index in range(count):
        path = profile_dir / f"shard_{index:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != PROFILE_FORMAT or payload.get("status") != "complete":
            raise ValueError(f"incomplete profile shard: {path}")
        expected = list(_assigned_layers(index, count))
        if payload.get("assigned_layers") != expected:
            raise ValueError(f"profile shard layer assignment changed: {path}")
        if reference is None:
            reference = payload["configuration"]
        elif payload["configuration"] != reference:
            raise ValueError("profile shard configurations differ")
        shards.append(payload)
    records = sorted(
        [row for shard in shards for row in shard["records"]],
        key=lambda row: (row["layer"], row["candidate_rank"]),
    )
    expected_records = NUM_LAYERS * (len(ranks) - 1)
    keys = {(int(row["layer"]), int(row["candidate_rank"])) for row in records}
    if len(records) != expected_records or len(keys) != expected_records:
        raise ValueError("profile shards do not contain every layer/rank intervention")
    assert reference is not None
    return records, shards, reference


def _allocate(
    records: Sequence[Mapping[str, Any]],
    *,
    ranks: Sequence[int],
    anchor_rank: int,
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
                else float(indexed[(layer, rank)]["terminal_kl_delta"]["mean"])
            )
            layer_options.append(
                MetricRankOption(
                    option_id=f"layer_{layer:02d}.r{rank}",
                    source_family="qwen35_layer_terminal_kl",
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
        "predicted_additive_kl_delta": allocation.total_cost,
        "changed_layers": allocation.changed_coordinates,
        "total_absolute_rank_deviation": allocation.total_absolute_rank_deviation,
        "contributions": [
            {
                "layer": layer,
                "rank": option.rank,
                "predicted_kl_delta": option.scalar_cost,
            }
            for layer, option in enumerate(allocation.selected_options)
        ],
    }


def _accounting(schedule: Sequence[int], *, anchor_rank: int) -> dict[str, Any]:
    total = sum(map(int, schedule))
    if len(schedule) != NUM_LAYERS or total != NUM_LAYERS * anchor_rank:
        raise ValueError("schedule violates the exact average-rank budget")
    histogram = {str(rank): list(schedule).count(rank) for rank in sorted(set(schedule))}
    relative_to_dense_allreduce = total / (NUM_LAYERS * 2 * LOCAL_WIDTH)
    return {
        "layer_ranks": list(map(int, schedule)),
        "rank_histogram": histogram,
        "layer_rank_sum": total,
        "average_local_rank": total / NUM_LAYERS,
        "gathered_width_by_layer": [TP_SIZE * int(rank) for rank in schedule],
        "communication_fraction_of_dense_allreduce": relative_to_dense_allreduce,
        "communication_reduction_fraction": 1.0 - relative_to_dense_allreduce,
    }


def _selected_payload(
    *,
    kind: str,
    schedule: Sequence[int],
    banks: Mapping[int, Mapping[int, Mapping[str, Any]]],
    model_path: Path,
) -> dict[str, Any]:
    selected_layers = _expected_layers(kind)
    return {
        "format": GDN_FACTOR_FORMAT if kind == "gdn" else FULL_FACTOR_FORMAT,
        "schema_version": 1,
        "model_path": str(model_path),
        "tp_size": TP_SIZE,
        "rank_schedule": {str(layer): int(schedule[layer]) for layer in selected_layers},
        "objective": "global_kl_selected_decoder_closed_c1_als",
        "recurrent_state_compressed": False,
        "attention_cache_compressed": False,
        "layers": [banks[int(schedule[layer])][layer] for layer in selected_layers],
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _summary(result: Mapping[str, Any]) -> str:
    schedule = result["selection"]["selected_schedule"]
    confirmation = result["confirmation"]
    uniform = confirmation["uniform_r192"]
    ragged = confirmation["mean_dp_ragged"]
    deltas = _paired_delta(ragged, uniform)
    kl_delta = deltas["terminal_kl"]
    nll_delta = deltas["nll"]
    kl_wins = sum(value < 0 for value in kl_delta["values"])
    nll_wins = sum(value < 0 for value in nll_delta["values"])
    lines = [
        "# Qwen3.5-9B C1 layer Global-KL allocation",
        "",
        f"- Average local rank: `{result['accounting']['average_local_rank']}`",
        f"- Communication reduction: `{100 * result['accounting']['communication_reduction_fraction']:.2f}%`",
        f"- Changed layers: `{result['selection']['changed_layers']}`",
        f"- Predicted additive profile delta-KL: `{result['selection']['predicted_additive_kl_delta']:.8g}`",
        "",
        "## Independent confirmation",
        "",
        f"- Windows: `{confirmation['windows']}` starting at offset `{confirmation['offset']}`; not used for schedule selection.",
        f"- Uniform-r192 KL: `{uniform['terminal_kl']['mean']:.8g}`",
        f"- Ragged KL: `{ragged['terminal_kl']['mean']:.8g}`",
        f"- Paired delta-KL: `{kl_delta['mean']:.8g} +/- {kl_delta['paired_standard_error']:.8g}`; improved on `{kl_wins}/{confirmation['windows']}` windows.",
        f"- Uniform-r192 PPL: `{uniform['nll']['perplexity']:.8g}`",
        f"- Ragged PPL: `{ragged['nll']['perplexity']:.8g}`",
        f"- Paired delta-NLL: `{nll_delta['mean']:.8g} +/- {nll_delta['paired_standard_error']:.8g}`; improved on `{nll_wins}/{confirmation['windows']}` windows.",
        "",
        "## Selected artifacts",
        "",
        f"- GDN: `{result['selected_factor_artifacts']['gdn']}`",
        f"- Full attention: `{result['selected_factor_artifacts']['full_attention']}`",
        "",
        "## Layer schedule",
        "",
        "| Layer | Type | Rank |",
        "|---:|:---|---:|",
    ]
    for layer, rank in enumerate(schedule):
        kind = "GDN" if layer in GDN_LAYERS else "Full"
        lines.append(f"| {layer} | {kind} | {rank} |")
    lines.extend(["", "## Command", "", f"`{result['command']}`", ""])
    return "\n".join(lines)


@torch.inference_mode()
def _finalize(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("Qwen3.5 Global-KL currently requires batch size 1")
    torch.set_num_threads(args.torch_num_threads)
    ranks = _parse_ranks(args.candidate_ranks, anchor_rank=args.anchor_rank)
    profile_dir = Path(args.profile_dir).expanduser().resolve()
    records, shards, profile_configuration = _merge_shards(
        profile_dir,
        count=args.profile_shard_count,
        ranks=ranks,
    )
    schedule, allocation = _allocate(
        records,
        ranks=ranks,
        anchor_rank=args.anchor_rank,
    )
    accounting = _accounting(schedule, anchor_rank=args.anchor_rank)
    banks, paths = _load_banks(
        args,
        ranks=ranks,
        required_layers=tuple(range(NUM_LAYERS)),
        include_anchor_all_layers=True,
    )
    samples, records_windows, windows_manifest = _load_window_split(
        Path(args.windows).expanduser().resolve(),
        sample_offset=args.confirmation_offset,
        num_samples=args.confirmation_windows,
    )
    expected_configuration = _configuration(
        args,
        ranks=ranks,
        paths=paths,
        windows_manifest=windows_manifest,
    )
    if profile_configuration != expected_configuration:
        raise ValueError("finalize configuration differs from profile shards")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    started = time.perf_counter()
    model = _load_model(args)
    layers = qwen35_decoder_layers(model)
    dense_modules = {
        layer_index: getattr(*_projection_parent(layers, layer_index))
        for layer_index in range(NUM_LAYERS)
    }
    teacher = _capture_teacher(
        model,
        samples,
        device=args.device,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    uniform = [args.anchor_rank] * NUM_LAYERS
    confirmation = {}
    for label, candidate_schedule in (
        ("uniform_r192", uniform),
        ("mean_dp_ragged", schedule),
    ):
        _install_schedule(
            layers,
            dense_modules=dense_modules,
            banks=banks,
            schedule=candidate_schedule,
        )
        confirmation[label] = _evaluate_teacher(
            model,
            teacher,
            device=args.device,
            vocab_chunk_size=args.vocab_chunk_size,
            label=label,
        )
    output_dir.mkdir(parents=True)
    model_path = Path(args.model_path).expanduser().resolve()
    gdn_path = output_dir / "gdn_selected_factors.pt"
    full_path = output_dir / "full_selected_factors.pt"
    _atomic_torch_save(
        gdn_path,
        _selected_payload(
            kind="gdn", schedule=schedule, banks=banks, model_path=model_path
        ),
    )
    _atomic_torch_save(
        full_path,
        _selected_payload(
            kind="full", schedule=schedule, banks=banks, model_path=model_path
        ),
    )
    result = {
        "format": RESULT_FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model_path": str(model_path),
        "profile": {
            "windows": args.profile_windows,
            "offset": args.profile_offset,
            "records": records,
            "shards": [
                {
                    "index": shard["profile_shard_index"],
                    "assigned_layers": shard["assigned_layers"],
                    "elapsed_seconds": shard["elapsed_seconds"],
                }
                for shard in shards
            ],
        },
        "selection": {
            "objective": "minimum additive mean terminal KL delta",
            "constraint": "exact sum of 32 per-layer local ranks",
            "target_layer_rank_sum": NUM_LAYERS * args.anchor_rank,
            "candidate_ranks": list(ranks),
            "selected_schedule": list(schedule),
            **allocation,
        },
        "accounting": accounting,
        "confirmation": {
            "windows": args.confirmation_windows,
            "offset": args.confirmation_offset,
            "sample_indices": [
                int(record["sample_index"]) for record in records_windows
            ],
            "not_used_for_schedule_selection": True,
            **confirmation,
        },
        "selected_factor_artifacts": {
            "gdn": str(gdn_path),
            "full_attention": str(full_path),
        },
        "environment": {
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
    }
    _atomic_json(output_dir / "result.json", result)
    (output_dir / "summary.md").write_text(_summary(result), encoding="utf-8")
    print(f"[Q35 Global-KL] wrote {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.stage == "profile":
        _profile(args)
    else:
        _finalize(args)


if __name__ == "__main__":
    main()
