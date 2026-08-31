#!/usr/bin/env python3
"""Evaluate Qwen3.5 post-gate Top-K on WikiText-2 and local MCQ tasks.

The evaluator compares a dense checkpoint with full-attention-only,
GDN-only, and combined interventions.  Top-K is applied immediately before
the corresponding output projection, after the checkpoint's exact sigmoid or
gated-RMSNorm/SiLU path.  Model weights and recurrent states are untouched.

This is an end-to-end quality oracle.  The current hook reconstructs a dense
zero-filled tensor and calls the original dense linear layer, so reported
wall time is not a sparse-kernel or collective benchmark.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import save_file
import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen35_postgate_topk import (  # noqa: E402
    Qwen35PostGateTopKRuntime,
)
from evaluation.analyze_qwen35_gated_attention_sparsity import (  # noqa: E402
    _git_commit,
    _installed_version,
)
from scripts.eval_qwen35_mlp_tp_cp_mcq import (  # noqa: E402
    _evaluate_task,
    _named_path,
)
from scripts.eval_qwen35_projected_gdn_nll import (  # noqa: E402
    _dtype,
    _evaluate,
)


FORMAT = "basisserve.qwen35.postgate_topk_crossdomain.v1"
INTERVENTIONS = ("full", "gdn", "both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wikitext-file")
    parser.add_argument(
        "--mcq-file",
        action="append",
        default=[],
        help="repeatable NAME=PATH JSONL or a bare JSONL path",
    )
    parser.add_argument("--interventions", default=",".join(INTERVENTIONS))
    parser.add_argument("--keep-ratios", default="0.5,0.75")
    parser.add_argument(
        "--mixed-keep-ratio",
        action="append",
        default=[],
        metavar="FULL:GDN",
        help=(
            "repeatable asymmetric full-attention:GDN retention pair, for example "
            "0.75:0.5"
        ),
    )
    parser.add_argument(
        "--selection-scope",
        choices=("global", "source_local"),
        default="source_local",
    )
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--ppl-sequence-length", type=int, default=512)
    parser.add_argument("--ppl-max-tokens", type=int, default=8192)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument("--mcq-max-examples", type=int, default=100)
    parser.add_argument("--choice-prefix", default=" ")
    parser.add_argument("--normalize", choices=("none", "length"), default="length")
    parser.add_argument("--no-add-special-tokens", action="store_true")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the variant plan without loading the model",
    )
    return parser.parse_args()


def _csv_values(raw: str, *, allowed: set[str]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(piece.strip() for piece in raw.split(",") if piece.strip()))
    unknown = sorted(set(values) - allowed)
    if not values or unknown:
        raise ValueError(f"invalid values {unknown}; allowed={sorted(allowed)}")
    return values


def _ratios(raw: str) -> tuple[float, ...]:
    values = tuple(sorted(set(float(piece.strip()) for piece in raw.split(",") if piece.strip())))
    if not values or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in values):
        raise ValueError("keep ratios must be finite values in (0,1)")
    return values


def _mixed_ratios(raw_values: Sequence[str]) -> tuple[tuple[float, float], ...]:
    values: list[tuple[float, float]] = []
    for raw in raw_values:
        pieces = tuple(piece.strip() for piece in raw.split(":"))
        if len(pieces) != 2 or not all(pieces):
            raise ValueError(
                f"mixed keep ratio {raw!r} must have FULL:GDN form"
            )
        pair = (float(pieces[0]), float(pieces[1]))
        if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in pair):
            raise ValueError("mixed keep ratios must be finite values in (0,1)")
        if pair[0] == pair[1]:
            raise ValueError(
                f"mixed keep ratio {raw!r} is symmetric; use --keep-ratios"
            )
        if pair not in values:
            values.append(pair)
    return tuple(values)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _ratio_tag(ratio: float) -> str:
    return f"{ratio * 100:.6g}".replace(".", "p")


def _variant_name(intervention: str, ratio: float, selection_scope: str) -> str:
    return f"{intervention}_topk_{_ratio_tag(ratio)}_{selection_scope}"


def _mixed_variant_name(
    full_ratio: float,
    gdn_ratio: float,
    selection_scope: str,
) -> str:
    return (
        f"full_topk_{_ratio_tag(full_ratio)}_"
        f"gdn_topk_{_ratio_tag(gdn_ratio)}_{selection_scope}"
    )


def _variant_plan(
    interventions: Sequence[str],
    ratios: Sequence[float],
    selection_scope: str,
    mixed_ratios: Sequence[tuple[float, float]] = (),
) -> tuple[dict[str, Any], ...]:
    standard = tuple(
        {
            "name": _variant_name(intervention, ratio, selection_scope),
            "intervention": intervention,
            "keep_ratio": ratio,
            "selection_scope": selection_scope,
        }
        for intervention in interventions
        for ratio in ratios
    )
    mixed = tuple(
        {
            "name": _mixed_variant_name(full_ratio, gdn_ratio, selection_scope),
            "intervention": "mixed",
            "family_keep_ratios": {
                "full": full_ratio,
                "gdn": gdn_ratio,
            },
            "selection_scope": selection_scope,
        }
        for full_ratio, gdn_ratio in mixed_ratios
    )
    return standard + mixed


def _runtime_specs(spec: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    if spec["intervention"] != "mixed":
        return ((str(spec["intervention"]), float(spec["keep_ratio"])),)
    ratios = spec["family_keep_ratios"]
    return (("full", float(ratios["full"])), ("gdn", float(ratios["gdn"])))


def _wikitext_samples(
    tokenizer: Any,
    path: Path,
    *,
    sequence_length: int,
    maximum_tokens: int,
) -> tuple[tuple[Tensor, ...], dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    token_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].squeeze(0)
    available = int(token_ids.numel())
    used = min(available, maximum_tokens)
    used -= used % sequence_length
    if used < sequence_length:
        raise ValueError(
            f"WikiText file supplies only {available} tokens, below one complete window"
        )
    token_ids = token_ids[:used]
    samples = tuple(token_ids[start : start + sequence_length].contiguous() for start in range(0, used, sequence_length))
    return samples, {
        "path": str(path),
        "sha256": _file_sha256(path),
        "available_tokens": available,
        "used_input_tokens": used,
        "sequence_length": sequence_length,
        "windows": len(samples),
        "evaluated_next_tokens": len(samples) * (sequence_length - 1),
    }


def _serializable_ppl(
    result: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"token_nlls", "top1_tokens"}
    }
    if baseline is not None:
        differences = result["token_nlls"] - baseline["token_nlls"]
        window_differences = [
            candidate - teacher
            for teacher, candidate in zip(
                baseline["window_nlls"], result["window_nlls"], strict=True
            )
        ]
        payload.update(
            {
                "delta_mean_nll": float(result["mean_nll"] - baseline["mean_nll"]),
                "paired_window_standard_error": (
                    statistics.stdev(window_differences) / math.sqrt(len(window_differences))
                    if len(window_differences) > 1
                    else 0.0
                ),
                "target_nll_mae": float(differences.abs().mean()),
                "target_nll_rmse": float(differences.square().mean().sqrt()),
                "target_nll_max_abs": float(differences.abs().max()),
                "top1_agreement": float(
                    (result["top1_tokens"] == baseline["top1_tokens"]).float().mean()
                ),
            }
        )
    return payload


def _mcq_paired_metrics(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_rows = candidate["results"]
    baseline_rows = baseline["results"]
    if len(candidate_rows) != len(baseline_rows):
        raise ValueError("candidate and dense MCQ result lengths differ")
    agreements = 0
    score_differences: list[float] = []
    margin_differences: list[float] = []
    dense_to_wrong = 0
    candidate_rescue = 0
    for dense_row, candidate_row in zip(baseline_rows, candidate_rows, strict=True):
        if int(dense_row["idx"]) != int(candidate_row["idx"]):
            raise ValueError("candidate and dense MCQ row indices differ")
        agreements += int(int(dense_row["pred"]) == int(candidate_row["pred"]))
        dense_to_wrong += int(bool(dense_row["correct"]) and not bool(candidate_row["correct"]))
        candidate_rescue += int(not bool(dense_row["correct"]) and bool(candidate_row["correct"]))
        dense_scores = tuple(map(float, dense_row["scores"]))
        candidate_scores = tuple(map(float, candidate_row["scores"]))
        if len(dense_scores) != len(candidate_scores):
            raise ValueError("candidate and dense MCQ choice counts differ")
        score_differences.extend(c - d for d, c in zip(dense_scores, candidate_scores, strict=True))
        if len(dense_scores) >= 2:
            dense_order = sorted(dense_scores, reverse=True)
            candidate_order = sorted(candidate_scores, reverse=True)
            margin_differences.append(
                (candidate_order[0] - candidate_order[1])
                - (dense_order[0] - dense_order[1])
            )
    score_tensor = torch.tensor(score_differences, dtype=torch.float64)
    margin_tensor = torch.tensor(margin_differences, dtype=torch.float64)
    answered = int(candidate["answered"])
    return {
        "accuracy_delta": float(candidate["accuracy"] - baseline["accuracy"]),
        "prediction_agreement": agreements / max(answered, 1),
        "dense_correct_to_candidate_wrong": dense_to_wrong,
        "candidate_rescues": candidate_rescue,
        "choice_score_delta_mean": float(score_tensor.mean()) if score_tensor.numel() else 0.0,
        "choice_score_delta_mae": float(score_tensor.abs().mean()) if score_tensor.numel() else 0.0,
        "choice_score_delta_rmse": float(score_tensor.square().mean().sqrt()) if score_tensor.numel() else 0.0,
        "choice_margin_delta_mean": float(margin_tensor.mean()) if margin_tensor.numel() else 0.0,
    }


def _aggregate_mcq(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [task for task in tasks if task["accuracy"] is not None]
    if not valid:
        raise ValueError("MCQ suite contains no answered task")
    answered = sum(int(task["answered"]) for task in valid)
    correct = sum(int(task["correct"]) for task in valid)
    return {
        "macro_accuracy": sum(float(task["accuracy"]) for task in valid) / len(valid),
        "micro_accuracy": correct / max(answered, 1),
        "answered": answered,
        "correct": correct,
        "num_tasks": len(valid),
    }


def _profile_key(dataset: str, variant: str, tensor_name: str) -> str:
    safe_dataset = dataset.replace("/", "_").replace(" ", "_")
    return f"{safe_dataset}__{variant}__{tensor_name}"


def _capture_profile(
    runtime: Qwen35PostGateTopKRuntime,
    *,
    dataset: str,
    variant: str,
    frequency_bank: dict[str, Tensor],
) -> list[dict[str, Any]]:
    summaries, tensors = runtime.profile_snapshot()
    for name, value in tensors.items():
        frequency_bank[_profile_key(dataset, variant, name)] = value
    return [{"dataset": dataset, "variant": variant, **row} for row in summaries]


def _wire_accounting(record: Any) -> dict[str, Any]:
    realized_ratio = float(record.realized_ratio)
    payload: dict[str, Any] = {
        "values_only_payload_ratio_vs_dense_bf16": realized_ratio,
        "uint16_index_payload_ratio_vs_dense_bf16": 2.0 * realized_ratio,
        "bitmask_payload_ratio_vs_dense_bf16": realized_ratio + 1.0 / 16.0,
    }
    if record.kept_per_source is not None:
        source_width = int(record.width) // int(record.tp_size)
        mask_bytes = (source_width + 7) // 8
        dense_bytes = 2 * source_width
        value_bytes = 2 * int(record.kept_per_source)
        payload.update(
            {
                "source_width": source_width,
                "dense_bf16_bytes_per_source": dense_bytes,
                "bitmask_bytes_per_source": mask_bytes,
                "selected_bf16_value_bytes_per_source": value_bytes,
                "bitmask_packet_bytes_per_source": mask_bytes + value_bytes,
            }
        )
    return payload


def _family_key(layer_kind: str) -> str:
    if layer_kind == "full_attention":
        return "full"
    if layer_kind == "gdn":
        return "gdn"
    raise ValueError(f"unknown layer kind {layer_kind!r}")


def _variant_metadata(
    spec: Mapping[str, Any],
    records: Sequence[Any],
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"variant {spec['name']} installed no hooks")
    grouped: dict[str, list[Any]] = {}
    for record in records:
        grouped.setdefault(_family_key(str(record.layer_kind)), []).append(record)
    per_family: dict[str, Any] = {}
    for family in ("full", "gdn"):
        family_records = grouped.get(family)
        if not family_records:
            continue
        signatures = {
            (
                record.width,
                record.keep_ratio,
                record.kept_per_vector,
                record.kept_per_source,
                record.realized_ratio,
            )
            for record in family_records
        }
        if len(signatures) != 1:
            raise ValueError(
                f"variant {spec['name']} has inconsistent {family} hook shapes"
            )
        representative = family_records[0]
        per_family[family] = {
            "keep_ratio": float(representative.keep_ratio),
            "realized_ratio": float(representative.realized_ratio),
            "layer_count": len(family_records),
            "layers": [record.layer_index for record in family_records],
            "width": int(representative.width),
            "kept_per_vector": int(representative.kept_per_vector),
            "kept_per_source": representative.kept_per_source,
            "wire_accounting": _wire_accounting(representative),
        }
    packet_bytes = [
        per_family[_family_key(str(record.layer_kind))]["wire_accounting"].get(
            "bitmask_packet_bytes_per_source"
        )
        for record in records
    ]
    metadata: dict[str, Any] = {
        **spec,
        "layer_count": len(records),
        "layers": [record.layer_index for record in records],
        "layer_kinds": sorted({record.layer_kind for record in records}),
        "per_family": per_family,
        "average_bitmask_payload_ratio_vs_dense_bf16": statistics.fmean(
            float(
                per_family[_family_key(str(record.layer_kind))]["wire_accounting"]
                ["bitmask_payload_ratio_vs_dense_bf16"]
            )
            for record in records
        ),
    }
    if all(value is not None for value in packet_bytes):
        metadata["average_bitmask_packet_bytes_per_source_per_layer"] = (
            statistics.fmean(float(value) for value in packet_bytes)
        )
    realized = {float(record.realized_ratio) for record in records}
    if len(realized) == 1:
        representative = records[0]
        metadata.update(
            {
                "width": int(representative.width),
                "kept_per_vector": int(representative.kept_per_vector),
                "kept_per_source": representative.kept_per_source,
                "realized_ratio": float(representative.realized_ratio),
                "wire_accounting": _wire_accounting(representative),
            }
        )
    return metadata


def _family_k_label(metadata: Mapping[str, Any]) -> str:
    labels = []
    for family, prefix in (("full", "F"), ("gdn", "G")):
        if family in metadata["per_family"]:
            labels.append(
                f"{prefix}{metadata['per_family'][family]['kept_per_source']}"
            )
    return "/".join(labels)


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3.5 post-gate Top-K cross-domain quality",
        "",
        (
            "Exact checkpoint gates and states are retained. Top-K is applied to the "
            "post-gate wire immediately before the dense output projection. The hook "
            "is a quality oracle and is not a sparse-kernel timing result. Packet bytes "
            "assume fixed-cardinality BF16 values plus one source-local bitmask."
        ),
        "",
    ]
    if payload.get("ppl"):
        lines.extend(
            [
                "## WikiText-2",
                "",
                "| Variant | K/source | Packet B/source/layer | PPL | Delta NLL | Paired SE | Top-1 agreement |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        dense = payload["ppl"]["dense"]
        lines.append(f"| dense | 512 | 1024 | {dense['perplexity']:.7g} | 0 | 0 | 1 |")
        metadata_by_name = {row["name"]: row for row in payload["variants"]}
        for name, row in payload["ppl"]["candidates"].items():
            metadata = metadata_by_name[name]
            packet_bytes = metadata.get(
                "average_bitmask_packet_bytes_per_source_per_layer"
            )
            packet_label = (
                "n/a" if packet_bytes is None else f"{float(packet_bytes):.0f}"
            )
            lines.append(
                f"| {name} | {_family_k_label(metadata)} | {packet_label} | "
                f"{row['perplexity']:.7g} | {row['delta_mean_nll']:+.7g} | "
                f"{row['paired_window_standard_error']:.7g} | "
                f"{row['top1_agreement']:.6f} |"
            )
        lines.append("")
    if payload.get("mcq"):
        task_names = [task["task"] for task in payload["mcq"]["dense"]["tasks"]]
        lines.extend(
            [
                "## MCQ",
                "",
                "| Variant | Macro | " + " | ".join(task_names) + " |",
                "|---|---:|" + "---:|" * len(task_names),
            ]
        )
        dense = payload["mcq"]["dense"]
        dense_by_name = {task["task"]: task for task in dense["tasks"]}
        lines.append(
            "| dense | "
            f"{dense['aggregate']['macro_accuracy']:.4f} | "
            + " | ".join(f"{dense_by_name[name]['accuracy']:.4f}" for name in task_names)
            + " |"
        )
        for name, suite in payload["mcq"]["candidates"].items():
            by_name = {task["task"]: task for task in suite["tasks"]}
            lines.append(
                f"| {name} | {suite['aggregate']['macro_accuracy']:.4f} | "
                + " | ".join(f"{by_name[task]['accuracy']:.4f}" for task in task_names)
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    interventions = _csv_values(args.interventions, allowed=set(INTERVENTIONS))
    ratios = _ratios(args.keep_ratios)
    mixed_ratios = _mixed_ratios(args.mixed_keep_ratio)
    variants = _variant_plan(
        interventions,
        ratios,
        args.selection_scope,
        mixed_ratios,
    )
    if args.tp_size <= 0 or args.torch_num_threads <= 0:
        raise ValueError("TP size and thread count must be positive")
    if args.ppl_sequence_length <= 1 or args.ppl_max_tokens < args.ppl_sequence_length:
        raise ValueError("invalid WikiText token/window controls")
    if args.ppl_batch_size <= 0 or args.mcq_max_examples <= 0:
        raise ValueError("batch size and MCQ example count must be positive")

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
    wikitext_path = (
        None if args.wikitext_file is None else Path(args.wikitext_file).expanduser().resolve()
    )
    if wikitext_path is not None and not wikitext_path.is_file():
        raise FileNotFoundError(wikitext_path)
    tasks = tuple(_named_path(raw) for raw in args.mcq_file)
    if len({name for name, _ in tasks}) != len(tasks):
        raise ValueError("MCQ task names must be unique")
    if wikitext_path is None and not tasks:
        raise ValueError("at least one WikiText or MCQ input is required")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite cross-domain result: {output_dir}")

    plan = {
        "model": str(model_path),
        "wikitext": None if wikitext_path is None else str(wikitext_path),
        "mcq": [{"name": name, "path": str(path)} for name, path in tasks],
        "variants": [{"name": "dense"}, *variants],
        "tp_size": args.tp_size,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        return

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("cross-domain Qwen3.5 evaluation requires CUDA")
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))

    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=args.local_files_only,
    )
    samples: tuple[Tensor, ...] | None = None
    wikitext_metadata: dict[str, Any] | None = None
    if wikitext_path is not None:
        samples, wikitext_metadata = _wikitext_samples(
            tokenizer,
            wikitext_path,
            sequence_length=args.ppl_sequence_length,
            maximum_tokens=args.ppl_max_tokens,
        )
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_path),
        dtype=_dtype(args.dtype),
        device_map={"": args.device},
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    ).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    dense_ppl: dict[str, Any] | None = None
    dense_ppl_raw: dict[str, Any] | None = None
    if samples is not None:
        dense_ppl_raw = _evaluate(
            model,
            samples,
            batch_size=args.ppl_batch_size,
            device=args.device,
            label="dense",
        )
        dense_ppl = _serializable_ppl(dense_ppl_raw, baseline=None)

    dense_mcq_tasks: list[dict[str, Any]] = []
    for name, path in tasks:
        dense_mcq_tasks.append(
            _evaluate_task(
                model,
                tokenizer,
                name=name,
                path=path,
                max_examples=args.mcq_max_examples,
                choice_prefix=args.choice_prefix,
                normalize=args.normalize,
                add_special_tokens=not args.no_add_special_tokens,
                device=torch.device(args.device),
                variant="dense",
            )
        )
    dense_mcq = (
        None
        if not dense_mcq_tasks
        else {"aggregate": _aggregate_mcq(dense_mcq_tasks), "tasks": dense_mcq_tasks}
    )

    candidate_ppl: dict[str, Any] = {}
    candidate_mcq: dict[str, Any] = {}
    profiles: list[dict[str, Any]] = []
    frequency_bank: dict[str, Tensor] = {}
    variant_metadata: list[dict[str, Any]] = []
    for variant_index, spec in enumerate(variants, start=1):
        name = str(spec["name"])
        print(
            f"[Variant] {variant_index}/{len(variants)} name={name}",
            flush=True,
        )
        runtimes = [
            Qwen35PostGateTopKRuntime(
                model,
                intervention=intervention,
                keep_ratio=keep_ratio,
                selection_scope=spec["selection_scope"],
                tp_size=args.tp_size,
                profile=True,
            )
            for intervention, keep_ratio in _runtime_specs(spec)
        ]
        with ExitStack() as stack:
            for runtime in runtimes:
                stack.enter_context(runtime)
            records = [record for runtime in runtimes for record in runtime.records]
            variant_metadata.append(_variant_metadata(spec, records))
            if samples is not None:
                assert dense_ppl_raw is not None
                for runtime in runtimes:
                    runtime.reset_profile()
                candidate_raw = _evaluate(
                    model,
                    samples,
                    batch_size=args.ppl_batch_size,
                    device=args.device,
                    label=name,
                )
                candidate_ppl[name] = _serializable_ppl(
                    candidate_raw,
                    baseline=dense_ppl_raw,
                )
                for runtime in runtimes:
                    profiles.extend(
                        _capture_profile(
                            runtime,
                            dataset="wikitext2",
                            variant=name,
                            frequency_bank=frequency_bank,
                        )
                    )
                del candidate_raw
            if tasks:
                assert dense_mcq is not None
                dense_by_name = {task["task"]: task for task in dense_mcq["tasks"]}
                task_results: list[dict[str, Any]] = []
                for task_name, task_path in tasks:
                    for runtime in runtimes:
                        runtime.reset_profile()
                    task_result = _evaluate_task(
                        model,
                        tokenizer,
                        name=task_name,
                        path=task_path,
                        max_examples=args.mcq_max_examples,
                        choice_prefix=args.choice_prefix,
                        normalize=args.normalize,
                        add_special_tokens=not args.no_add_special_tokens,
                        device=torch.device(args.device),
                        variant=name,
                    )
                    task_result["paired_vs_dense"] = _mcq_paired_metrics(
                        task_result,
                        dense_by_name[task_name],
                    )
                    task_results.append(task_result)
                    for runtime in runtimes:
                        profiles.extend(
                            _capture_profile(
                                runtime,
                                dataset=f"mcq_{task_name}",
                                variant=name,
                                frequency_bank=frequency_bank,
                            )
                        )
                candidate_mcq[name] = {
                    "aggregate": _aggregate_mcq(task_results),
                    "tasks": task_results,
                }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True)
    frequency_path = output_dir / "selection_frequency.safetensors"
    temporary_frequency = frequency_path.with_suffix(".safetensors.tmp")
    save_file(frequency_bank, str(temporary_frequency))
    os.replace(temporary_frequency, frequency_path)
    payload: dict[str, Any] = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "git_commit": _git_commit(),
        "elapsed_seconds": elapsed,
        "model": {
            "path": str(model_path),
            "config_sha256": _file_sha256(model_path / "config.json"),
            "weights_updated": False,
        },
        "protocol": {
            "exact_checkpoint_gate": True,
            "exact_attention_and_recurrent_states": True,
            "intervention_point": "post_gate_pre_output_projection",
            "quality_oracle_dense_zero_fill": True,
            "sparse_kernel_or_collective_executed": False,
            "fixed_cardinality_per_source": args.selection_scope == "source_local",
            "collective_candidate": (
                "fixed_size_packed_allgather"
                if args.selection_scope == "source_local"
                else None
            ),
            "selection_scope": args.selection_scope,
            "tp_size": args.tp_size,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        },
        "variants": variant_metadata,
        "selection_frequency": {
            "path": str(frequency_path),
            "sha256": _file_sha256(frequency_path),
            "tensor_count": len(frequency_bank),
        },
        "profiles": profiles,
        "ppl": (
            None
            if dense_ppl is None
            else {
                "dataset": "wikitext2",
                "source": wikitext_metadata,
                "dense": dense_ppl,
                "candidates": candidate_ppl,
            }
        ),
        "mcq": (
            None
            if dense_mcq is None
            else {
                "normalize": args.normalize,
                "choice_prefix": args.choice_prefix,
                "maximum_examples_per_task": args.mcq_max_examples,
                "sources": [
                    {"task": name, "path": str(path), "sha256": _file_sha256(path)}
                    for name, path in tasks
                ],
                "dense": dense_mcq,
                "candidates": candidate_mcq,
            }
        ),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "safetensors": _installed_version("safetensors"),
        },
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    _atomic_json(output_dir / "results.json", payload)
    _atomic_text(output_dir / "summary.md", _summary_markdown(payload))
    print(f"[Done] elapsed={elapsed:.2f}s output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
