#!/usr/bin/env python3
"""Compare frozen Qwen3-32B per-layer C1 schedules on full C4 documents."""

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
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import allocate_qwen3_32b_c1_tp_source_global_kl as common  # noqa: E402
from evaluation import run_qwen3_32b_c1_tp_source_global_kl_sharded as runtime  # noqa: E402
from evaluation.eval_qwen3_32b_c1_wikitext import (  # noqa: E402
    _decoder_layers,
    _load_external_layer_schedule,
    _load_layer_allocation_results,
    _sha256,
    activate_model_profile as activate_evaluator_profile,
    install_layer_allocation_schedule,
    load_layer_allocation_reconstruction_state,
)


FORMAT = "basisserve.qwen3_32b.gqa_c1.layer_schedules_c4_kl.v1"
MODEL_LABEL = "Qwen3-32B"


def activate_model_profile(name: str) -> None:
    global FORMAT, MODEL_LABEL
    common.activate_model_profile(name)
    runtime.activate_model_profile(name)
    activate_evaluator_profile(name)
    if name == "qwen3_32b":
        slug = "qwen3_32b"
        MODEL_LABEL = "Qwen3-32B"
    elif name == "qwen3_8b":
        slug = "qwen3_8b"
        MODEL_LABEL = "Qwen3-8B-Base"
    else:
        raise ValueError(f"unknown Qwen3 C1 model profile: {name}")
    FORMAT = f"basisserve.{slug}.gqa_c1.layer_schedules_c4_kl.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--allocation-dir", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--schedule-names",
        default="uniform_anchor,mean_dp,ucb_dp",
        help="Comma-separated recorded schedules to compare.",
    )
    parser.add_argument(
        "--schedule-file",
        type=Path,
        help="Authenticated external schedule to append to the comparison.",
    )
    parser.add_argument("--window-start", type=int, default=320)
    parser.add_argument("--profile-windows", type=int, default=8)
    parser.add_argument("--confirmation-windows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _metric_subset(
    metrics: Mapping[str, Any],
    start: int,
    stop: int,
) -> dict[str, Any]:
    return {
        name: common._paired(row["values"][start:stop])
        for name, row in metrics.items()
    }


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.profile_windows,
        args.confirmation_windows,
        args.sequence_length,
        args.batch_size,
        args.vocab_chunk_size,
        args.torch_num_threads,
        args.max_memory_per_gpu_gib,
    )
    if min(positive) <= 0:
        raise ValueError("sample and compute arguments must be positive")
    if args.window_start < common.FRESH_WINDOW_START:
        raise ValueError("C4 schedule windows must remain disjoint from ALS documents")
    if args.sequence_length != 2048:
        raise ValueError("this controlled comparison requires full 2048-token C4 documents")


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    _validate_args(args)
    torch.set_num_threads(args.torch_num_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("C4 schedule evaluation requires CUDA")
    torch.cuda.set_device(0)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)

    started = time.perf_counter()
    model_path = Path(args.model).expanduser().resolve()
    allocation_dir = args.allocation_dir.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = _load_layer_allocation_results(allocation_dir, model_path)
    schedule_names = tuple(
        name.strip() for name in args.schedule_names.split(",") if name.strip()
    )
    if not schedule_names or len(schedule_names) != len(set(schedule_names)):
        raise ValueError("schedule names must be non-empty and unique")
    if any(name not in result["schedules"] for name in schedule_names):
        raise ValueError("requested recorded schedule does not exist")
    external_schedule = None
    if args.schedule_file is not None:
        schedule_path = args.schedule_file.expanduser().resolve()
        external_name, external_row, external_schedule = (
            _load_external_layer_schedule(
                schedule_path,
                allocation_result_path=allocation_dir / "result.json",
                result=result,
            )
        )
        result["schedules"][external_name] = external_row
        schedule_names = (*schedule_names, external_name)
    profile, confirmation, provenance = common._select_fresh_windows(
        args.windows,
        window_start=args.window_start,
        profile_windows=args.profile_windows,
        confirmation_windows=args.confirmation_windows,
        sequence_length=args.sequence_length,
    )
    sequences = torch.cat((profile, confirmation), dim=0)
    model = runtime._load_model(args, model_path)
    teacher = common._capture_teacher(
        model,
        sequences,
        batch_size=args.batch_size,
        vocab_chunk_size=args.vocab_chunk_size,
        label="C4 seq2048 schedule comparison",
    )
    dense_v_weights = tuple(
        layer.self_attn.v_proj.weight.detach().cpu().clone()
        for layer in _decoder_layers(model)
    )
    reconstruction_state = load_layer_allocation_reconstruction_state(
        result,
        model_path=model_path,
    )

    schedule_rows = {}
    profile_stop = args.profile_windows
    all_stop = args.profile_windows + args.confirmation_windows
    profile_label = f"profile_{args.profile_windows}"
    confirmation_label = f"confirmation_{args.confirmation_windows}"
    all_label = f"all_{len(sequences)}"
    for schedule_name in schedule_names:
        schedule_started = time.perf_counter()
        installation = install_layer_allocation_schedule(
            model,
            allocation_dir,
            result,
            schedule_name=schedule_name,
            dense_v_weights=dense_v_weights,
            reconstruction_state=reconstruction_state,
        )
        metrics = common._evaluate_teacher_metrics(
            model,
            teacher,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        schedule_rows[schedule_name] = {
            "accounting": result["schedules"][schedule_name]["accounting"],
            "metrics": {
                profile_label: _metric_subset(metrics, 0, profile_stop),
                confirmation_label: _metric_subset(metrics, profile_stop, all_stop),
                all_label: metrics,
            },
            "installation": installation,
            "elapsed_seconds": time.perf_counter() - schedule_started,
        }
        mean_kl = metrics["terminal_kl"]["mean"]
        print(f"[C4 seq2048] {schedule_name} mean_kl={mean_kl:.9g}", flush=True)
        torch.cuda.empty_cache()

    rankings = {}
    for split in (profile_label, confirmation_label, all_label):
        rankings[split] = sorted(
            (
                {
                    "schedule": name,
                    "mean_terminal_kl": schedule_rows[name]["metrics"][split][
                        "terminal_kl"
                    ]["mean"],
                }
                for name in schedule_names
            ),
            key=lambda row: row["mean_terminal_kl"],
        )

    source_result_path = allocation_dir / "result.json"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "allocation": {
            "path": str(source_result_path),
            "sha256": _sha256(source_result_path),
            "original_selected_candidate": result["selection"]["selected_candidate"],
            "factor_stage": result["profile"]["factor_stage"],
        },
        "dataset": {
            "name": "c4_train_fresh_documents",
            "windows_provenance": provenance,
            "profile_windows": args.profile_windows,
            "confirmation_windows": args.confirmation_windows,
            "sequence_length": args.sequence_length,
            "predicted_tokens_per_document": args.sequence_length - 1,
            "batch_size": args.batch_size,
            "forward_batches_per_schedule": math.ceil(len(sequences) / args.batch_size),
        },
        "metric_protocol": {
            "terminal_kl_legacy_field_name": (
                "mean dense-teacher KL over every next-token position in each document"
            ),
            "paired_standard_error_unit": "C4 document",
            "teacher": "uncompressed dense Qwen3-32B BF16 SDPA",
        },
        "schedules": schedule_rows,
        "rankings_lowest_kl_first": rankings,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(torch.cuda.device_count())
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    payload["metric_protocol"]["teacher"] = (
        f"uncompressed dense {MODEL_LABEL} BF16 SDPA"
    )
    if external_schedule is not None:
        payload["allocation"]["external_schedule"] = {
            "path": str(schedule_path),
            "sha256": _sha256(schedule_path),
            "format": external_schedule["format"],
            "method": external_schedule["method"],
        }
    common._atomic_json(output_path, payload)
    print(f"[C4 seq2048] wrote {output_path}", flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
