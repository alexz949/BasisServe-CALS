#!/usr/bin/env python3
"""Run the complete Qwen3-8B ICLR quality protocol on one matrix arm."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.build_qwen3_8b_iclr_v_checkpoint import (  # noqa: E402
    FORMAT as CHECKPOINT_FORMAT,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)
from evaluation.eval_gqa_palu_m_wikitext import (  # noqa: E402
    _atomic_json,
    _decoder_layers,
    _load_checkpoint,
    _sha256,
    install_palu_m_factors,
)
from evaluation.eval_qwen3_32b_c1_c4_ppl_shard import (  # noqa: E402
    _evaluate_document_ppl,
    _load_windows,
)


QUALITY_PROFILES = {
    "qwen3_8b": {
        "label": "Qwen3-8B-Base",
        "checkpoint": "basisserve.qwen3_8b.iclr_v_factors.v1",
        "quality": "basisserve.qwen3_8b.iclr_quality.v1",
        "slug": "qwen3_8b",
    },
    "llama31_8b": {
        "label": "Llama-3.1-8B",
        "checkpoint": "basisserve.llama31_8b.iclr_v_factors.v1",
        "quality": "basisserve.llama31_8b.iclr_quality.v1",
        "slug": "llama31_8b",
    },
}
FORMAT = ""
DESCRIPTION = ""
STAGE_FORMATS: dict[str, str] = {}


def activate_quality_profile(name: str) -> None:
    profile = QUALITY_PROFILES[name]
    slug = str(profile["slug"])
    global CHECKPOINT_FORMAT, DESCRIPTION, FORMAT, STAGE_FORMATS
    CHECKPOINT_FORMAT = str(profile["checkpoint"])
    DESCRIPTION = f"Run the complete {profile['label']} ICLR quality protocol."
    FORMAT = str(profile["quality"])
    STAGE_FORMATS = {
        "wikitext2": f"basisserve.{slug}.iclr_quality.wikitext2.v1",
        "c4": f"basisserve.{slug}.iclr_quality.c4_validation.v1",
        "commonsense": f"basisserve.{slug}.iclr_quality.commonsense.v1",
    }


activate_quality_profile("qwen3_8b")
TASKS = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "boolq",
    "openbookqa",
)


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _cuda_device(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    if text.startswith("cuda:") and text.removeprefix("cuda:").isdigit():
        return int(text.removeprefix("cuda:"))
    return None


def _model_cuda_devices(model: torch.nn.Module) -> list[int]:
    device_map = getattr(model, "hf_device_map", {})
    devices = {
        device
        for value in device_map.values()
        if (device := _cuda_device(value)) is not None
    }
    return sorted(devices)


def _accuracy_metric(metrics: Mapping[str, Any]) -> tuple[str, float] | None:
    for metric in ("acc_norm", "acc"):
        for key, value in metrics.items():
            if key.split(",", 1)[0] == metric and "stderr" not in key:
                return metric, float(value)
    return None


def summarize_commonsense(
    evaluation: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], float] | None:
    results = evaluation.get("results", {})
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        selected = _accuracy_metric(results.get(task, {}))
        if not _check(selected is not None, f"no acc_norm or acc metric for {task}"):
            return None
        metric, value = selected
        rows.append({"task": task, "metric": metric, "value": value})
    return rows, sum(row["value"] for row in rows) / len(rows)


def _load_stage(
    path: Path,
    *,
    stage: str,
    run_id: str,
    checkpoint_manifest_sha256: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = all(
        (
            payload.get("format") == STAGE_FORMATS[stage],
            payload.get("status") == "complete",
            payload.get("run_id") == run_id,
            payload.get("checkpoint", {}).get("manifest_sha256")
            == checkpoint_manifest_sha256,
        )
    )
    if not _check(matches, f"existing {stage} stage does not match this checkpoint"):
        return {"status": "mismatch"}
    print(f"[Resume] reusing {path}", flush=True)
    return payload


def _stage_base(
    *,
    stage: str,
    run_id: str,
    checkpoint_dir: Path,
    checkpoint_manifest_sha256: str,
    artifact_sha256: str | None,
    compression: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": STAGE_FORMATS[stage],
        "status": "complete",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "manifest_sha256": checkpoint_manifest_sha256,
            "artifact_sha256": artifact_sha256,
            "format": CHECKPOINT_FORMAT,
        },
        "compression": compression,
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> int:
    valid = all(
        (
            _check(torch.cuda.is_available(), "CUDA is required"),
            _check(torch.cuda.device_count() == 4, "exactly four L40S GPUs must be visible"),
            _check(args.batch_size > 0, "batch size must be positive"),
            _check(args.lm_eval_batch_size > 0, "lm-eval batch size must be positive"),
        )
    )
    if not valid:
        return 2
    gpu_names = [torch.cuda.get_device_name(index) for index in range(4)]
    if not _check(all(name == "NVIDIA L40S" for name in gpu_names), f"unexpected GPUs: {gpu_names}"):
        return 2
    torch.set_num_threads(args.torch_num_threads)
    for index in range(4):
        torch.cuda.reset_peak_memory_stats(index)

    model_path = Path(args.model).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "result.json"
    if not _check(not final_path.exists(), f"final result already exists: {final_path}"):
        return 2

    manifest_path = checkpoint_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest.get("run_id"))
    checkpoint_manifest_sha256 = _sha256(manifest_path)
    valid_manifest = all(
        (
            _check(manifest.get("format") == CHECKPOINT_FORMAT, "checkpoint format mismatch"),
            _check(manifest.get("status") == "complete", "checkpoint is incomplete"),
            _check(run_id == args.run_id, "run ID differs from checkpoint manifest"),
            _check(
                manifest.get("model", {}).get("config_sha256")
                == _sha256(model_path / "config.json"),
                "checkpoint belongs to another model config",
            ),
        )
    )
    if not valid_manifest:
        return 2
    compression = manifest["compression"]
    dense = compression["method"] == "dense"
    artifact_sha256 = None if dense else str(manifest["artifact"]["sha256"])

    stage_paths = {
        stage: output_dir / f"{stage}.json" for stage in STAGE_FORMATS
    }
    stages = {
        stage: _load_stage(
            path,
            stage=stage,
            run_id=run_id,
            checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        )
        for stage, path in stage_paths.items()
    }
    if any(stage is not None and stage.get("status") == "mismatch" for stage in stages.values()):
        return 2

    c4_sequences, c4_provenance = _load_windows(
        Path(args.c4_windows), model_path=model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    dtype = torch.bfloat16
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map="balanced",
        max_memory={index: f"{args.max_memory_per_gpu_gib}GiB" for index in range(4)},
    ).eval()
    model.config.use_cache = False
    used_devices = _model_cuda_devices(model)
    if not _check(used_devices == [0, 1, 2, 3], f"model uses CUDA devices {used_devices}"):
        return 2

    installation: list[dict[str, Any]] = []
    if not dense:
        loaded_manifest, factors = _load_checkpoint(checkpoint_dir, model_path)
        layer_ranks = [
            [int(rank) for rank in ranks]
            for ranks in loaded_manifest["compression"]["layer_ranks"]
        ]
        installation = install_palu_m_factors(
            model,
            factors,
            layer_ranks=layer_ranks,
            head_dim=int(compression["head_dim"]),
            require_cuda_resident=True,
        )
        del factors

    common = {
        "command": shlex.join(sys.argv),
        "model": manifest["model"],
        "runtime": {
            "model_dtype": str(dtype),
            "attention_implementation": "sdpa",
            "device_map": "balanced",
            "cuda_devices_used": used_devices,
            "cuda_device_names": gpu_names,
            "installation": installation,
        },
    }

    if stages["wikitext2"] is None:
        wiki_started = time.perf_counter()
        wiki_metrics = _eval_ppl_fp32_loss(
            model,
            tokenizer,
            dataset="wikitext2",
            split="test",
            seqlen=2048,
            batch_size=args.batch_size,
            max_samples=None,
            max_tokens=None,
        )
        stages["wikitext2"] = {
            **_stage_base(
                stage="wikitext2",
                run_id=run_id,
                checkpoint_dir=checkpoint_dir,
                checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                artifact_sha256=artifact_sha256,
                compression=compression,
            ),
            **common,
            "protocol": {"full_test_corpus": True, "loss_dtype": "float32"},
            "metrics": wiki_metrics,
            "elapsed_seconds": time.perf_counter() - wiki_started,
        }
        _atomic_json(stage_paths["wikitext2"], stages["wikitext2"])
        print(f"[WikiText-2] ppl={wiki_metrics['ppl']:.9f}", flush=True)

    if stages["c4"] is None:
        c4_started = time.perf_counter()
        c4_metrics = _evaluate_document_ppl(
            model,
            c4_sequences,
            batch_size=args.batch_size,
            label=run_id,
        )
        stages["c4"] = {
            **_stage_base(
                stage="c4",
                run_id=run_id,
                checkpoint_dir=checkpoint_dir,
                checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                artifact_sha256=artifact_sha256,
                compression=compression,
            ),
            **common,
            "protocol": {
                "split": "validation",
                "documents": 128,
                "sequence_length": 2048,
                "cross_document_transitions": False,
                "loss_dtype": "float32",
            },
            "windows": c4_provenance,
            "metrics": c4_metrics,
            "elapsed_seconds": time.perf_counter() - c4_started,
        }
        _atomic_json(stage_paths["c4"], stages["c4"])
        print(f"[C4] ppl={c4_metrics['ppl']:.9f}", flush=True)

    if stages["commonsense"] is None:
        commonsense_started = time.perf_counter()
        model.config.use_cache = True
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.lm_eval_batch_size,
            max_length=4096,
            add_bos_token=False,
        )
        evaluation = lm_eval.simple_evaluate(
            model=lm,
            tasks=list(TASKS),
            num_fewshot=0,
            task_manager=TaskManager(),
            log_samples=False,
        )
        if not _check(evaluation is not None, "lm-eval returned no results"):
            return 2
        summarized = summarize_commonsense(evaluation)
        if summarized is None:
            return 2
        task_rows, average_accuracy = summarized
        stages["commonsense"] = {
            **_stage_base(
                stage="commonsense",
                run_id=run_id,
                checkpoint_dir=checkpoint_dir,
                checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                artifact_sha256=artifact_sha256,
                compression=compression,
            ),
            **common,
            "protocol": {
                "tasks": list(TASKS),
                "num_fewshot": 0,
                "batch_size": args.lm_eval_batch_size,
                "max_length": 4096,
                "metric_selection": "acc_norm when present, otherwise acc",
            },
            "task_accuracy": task_rows,
            "average_accuracy": average_accuracy,
            "evaluation": evaluation,
            "elapsed_seconds": time.perf_counter() - commonsense_started,
        }
        _write_json(stage_paths["commonsense"], stages["commonsense"])
        print(f"[Commonsense] average_accuracy={average_accuracy:.9f}", flush=True)

    wiki = stages["wikitext2"]
    c4 = stages["c4"]
    commonsense = stages["commonsense"]
    if not _check(
        wiki is not None and c4 is not None and commonsense is not None,
        "one or more quality stages are incomplete",
    ):
        return 2
    result = {
        "format": FORMAT,
        "status": "complete",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model": manifest["model"],
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "manifest_sha256": checkpoint_manifest_sha256,
            "artifact_sha256": artifact_sha256,
            "format": CHECKPOINT_FORMAT,
        },
        "compression": compression,
        "metrics": {
            "wikitext2_ppl": wiki["metrics"]["ppl"],
            "c4_validation_128_ppl": c4["metrics"]["ppl"],
            "task_accuracy": commonsense["task_accuracy"],
            "average_accuracy": commonsense["average_accuracy"],
        },
        "stages": {
            stage: {
                "file": path.name,
                "sha256": _sha256(path),
            }
            for stage, path in stage_paths.items()
        },
        "elapsed_seconds_this_attempt": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "lm_eval": importlib.metadata.version("lm-eval"),
            "cuda_devices": gpu_names,
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(4)
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(final_path, result)
    print(
        f"[Result] run_id={run_id} wiki_ppl={result['metrics']['wikitext2_ppl']:.9f} "
        f"c4_ppl={result['metrics']['c4_validation_128_ppl']:.9f} "
        f"average_accuracy={result['metrics']['average_accuracy']:.9f}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--c4-windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lm-eval-batch-size", type=int, default=8)
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--torch-num-threads", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(evaluate(parse_args()))
