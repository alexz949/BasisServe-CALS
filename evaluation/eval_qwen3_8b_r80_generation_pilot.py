#!/usr/bin/env python3
"""Evaluate Qwen3-8B R80 pilot arms on IFEval and GSM8K."""

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
from lm_eval.utils import make_table
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import eval_qwen3_8b_iclr_quality as quality  # noqa: E402


FORMAT = "basisserve.qwen3_8b.r80_generation_pilot.v1"
STAGE_FORMAT = "basisserve.qwen3_8b.r80_generation_pilot.task.v1"
TASKS = ("ifeval", "gsm8k")
TASK_FEWSHOT = {"ifeval": 0, "gsm8k": 5}
TASK_MAX_GEN_TOKS = {"ifeval": 1280, "gsm8k": 512}
EXPECTED_ARMS = {
    "Q3-8B-Dense": {"method": "dense"},
    "Q3-8B-C1-R80": {
        "method": "c1-two-sided-kl",
        "equivalent_rank_target": 80,
    },
    "Q3-8B-C1U-R80": {
        "method": "c1-uniform",
        "equivalent_rank_target": 80,
    },
    "Q3-8B-PALUM-R80": {
        "method": "palu-fisher",
        "equivalent_rank_target": 80,
        "head_group_size": 1,
    },
    "Q3-8B-PALUG2-R80": {
        "method": "palu-fisher",
        "equivalent_rank_target": 80,
        "head_group_size": 2,
    },
    "Q3-8B-PALUG4-R80": {
        "method": "palu-fisher",
        "equivalent_rank_target": 80,
        "head_group_size": 4,
    },
}


def _metric(
    task_metrics: Mapping[str, Any],
    metric: str,
    filter_name: str,
) -> float | None:
    value = task_metrics.get(f"{metric},{filter_name}")
    return None if value is None else float(value)


def summarize_task(
    task: str,
    evaluation: Mapping[str, Any],
) -> dict[str, float] | None:
    task_metrics = evaluation.get("results", {}).get(task, {})
    if task == "ifeval":
        selected = {
            "prompt_level_strict_accuracy": _metric(
                task_metrics, "prompt_level_strict_acc", "none"
            ),
            "instruction_level_strict_accuracy": _metric(
                task_metrics, "inst_level_strict_acc", "none"
            ),
            "prompt_level_loose_accuracy": _metric(
                task_metrics, "prompt_level_loose_acc", "none"
            ),
            "instruction_level_loose_accuracy": _metric(
                task_metrics, "inst_level_loose_acc", "none"
            ),
        }
    else:
        selected = {
            "strict_exact_match": _metric(
                task_metrics, "exact_match", "strict-match"
            ),
            "flexible_exact_match": _metric(
                task_metrics, "exact_match", "flexible-extract"
            ),
        }
    if not quality._check(
        all(value is not None for value in selected.values()),
        f"missing expected metrics for {task}: {task_metrics}",
    ):
        return None
    return {name: float(value) for name, value in selected.items()}


def _stage_matches(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    task: str,
    checkpoint_manifest_sha256: str,
    limit: float | None,
) -> bool:
    return all(
        (
            payload.get("format") == STAGE_FORMAT,
            payload.get("status") == "complete",
            payload.get("run_id") == run_id,
            payload.get("task") == task,
            payload.get("checkpoint", {}).get("manifest_sha256")
            == checkpoint_manifest_sha256,
            payload.get("protocol", {}).get("limit") == limit,
            payload.get("protocol", {}).get("max_gen_toks")
            == TASK_MAX_GEN_TOKS[task],
        )
    )


def _validate_arm(run_id: str, compression: Mapping[str, Any]) -> bool:
    expected = EXPECTED_ARMS[run_id]
    return all(
        quality._check(
            compression.get(key) == value,
            f"{run_id} has {key}={compression.get(key)!r}, expected {value!r}",
        )
        for key, value in expected.items()
    )


def clear_model_generation_length_override(model: torch.nn.Module) -> int | None:
    """Let lm-eval's per-task max_gen_toks control generation length."""

    source_value = model.generation_config.max_new_tokens
    model.generation_config.max_new_tokens = None
    return None if source_value is None else int(source_value)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> int:
    valid = all(
        (
            quality._check(torch.cuda.is_available(), "CUDA is required"),
            quality._check(
                torch.cuda.device_count() == 1,
                "exactly one L40S GPU must be visible per pilot arm",
            ),
            quality._check(
                args.lm_eval_batch_size > 0,
                "lm-eval batch size must be positive",
            ),
            quality._check(args.max_length > 0, "max length must be positive"),
        )
    )
    if not valid:
        return 2
    gpu_names = [torch.cuda.get_device_name(0)]
    if not quality._check(
        all(name == "NVIDIA L40S" for name in gpu_names),
        f"unexpected GPUs: {gpu_names}",
    ):
        return 2
    torch.set_num_threads(args.torch_num_threads)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)

    model_path = Path(args.model).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "manifest.json"
    if not quality._check(manifest_path.is_file(), f"missing {manifest_path}"):
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_manifest_sha256 = quality._sha256(manifest_path)
    run_id = str(manifest.get("run_id"))
    valid_manifest = all(
        (
            quality._check(
                manifest.get("format") == quality.CHECKPOINT_FORMAT,
                "checkpoint format mismatch",
            ),
            quality._check(
                manifest.get("status") == "complete",
                "checkpoint is incomplete",
            ),
            quality._check(run_id == args.run_id, "run ID mismatch"),
            quality._check(run_id in EXPECTED_ARMS, "run is not an R80 pilot arm"),
            quality._check(
                manifest.get("model", {}).get("config_sha256")
                == quality._sha256(model_path / "config.json"),
                "checkpoint belongs to another model config",
            ),
        )
    )
    if not valid_manifest or not _validate_arm(run_id, manifest["compression"]):
        return 2

    final_path = output_dir / "result.json"
    if final_path.is_file():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if quality._check(
            existing.get("format") == FORMAT
            and existing.get("status") == "complete"
            and existing.get("run_id") == run_id
            and existing.get("checkpoint", {}).get("manifest_sha256")
            == checkpoint_manifest_sha256
            and existing.get("protocol", {}).get("limit") == args.limit
            and existing.get("protocol", {}).get("task_max_gen_toks")
            == TASK_MAX_GEN_TOKS,
            f"existing final result does not match this run: {final_path}",
        ):
            print(f"[Resume] complete result already exists: {final_path}", flush=True)
            return 0
        return 2

    compression = manifest["compression"]
    dense = compression["method"] == "dense"
    artifact_sha256 = None if dense else str(manifest["artifact"]["sha256"])
    stage_paths = {task: output_dir / f"{task}.json" for task in TASKS}
    stages: dict[str, dict[str, Any] | None] = {}
    for task, path in stage_paths.items():
        if not path.is_file():
            stages[task] = None
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not quality._check(
            _stage_matches(
                payload,
                run_id=run_id,
                task=task,
                checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                limit=args.limit,
            ),
            f"existing stage does not match this run: {path}",
        ):
            return 2
        stages[task] = payload
        print(f"[Resume] reusing {path}", flush=True)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map="balanced",
        max_memory={0: f"{args.max_memory_per_gpu_gib}GiB"},
    ).eval()
    model.config.use_cache = True
    source_max_new_tokens = clear_model_generation_length_override(model)
    used_devices = sorted(
        {
            int(parameter.device.index)
            for parameter in model.parameters()
            if parameter.device.type == "cuda"
            and parameter.device.index is not None
        }
    )
    if not quality._check(
        used_devices == [0],
        f"model uses CUDA devices {used_devices}",
    ):
        return 2

    installation: list[dict[str, Any]] = []
    if compression["method"] == "c1-two-sided-kl":
        installed = quality.install_c1_allocation(model, checkpoint_dir, manifest)
        if installed is None:
            return 2
        installation = installed
    elif compression["method"] == "c1-uniform":
        installed = quality.install_c1_uniform_factor_bank(
            model, checkpoint_dir, manifest
        )
        if installed is None:
            return 2
        installation = installed
    elif not dense:
        loaded_manifest, factors = quality._load_checkpoint(
            checkpoint_dir, model_path
        )
        installation = quality.install_palu_m_factors(
            model,
            factors,
            layer_ranks=[
                [int(rank) for rank in ranks]
                for ranks in loaded_manifest["compression"]["layer_ranks"]
            ],
            head_dim=int(compression["head_dim"]),
            require_cuda_resident=True,
        )
        del factors

    runtime = {
        "model_dtype": str(torch.bfloat16),
        "attention_implementation": "sdpa",
        "device_map": "balanced",
        "cuda_devices_used": used_devices,
        "cuda_device_names": gpu_names,
        "model_generation_config_source_max_new_tokens": source_max_new_tokens,
        "model_generation_config_runtime_max_new_tokens": (
            model.generation_config.max_new_tokens
        ),
        "installation": installation,
    }
    checkpoint = {
        "directory": str(checkpoint_dir),
        "manifest_sha256": checkpoint_manifest_sha256,
        "artifact_sha256": artifact_sha256,
        "format": quality.CHECKPOINT_FORMAT,
    }
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.lm_eval_batch_size,
        max_length=args.max_length,
        add_bos_token=False,
    )
    task_manager = TaskManager()
    for task in TASKS:
        if stages[task] is not None:
            continue
        task_started = time.perf_counter()
        evaluation = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=None,
            task_manager=task_manager,
            limit=args.limit,
            log_samples=True,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            gen_kwargs={"max_gen_toks": TASK_MAX_GEN_TOKS[task]},
        )
        if not quality._check(
            evaluation is not None,
            f"lm-eval returned no result for {task}",
        ):
            return 2
        metrics = summarize_task(task, evaluation)
        if metrics is None:
            return 2
        stage = {
            "format": STAGE_FORMAT,
            "status": "complete",
            "run_id": run_id,
            "task": task,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "model": manifest["model"],
            "checkpoint": checkpoint,
            "compression": compression,
            "runtime": runtime,
            "protocol": {
                "task": task,
                "task_default_num_fewshot": TASK_FEWSHOT[task],
                "use_task_default_num_fewshot": True,
                "batch_size": args.lm_eval_batch_size,
                "max_length": args.max_length,
                "max_gen_toks": TASK_MAX_GEN_TOKS[task],
                "limit": args.limit,
                "log_samples": True,
                "apply_chat_template": False,
                "fewshot_as_multiturn": False,
                "generation_configuration_source": "lm-eval 0.4.11 task YAML",
            },
            "metrics": metrics,
            "evaluation": evaluation,
            "elapsed_seconds": time.perf_counter() - task_started,
        }
        quality._write_json(stage_paths[task], stage)
        stages[task] = stage
        print(make_table(evaluation), flush=True)
        print(f"[{task}] metrics={metrics}", flush=True)

    if not quality._check(
        all(stages[task] is not None for task in TASKS),
        "one or more pilot stages are incomplete",
    ):
        return 2
    result = {
        "format": FORMAT,
        "status": "complete",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model": manifest["model"],
        "checkpoint": checkpoint,
        "compression": compression,
        "protocol": {
            "tasks": list(TASKS),
            "task_default_num_fewshot": TASK_FEWSHOT,
            "batch_size": args.lm_eval_batch_size,
            "max_length": args.max_length,
            "task_max_gen_toks": TASK_MAX_GEN_TOKS,
            "limit": args.limit,
            "log_samples": True,
            "apply_chat_template": False,
            "generation_configuration_source": "lm-eval 0.4.11 task YAML",
        },
        "metrics": {task: stages[task]["metrics"] for task in TASKS},
        "stages": {
            task: {
                "file": stage_paths[task].name,
                "sha256": quality._sha256(stage_paths[task]),
            }
            for task in TASKS
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
                for index in range(torch.cuda.device_count())
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    quality._write_json(final_path, result)
    print(f"[Result] wrote {final_path}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", choices=tuple(EXPECTED_ARMS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lm-eval-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--torch-num-threads", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(evaluate(parse_args()))
