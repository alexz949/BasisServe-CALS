#!/usr/bin/env python3
"""Evaluate the fused STAR-KV V50 adaptive checkpoint on seven MCQ tasks."""

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
import time

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.eval_qwen3_8b_iclr_quality import TASKS, summarize_commonsense
from evaluation.eval_starkv_v50_adaptive_ppl import _sha256

sys.path.insert(0, str(ROOT / "external/STAR-KV"))
from model import build_fused_from_state_dict


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _summary(payload: dict) -> str:
    rows = [
        "# STAR-KV V-only adaptive MCQ",
        "",
        f"- Average accuracy: `{payload['metrics']['average_accuracy']:.8f}`",
        f"- Actual V-cache compression: `{payload['checkpoint']['training']['budget']['v_cache_compression']:.4%}`",
        "",
        "| Task | Metric | Accuracy |",
        "|---|---|---:|",
    ]
    for row in payload["metrics"]["task_accuracy"]:
        rows.append(f"| {row['task']} | {row['metric']} | {row['value']:.8f} |")
    rows.extend(["", "## Command", "", "```bash", payload["command"], "```", ""])
    return "\n".join(rows)


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lm-eval-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(0)
    if not all(
        (
            _check(torch.cuda.device_count() == 1, "exactly one CUDA device is required"),
            _check(args.lm_eval_batch_size > 0, "batch size must be positive"),
            _check(args.limit is None or args.limit > 0, "limit must be positive"),
        )
    ):
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not _check(not (args.output_dir / "result.json").exists(), "result already exists"):
        return 2

    checkpoint_path = args.checkpoint.resolve()
    training = json.loads((checkpoint_path / "result.json").read_text())
    if not all(
        (
            _check(training["status"] == "complete", "training is incomplete"),
            _check(
                training["protocol"]["method"]
                == "STAR-KV V-only adaptive-rank adaptation",
                "checkpoint method mismatch",
            ),
        )
    ):
        return 2

    base_manifest = json.loads(
        (ROOT / "ICLR-results/qwen3-8b/checkpoints/Q3-8B-Dense/manifest.json").read_text()
    )
    model_path = Path(base_manifest["model"]["path"])
    if not all(
        (
            _check(training["protocol"]["model"] == str(model_path), "model path mismatch"),
            _check(
                _sha256(model_path / "config.json") == base_manifest["model"]["config_sha256"],
                "model config hash mismatch",
            ),
        )
    ):
        return 2

    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    artifact = checkpoint_path / "fused.pt"
    state = torch.load(artifact, map_location="cpu", mmap=True, weights_only=True)
    skip_layers = tuple(training["protocol"]["skip_layers"])
    build_fused_from_state_dict(model, model.config, state, skip_layers=skip_layers)
    model.load_state_dict(state, strict=True)
    actual_ranks = [
        layer.self_attn.v_proj.VS.out_features
        if hasattr(layer.self_attn.v_proj, "VS")
        else layer.self_attn.v_proj.out_features
        for layer in model.model.layers
    ]
    if not _check(actual_ranks == training["budget"]["ranks"], "fused rank profile mismatch"):
        return 2
    del state
    gc.collect()

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
        limit=args.limit,
    )
    if not _check(evaluation is not None, "lm-eval returned no results"):
        return 2
    summarized = summarize_commonsense(evaluation, TASKS)
    if not _check(summarized is not None, "MCQ metrics could not be summarized"):
        return 2
    task_rows, average_accuracy = summarized
    metrics = {
        "task_accuracy": task_rows,
        "average_accuracy": average_accuracy,
        "evaluation": evaluation,
    }
    result = {
        "status": "complete",
        "arm": "star_v50_adaptive",
        "command": shlex.join(sys.argv),
        "checkpoint": {
            "artifact": str(artifact),
            "sha256": _sha256(artifact),
            "model": base_manifest["model"],
            "training": training,
        },
        "protocol": {
            "tasks": list(TASKS),
            "num_fewshot": 0,
            "lm_eval_batch_size": args.lm_eval_batch_size,
            "max_length": 4096,
            "limit": args.limit,
            "metric_selection": "acc_norm when present, otherwise acc",
            "quality_only": True,
        },
        "environment": {
            "conda": os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name,
            "python": sys.executable,
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "lm_eval": importlib.metadata.version("lm-eval"),
            "gpu": torch.cuda.get_device_name(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    _atomic_json(args.output_dir / "result.json", result)
    (args.output_dir / "summary.md").write_text(_summary(result), encoding="utf-8")
    print(json.dumps({"status": "complete", "average_accuracy": average_accuracy}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
