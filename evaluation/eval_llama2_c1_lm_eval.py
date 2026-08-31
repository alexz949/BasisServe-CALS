#!/usr/bin/env python3
"""Run lm-evaluation-harness on dense or folded-C1 Llama-2-7B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_llama2_mha_v25_comparison import (  # noqa: E402
    _install_c1,
    _sha256,
    _validate_model_config,
)


DEFAULT_TASKS = (
    "openbookqa",
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
)


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("dense", "c1_joint"), required=True)
    parser.add_argument("--factor-dir")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--batch-size", default="8")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=float)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.arm == "c1_joint" and args.factor_dir is None:
        raise ValueError("c1_joint requires --factor-dir")
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.torch_num_threads)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=_dtype(args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).to(device).eval()
    _validate_model_config(model.config)
    install_records: list[dict[str, Any]] = []
    factor_provenance: dict[str, Any] | None = None
    if args.arm == "c1_joint":
        factor_dir = Path(args.factor_dir).expanduser().resolve()
        install_records, fit_results = _install_c1(model, factor_dir, model_path)
        factor_provenance = {
            "results_json": str(factor_dir / "results.json"),
            "results_sha256": _sha256(factor_dir / "results.json"),
            "fit_config": fit_results["fit_config"],
            "aggregate": fit_results["aggregate"],
        }
    model.config.use_cache = True
    task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    if not task_names:
        raise ValueError("--tasks must contain at least one task")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=str(device),
        batch_size=args.batch_size,
        max_length=args.max_length,
        add_bos_token=False,
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_names,
        task_manager=lm_eval.tasks.TaskManager(),
        limit=args.limit,
        log_samples=args.log_samples,
    )
    if results is None:
        raise RuntimeError("lm-evaluation-harness returned no results")
    payload = {
        "format": "cals.llama2_7b.c1_lm_eval.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "lm_eval": getattr(lm_eval, "__version__", "unknown"),
        },
        "model": str(model_path),
        "arm": args.arm,
        "dtype": args.dtype,
        "tasks": task_names,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "limit": args.limit,
        "factor_provenance": factor_provenance,
        "install_records": install_records,
        "evaluation": results,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(make_table(results), flush=True)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
