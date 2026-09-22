#!/usr/bin/env python3
"""Evaluate one communication-matched Wo-C1 AllGather/AllReduce arm."""

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

from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
    _sha256,
)
from evaluation.eval_qwen3_8b_iclr_quality import (  # noqa: E402
    TASKS,
    summarize_commonsense,
)
from evaluation.eval_qwen3_8b_wo_c1_lr_ar_quality import (  # noqa: E402
    _install_arm,
    _validate_phase1,
)


FORMAT = "basisserve.qwen3_8b.wo_c1_ag_ar_mcq.v1"
ARM_TO_FACTOR = {
    "c1_allgather": "wo_c1_ag",
    "lr_allreduce": "wo_lr_ar_wire",
}


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _csv_tasks(raw: str) -> tuple[str, ...] | None:
    tasks = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not _check(bool(tasks), "at least one MCQ task is required"):
        return None
    if not _check(len(tasks) == len(set(tasks)), "MCQ tasks must be unique"):
        return None
    if not _check(not (set(tasks) - set(TASKS)), "unsupported MCQ task"):
        return None
    return tasks


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    protocol = payload["protocol"]
    lines = [
        "# Qwen3-8B Wo-only AllGather/AllReduce quality",
        "",
        f"- Arm: `{payload['arm']}`",
        f"- C1 AllGather source rank: `{protocol['source_rank']}`",
        f"- Communication-matched AllReduce rank: `{protocol['wire_allreduce_rank']}`",
        f"- Ideal ring bytes/row/rank: `{protocol['ring_bytes_per_row_per_rank']:.0f}`",
    ]
    if payload.get("wikitext2") is not None:
        lines.append(f"- WikiText-2 PPL: `{payload['wikitext2']['ppl']:.8f}`")
    if payload.get("mcq") is not None:
        lines.extend(
            [
                f"- MCQ average: `{payload['mcq']['average_accuracy']:.8f}`",
                "",
                "| Task | Metric | Accuracy |",
                "|---|---|---:|",
            ]
        )
        for row in payload["mcq"]["task_accuracy"]:
            lines.append(
                f"| {row['task']} | {row['metric']} | {float(row['value']):.8f} |"
            )
    lines.extend(["", "## Command", "", "```bash", payload["command"], "```", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARM_TO_FACTOR), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--limit", type=float)
    parser.add_argument("--lm-eval-batch-size", type=int, default=8)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-mcq", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> int:
    tasks = _csv_tasks(args.tasks)
    valid = all(
        (
            tasks is not None,
            _check(torch.cuda.is_available(), "CUDA is required"),
            _check(args.lm_eval_batch_size > 0, "lm-eval batch size must be positive"),
            _check(args.ppl_batch_size > 0, "PPL batch size must be positive"),
            _check(not (args.skip_ppl and args.skip_mcq), "both stages are disabled"),
            _check(args.limit is None or args.limit > 0, "limit must be positive"),
        )
    )
    if not valid:
        return 2
    tasks = tuple(tasks or ())
    model_path = args.model.expanduser().resolve()
    factor_dir = args.factor_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    result_path = output_dir / "results.json"
    if not _check(not result_path.exists(), f"result already exists: {result_path}"):
        return 2

    phase1, records = _validate_phase1(factor_dir, model_path=model_path)
    run_signature = phase1["method"]["run_signature"]
    source_rank = int(run_signature["source_rank"])
    first_layer = phase1["layers"][0]
    geometry = first_layer["geometry"]
    communication = first_layer["communication"]
    wire_rank = int(geometry["wire_matched_lr_ar_rank"])
    ring_bytes = float(communication["c1_allgather"])
    if not _check(
        ring_bytes == float(communication["lr_ar_wire_matched"]),
        "AllGather and AllReduce ring traffic are not matched",
    ):
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": str(device)},
    ).eval()
    model.config.use_cache = False
    installation = _install_arm(
        model,
        arm=ARM_TO_FACTOR[args.arm],
        phase1_dir=factor_dir,
        records=records,
    )

    wikitext2 = None
    if not args.skip_ppl:
        ppl_started = time.perf_counter()
        wikitext2 = _eval_ppl_fp32_loss(
            model,
            tokenizer,
            dataset="wikitext2",
            split="test",
            seqlen=2048,
            batch_size=args.ppl_batch_size,
            max_samples=None,
            max_tokens=None,
        )
        wikitext2["elapsed_seconds"] = time.perf_counter() - ppl_started
        print(f"[WikiText-2] arm={args.arm} ppl={wikitext2['ppl']:.9f}", flush=True)

    mcq = None
    if not args.skip_mcq:
        mcq_started = time.perf_counter()
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
            tasks=list(tasks),
            num_fewshot=0,
            task_manager=TaskManager(),
            log_samples=False,
            limit=args.limit,
        )
        if not _check(evaluation is not None, "lm-eval returned no results"):
            return 2
        summarized = summarize_commonsense(evaluation, tasks)
        if not _check(summarized is not None, "MCQ metrics could not be summarized"):
            return 2
        task_rows, average_accuracy = summarized or ([], 0.0)
        mcq = {
            "task_accuracy": task_rows,
            "average_accuracy": average_accuracy,
            "evaluation": evaluation,
            "elapsed_seconds": time.perf_counter() - mcq_started,
        }
        print(f"[MCQ] arm={args.arm} average={average_accuracy:.9f}", flush=True)

    payload = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "arm": args.arm,
        "model": phase1["model"],
        "factor_bank": {
            "directory": str(factor_dir),
            "results_sha256": _sha256(factor_dir / "results.json"),
            "format": phase1["format"],
        },
        "protocol": {
            "scope": phase1["method"]["scope"],
            "tp_size": int(run_signature["tp_size"]),
            "source_rank": source_rank,
            "source_width": int(geometry["source_width"]),
            "retained_ratio": source_rank / int(geometry["source_width"]),
            "wire_allreduce_rank": wire_rank,
            "ring_bytes_per_row_per_rank": ring_bytes,
            "quality_runtime": "factorized maps folded into equivalent BF16 o_proj weights",
            "tasks": list(tasks) if not args.skip_mcq else [],
            "num_fewshot": 0,
            "lm_eval_batch_size": args.lm_eval_batch_size,
            "max_length": 4096,
            "limit": args.limit,
            "metric_selection": "acc_norm when present, otherwise acc",
            "wikitext2_sequence_length": None if args.skip_ppl else 2048,
            "wikitext2_batch_size": None if args.skip_ppl else args.ppl_batch_size,
        },
        "installation": installation,
        "wikitext2": wikitext2,
        "mcq": mcq,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name,
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "lm_eval": importlib.metadata.version("lm-eval"),
            "device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    _atomic_json(result_path, payload)
    (output_dir / "summary.md").write_text(
        _summary_markdown(payload), encoding="utf-8"
    )
    print(f"[Complete] {result_path}", flush=True)
    return 0


def main() -> None:
    status = evaluate(parse_args())
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()
