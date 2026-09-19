#!/usr/bin/env python3
"""Evaluate Dense, C1, group-SVD, and PaLU with one Section-3 quality path."""

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
from evaluation.eval_gqa_palu_m_wikitext import (  # noqa: E402
    _load_checkpoint as _load_palu_checkpoint,
    install_palu_m_factors,
)
from evaluation.eval_qwen3_32b_c1_wikitext import (  # noqa: E402
    _load_results as _load_c1_results,
    activate_model_profile as _activate_c1_model_profile,
    install_c1_factors,
)
from evaluation.eval_qwen3_8b_iclr_quality import (  # noqa: E402
    TASKS,
    summarize_commonsense,
)


FORMAT = "basisserve.qwen3_8b.section3.short_quality.v1"
_activate_c1_model_profile("qwen3_8b")


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


def _tasks(raw: str) -> tuple[str, ...] | None:
    tasks = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not _check(bool(tasks), "at least one task is required"):
        return None
    if not _check(len(tasks) == len(set(tasks)), "tasks must be unique"):
        return None
    if not _check(not (set(tasks) - set(TASKS)), "unsupported task requested"):
        return None
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("dense", "c1", "group_svd", "palu_m", "palu_g4"),
        required=True,
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--limit", type=float)
    parser.add_argument("--lm-eval-batch-size", type=int, default=8)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument("--ppl-max-samples", type=int)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-mcq", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _install(
    model: torch.nn.Module,
    *,
    method: str,
    checkpoint_dir: Path | None,
    model_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if method == "dense":
        return [], {"method": "dense", "rank": 128}
    assert checkpoint_dir is not None
    if method in {"c1", "group_svd"}:
        result = _load_c1_results(checkpoint_dir, model_path)
        installation = install_c1_factors(model, checkpoint_dir, result)
        return installation, {
            "method": method,
            "rank": int(result["fit_config"]["cache_rank_per_head"]),
            "checkpoint_format": result["format"],
            "checkpoint_sha256": _sha256(checkpoint_dir / "results.json"),
            "heldout_rel_mse": result.get("aggregate", {}).get(
                "mean_heldout_factor_dtype_relative_mse"
            ),
        }
    manifest, factors = _load_palu_checkpoint(checkpoint_dir, model_path)
    compression = manifest["compression"]
    layer_ranks = [
        [int(rank) for rank in ranks] for ranks in compression["layer_ranks"]
    ]
    installation = install_palu_m_factors(
        model,
        factors,
        layer_ranks=layer_ranks,
        head_dim=int(compression["head_dim"]),
        require_cuda_resident=True,
    )
    return installation, {
        "method": method,
        "rank": int(compression["equivalent_rank_target"]),
        "group_size": int(compression["head_group_size"]),
        "nominal_group_rank": int(compression["nominal_group_rank"]),
        "checkpoint_format": manifest["format"],
        "checkpoint_sha256": _sha256(checkpoint_dir / "manifest.json"),
    }


def _summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3-8B Section-3 short-context quality",
        "",
        f"- Method: `{payload['method']}`",
        f"- Rank: `{payload['checkpoint']['rank']}`",
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


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> int:
    tasks = _tasks(args.tasks)
    checkpoint_required = args.method != "dense"
    valid = all(
        (
            tasks is not None,
            _check(torch.cuda.is_available(), "CUDA is required"),
            _check(
                checkpoint_required == (args.checkpoint_dir is not None),
                "checkpoint-dir is required exactly for compressed methods",
            ),
            _check(not (args.skip_ppl and args.skip_mcq), "both stages are disabled"),
            _check(args.lm_eval_batch_size > 0, "MCQ batch size must be positive"),
            _check(args.ppl_batch_size > 0, "PPL batch size must be positive"),
        )
    )
    if not valid:
        return 2
    tasks = tuple(tasks or ())
    model_path = args.model.expanduser().resolve()
    checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir is not None
        else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    result_path = output_dir / "results.json"
    if not _check(not result_path.exists(), f"result already exists: {result_path}"):
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
    installation, checkpoint = _install(
        model,
        method=args.method,
        checkpoint_dir=checkpoint_dir,
        model_path=model_path,
    )
    wikitext2 = None
    if not args.skip_ppl:
        stage_started = time.perf_counter()
        wikitext2 = _eval_ppl_fp32_loss(
            model,
            tokenizer,
            dataset="wikitext2",
            split="test",
            seqlen=2048,
            batch_size=args.ppl_batch_size,
            max_samples=args.ppl_max_samples,
            max_tokens=None,
        )
        wikitext2["elapsed_seconds"] = time.perf_counter() - stage_started
        print(f"[WikiText-2] method={args.method} ppl={wikitext2['ppl']:.9f}", flush=True)
    mcq = None
    if not args.skip_mcq:
        stage_started = time.perf_counter()
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
        task_rows, average = summarized or ([], 0.0)
        mcq = {
            "task_accuracy": task_rows,
            "average_accuracy": average,
            "evaluation": evaluation,
            "elapsed_seconds": time.perf_counter() - stage_started,
        }
        print(f"[MCQ] method={args.method} average={average:.9f}", flush=True)
    payload = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "method": args.method,
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "checkpoint": {
            "directory": str(checkpoint_dir) if checkpoint_dir is not None else None,
            **checkpoint,
        },
        "protocol": {
            "dense_k": True,
            "full_dense_attention": True,
            "routing": False,
            "uniform_value_rank": True,
            "tasks": list(tasks) if not args.skip_mcq else [],
            "num_fewshot": 0,
            "lm_eval_batch_size": args.lm_eval_batch_size,
            "limit": args.limit,
            "wikitext2_sequence_length": None if args.skip_ppl else 2048,
            "wikitext2_batch_size": None if args.skip_ppl else args.ppl_batch_size,
            "wikitext2_max_samples": args.ppl_max_samples,
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
    (output_dir / "summary.md").write_text(_summary(payload), encoding="utf-8")
    print(f"[Complete] {result_path}", flush=True)
    return 0


def main() -> None:
    status = evaluate(parse_args())
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()
