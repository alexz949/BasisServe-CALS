#!/usr/bin/env python3
"""Evaluate teacher-forced C1 NLL/PPL under sequential and block KV schedules."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from basisserve.checkpoint.gqa_vo_qwen3 import (  # noqa: E402
    install_qwen3_gqa_vo_als_export,
)
from basisserve.core.c1_block_commit import (  # noqa: E402
    TargetScheduleResult,
    block_scheduled_teacher_forced_nll,
)
from basisserve.core.c1_shadow_kv import (  # noqa: E402
    C1ShadowKeyValueCache,
    ShadowKeyConfig,
)
from basisserve.core.c1_tp_decode import file_sha256  # noqa: E402
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids  # noqa: E402


FORMAT = "basisserve.c1_block_scheduled_teacher_forced_ppl.v1"


def _parse_int_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 1 for item in result):
        raise ValueError("block lengths must be a non-empty list of integers above one")
    if len(result) != len(set(result)):
        raise ValueError("block lengths must be unique")
    return result


def _dtype(value: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    return float(torch.quantile(tensor, quantile, interpolation="linear"))


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _cache(num_layers: int, group_size: int) -> C1ShadowKeyValueCache:
    return C1ShadowKeyValueCache(
        num_layers=num_layers,
        config=ShadowKeyConfig(
            bits=16,
            group_size=group_size,
            recent_exact_window=0,
        ),
    )


def _paired_record(
    sequential: TargetScheduleResult,
    block: TargetScheduleResult,
) -> dict[str, Any]:
    if sequential.evaluated_tokens != block.evaluated_tokens:
        raise RuntimeError("PPL schedules evaluated different token counts")
    deltas = [
        observed - reference
        for reference, observed in zip(
            sequential.token_nlls,
            block.token_nlls,
            strict=True,
        )
    ]
    agreements = sum(
        reference == observed
        for reference, observed in zip(
            sequential.top1_token_ids,
            block.top1_token_ids,
            strict=True,
        )
    )
    first_disagreement = next(
        (
            index
            for index, (reference, observed) in enumerate(
                zip(
                    sequential.top1_token_ids,
                    block.top1_token_ids,
                    strict=True,
                )
            )
            if reference != observed
        ),
        None,
    )
    mean_sequential = sequential.mean_nll
    mean_block = block.mean_nll
    return {
        "block_length": block.block_length,
        "evaluated_tokens": block.evaluated_tokens,
        "blocks": block.block_count,
        "sequential_nll_sum": sequential.nll_sum,
        "block_nll_sum": block.nll_sum,
        "sequential_mean_nll": mean_sequential,
        "block_mean_nll": mean_block,
        "mean_nll_delta": mean_block - mean_sequential,
        "ppl_ratio_block_over_sequential": math.exp(mean_block - mean_sequential),
        "token_nll_delta": _distribution(deltas),
        "top1_agreements": agreements,
        "top1_comparisons": block.evaluated_tokens,
        "top1_agreement": agreements / block.evaluated_tokens,
        "first_top1_disagreement": first_disagreement,
    }


def _aggregate(records: Sequence[dict[str, Any]], block_length: int) -> dict[str, Any]:
    tokens = sum(int(record["evaluated_tokens"]) for record in records)
    sequential_nll = sum(float(record["sequential_nll_sum"]) for record in records)
    block_nll = sum(float(record["block_nll_sum"]) for record in records)
    agreements = sum(int(record["top1_agreements"]) for record in records)
    sequential_mean = sequential_nll / tokens
    block_mean = block_nll / tokens
    return {
        "block_length": int(block_length),
        "samples": len(records),
        "evaluated_tokens": tokens,
        "blocks": sum(int(record["blocks"]) for record in records),
        "sequential_nll_sum": sequential_nll,
        "block_nll_sum": block_nll,
        "sequential_mean_nll": sequential_mean,
        "block_mean_nll": block_mean,
        "sequential_ppl": math.exp(sequential_mean),
        "block_ppl": math.exp(block_mean),
        "mean_nll_delta": block_mean - sequential_mean,
        "ppl_ratio_block_over_sequential": math.exp(block_mean - sequential_mean),
        "top1_agreement": agreements / tokens,
        "samples_with_top1_disagreement": sum(
            record["first_top1_disagreement"] is not None for record in records
        ),
        "sample_mean_nll_delta": _distribution(
            [float(record["mean_nll_delta"]) for record in records]
        ),
        "sample_ppl_ratio": _distribution(
            [float(record["ppl_ratio_block_over_sequential"]) for record in records]
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    lines = [
        "# C1 block-scheduled teacher-forced NLL/PPL",
        "",
        "Fixed corpus tokens are scored under a one-token sequential exact-C1 "
        "cache schedule and under direct exact block-KV commits. Shadow-Key drafting "
        "is not used.",
        "",
        f"Dataset: `{metadata['dataset']}`; split: `{metadata['split']}`; "
        f"sequence length: `{metadata['sequence_length']}`; samples: "
        f"`{metadata['samples']}`.",
        "",
        "| Block | Tokens | Sequential PPL | Block PPL | PPL ratio | Mean NLL delta | "
        "Top-1 agreement | Samples diverged |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['block_length']} | {row['evaluated_tokens']} | "
            f"{row['sequential_ppl']:.8f} | {row['block_ppl']:.8f} | "
            f"{row['ppl_ratio_block_over_sequential']:.9f} | "
            f"{row['mean_nll_delta']:.9e} | {row['top1_agreement']:.8f} | "
            f"{row['samples_with_top1_disagreement']} |"
        )
    lines.extend(
        [
            "",
            "This is schedule-conditioned teacher-forced PPL. It isolates numerical "
            "execution-order effects and is not a free-running generation or MCQ result.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--c1-export", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--prefill-tokens", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--block-lengths", default="2,4,8,16")
    parser.add_argument("--cache-group-size", type=int, default=32)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    model_path = Path(args.model_path).expanduser().resolve()
    c1_export = Path(args.c1_export).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    if args.sequence_length <= 1:
        raise ValueError("sequence-length must be greater than one")
    if not 0 < args.prefill_tokens < args.sequence_length:
        raise ValueError("prefill-tokens must leave at least one scored token")
    if args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    block_lengths = _parse_int_csv(args.block_lengths)
    dtype = _dtype(args.dtype)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "the real-model block-scheduled PPL evaluation requires CUDA"
        )
    model_config_path = model_path / "config.json"
    export_manifest_path = c1_export / "results.json"
    if not model_config_path.is_file():
        raise FileNotFoundError(model_config_path)
    if not export_manifest_path.is_file():
        raise FileNotFoundError(export_manifest_path)
    manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    model_hash = file_sha256(model_config_path)
    if manifest.get("fit_config", {}).get("model_config_sha256") != model_hash:
        raise ValueError("C1 export belongs to another model config")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    input_ids = _token_ids(tokenizer, args.dataset, args.split, args.max_tokens)
    complete_samples = int(input_ids.numel() // args.sequence_length)
    samples = min(complete_samples, args.max_samples)
    if samples <= 0:
        raise ValueError("dataset contains no complete evaluation sequence")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    model.eval()
    replacements = install_qwen3_gqa_vo_als_export(model, c1_export)
    num_layers = int(model.config.num_hidden_layers)
    records_by_length: dict[int, list[dict[str, Any]]] = {
        length: [] for length in block_lengths
    }
    sample_records: list[dict[str, Any]] = []
    for sample_index in range(samples):
        start = sample_index * args.sequence_length
        end = start + args.sequence_length
        sample = input_ids[:, start:end].to(device)
        sequential = block_scheduled_teacher_forced_nll(
            model,
            sample,
            cache=_cache(num_layers, args.cache_group_size),
            prefill_length=args.prefill_tokens,
            block_length=1,
        )
        configurations = []
        for block_length in block_lengths:
            block = block_scheduled_teacher_forced_nll(
                model,
                sample,
                cache=_cache(num_layers, args.cache_group_size),
                prefill_length=args.prefill_tokens,
                block_length=block_length,
            )
            record = _paired_record(sequential, block)
            records_by_length[block_length].append(record)
            configurations.append(record)
        sample_records.append(
            {
                "sample_index": sample_index,
                "source_token_start": start,
                "source_token_end": end,
                "configurations": configurations,
            }
        )
        print(f"[PPL] sample={sample_index + 1}/{samples}", flush=True)

    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "metadata": {
            "cli_parameters": vars(args),
            "model_path": str(model_path),
            "model_config_sha256": model_hash,
            "c1_export": str(c1_export),
            "c1_export_manifest_sha256": file_sha256(export_manifest_path),
            "dataset": args.dataset,
            "split": args.split,
            "sequence_length": args.sequence_length,
            "prefill_tokens": args.prefill_tokens,
            "samples": samples,
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_worktree_dirty": bool(_git_value("status", "--porcelain")),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "dtype": str(dtype),
            "attention_backend": "eager",
            "c1_replaced_layers": len(replacements),
            "shadow_key_used": False,
            "source_sha256": {
                "c1_shadow_kv.py": file_sha256(
                    REPO_ROOT / "basisserve/core/c1_shadow_kv.py"
                ),
                "c1_block_commit.py": file_sha256(
                    REPO_ROOT / "basisserve/core/c1_block_commit.py"
                ),
                "eval_c1_block_scheduled_ppl.py": file_sha256(Path(__file__)),
            },
        },
        "aggregate": [
            _aggregate(records_by_length[length], length) for length in block_lengths
        ],
        "samples": sample_records,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "teacher-forced fixed-token evaluation; not free-running generation",
            "exact block schedule only; Shadow-Key proposal error is excluded",
            "single GPU, batch size one, eager attention",
            "sequential BF16 reference requires one target forward per scored token",
            "reference Python/Hugging Face timings are not serving throughput",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
