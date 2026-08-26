#!/usr/bin/env python3
"""Compare exact-proposal C1 block commits with sequential BF16 execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any, Iterable, Sequence


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
    CacheDriftRecord,
    ExactBlockScheduleComparison,
    compare_exact_block_and_sequential_schedules,
    transactional_exact_c1_greedy_trace,
)
from basisserve.core.c1_shadow_kv import (  # noqa: E402
    C1ShadowKeyValueCache,
    ShadowKeyConfig,
)
from basisserve.core.c1_tp_decode import file_sha256  # noqa: E402


FORMAT = "basisserve.c1_exact_block_commit_numerics.v1"


def _parse_int_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise ValueError("block lengths must be a non-empty positive integer list")
    if len(result) != len(set(result)):
        raise ValueError("block lengths must be unique")
    return result


def _dtype(value: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]


def _load_prompts(path: Path) -> tuple[str, ...]:
    prompts: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if isinstance(payload, str):
            prompt = payload
        elif isinstance(payload, dict) and isinstance(payload.get("prompt"), str):
            prompt = payload["prompt"]
        else:
            raise ValueError(
                f"prompt record {line_number} must be a string or contain 'prompt'"
            )
        if not prompt:
            raise ValueError(f"prompt record {line_number} is empty")
        prompts.append(prompt)
    if not prompts:
        raise ValueError("prompt file contains no prompts")
    return tuple(prompts)


def _git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


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
        "maximum": max(values) if values else None,
    }


def _cache_summary(records: Sequence[CacheDriftRecord]) -> dict[str, Any]:
    key_relative = [record.key_relative_l2 for record in records]
    key_absolute = [record.key_maximum_absolute_error for record in records]
    key_cosine = [record.key_cosine_similarity for record in records]
    value_relative = [record.c1_value_relative_l2 for record in records]
    value_absolute = [record.c1_value_maximum_absolute_error for record in records]
    value_cosine = [record.c1_value_cosine_similarity for record in records]
    return {
        "records": len(records),
        "key_relative_l2": _distribution(key_relative),
        "key_maximum_absolute_error": _distribution(key_absolute),
        "key_minimum_cosine_similarity": min(key_cosine) if key_cosine else None,
        "c1_value_relative_l2": _distribution(value_relative),
        "c1_value_maximum_absolute_error": _distribution(value_absolute),
        "c1_value_minimum_cosine_similarity": (
            min(value_cosine) if value_cosine else None
        ),
    }


def _layerwise_cache_summary(
    records: Sequence[CacheDriftRecord],
) -> list[dict[str, Any]]:
    by_layer: dict[int, list[CacheDriftRecord]] = {}
    for record in records:
        by_layer.setdefault(record.layer_index, []).append(record)
    return [
        {"layer_index": layer, **_cache_summary(layer_records)}
        for layer, layer_records in sorted(by_layer.items())
    ]


def _comparison_record(comparison: ExactBlockScheduleComparison) -> dict[str, Any]:
    token_delta = [
        block - sequential
        for sequential, block in zip(
            comparison.sequential_token_nlls,
            comparison.block_token_nlls,
            strict=True,
        )
    ]
    mean_sequential = comparison.sequential_nll_sum / comparison.evaluated_tokens
    mean_block = comparison.block_nll_sum / comparison.evaluated_tokens
    worst_cache = sorted(
        comparison.cache_drift,
        key=lambda record: max(record.key_relative_l2, record.c1_value_relative_l2),
        reverse=True,
    )[:20]
    return {
        "block_length": comparison.block_length,
        "evaluated_tokens": comparison.evaluated_tokens,
        "block_count": comparison.block_count,
        "sequential_mean_nll": mean_sequential,
        "block_mean_nll": mean_block,
        "mean_nll_delta": mean_block - mean_sequential,
        "ppl_ratio_block_over_sequential": math.exp(mean_block - mean_sequential),
        "token_nll_delta": _distribution(token_delta),
        "top1_agreement": comparison.top1_agreement,
        "top1_agreements": comparison.top1_agreements,
        "top1_comparisons": comparison.top1_comparisons,
        "first_top1_disagreement": comparison.first_top1_disagreement,
        "sequential_label_top1_matches": comparison.sequential_label_top1_matches,
        "block_label_top1_matches": comparison.block_label_top1_matches,
        "kl_sequential_to_block": _distribution(comparison.kl_sequential_to_block),
        "logit_relative_l2_by_block": _distribution(
            comparison.logit_relative_l2_by_block
        ),
        "logit_maximum_absolute_error_by_block": _distribution(
            comparison.logit_maximum_absolute_error_by_block
        ),
        "mean_top5_overlap": comparison.mean_top5_overlap,
        "sequential_top1_margin": _distribution(comparison.sequential_top1_margins),
        "block_top1_margin": _distribution(comparison.block_top1_margins),
        "cache_drift": _cache_summary(comparison.cache_drift),
        "layerwise_cache_drift": _layerwise_cache_summary(comparison.cache_drift),
        "worst_cache_drift_records": [asdict(record) for record in worst_cache],
    }


def _aggregate(
    comparisons: Iterable[ExactBlockScheduleComparison],
    *,
    block_length: int,
) -> dict[str, Any]:
    rows = tuple(comparisons)
    evaluated = sum(row.evaluated_tokens for row in rows)
    sequential_nll = sum(row.sequential_nll_sum for row in rows)
    block_nll = sum(row.block_nll_sum for row in rows)
    top1_agreements = sum(row.top1_agreements for row in rows)
    top1_comparisons = sum(row.top1_comparisons for row in rows)
    token_delta = [
        block - sequential
        for row in rows
        for sequential, block in zip(
            row.sequential_token_nlls,
            row.block_token_nlls,
            strict=True,
        )
    ]
    kl = [value for row in rows for value in row.kl_sequential_to_block]
    cache = [record for row in rows for record in row.cache_drift]
    mean_sequential = sequential_nll / evaluated
    mean_block = block_nll / evaluated
    return {
        "block_length": int(block_length),
        "prompts": len(rows),
        "evaluated_tokens": evaluated,
        "blocks": sum(row.block_count for row in rows),
        "sequential_mean_nll": mean_sequential,
        "block_mean_nll": mean_block,
        "mean_nll_delta": mean_block - mean_sequential,
        "ppl_ratio_block_over_sequential": math.exp(mean_block - mean_sequential),
        "token_nll_delta": _distribution(token_delta),
        "top1_agreement": top1_agreements / top1_comparisons,
        "top1_disagreement_prompts": sum(
            row.first_top1_disagreement is not None for row in rows
        ),
        "sequential_exact_proposal_matches": sum(
            row.sequential_label_top1_matches for row in rows
        ),
        "block_exact_proposal_matches": sum(
            row.block_label_top1_matches for row in rows
        ),
        "kl_sequential_to_block": _distribution(kl),
        "mean_top5_overlap": (
            sum(row.top5_overlap_fraction_sum for row in rows) / top1_comparisons
        ),
        "cache_drift": _cache_summary(cache),
        "layerwise_cache_drift": _layerwise_cache_summary(cache),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Exact-proposal C1 block-commit numerical oracle",
        "",
        "This isolates block-versus-sequential BF16 execution. Candidate tokens "
        "come from the transactional one-token exact-Key C1 path used by strict "
        "replay; Shadow-Key is not used.",
        "",
        "| Block | Tokens | Top-1 agreement | Prompts diverged | Mean NLL delta | "
        "PPL ratio | Mean KL |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['block_length']} | {row['evaluated_tokens']} | "
            f"{row['top1_agreement']:.8f} | {row['top1_disagreement_prompts']} | "
            f"{row['mean_nll_delta']:.9e} | "
            f"{row['ppl_ratio_block_over_sequential']:.9f} | "
            f"{row['kl_sequential_to_block']['mean']:.9e} |"
        )
    lines.extend(
        [
            "",
            "A top-1 disagreement is an implementation-schedule difference, not by "
            "itself a task failure. This report contains no Shadow-Key proposal error "
            "and no production throughput measurement.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--c1-export", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--block-lengths", default="2,4,8,16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
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
    model_path = Path(args.model_path).expanduser().resolve()
    c1_export = Path(args.c1_export).expanduser().resolve()
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    block_lengths = _parse_int_csv(args.block_lengths)
    prompts = _load_prompts(prompt_path)
    dtype = _dtype(args.dtype)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real-model block-commit oracle requires CUDA")
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    model.eval()
    replacements = install_qwen3_gqa_vo_als_export(model, c1_export)
    num_layers = int(model.config.num_hidden_layers)
    cache_config = ShadowKeyConfig(
        bits=16,
        group_size=args.cache_group_size,
        recent_exact_window=0,
    )
    comparisons_by_length: dict[int, list[ExactBlockScheduleComparison]] = {
        length: [] for length in block_lengths
    }
    prompt_records: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_ids = encoded.input_ids.to(device)
        sequential_cache = C1ShadowKeyValueCache(
            num_layers=num_layers,
            config=cache_config,
        )
        sequential_trace = transactional_exact_c1_greedy_trace(
            model,
            prompt_ids,
            cache=sequential_cache,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        exact_ids = sequential_trace.token_ids
        if int(exact_ids.shape[1]) == int(prompt_ids.shape[1]):
            raise RuntimeError(
                "exact proposal generator emitted no continuation tokens"
            )
        configuration_records = []
        for block_length in block_lengths:
            comparison = compare_exact_block_and_sequential_schedules(
                model,
                exact_ids,
                sequential_cache=sequential_cache,
                block_cache=C1ShadowKeyValueCache(
                    num_layers=num_layers,
                    config=cache_config,
                ),
                prefill_length=int(prompt_ids.shape[1]),
                block_length=block_length,
                sequential_reference=sequential_trace.schedule,
            )
            if comparison.sequential_label_top1_matches != comparison.evaluated_tokens:
                raise RuntimeError(
                    "sequential schedule did not reproduce the exact proposal tokens"
                )
            comparisons_by_length[block_length].append(comparison)
            configuration_records.append(_comparison_record(comparison))
        prompt_records.append(
            {
                "prompt_index": prompt_index,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_tokens": int(prompt_ids.shape[1]),
                "exact_continuation_tokens": int(
                    exact_ids.shape[1] - prompt_ids.shape[1]
                ),
                "configurations": configuration_records,
            }
        )

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
            "prompt_file": str(prompt_path),
            "prompt_file_sha256": file_sha256(prompt_path),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_worktree_dirty": bool(_git_value("status", "--porcelain")),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "dtype": str(dtype),
            "attention_backend": "eager",
            "block_lengths": list(block_lengths),
            "max_new_tokens": args.max_new_tokens,
            "c1_replaced_layers": len(replacements),
            "proposal_source": "transactional one-token sequential exact-Key C1 greedy",
            "shadow_key_used": False,
            "source_sha256": {
                "c1_shadow_kv.py": file_sha256(
                    REPO_ROOT / "basisserve/core/c1_shadow_kv.py"
                ),
                "c1_block_commit.py": file_sha256(
                    REPO_ROOT / "basisserve/core/c1_block_commit.py"
                ),
                "eval_c1_block_commit_numerics.py": file_sha256(Path(__file__)),
            },
        },
        "aggregate": [
            _aggregate(
                comparisons_by_length[block_length],
                block_length=block_length,
            )
            for block_length in block_lengths
        ],
        "prompts": prompt_records,
        "limitations": [
            "exact-proposal numerical isolation only; Shadow-Key is not exercised",
            "single GPU, batch size one, eager attention",
            "reference Python/Hugging Face execution is not a throughput benchmark",
            "task-level answer and MCQ accuracy are not measured",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
