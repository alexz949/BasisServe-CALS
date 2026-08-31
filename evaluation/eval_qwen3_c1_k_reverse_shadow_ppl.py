#!/usr/bin/env python3
"""End-to-end teacher-forced NLL/PPL for block-scheduled Reverse ShadowKV."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any

from safetensors.torch import load_file
import torch
import transformers
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import (  # noqa: E402
    GQATiedVOQwen3Attention,
    install_qwen3_gqa_vo_als_export,
)
from basisserve.core.c1_block_commit import (  # noqa: E402
    TargetScheduleResult,
    block_scheduled_teacher_forced_nll,
)
from basisserve.core.c1_k_reverse_shadow import ReverseShadowConfig  # noqa: E402
from evaluation.eval_c1_block_scheduled_ppl import (  # noqa: E402
    _cache,
    _distribution,
    _dtype,
)


FORMAT = "basisserve.qwen3_8b.c1_k_reverse_shadow_block_ppl.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise ValueError("budgets must be unique positive integers")
    return result


def _parse_layers(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if any(item < 0 for item in result) or len(set(result)) != len(result):
        raise ValueError("full-exact layers must be unique nonnegative integers")
    return result


def _parse_quest_supports(value: str) -> tuple[str, ...]:
    result = tuple(item for item in value.split(",") if item)
    allowed = {"physical_shared", "per_query_head"}
    if not result or len(set(result)) != len(result) or not set(result) <= allowed:
        raise ValueError(
            "QUEST supports must be unique values from physical_shared,per_query_head"
        )
    return result


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _attention_modules(model: torch.nn.Module) -> list[GQATiedVOQwen3Attention]:
    modules = [layer.self_attn for layer in model.model.layers]
    if not all(isinstance(module, GQATiedVOQwen3Attention) for module in modules):
        raise TypeError("every Qwen3 attention layer must contain the C1 replacement")
    return modules


def _set_policy(
    modules: list[GQATiedVOQwen3Attention],
    *,
    budget: int | None,
    full_exact_layers: set[int],
    page_size: int,
    landmark_dtype: str,
    quest_support: str,
    force_last_page: bool,
) -> None:
    for layer, module in enumerate(modules):
        config = None
        if budget is not None and layer not in full_exact_layers:
            config = ReverseShadowConfig(
                page_size=page_size,
                exact_token_budget=budget,
                recent_exact_window=1 if force_last_page else 0,
                landmarks_per_page=1,
                selector="quest_minmax",
                landmark_dtype=landmark_dtype,
                quest_support=quest_support,
            )
        module.set_reverse_shadow_config(config)


def _runtime_statistics(
    modules: list[GQATiedVOQwen3Attention], full_exact_layers: set[int]
) -> dict[str, float]:
    totals = {
        "queries": 0.0,
        "physical_valid_tokens": 0.0,
        "query_valid_tokens": 0.0,
        "selected_tokens": 0.0,
        "query_selected_tokens": 0.0,
        "selected_pages": 0.0,
        "logical_selected_pages": 0.0,
        "cpu_exact_key_bytes_fetched": 0.0,
        "selection_qk_flops": 0.0,
        "sparse_exact_qk_flops": 0.0,
        "sparse_c1_value_flops": 0.0,
        "resident_selector_metadata_bytes": 0.0,
    }
    for layer, module in enumerate(modules):
        if layer in full_exact_layers:
            continue
        row = module.reverse_shadow_statistics()
        for name in (
            "queries",
            "physical_valid_tokens",
            "query_valid_tokens",
            "selected_tokens",
            "query_selected_tokens",
            "selected_pages",
            "logical_selected_pages",
            "cpu_exact_key_bytes_fetched",
            "selection_qk_flops",
            "sparse_exact_qk_flops",
            "sparse_c1_value_flops",
        ):
            totals[name] += row[name]
        totals["resident_selector_metadata_bytes"] += row[
            "maximum_resident_selector_metadata_bytes"
        ]
    totals["selected_token_fraction"] = (
        totals["selected_tokens"] / totals["physical_valid_tokens"]
        if totals["physical_valid_tokens"]
        else 0.0
    )
    totals["query_selected_token_fraction"] = (
        totals["query_selected_tokens"] / totals["query_valid_tokens"]
        if totals["query_valid_tokens"]
        else 0.0
    )
    return totals


def _paired_record(
    dense: TargetScheduleResult,
    sparse: TargetScheduleResult,
    *,
    budget: int,
    quest_support: str,
    runtime: dict[str, float],
) -> dict[str, Any]:
    if dense.evaluated_tokens != sparse.evaluated_tokens:
        raise RuntimeError("dense and sparse schedules evaluated different tokens")
    deltas = [
        candidate - reference
        for reference, candidate in zip(
            dense.token_nlls, sparse.token_nlls, strict=True
        )
    ]
    agreements = sum(
        reference == candidate
        for reference, candidate in zip(
            dense.top1_token_ids, sparse.top1_token_ids, strict=True
        )
    )
    first_disagreement = next(
        (
            index
            for index, (reference, candidate) in enumerate(
                zip(
                    dense.top1_token_ids,
                    sparse.top1_token_ids,
                    strict=True,
                )
            )
            if reference != candidate
        ),
        None,
    )
    return {
        "quest_support": quest_support,
        "exact_token_budget": budget,
        "evaluated_tokens": dense.evaluated_tokens,
        "blocks": sparse.block_count,
        "dense_nll_sum": dense.nll_sum,
        "sparse_nll_sum": sparse.nll_sum,
        "dense_mean_nll": dense.mean_nll,
        "sparse_mean_nll": sparse.mean_nll,
        "mean_nll_delta": sparse.mean_nll - dense.mean_nll,
        "ppl_ratio_sparse_over_dense": math.exp(sparse.mean_nll - dense.mean_nll),
        "token_nll_delta": _distribution(deltas),
        "top1_agreements": agreements,
        "top1_comparisons": dense.evaluated_tokens,
        "top1_agreement": agreements / dense.evaluated_tokens,
        "first_top1_disagreement": first_disagreement,
        "runtime_logical": runtime,
    }


def _aggregate(
    records: list[dict[str, Any]], quest_support: str, budget: int
) -> dict[str, Any]:
    tokens = sum(int(row["evaluated_tokens"]) for row in records)
    dense_nll = sum(float(row["dense_nll_sum"]) for row in records)
    sparse_nll = sum(float(row["sparse_nll_sum"]) for row in records)
    dense_mean = dense_nll / tokens
    sparse_mean = sparse_nll / tokens
    sample_deltas = [float(row["mean_nll_delta"]) for row in records]
    paired_se = (
        statistics.stdev(sample_deltas) / math.sqrt(len(sample_deltas))
        if len(sample_deltas) > 1
        else None
    )
    runtime_names = tuple(records[0]["runtime_logical"])
    runtime = {}
    for name in runtime_names:
        values = [float(row["runtime_logical"][name]) for row in records]
        runtime[name] = (
            max(values) if name == "resident_selector_metadata_bytes" else sum(values)
        )
    runtime["selected_token_fraction"] = (
        runtime["selected_tokens"] / runtime["physical_valid_tokens"]
        if runtime["physical_valid_tokens"]
        else 0.0
    )
    runtime["query_selected_token_fraction"] = (
        runtime["query_selected_tokens"] / runtime["query_valid_tokens"]
        if runtime["query_valid_tokens"]
        else 0.0
    )
    agreements = sum(int(row["top1_agreements"]) for row in records)
    return {
        "quest_support": quest_support,
        "exact_token_budget": budget,
        "samples": len(records),
        "evaluated_tokens": tokens,
        "dense_mean_nll": dense_mean,
        "sparse_mean_nll": sparse_mean,
        "dense_ppl": math.exp(dense_mean),
        "sparse_ppl": math.exp(sparse_mean),
        "mean_nll_delta": sparse_mean - dense_mean,
        "paired_sample_standard_error": paired_se,
        "ppl_ratio_sparse_over_dense": math.exp(sparse_mean - dense_mean),
        "top1_agreement": agreements / tokens,
        "samples_with_top1_disagreement": sum(
            row["first_top1_disagreement"] is not None for row in records
        ),
        "sample_mean_nll_delta": _distribution(sample_deltas),
        "runtime_logical": runtime,
    }


def _markdown(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    lines = [
        "# Qwen3 C1 K-only Reverse ShadowKV block-scheduled NLL/PPL",
        "",
        "Dense-C1 and Reverse ShadowKV score identical fixed tokens with the "
        "same direct target-block commit schedule.",
        "",
        f"Windows: `{metadata['samples']} x {metadata['sequence_length']}`; "
        f"block length: `{metadata['block_length']}`; full-exact layers: "
        f"`{metadata['full_exact_layers']}`.",
        "",
        "| QUEST support | Budget | Tokens | Dense-C1 PPL | Sparse PPL | "
        "PPL ratio | Mean NLL delta | Paired SE | Physical K fraction | "
        "Query support fraction | Top-1 agreement |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        paired_se = row["paired_sample_standard_error"]
        runtime = row["runtime_logical"]
        lines.append(
            f"| {row['quest_support']} | {row['exact_token_budget']} | "
            f"{row['evaluated_tokens']} | "
            f"{row['dense_ppl']:.8f} | {row['sparse_ppl']:.8f} | "
            f"{row['ppl_ratio_sparse_over_dense']:.8f} | "
            f"{row['mean_nll_delta']:+.8e} | "
            f"{'n/a' if paired_se is None else f'{paired_se:.3e}'} | "
            f"{runtime['selected_token_fraction']:.8f} | "
            f"{runtime['query_selected_token_fraction']:.8f} | "
            f"{row['top1_agreement']:.8f} |"
        )
    lines.extend(
        [
            "",
            "This is an all-layer, end-to-end teacher-forced quality oracle. "
            "Exact K remains on GPU and QUEST metadata is rebuilt in Python; "
            "runtime is therefore not a CPU-offload latency measurement.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--prefill-tokens", type=int, default=1)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--exact-token-budgets", default="512,1024")
    parser.add_argument(
        "--quest-supports",
        default="physical_shared,per_query_head",
        help="comma-separated QUEST selection granularities",
    )
    parser.add_argument(
        "--force-last-page",
        action="store_true",
        help="reserve one page from the budget for the current final page",
    )
    parser.add_argument("--full-exact-layers", default="0,1")
    parser.add_argument(
        "--landmark-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    model_path = args.model_path.expanduser().resolve()
    c1_export = args.c1_export.expanduser().resolve()
    windows_path = args.windows.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    if min(
        args.samples,
        args.sequence_length,
        args.prefill_tokens,
        args.block_length,
        args.page_size,
    ) <= 0:
        raise ValueError("evaluation sizes must be positive")
    if args.prefill_tokens >= args.sequence_length:
        raise ValueError("prefill must leave scored tokens")
    budgets = _parse_positive_ints(args.exact_token_budgets)
    quest_supports = _parse_quest_supports(args.quest_supports)
    full_exact_layers = set(_parse_layers(args.full_exact_layers))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the end-to-end Reverse ShadowKV PPL oracle needs CUDA")

    windows_manifest_path = windows_path.parent / "manifest.json"
    windows_manifest = json.loads(windows_manifest_path.read_text(encoding="utf-8"))
    if _sha256(windows_path) != windows_manifest["artifact"]["sha256"]:
        raise ValueError("window artifact hash differs from its manifest")
    if _sha256(model_path / "config.json") != windows_manifest["model"][
        "config_sha256"
    ]:
        raise ValueError("windows belong to another model")
    input_ids = load_file(str(windows_path), device="cpu")["input_ids"]
    stop = args.window_start + args.samples
    if (
        int(input_ids.shape[1]) < args.sequence_length
        or args.window_start < 0
        or stop > len(input_ids)
    ):
        raise ValueError("requested windows are outside the fixed bank")
    input_ids = input_ids[
        args.window_start:stop, : args.sequence_length
    ].to(torch.long)

    dtype = _dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    model.eval()
    replacements = install_qwen3_gqa_vo_als_export(model, c1_export)
    modules = _attention_modules(model)
    if any(layer < 0 or layer >= len(modules) for layer in full_exact_layers):
        raise ValueError("full-exact layer index is outside the model")

    records_by_configuration: dict[tuple[str, int], list[dict[str, Any]]] = {
        (support, budget): []
        for support in quest_supports
        for budget in budgets
    }
    sample_records = []
    for sample_offset, sample in enumerate(input_ids):
        sample_index = args.window_start + sample_offset
        tokens = sample.unsqueeze(0).to(device)
        _set_policy(
            modules,
            budget=None,
            full_exact_layers=full_exact_layers,
            page_size=args.page_size,
            landmark_dtype=args.landmark_dtype,
            quest_support="physical_shared",
            force_last_page=False,
        )
        dense = block_scheduled_teacher_forced_nll(
            model,
            tokens,
            cache=_cache(len(modules), 32),
            prefill_length=args.prefill_tokens,
            block_length=args.block_length,
        )
        configurations = []
        for quest_support in quest_supports:
            for budget in budgets:
                _set_policy(
                    modules,
                    budget=None,
                    full_exact_layers=full_exact_layers,
                    page_size=args.page_size,
                    landmark_dtype=args.landmark_dtype,
                    quest_support=quest_support,
                    force_last_page=args.force_last_page,
                )

                def enable_reverse_shadow(
                    cache: object,
                    *,
                    selected_budget: int = budget,
                    selected_support: str = quest_support,
                ) -> None:
                    del cache
                    _set_policy(
                        modules,
                        budget=selected_budget,
                        full_exact_layers=full_exact_layers,
                        page_size=args.page_size,
                        landmark_dtype=args.landmark_dtype,
                        quest_support=selected_support,
                        force_last_page=args.force_last_page,
                    )

                sparse = block_scheduled_teacher_forced_nll(
                    model,
                    tokens,
                    cache=_cache(len(modules), 32),
                    prefill_length=args.prefill_tokens,
                    block_length=args.block_length,
                    post_prefill_callback=enable_reverse_shadow,
                )
                runtime = _runtime_statistics(modules, full_exact_layers)
                expected_queries = (
                    args.sequence_length - args.prefill_tokens
                ) * (len(modules) - len(full_exact_layers))
                if runtime["queries"] != expected_queries:
                    raise RuntimeError(
                        "Reverse ShadowKV query accounting is incomplete: "
                        f"{runtime['queries']} vs {expected_queries}"
                    )
                record = _paired_record(
                    dense,
                    sparse,
                    budget=budget,
                    quest_support=quest_support,
                    runtime=runtime,
                )
                records_by_configuration[(quest_support, budget)].append(record)
                configurations.append(record)
                print(
                    f"[Reverse ShadowKV PPL] sample={sample_index} "
                    f"support={quest_support} budget={budget} "
                    f"delta_nll={record['mean_nll_delta']:+.6e} "
                    f"top1={record['top1_agreement']:.6f}",
                    flush=True,
                )
        sample_records.append(
            {"sample_index": sample_index, "configurations": configurations}
        )

    c1_manifest_path = c1_export / "results.json"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "metadata": {
            "model_path": str(model_path),
            "model_config_sha256": _sha256(model_path / "config.json"),
            "c1_export": str(c1_export),
            "c1_export_sha256": _sha256(c1_manifest_path),
            "windows": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest_sha256": _sha256(windows_manifest_path),
            "window_start": args.window_start,
            "samples": args.samples,
            "sequence_length": args.sequence_length,
            "prefill_tokens": args.prefill_tokens,
            "block_length": args.block_length,
            "page_size": args.page_size,
            "quest_supports": list(quest_supports),
            "force_last_page": args.force_last_page,
            "full_exact_layers": sorted(full_exact_layers),
            "sparse_layers": [
                layer for layer in range(len(modules)) if layer not in full_exact_layers
            ],
            "dtype": str(dtype),
            "landmark_dtype": args.landmark_dtype,
            "c1_replaced_layers": len(replacements),
            "reverse_shadow_enabled_after_exact_prefill": True,
            "cuda_device": torch.cuda.get_device_name(device),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "aggregate": [
            _aggregate(
                records_by_configuration[(support, budget)], support, budget
            )
            for support in quest_supports
            for budget in budgets
        ],
        "samples": sample_records,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "teacher-forced fixed-token evaluation; not free-running generation",
            "exact K is GPU resident in this correctness oracle, not CPU offloaded",
            "QUEST metadata is rebuilt in Python for every causal query",
            "Python/Hugging Face timing is not serving throughput",
        ],
    }
    _atomic_text(output_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(output_markdown, _markdown(payload))
    print(json.dumps(payload["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    evaluate(_parser().parse_args())
