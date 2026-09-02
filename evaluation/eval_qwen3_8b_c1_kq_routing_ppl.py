#!/usr/bin/env python3
"""Window NLL/PPL for C1-V64 plus KQ-SVD-routed exact-Key pages."""

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
from basisserve.core.c1_k_routing_sidecar import routing_storage_ratio  # noqa: E402
from evaluation.eval_c1_block_scheduled_ppl import (  # noqa: E402
    _cache,
    _distribution,
    _dtype,
)


FORMAT = "basisserve.qwen3_8b.c1_kq_routing_window_ppl.v1"
FACTOR_FORMAT = "basisserve.qwen3_8b.pairwise_kq_svd.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_positive_ints(value: str, *, name: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique positive integers")
    return result


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _attention_modules(model: torch.nn.Module) -> list[GQATiedVOQwen3Attention]:
    modules = [layer.self_attn for layer in model.model.layers]
    if not all(isinstance(module, GQATiedVOQwen3Attention) for module in modules):
        raise TypeError("every Qwen3 layer must contain the C1 attention replacement")
    return modules


def _load_routing_factors(
    factor_dir: Path,
    *,
    model_path: Path,
    layers: int,
    kv_heads: int,
    query_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], Path, Path]:
    result_path = factor_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("format") != FACTOR_FORMAT or result.get("status") != "complete":
        raise ValueError("pairwise KQ-SVD factor result is incomplete or incompatible")
    if result["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("KQ-SVD factors belong to another model")
    factor_path = factor_dir / result["artifacts"]["factors"]["file"]
    if _sha256(factor_path) != result["artifacts"]["factors"]["sha256"]:
        raise ValueError("KQ-SVD factor hash mismatch")
    tensors = load_file(str(factor_path), device="cpu")
    if not {
        "independent_key_projector",
        "independent_query_projector",
    } <= set(tensors):
        raise ValueError("pairwise checkpoint lacks independent KQ-SVD factors")
    key = tensors["independent_key_projector"].contiguous()
    query = tensors["independent_query_projector"].contiguous()
    if tuple(key.shape[:3]) != (layers, kv_heads, head_dim):
        raise ValueError("KQ-SVD Key factors have incompatible model geometry")
    if (
        query.ndim != 4
        or int(query.shape[0]) != layers
        or int(query.shape[1]) not in (kv_heads, query_heads)
        or int(query.shape[2]) != head_dim
        or int(query.shape[3]) != int(key.shape[3])
    ):
        raise ValueError("KQ-SVD Query factors have incompatible model geometry")
    return key, query, result, result_path, factor_path


def _schedule(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    layers: int,
    prefill_tokens: int,
    block_length: int,
    post_prefill_callback: Any | None = None,
) -> TargetScheduleResult:
    return block_scheduled_teacher_forced_nll(
        model,
        tokens,
        cache=_cache(layers, 32),
        prefill_length=prefill_tokens,
        block_length=block_length,
        post_prefill_callback=post_prefill_callback,
    )


def _result_summary(result: TargetScheduleResult) -> dict[str, Any]:
    return {
        "evaluated_tokens": result.evaluated_tokens,
        "blocks": result.block_count,
        "nll_sum": result.nll_sum,
        "mean_nll": result.mean_nll,
        "ppl": math.exp(result.mean_nll),
    }


def _comparison(
    reference: TargetScheduleResult,
    candidate: TargetScheduleResult,
) -> dict[str, Any]:
    if reference.evaluated_tokens != candidate.evaluated_tokens:
        raise RuntimeError("paired schedules evaluated different token counts")
    deltas = [
        observed - expected
        for expected, observed in zip(
            reference.token_nlls, candidate.token_nlls, strict=True
        )
    ]
    agreements = sum(
        expected == observed
        for expected, observed in zip(
            reference.top1_token_ids, candidate.top1_token_ids, strict=True
        )
    )
    return {
        "mean_nll_delta": candidate.mean_nll - reference.mean_nll,
        "ppl_ratio": math.exp(candidate.mean_nll - reference.mean_nll),
        "top1_agreements": agreements,
        "top1_comparisons": reference.evaluated_tokens,
        "top1_agreement": agreements / reference.evaluated_tokens,
        "token_nll_delta": _distribution(deltas),
    }


def _runtime_statistics(
    modules: list[GQATiedVOQwen3Attention],
) -> dict[str, float]:
    sum_names = (
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
        "adaptive_eligible_query_heads",
        "adaptive_refined_query_heads",
        "adaptive_tail_mass_ratio_sum",
    )
    totals = {name: 0.0 for name in sum_names}
    totals["resident_selector_metadata_bytes"] = 0.0
    for module in modules:
        row = module.reverse_shadow_statistics()
        for name in sum_names:
            totals[name] += float(row[name])
        totals["resident_selector_metadata_bytes"] += float(
            row["maximum_resident_selector_metadata_bytes"]
        )
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
    totals["adaptive_refinement_fraction"] = (
        totals["adaptive_refined_query_heads"]
        / totals["adaptive_eligible_query_heads"]
        if totals["adaptive_eligible_query_heads"]
        else 0.0
    )
    totals["adaptive_mean_tail_mass_ratio"] = (
        totals["adaptive_tail_mass_ratio_sum"]
        / totals["adaptive_eligible_query_heads"]
        if totals["adaptive_eligible_query_heads"]
        else 0.0
    )
    return totals


def _aggregate_results(results: list[TargetScheduleResult]) -> dict[str, Any]:
    tokens = sum(result.evaluated_tokens for result in results)
    nll_sum = sum(result.nll_sum for result in results)
    mean_nll = nll_sum / tokens
    return {
        "samples": len(results),
        "evaluated_tokens": tokens,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
    }


def _aggregate_comparison(
    references: list[TargetScheduleResult],
    candidates: list[TargetScheduleResult],
) -> dict[str, Any]:
    reference = _aggregate_results(references)
    candidate = _aggregate_results(candidates)
    agreements = sum(
        expected == observed
        for reference_result, candidate_result in zip(
            references, candidates, strict=True
        )
        for expected, observed in zip(
            reference_result.top1_token_ids,
            candidate_result.top1_token_ids,
            strict=True,
        )
    )
    sample_deltas = [
        candidate_result.mean_nll - reference_result.mean_nll
        for reference_result, candidate_result in zip(
            references, candidates, strict=True
        )
    ]
    return {
        "mean_nll_delta": candidate["mean_nll"] - reference["mean_nll"],
        "ppl_ratio": math.exp(candidate["mean_nll"] - reference["mean_nll"]),
        "top1_agreement": agreements / reference["evaluated_tokens"],
        "paired_sample_standard_error": (
            statistics.stdev(sample_deltas) / math.sqrt(len(sample_deltas))
            if len(sample_deltas) > 1
            else None
        ),
        "sample_mean_nll_delta": _distribution(sample_deltas),
    }


def _merge_runtime(rows: list[dict[str, float]]) -> dict[str, float]:
    merged = {name: 0.0 for name in rows[0]}
    for name in merged:
        values = [row[name] for row in rows]
        merged[name] = (
            max(values) if name == "resident_selector_metadata_bytes" else sum(values)
        )
    merged["selected_token_fraction"] = (
        merged["selected_tokens"] / merged["physical_valid_tokens"]
        if merged["physical_valid_tokens"]
        else 0.0
    )
    merged["query_selected_token_fraction"] = (
        merged["query_selected_tokens"] / merged["query_valid_tokens"]
        if merged["query_valid_tokens"]
        else 0.0
    )
    merged["adaptive_refinement_fraction"] = (
        merged.get("adaptive_refined_query_heads", 0.0)
        / merged["adaptive_eligible_query_heads"]
        if merged.get("adaptive_eligible_query_heads", 0.0)
        else 0.0
    )
    merged["adaptive_mean_tail_mass_ratio"] = (
        merged.get("adaptive_tail_mass_ratio_sum", 0.0)
        / merged["adaptive_eligible_query_heads"]
        if merged.get("adaptive_eligible_query_heads", 0.0)
        else 0.0
    )
    return merged


def _set_routing_rank(
    modules: list[GQATiedVOQwen3Attention],
    key_factors: torch.Tensor,
    query_factors: torch.Tensor,
    rank: int,
) -> None:
    for layer, module in enumerate(modules):
        module.set_reverse_shadow_config(None)
        module.set_routing_projectors(
            key_factors[layer, ..., :rank],
            query_factors[layer, ..., :rank],
        )


def _set_sparse_policy(
    modules: list[GQATiedVOQwen3Attention],
    *,
    budget: int | None,
    page_size: int,
    force_last_page: bool,
    adaptive_max_budget: int | None = None,
    adaptive_tail_mass_ratio: float | None = None,
    pinned_prefix_pages: int = 0,
) -> None:
    for module in modules:
        module.set_reverse_shadow_config(
            None
            if budget is None
            else ReverseShadowConfig(
                page_size=page_size,
                exact_token_budget=budget,
                recent_exact_window=1 if force_last_page else 0,
                selector="kq_svd",
                landmark_dtype="bfloat16",
                adaptive_max_token_budget=adaptive_max_budget,
                adaptive_tail_mass_ratio_threshold=adaptive_tail_mass_ratio,
                pinned_prefix_pages=pinned_prefix_pages,
            )
        )


def _markdown(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    baseline = payload["baseline"]
    corpus_coverage = metadata.get("corpus_coverage")
    if corpus_coverage and corpus_coverage["all_complete_blocks"]:
        evaluation_label = "full-corpus block"
    else:
        evaluation_label = (
            "full-window"
            if metadata["prefill_tokens"] == 1
            else "suffix-conditioned"
        )
    lines = [
        f"# Qwen3-8B C1-V64 + KQ-SVD exact-Key routing {evaluation_label} PPL",
        "",
        "The same fixed evaluation tokens are scored under BF16 dense KV, full C1-V64 "
        "with exact QK, and KQ-SVD page routing followed by exact QK over fetched pages.",
        "",
        f"Windows: `{metadata['samples']} x {metadata['sequence_length']}`; scored tokens: "
        f"`{metadata['sequence_length'] - metadata['prefill_tokens']}` tokens/window; "
        f"block: `{metadata['block_length']}`; page: `{metadata['page_size']}`.",
        "",
        f"BF16 dense {evaluation_label} PPL: `{baseline['bf16_dense']['ppl']:.8f}`; "
        f"full C1 exact-QK {evaluation_label} PPL: `{baseline['c1_exact_qk']['ppl']:.8f}` "
        f"(ratio `{baseline['c1_vs_bf16']['ppl_ratio']:.8f}`).",
        "",
        "| R | B | Sparse PPL | Sparse/C1 | Sparse/BF16 | NLL delta vs C1 | "
        "Top-1 vs C1 | Physical K fraction | Exact-K MiB/token | GPU KV ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        runtime = row["runtime_logical"]
        lines.append(
            f"| {row['routing_rank']} | {row['nominal_token_budget']} | "
            f"{row['sparse']['ppl']:.8f} | "
            f"{row['sparse_vs_c1']['ppl_ratio']:.8f} | "
            f"{row['sparse_vs_bf16']['ppl_ratio']:.8f} | "
            f"{row['sparse_vs_c1']['mean_nll_delta']:+.8e} | "
            f"{row['sparse_vs_c1']['top1_agreement']:.8f} | "
            f"{runtime['selected_token_fraction']:.8f} | "
            f"{runtime['cpu_exact_key_bytes_fetched'] / row['sparse']['evaluated_tokens'] / (1 << 20):.3f} | "
            f"{row['persistent_gpu_scalar_ratio']:.4f} |"
        )
    lines.append("")
    if corpus_coverage and corpus_coverage["all_complete_blocks"]:
        lines.append(
            "This is non-overlapping block-chunked teacher-forced PPL over every complete "
            f"{metadata['sequence_length']}-token block in WikiText-2 test: "
            f"{corpus_coverage['covered_tokens']:,}/{corpus_coverage['tokenized_total']:,} "
            f"tokenizer tokens are covered, the final {corpus_coverage['trailing_tokens']} "
            "tokens are dropped, and context resets at each block boundary. "
            "Exact K remains physically GPU-resident in this correctness oracle; reported "
            "traffic is the logical page-store read volume."
        )
    else:
        lines.append(
            f"This is {evaluation_label} teacher-forced PPL, not full-corpus WikiText PPL. "
            "Exact K remains physically GPU-resident in this correctness oracle; reported "
            "traffic is the logical page-store read volume."
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path, required=True)
    parser.add_argument("--routing-factors", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--prefill-tokens", type=int, default=3968)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--ranks", default="32,64")
    parser.add_argument("--exact-token-budgets", default="512,1024")
    parser.add_argument("--force-last-page", action="store_true")
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
    factor_dir = args.routing_factors.expanduser().resolve()
    windows_path = args.windows.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    ranks = _parse_positive_ints(args.ranks, name="ranks")
    budgets = _parse_positive_ints(
        args.exact_token_budgets, name="exact-token budgets"
    )
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
    if args.window_start < 0 or any(budget % args.page_size for budget in budgets):
        raise ValueError("window start must be nonnegative and budgets page-aligned")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("end-to-end KQ routing PPL requires CUDA")

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
    if stop > len(input_ids) or int(input_ids.shape[1]) < args.sequence_length:
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
    layers = int(model.config.num_hidden_layers)
    kv_heads = int(model.config.num_key_value_heads)
    query_heads = int(model.config.num_attention_heads)
    head_dim = int(
        getattr(model.config, "head_dim", model.config.hidden_size // query_heads)
    )
    key_factors, query_factors, factor_result, factor_result_path, factor_path = (
        _load_routing_factors(
            factor_dir,
            model_path=model_path,
            layers=layers,
            kv_heads=kv_heads,
            query_heads=query_heads,
            head_dim=head_dim,
        )
    )
    if max(ranks) > int(key_factors.shape[-1]):
        raise ValueError("requested routing rank exceeds the calibrated factors")

    dense_results = []
    for sample_offset, sample in enumerate(input_ids):
        result = _schedule(
            model,
            sample.unsqueeze(0).to(device),
            layers=layers,
            prefill_tokens=args.prefill_tokens,
            block_length=args.block_length,
        )
        dense_results.append(result)
        print(
            f"[KQ routing PPL] sample={args.window_start + sample_offset} "
            f"baseline=bf16_dense mean_nll={result.mean_nll:.8f}",
            flush=True,
        )

    replacements = install_qwen3_gqa_vo_als_export(model, c1_export)
    modules = _attention_modules(model)
    c1_results = []
    for sample_offset, sample in enumerate(input_ids):
        _set_sparse_policy(
            modules,
            budget=None,
            page_size=args.page_size,
            force_last_page=False,
        )
        result = _schedule(
            model,
            sample.unsqueeze(0).to(device),
            layers=layers,
            prefill_tokens=args.prefill_tokens,
            block_length=args.block_length,
        )
        c1_results.append(result)
        print(
            f"[KQ routing PPL] sample={args.window_start + sample_offset} "
            f"baseline=c1_exact_qk mean_nll={result.mean_nll:.8f}",
            flush=True,
        )

    sparse_results: dict[tuple[int, int], list[TargetScheduleResult]] = {
        (rank, budget): [] for rank in ranks for budget in budgets
    }
    runtimes: dict[tuple[int, int], list[dict[str, float]]] = {
        (rank, budget): [] for rank in ranks for budget in budgets
    }
    sample_records = []
    for sample_offset, sample in enumerate(input_ids):
        sample_index = args.window_start + sample_offset
        configurations = []
        tokens = sample.unsqueeze(0).to(device)
        for rank in ranks:
            _set_routing_rank(modules, key_factors, query_factors, rank)
            for budget in budgets:
                _set_sparse_policy(
                    modules,
                    budget=None,
                    page_size=args.page_size,
                    force_last_page=False,
                )

                def enable_sparse(
                    cache: object,
                    *,
                    selected_budget: int = budget,
                ) -> None:
                    del cache
                    _set_sparse_policy(
                        modules,
                        budget=selected_budget,
                        page_size=args.page_size,
                        force_last_page=args.force_last_page,
                    )

                sparse = _schedule(
                    model,
                    tokens,
                    layers=layers,
                    prefill_tokens=args.prefill_tokens,
                    block_length=args.block_length,
                    post_prefill_callback=enable_sparse,
                )
                runtime = _runtime_statistics(modules)
                expected_queries = (
                    args.sequence_length - args.prefill_tokens
                ) * layers
                if runtime["queries"] != expected_queries:
                    raise RuntimeError(
                        "KQ routing query accounting is incomplete: "
                        f"{runtime['queries']} vs {expected_queries}"
                    )
                sparse_results[(rank, budget)].append(sparse)
                runtimes[(rank, budget)].append(runtime)
                record = {
                    "routing_rank": rank,
                    "nominal_token_budget": budget,
                    "bf16_dense": _result_summary(dense_results[sample_offset]),
                    "c1_exact_qk": _result_summary(c1_results[sample_offset]),
                    "sparse": _result_summary(sparse),
                    "c1_vs_bf16": _comparison(
                        dense_results[sample_offset], c1_results[sample_offset]
                    ),
                    "sparse_vs_c1": _comparison(c1_results[sample_offset], sparse),
                    "sparse_vs_bf16": _comparison(
                        dense_results[sample_offset], sparse
                    ),
                    "runtime_logical": runtime,
                }
                configurations.append(record)
                print(
                    f"[KQ routing PPL] sample={sample_index} rank={rank} "
                    f"budget={budget} delta_vs_c1="
                    f"{record['sparse_vs_c1']['mean_nll_delta']:+.8e} "
                    f"top1_vs_c1={record['sparse_vs_c1']['top1_agreement']:.6f}",
                    flush=True,
                )
        sample_records.append(
            {"sample_index": sample_index, "configurations": configurations}
        )

    baseline = {
        "bf16_dense": _aggregate_results(dense_results),
        "c1_exact_qk": _aggregate_results(c1_results),
        "c1_vs_bf16": _aggregate_comparison(dense_results, c1_results),
    }
    aggregate = []
    for rank in ranks:
        for budget in budgets:
            candidates = sparse_results[(rank, budget)]
            aggregate.append(
                {
                    "routing_rank": rank,
                    "nominal_token_budget": budget,
                    "persistent_gpu_scalar_ratio": routing_storage_ratio(
                        value_rank=64,
                        routing_rank=rank,
                        key_width=head_dim,
                        value_width=head_dim,
                    ),
                    "sparse": _aggregate_results(candidates),
                    "sparse_vs_c1": _aggregate_comparison(
                        c1_results, candidates
                    ),
                    "sparse_vs_bf16": _aggregate_comparison(
                        dense_results, candidates
                    ),
                    "runtime_logical": _merge_runtime(
                        runtimes[(rank, budget)]
                    ),
                }
            )

    c1_result_path = c1_export / "results.json"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "metadata": {
            "model_path": str(model_path),
            "model_config_sha256": _sha256(model_path / "config.json"),
            "c1_export": str(c1_export),
            "c1_export_sha256": _sha256(c1_result_path),
            "routing_factors": str(factor_dir),
            "routing_factor_result_sha256": _sha256(factor_result_path),
            "routing_factor_tensor_sha256": _sha256(factor_path),
            "routing_factor_format": factor_result["format"],
            "routing_factor_fit_config": factor_result.get("fit_config"),
            "windows": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest_sha256": _sha256(windows_manifest_path),
            "window_start": args.window_start,
            "samples": args.samples,
            "sequence_length": args.sequence_length,
            "prefill_tokens": args.prefill_tokens,
            "block_length": args.block_length,
            "page_size": args.page_size,
            "ranks": list(ranks),
            "nominal_token_budgets": list(budgets),
            "force_last_page": args.force_last_page,
            "layers": layers,
            "dtype": str(dtype),
            "c1_replaced_layers": len(replacements),
            "cuda_device": torch.cuda.get_device_name(device),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "baseline": baseline,
        "aggregate": aggregate,
        "samples": sample_records,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "fixed-window teacher-forced PPL, not full-corpus PPL",
            "exact K is physically GPU resident; CPU fetch traffic is logical",
            "routing sidecars are rebuilt in Python for each target block",
            "Python/Hugging Face timing is not serving throughput",
        ],
    }
    _atomic_text(output_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(output_markdown, _markdown(payload))
    print(json.dumps({"baseline": baseline, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    evaluate(_parser().parse_args())
