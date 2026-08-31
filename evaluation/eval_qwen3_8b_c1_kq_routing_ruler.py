#!/usr/bin/env python3
"""RULER-v1 generation for C1-V plus KQ-routed exact-Key pages."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import (  # noqa: E402
    GQATiedVOQwen3Attention,
    install_qwen3_gqa_vo_als_export,
)
from basisserve.core.c1_k_routing_sidecar import (  # noqa: E402
    ROUTING_PROXY_IMPLEMENTATION,
    RoutingDynamicCache,
    routing_storage_ratio,
)
from evaluation.eval_c1_block_scheduled_ppl import _dtype  # noqa: E402
from evaluation.eval_qwen3_8b_c1_kq_routing_ppl import (  # noqa: E402
    _attention_modules,
    _load_routing_factors,
    _merge_runtime,
    _runtime_statistics,
    _set_routing_rank,
    _set_sparse_policy,
)
from evaluation.eval_qwen3_c1_quest_ruler import (  # noqa: E402
    _atomic_json,
    _atomic_text,
    _build_work,
    _eos_ids,
    _fingerprint,
    _greedy_continuation,
    _load_dataset_manifest,
    _sha256,
)
from evaluation.eval_qwen3_dense_ruler import (  # noqa: E402
    ARM as BF16_ARM,
    FORMAT as DENSE_FORMAT,
    _qwen3_config,
)
from evaluation.ruler_v1 import (  # noqa: E402
    paired_summary,
    parse_tasks,
    ruler_prompt,
    sample_score,
    summarize_arm,
)


FORMAT = "basisserve.qwen3_8b.c1_kq_routing.ruler_v1.v1"
RANK_FORMAT = "basisserve.qwen3_8b.c1_kq_routing.ruler_v1.rank.v1"


@dataclass(frozen=True)
class SparsePolicy:
    arm: str
    token_budget: int
    adaptive_max_token_budget: int | None = None
    adaptive_tail_mass_ratio: float | None = None


def _arm_names(
    value_rank: int,
    routing_rank: int,
    exact_token_budget: int,
) -> tuple[str, str]:
    if min(value_rank, routing_rank, exact_token_budget) <= 0:
        raise ValueError(
            "C1 rank, routing rank, and exact-token budget must be positive"
        )
    return (
        f"dense_k_c1_v{value_rank}",
        f"kq_r{routing_rank}_b{exact_token_budget}_exact_k_c1_v{value_rank}",
    )


def _ratio_slug(value: float) -> str:
    if not 0.0 < value <= 1.0:
        raise ValueError("adaptive tail mass ratio must lie in (0, 1]")
    return format(value, ".6g").replace(".", "p")


def _adaptive_arm_name(
    value_rank: int,
    routing_rank: int,
    base_token_budget: int,
    max_token_budget: int,
    tail_mass_ratio: float,
) -> str:
    if min(value_rank, routing_rank, base_token_budget, max_token_budget) <= 0:
        raise ValueError("adaptive arm geometry must be positive")
    if max_token_budget <= base_token_budget:
        raise ValueError("adaptive maximum budget must exceed base budget")
    return (
        f"kq_r{routing_rank}_b{base_token_budget}to{max_token_budget}"
        f"_tail{_ratio_slug(tail_mass_ratio)}_exact_k_c1_v{value_rank}"
    )


def _parse_adaptive_tail_ratios(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    ratios = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not ratios or any(not 0.0 < ratio <= 1.0 for ratio in ratios):
        raise ValueError("adaptive tail mass ratios must lie in (0, 1]")
    if len(ratios) != len(set(ratios)):
        raise ValueError("adaptive tail mass ratios must be unique")
    return ratios


def _sparse_policies(
    *,
    value_rank: int,
    routing_rank: int,
    base_token_budget: int,
    adaptive_max_token_budget: int | None,
    adaptive_tail_mass_ratios: tuple[float, ...],
) -> tuple[SparsePolicy, ...]:
    _, base_arm = _arm_names(value_rank, routing_rank, base_token_budget)
    policies = [SparsePolicy(arm=base_arm, token_budget=base_token_budget)]
    if not adaptive_tail_mass_ratios:
        if adaptive_max_token_budget is not None:
            raise ValueError(
                "adaptive maximum budget requires adaptive tail mass ratios"
            )
        return tuple(policies)
    if adaptive_max_token_budget is None:
        raise ValueError("adaptive tail mass ratios require a maximum budget")
    if adaptive_max_token_budget <= base_token_budget:
        raise ValueError("adaptive maximum budget must exceed base budget")
    for ratio in adaptive_tail_mass_ratios:
        policies.append(
            SparsePolicy(
                arm=_adaptive_arm_name(
                    value_rank,
                    routing_rank,
                    base_token_budget,
                    adaptive_max_token_budget,
                    ratio,
                ),
                token_budget=base_token_budget,
                adaptive_max_token_budget=adaptive_max_token_budget,
                adaptive_tail_mass_ratio=ratio,
            )
        )
    _, maximum_arm = _arm_names(
        value_rank,
        routing_rank,
        adaptive_max_token_budget,
    )
    policies.append(
        SparsePolicy(arm=maximum_arm, token_budget=adaptive_max_token_budget)
    )
    return tuple(policies)


def _load_rank_state(
    path: Path,
    *,
    fingerprint: str,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "format": RANK_FORMAT,
            "fingerprint": fingerprint,
            "rank": rank,
            "world_size": world_size,
            "records": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = (
        payload.get("format"),
        payload.get("fingerprint"),
        payload.get("rank"),
        payload.get("world_size"),
    )
    expected = (RANK_FORMAT, fingerprint, rank, world_size)
    if observed != expected:
        raise ValueError(f"rank resume state is incompatible: {path}")
    keys = [record["key"] for record in payload["records"]]
    if len(keys) != len(set(keys)):
        raise ValueError(f"rank resume state contains duplicate samples: {path}")
    return payload


def _load_dense_baseline(
    path: Path,
    *,
    model_path: Path,
    dataset_manifest_sha256: str,
    sequence_length: int,
    samples_per_task: int,
    task_names: list[str],
    rope_scaling: dict[str, float | int | str] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != DENSE_FORMAT or payload.get("status") != "complete":
        raise ValueError("dense RULER baseline is incomplete or incompatible")
    expected = {
        "model_config_sha256": _sha256(model_path / "config.json"),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "sequence_length": sequence_length,
        "arm": BF16_ARM,
        "rope_scaling": rope_scaling,
    }
    for name, value in expected.items():
        if payload["metadata"].get(name) != value:
            raise ValueError(
                f"dense RULER baseline {name}={payload['metadata'].get(name)!r}, "
                f"expected {value!r}"
            )
    baseline_tasks = list(payload["metadata"].get("tasks", ()))
    if any(task not in baseline_tasks for task in task_names):
        raise ValueError("dense RULER baseline does not contain every requested task")
    if int(payload["metadata"].get("samples_per_task", 0)) < samples_per_task:
        raise ValueError("dense RULER baseline has fewer samples than requested")
    records = {
        record["key"]: record
        for record in payload["records"]
        if record["task"] in task_names
        and int(record["sample_ordinal"]) < samples_per_task
    }
    expected_records = len(task_names) * samples_per_task
    if len(records) != expected_records:
        raise ValueError(
            f"dense RULER baseline has {len(records)} rows, expected {expected_records}"
        )
    if any(BF16_ARM not in record["arms"] for record in records.values()):
        raise ValueError("dense RULER baseline is missing its dense arm")
    return payload, records


def _restore_prompt_cache(cache: Any, prompt_tokens: int) -> None:
    """Rollback decode-only cache appends between paired evaluation arms."""

    if prompt_tokens <= 0:
        raise ValueError("prompt cache length must be positive")
    if int(cache.get_seq_length()) < prompt_tokens:
        raise RuntimeError("decode arm unexpectedly shortened the prompt cache")
    cache.crop(prompt_tokens)
    if int(cache.get_seq_length()) != prompt_tokens:
        raise RuntimeError("cache transaction did not restore the prompt length")


def _assigned_work(
    work: list[tuple[Any, ...]],
    *,
    rank: int,
    world_size: int,
    rank_offset: int,
) -> list[tuple[Any, ...]]:
    """Assign independent samples with an explicit cyclic device offset."""

    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank geometry")
    if not 0 <= rank_offset < world_size:
        raise ValueError("assignment rank offset must lie in [0, world size)")
    return [row for row in work if (int(row[0]) + rank_offset) % world_size == rank]


def _set_backend(
    modules: list[GQATiedVOQwen3Attention],
    backend: str,
) -> None:
    if backend not in {
        "native",
        "sdpa",
        "triton",
        "cuda_dense",
        "cuda_sparse",
        "dense_prefill",
    }:
        raise ValueError(f"unsupported runtime attention backend {backend!r}")
    for module in modules:
        module.attention_backend = backend


def _chunked_exact_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    chunk_size: int,
    empty_cuda_cache_between_chunks: bool = False,
    incremental_routing_sidecar: bool = False,
) -> tuple[int, Any]:
    """Run mathematically dense causal C1 attention with bounded workspace."""

    if chunk_size <= 0:
        raise ValueError("prefill chunk size must be positive")
    cache = RoutingDynamicCache() if incremental_routing_sidecar else DynamicCache()
    final_logits: torch.Tensor | None = None
    for start in range(0, int(input_ids.shape[1]), chunk_size):
        stop = min(start + chunk_size, int(input_ids.shape[1]))
        output = model(
            input_ids=input_ids[:, start:stop],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        final_logits = output.logits[:, -1].clone()
        del output
        if empty_cuda_cache_between_chunks and input_ids.device.type == "cuda":
            torch.cuda.empty_cache()
    if final_logits is None:
        raise ValueError("RULER prompt is empty")
    return int(final_logits[0].argmax().item()), cache


def _evaluate_decode(
    *,
    model: torch.nn.Module,
    modules: list[GQATiedVOQwen3Attention],
    cache: Any,
    first_token: int,
    task: Any,
    references: list[str],
    tokenizer: Any,
    sparse: bool,
    page_size: int,
    token_budget: int,
    force_last_page: bool,
    device: torch.device,
    adaptive_max_token_budget: int | None = None,
    adaptive_tail_mass_ratio: float | None = None,
) -> dict[str, Any]:
    if sparse:
        _set_backend(modules, "native")
        _set_sparse_policy(
            modules,
            budget=token_budget,
            page_size=page_size,
            force_last_page=force_last_page,
            adaptive_max_budget=adaptive_max_token_budget,
            adaptive_tail_mass_ratio=adaptive_tail_mass_ratio,
        )
    else:
        _set_sparse_policy(
            modules,
            budget=None,
            page_size=page_size,
            force_last_page=False,
        )
        _set_backend(modules, "sdpa")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    generated_ids, cache = _greedy_continuation(
        model,
        cache,
        first_token,
        maximum_tokens=task.tokens_to_generate,
        eos_ids=_eos_ids(tokenizer, model),
        device=device,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    prediction = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    result = {
        "prediction": prediction,
        "generated_token_ids": generated_ids,
        "generated_tokens": len(generated_ids),
        "stopped_on_eos": generated_ids[-1] in _eos_ids(tokenizer, model),
        "score": sample_score(prediction, references, task.match_type),
        "elapsed_seconds": elapsed,
    }
    if sparse:
        result["runtime_logical"] = _runtime_statistics(modules)
    del cache
    return result


def _policy_label(policy: Mapping[str, Any], *, routing_rank: int) -> str:
    maximum = policy.get("adaptive_max_token_budget")
    ratio = policy.get("adaptive_tail_mass_ratio")
    if maximum is None:
        return f"R{routing_rank}/B{int(policy['token_budget'])}"
    return (
        f"R{routing_rank}/B{int(policy['token_budget'])}->{int(maximum)} "
        f"tail>={float(ratio):g}"
    )


def _adaptive_markdown(payload: Mapping[str, Any]) -> str:
    metadata = payload["metadata"]
    bf16_arm = metadata["arms"][0]
    c1_arm = metadata["arms"][1]
    policies = list(metadata["sparse_policies"])
    summaries = {row["arm"]: row for row in payload["summary"]["arms"]}
    labels = {
        bf16_arm: "BF16 dense",
        c1_arm: f"C1-V{int(metadata['value_rank'])} exact-K",
        **{
            policy["arm"]: _policy_label(
                policy, routing_rank=int(metadata["routing_rank"])
            )
            for policy in policies
        },
    }
    arms = list(metadata["arms"])
    header = "| Task | Samples | " + " | ".join(labels[arm] for arm in arms) + " |"
    alignment = "|:---|---:|" + "---:|" * len(arms)
    lines = [
        "# Qwen3-8B-Base adaptive KQ-routing RULER-v1 32K",
        "",
        (
            "Adaptive policies start from the base page budget and expand one "
            "Query head to the maximum budget when the proxy mass of the next "
            "page band exceeds the configured fraction of the base-band mass."
        ),
        "",
        header,
        alignment,
    ]
    for task in metadata["tasks"]:
        sample_count = int(summaries[bf16_arm]["tasks"][task]["samples"])
        values = " | ".join(
            f"{100.0 * summaries[arm]['tasks'][task]['accuracy']:.2f}%" for arm in arms
        )
        lines.append(f"| {task} | {sample_count} | {values} |")
    values = " | ".join(
        f"**{100.0 * summaries[arm]['task_balanced_accuracy']:.2f}%**" for arm in arms
    )
    lines.append(f"| **Task-balanced mean** | {len(payload['records'])} | {values} |")
    lines.extend(
        [
            "",
            "| Sparse policy | Accuracy | Selected exact-K tokens | "
            "Adaptive refinements | Logical exact-K MiB/decode token |",
            "|:---|---:|---:|---:|---:|",
        ]
    )
    runtimes = payload["summary"]["runtime_logical_by_arm"]
    layers = int(metadata["layers"])
    for policy in policies:
        arm = policy["arm"]
        runtime = runtimes[arm]
        decode_queries = runtime["queries"] / layers
        exact_k_mib = (
            runtime["cpu_exact_key_bytes_fetched"] / decode_queries / (1 << 20)
            if decode_queries
            else 0.0
        )
        refinement = (
            f"{100.0 * runtime['adaptive_refinement_fraction']:.2f}%"
            if policy.get("adaptive_max_token_budget") is not None
            else "--"
        )
        lines.append(
            f"| {labels[arm]} | "
            f"{100.0 * summaries[arm]['task_balanced_accuracy']:.2f}% | "
            f"{100.0 * runtime['selected_token_fraction']:.4f}% | "
            f"{refinement} | {exact_k_mib:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "Prompt prefill is shared within each sample and uses dense "
                f"C1-V{int(metadata['value_rank'])} attention. Exact K remains "
                "GPU-resident, so exact-K traffic is logical rather than measured PCIe."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown(payload: Mapping[str, Any]) -> str:
    metadata = payload["metadata"]
    if metadata.get("adaptive_budget_enabled"):
        return _adaptive_markdown(payload)
    bf16_arm, c1_arm, routing_arm = metadata["arms"]
    summaries = {row["arm"]: row for row in payload["summary"]["arms"]}
    bf16 = summaries[bf16_arm]
    c1 = summaries[c1_arm]
    routing = summaries[routing_arm]
    paired = payload["summary"]["paired_routing_vs_c1"]
    runtime = payload["summary"]["runtime_logical"]
    value_rank = int(metadata["value_rank"])
    routing_rank = int(metadata["routing_rank"])
    exact_token_budget = int(metadata["exact_token_budget"])
    decode_queries = runtime["queries"] / metadata["layers"]
    exact_k_mib = (
        runtime["cpu_exact_key_bytes_fetched"] / decode_queries / (1 << 20)
        if decode_queries
        else 0.0
    )
    lines = [
        f"# Qwen3-8B-Base C1 KQ-routing RULER-v1 {metadata['sequence_length'] // 1024}K",
        "",
        f"BF16 dense, C1-V{value_rank} exact-QK, and C1-V{value_rank} with "
        f"R{routing_rank}/B{exact_token_budget} KQ routing use identical official "
        "base-model prompts and greedy decoding.",
        "",
        f"| Task | Samples | BF16 dense | C1 exact-QK | "
        f"R{routing_rank}/B{exact_token_budget} | Routing-C1 | Regressions | "
        "Improvements |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in metadata["tasks"]:
        dense_row = bf16["tasks"][task]
        c1_row = c1["tasks"][task]
        sparse_row = routing["tasks"][task]
        pair = paired["tasks"][task]
        lines.append(
            f"| {task} | {dense_row['samples']} | "
            f"{100 * dense_row['accuracy']:.2f}% | "
            f"{100 * c1_row['accuracy']:.2f}% | "
            f"{100 * sparse_row['accuracy']:.2f}% | "
            f"{100 * (sparse_row['accuracy'] - c1_row['accuracy']):+.2f} pp | "
            f"{pair['sparse_regressions']} | {pair['sparse_improvements']} |"
        )
    overall = paired["all_samples"]
    lines.extend(
        [
            f"| **Task-balanced mean** | {len(payload['records'])} | "
            f"**{100 * bf16['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * c1['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * routing['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * (routing['task_balanced_accuracy'] - c1['task_balanced_accuracy']):+.2f} pp** | "
            f"**{overall['sparse_regressions']}** | "
            f"**{overall['sparse_improvements']}** |",
            "",
            f"Logical exact-K traffic: `{exact_k_mib:.3f} MiB/decode token`; "
            f"physical selected-K fraction: `{runtime['selected_token_fraction']:.6f}`; "
            f"persistent GPU KV ratio: `{metadata['persistent_gpu_scalar_ratio']:.4f}`.",
            "",
            f"Prompt prefill uses chunked exact-QK C1-V{value_rank} SDPA. The first "
            "generated token "
            "comes from dense C1 prefill; KQ routing is enabled for subsequent "
            "decode tokens. Exact K remains physically GPU-resident, so traffic is "
            "a logical CPU page-store read volume rather than measured PCIe latency.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path, required=True)
    parser.add_argument("--routing-factors", type=Path, required=True)
    parser.add_argument("--dense-baseline", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument(
        "--assignment-rank-offset",
        type=int,
        default=0,
        help="cyclic offset for data-parallel sample ownership",
    )
    parser.add_argument(
        "--yarn-factor",
        type=float,
        help="static YaRN factor; required when sequence length exceeds native context",
    )
    parser.add_argument("--routing-rank", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--exact-token-budget", type=int, default=1024)
    parser.add_argument("--adaptive-max-token-budget", type=int)
    parser.add_argument(
        "--adaptive-tail-mass-ratios",
        default="",
        help=(
            "comma-separated next-band/base-band proxy mass thresholds; when "
            "set, also evaluates the fixed maximum-budget endpoint"
        ),
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument(
        "--empty-cache-between-prefill-chunks",
        action="store_true",
        help="release allocator fragments between long-context prefill chunks",
    )
    parser.add_argument(
        "--incremental-routing-sidecar",
        action="store_true",
        help="project each new post-RoPE K once and retain its routing code",
    )
    parser.add_argument("--force-last-page", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


@torch.inference_mode()
def main() -> None:
    args = _parser().parse_args()
    adaptive_tail_mass_ratios = _parse_adaptive_tail_ratios(
        args.adaptive_tail_mass_ratios
    )
    started = time.perf_counter()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not torch.cuda.is_available():
        raise RuntimeError("C1 KQ-routing RULER evaluation requires CUDA")
    if (
        min(
            args.sequence_length,
            args.samples_per_task,
            args.routing_rank,
            args.page_size,
            args.exact_token_budget,
            args.prefill_chunk_size,
        )
        <= 0
    ):
        raise ValueError("RULER and routing sizes must be positive")
    if args.exact_token_budget % args.page_size:
        raise ValueError("exact token budget must align to complete pages")
    if (
        args.adaptive_max_token_budget is not None
        and args.adaptive_max_token_budget % args.page_size
    ):
        raise ValueError("adaptive maximum token budget must align to complete pages")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        # Ranks evaluate independent samples and only synchronize before the
        # rank-0 JSON merge.  Keep that control-plane barrier on the CPU: a
        # lazy, first-use NCCL barrier can hang on nodes where peer-to-peer GPU
        # communication is unavailable even though the independent GPU work
        # itself completed successfully.
        dist.init_process_group(backend="gloo")

    model_path = args.model_path.expanduser().resolve()
    c1_export = args.c1_export.expanduser().resolve()
    factor_dir = args.routing_factors.expanduser().resolve()
    dense_baseline_path = args.dense_baseline.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_tasks(args.tasks)
    task_names = [task.name for task in tasks]
    effective_model_config, rope_scaling = _qwen3_config(
        model_path,
        sequence_length=args.sequence_length,
        yarn_factor=args.yarn_factor,
    )
    dataset_manifest, dataset_hash = _load_dataset_manifest(
        data_dir,
        sequence_length=args.sequence_length,
        samples=args.samples_per_task,
        tasks=tasks,
        tokenizer_path=model_path,
    )
    dense_baseline, dense_records = _load_dense_baseline(
        dense_baseline_path,
        model_path=model_path,
        dataset_manifest_sha256=dataset_hash,
        sequence_length=args.sequence_length,
        samples_per_task=args.samples_per_task,
        task_names=task_names,
        rope_scaling=rope_scaling,
    )
    work = _build_work(data_dir, tasks, args.samples_per_task)
    assigned = _assigned_work(
        work,
        rank=rank,
        world_size=world_size,
        rank_offset=args.assignment_rank_offset,
    )

    c1_result_path = c1_export / "results.json"
    c1_result = json.loads(c1_result_path.read_text(encoding="utf-8"))
    value_rank = int(c1_result["fit_config"]["cache_rank_per_head"])
    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    layers = int(model_config["num_hidden_layers"])
    kv_heads = int(model_config["num_key_value_heads"])
    query_heads = int(model_config["num_attention_heads"])
    head_dim = int(
        model_config.get("head_dim", model_config["hidden_size"] // query_heads)
    )
    if not 0 < value_rank <= head_dim:
        raise ValueError(f"C1 Value rank must lie in [1, {head_dim}], got {value_rank}")
    c1_arm, routing_arm = _arm_names(
        value_rank,
        args.routing_rank,
        args.exact_token_budget,
    )
    sparse_policies = _sparse_policies(
        value_rank=value_rank,
        routing_rank=args.routing_rank,
        base_token_budget=args.exact_token_budget,
        adaptive_max_token_budget=args.adaptive_max_token_budget,
        adaptive_tail_mass_ratios=adaptive_tail_mass_ratios,
    )
    if sparse_policies[0].arm != routing_arm:
        raise AssertionError("base sparse policy differs from routing arm")
    arms = (BF16_ARM, c1_arm, *(policy.arm for policy in sparse_policies))
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
    if args.routing_rank > int(key_factors.shape[-1]):
        raise ValueError("routing rank exceeds calibrated factor rank")

    protocol_identity = {
        "model_config_sha256": _sha256(model_path / "config.json"),
        "c1_result_sha256": _sha256(c1_result_path),
        "routing_factor_result_sha256": _sha256(factor_result_path),
        "routing_factor_tensor_sha256": _sha256(factor_path),
        "dense_baseline_sha256": _sha256(dense_baseline_path),
        "dataset_manifest_sha256": dataset_hash,
        "sequence_length": args.sequence_length,
        "tasks": task_names,
        "samples_per_task": args.samples_per_task,
        "arms": list(arms),
        "value_rank": value_rank,
        "routing_rank": args.routing_rank,
        "page_size": args.page_size,
        "exact_token_budget": args.exact_token_budget,
        "adaptive_max_token_budget": args.adaptive_max_token_budget,
        "adaptive_tail_mass_ratios": list(adaptive_tail_mass_ratios),
        "sparse_policies": [asdict(policy) for policy in sparse_policies],
        "prefill_chunk_size": args.prefill_chunk_size,
        "empty_cache_between_prefill_chunks": (args.empty_cache_between_prefill_chunks),
        "incremental_routing_sidecar": args.incremental_routing_sidecar,
        "routing_proxy_implementation": ROUTING_PROXY_IMPLEMENTATION,
        "force_last_page": args.force_last_page,
        "dtype": args.dtype,
        "assignment_rank_offset": args.assignment_rank_offset,
        "rope_scaling": rope_scaling,
        "effective_max_position_embeddings": int(
            effective_model_config.max_position_embeddings
        ),
        "world_size": world_size,
    }
    fingerprint = _fingerprint(protocol_identity)
    rank_path = output_dir / f"rank_{rank:02d}.json"
    rank_state = _load_rank_state(
        rank_path,
        fingerprint=fingerprint,
        rank=rank,
        world_size=world_size,
    )
    completed = {record["key"] for record in rank_state["records"]}
    pending = [row for row in assigned if f"{row[1].name}:{row[2]}" not in completed]
    print(
        f"[KQ routing RULER] rank={rank}/{world_size} assigned={len(assigned)} "
        f"completed={len(completed)} pending={len(pending)} device={device}",
        flush=True,
    )

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                config=effective_model_config,
                dtype=_dtype(args.dtype),
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        replacements = install_qwen3_gqa_vo_als_export(
            model,
            c1_export,
            attention_backend="sdpa",
        )
        modules = _attention_modules(model)
        if len(replacements) != layers or len(modules) != layers:
            raise RuntimeError("C1 replacement count differs from model layers")
        _set_routing_rank(
            modules,
            key_factors,
            query_factors,
            args.routing_rank,
        )
        rank_state["runtime"] = {
            "cuda_device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "prefill_attention_backend": "torch_sdpa_chunked_exact",
            "prefill_chunk_size": args.prefill_chunk_size,
            "empty_cache_between_prefill_chunks": (
                args.empty_cache_between_prefill_chunks
            ),
            "incremental_routing_sidecar": args.incremental_routing_sidecar,
            "rope_scaling": rope_scaling,
        }
        _atomic_json(rank_path, rank_state)

        for _, task, ordinal, source in pending:
            key = f"{task.name}:{ordinal}"
            input_ids = tokenizer(
                ruler_prompt(source),
                add_special_tokens=True,
                return_tensors="pt",
            )["input_ids"].to(device)
            prompt_tokens = int(input_ids.shape[1])
            if prompt_tokens + task.tokens_to_generate > args.sequence_length:
                raise ValueError(
                    f"RULER prompt {key} exceeds total length: {prompt_tokens} + "
                    f"{task.tokens_to_generate} > {args.sequence_length}"
                )
            _set_sparse_policy(
                modules,
                budget=None,
                page_size=args.page_size,
                force_last_page=False,
            )
            _set_backend(modules, "sdpa")
            torch.cuda.synchronize(device)
            prefill_started = time.perf_counter()
            first_token, routing_cache = _chunked_exact_prefill(
                model,
                input_ids,
                chunk_size=args.prefill_chunk_size,
                empty_cuda_cache_between_chunks=(
                    args.empty_cache_between_prefill_chunks
                ),
                incremental_routing_sidecar=args.incremental_routing_sidecar,
            )
            torch.cuda.synchronize(device)
            prefill_seconds = time.perf_counter() - prefill_started
            del input_ids
            references = list(source["outputs"])
            c1_result_row = _evaluate_decode(
                model=model,
                modules=modules,
                cache=routing_cache,
                first_token=first_token,
                task=task,
                references=references,
                tokenizer=tokenizer,
                sparse=False,
                page_size=args.page_size,
                token_budget=args.exact_token_budget,
                force_last_page=False,
                device=device,
            )
            _restore_prompt_cache(routing_cache, prompt_tokens)
            sparse_result_rows: dict[str, dict[str, Any]] = {}
            for policy in sparse_policies:
                sparse_result_rows[policy.arm] = _evaluate_decode(
                    model=model,
                    modules=modules,
                    cache=routing_cache,
                    first_token=first_token,
                    task=task,
                    references=references,
                    tokenizer=tokenizer,
                    sparse=True,
                    page_size=args.page_size,
                    token_budget=policy.token_budget,
                    force_last_page=args.force_last_page,
                    device=device,
                    adaptive_max_token_budget=(policy.adaptive_max_token_budget),
                    adaptive_tail_mass_ratio=policy.adaptive_tail_mass_ratio,
                )
                _restore_prompt_cache(routing_cache, prompt_tokens)
            bf16_result_row = copy.deepcopy(dense_records[key]["arms"][BF16_ARM])
            rank_state["records"].append(
                {
                    "key": key,
                    "task": task.name,
                    "task_family": task.family,
                    "sample_ordinal": ordinal,
                    "source_index": source.get("index"),
                    "declared_length": source.get("length"),
                    "prompt_tokens": prompt_tokens,
                    "tokens_to_generate": task.tokens_to_generate,
                    "match_type": task.match_type,
                    "references": references,
                    "prefill_seconds": prefill_seconds,
                    "first_generated_token_from_c1_exact_prefill": first_token,
                    "arms": {
                        BF16_ARM: bf16_result_row,
                        c1_arm: c1_result_row,
                        **sparse_result_rows,
                    },
                }
            )
            rank_state["elapsed_seconds"] = time.perf_counter() - started
            rank_state["maximum_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            _atomic_json(rank_path, rank_state)
            sparse_score_text = " ".join(
                f"{policy.arm}={sparse_result_rows[policy.arm]['score']:.3f}"
                for policy in sparse_policies
            )
            print(
                f"[KQ routing RULER] rank={rank} sample={key} prompt={prompt_tokens} "
                f"bf16={bf16_result_row['score']:.3f} "
                f"c1={c1_result_row['score']:.3f} "
                f"{sparse_score_text} "
                f"prefill={prefill_seconds:.2f}s "
                f"c1_decode={c1_result_row['elapsed_seconds']:.2f}s "
                f"sparse_decode={sum(row['elapsed_seconds'] for row in sparse_result_rows.values()):.2f}s",
                flush=True,
            )

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        rank_payloads = [
            _load_rank_state(
                output_dir / f"rank_{other_rank:02d}.json",
                fingerprint=fingerprint,
                rank=other_rank,
                world_size=world_size,
            )
            for other_rank in range(world_size)
        ]
        task_order = {task.name: index for index, task in enumerate(tasks)}
        records = sorted(
            (record for payload in rank_payloads for record in payload["records"]),
            key=lambda row: (task_order[row["task"]], row["sample_ordinal"]),
        )
        if len(records) != len(work):
            raise RuntimeError(
                f"KQ routing RULER merge has {len(records)} rows, expected {len(work)}"
            )
        runtime_by_arm = {
            policy.arm: _merge_runtime(
                [record["arms"][policy.arm]["runtime_logical"] for record in records]
            )
            for policy in sparse_policies
        }
        runtime = runtime_by_arm[routing_arm]
        payload = {
            "format": FORMAT,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "metadata": {
                **protocol_identity,
                "model_path": str(model_path),
                "c1_export": str(c1_export),
                "routing_factors": str(factor_dir),
                "routing_factor_format": factor_result["format"],
                "dense_baseline": str(dense_baseline_path),
                "data_dir": str(data_dir),
                "ruler_revision": dataset_manifest["ruler"]["revision"],
                "official_base_prompt": "input + answer_prefix",
                "greedy_decoding": True,
                "chunked_dense_c1_sdpa_prompt_prefill": True,
                "prefill_chunk_size": args.prefill_chunk_size,
                "empty_cache_between_prefill_chunks": (
                    args.empty_cache_between_prefill_chunks
                ),
                "incremental_routing_sidecar": args.incremental_routing_sidecar,
                "first_generated_token_from_dense_c1_prefill": True,
                "routing_enabled_during_decode": True,
                "adaptive_budget_enabled": bool(adaptive_tail_mass_ratios),
                "exact_k_gpu_resident": True,
                "layers": layers,
                "persistent_gpu_scalar_ratio": routing_storage_ratio(
                    value_rank=value_rank,
                    routing_rank=args.routing_rank,
                    key_width=head_dim,
                    value_width=head_dim,
                ),
                "rank_runtimes": [payload.get("runtime") for payload in rank_payloads],
            },
            "summary": {
                "arms": [summarize_arm(records, arm, tasks) for arm in arms],
                "paired_c1_vs_bf16": paired_summary(
                    records,
                    BF16_ARM,
                    c1_arm,
                    tasks,
                ),
                "paired_routing_vs_c1": paired_summary(
                    records,
                    c1_arm,
                    routing_arm,
                    tasks,
                ),
                "paired_routing_vs_bf16": paired_summary(
                    records,
                    BF16_ARM,
                    routing_arm,
                    tasks,
                ),
                "runtime_logical": runtime,
                "paired_sparse_vs_c1": {
                    policy.arm: paired_summary(
                        records,
                        c1_arm,
                        policy.arm,
                        tasks,
                    )
                    for policy in sparse_policies
                },
                "paired_sparse_vs_bf16": {
                    policy.arm: paired_summary(
                        records,
                        BF16_ARM,
                        policy.arm,
                        tasks,
                    )
                    for policy in sparse_policies
                },
                "runtime_logical_by_arm": runtime_by_arm,
            },
            "records": records,
            "elapsed_seconds": time.perf_counter() - started,
            "limitations": [
                "Qwen3-8B-Base is not the post-trained Qwen3-8B in the public RULER table",
                "the first generated token is computed by dense C1 prompt prefill",
                "exact K remains GPU resident in this correctness oracle",
                "logical exact-K traffic does not measure PCIe latency or page-cache hits",
                "Python reference routing time is not serving throughput",
                *(
                    [
                        "static YaRN changes RoPE at every position, including positions inside the native context",
                        "the R32 routing factors were calibrated at 32K without YaRN",
                    ]
                    if rope_scaling is not None
                    else []
                ),
            ],
        }
        _atomic_json(output_dir / "result.json", payload)
        _atomic_text(output_dir / "summary.md", _markdown(payload))
        print(f"[KQ routing RULER] result={output_dir / 'result.json'}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
