#!/usr/bin/env python3
"""Paired RULER-v1 evaluation of exact-K Store80 and Store80 Route32."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_joint_routing_payload_s80_qwen3 import (  # noqa: E402
    S80Qwen3Attention,
    install_qwen3_s80_factor_bank,
)
from basisserve.core.c1_k_reverse_shadow import ReverseShadowConfig  # noqa: E402
from evaluation.eval_c1_block_scheduled_ppl import _dtype  # noqa: E402
from evaluation.eval_qwen3_8b_c1_kq_routing_ruler import (  # noqa: E402
    _assigned_work,
    _chunked_exact_prefill,
    _load_dense_baseline,
    _restore_prompt_cache,
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
    ARM as DENSE_ARM,
    _qwen3_config,
)
from evaluation.ruler_v1 import (  # noqa: E402
    paired_summary,
    parse_tasks,
    ruler_prompt,
    sample_score,
    summarize_arm,
)


FORMAT = "basisserve.qwen3_8b.s80_route32.ruler_v1.v1"
RANK_FORMAT = "basisserve.qwen3_8b.s80_route32.ruler_v1.rank.v1"
EXACT_ARM = "store80_exact_k"
ROUTED_ARM = "store80_route32_exact_k"
DEFAULT_TASKS = (
    "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,"
    "niah_multikey_2,niah_multiquery,niah_multivalue,vt,fwe,qa_1,qa_2"
)


def _attention_modules(model: torch.nn.Module) -> list[S80Qwen3Attention]:
    modules = [layer.self_attn for layer in model.model.layers]
    assert all(isinstance(module, S80Qwen3Attention) for module in modules)
    return modules


def _set_sparse_policy(
    modules: Sequence[S80Qwen3Attention],
    *,
    token_budget: int | None,
    page_size: int,
    force_last_page: bool,
) -> None:
    config = (
        None
        if token_budget is None
        else ReverseShadowConfig(
            page_size=page_size,
            exact_token_budget=token_budget,
            recent_exact_window=1 if force_last_page else 0,
            selector="kq_svd",
            landmark_dtype="bfloat16",
        )
    )
    for module in modules:
        module.set_sparse_routing_config(config)


def _runtime_statistics(
    modules: Sequence[S80Qwen3Attention],
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
        row = module.sparse_routing_statistics()
        for name in sum_names:
            totals[name] += float(row[name])
        totals["resident_selector_metadata_bytes"] += float(
            row["maximum_resident_selector_metadata_bytes"]
        )
    valid = totals["physical_valid_tokens"]
    query_valid = totals["query_valid_tokens"]
    totals["selected_token_fraction"] = (
        totals["selected_tokens"] / valid if valid else 0.0
    )
    totals["query_selected_token_fraction"] = (
        totals["query_selected_tokens"] / query_valid if query_valid else 0.0
    )
    return totals


def _merge_runtime(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    merged = {name: 0.0 for name in rows[0]}
    for name in merged:
        values = [float(row[name]) for row in rows]
        merged[name] = (
            max(values) if name == "resident_selector_metadata_bytes" else sum(values)
        )
    valid = merged["physical_valid_tokens"]
    query_valid = merged["query_valid_tokens"]
    merged["selected_token_fraction"] = (
        merged["selected_tokens"] / valid if valid else 0.0
    )
    merged["query_selected_token_fraction"] = (
        merged["query_selected_tokens"] / query_valid if query_valid else 0.0
    )
    return merged


def _evaluate_arm(
    *,
    model: torch.nn.Module,
    modules: Sequence[S80Qwen3Attention],
    cache: Any,
    first_token: int,
    task: Any,
    references: Sequence[str],
    tokenizer: Any,
    sparse: bool,
    page_size: int,
    token_budget: int,
    force_last_page: bool,
    device: torch.device,
) -> dict[str, Any]:
    _set_sparse_policy(
        modules,
        token_budget=token_budget if sparse else None,
        page_size=page_size,
        force_last_page=force_last_page if sparse else False,
    )
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
        "elapsed_seconds": time.perf_counter() - started,
    }
    if sparse:
        result["runtime_logical"] = _runtime_statistics(modules)
    del cache
    return result


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
    assert (
        payload.get("format"),
        payload.get("fingerprint"),
        payload.get("rank"),
        payload.get("world_size"),
    ) == (RANK_FORMAT, fingerprint, rank, world_size)
    keys = [record["key"] for record in payload["records"]]
    assert len(keys) == len(set(keys))
    return payload


def _markdown(payload: Mapping[str, Any]) -> str:
    metadata = payload["metadata"]
    summaries = {row["arm"]: row for row in payload["summary"]["arms"]}
    dense = summaries[DENSE_ARM]
    exact = summaries[EXACT_ARM]
    routed = summaries[ROUTED_ARM]
    paired = payload["summary"]["paired_routed_vs_exact"]
    runtime = payload["summary"]["runtime_logical"]
    decode_queries = runtime["queries"] / int(metadata["layers"])
    exact_k_mib = (
        runtime["cpu_exact_key_bytes_fetched"] / decode_queries / (1 << 20)
        if decode_queries
        else 0.0
    )
    lines = [
        "# Qwen3-8B-Base Store80/Route32 RULER-v1 32K",
        "",
        "Store80 exact-K and Store80 Route32-selected exact-K use the same joint "
        "80-dimensional cache and identical greedy-decoding prompts. Route32 reads "
        "the first 32 orthogonal Store80 coordinates; it does not allocate a second "
        "routing sidecar.",
        "",
        "| Task | Samples | BF16 dense | Store80 exact-K | Route32/B"
        f"{metadata['exact_token_budget']} | Routing delta | Regressions | Improvements |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in metadata["tasks"]:
        dense_row = dense["tasks"][task]
        exact_row = exact["tasks"][task]
        routed_row = routed["tasks"][task]
        pair = paired["tasks"][task]
        lines.append(
            f"| {task} | {dense_row['samples']} | "
            f"{100 * dense_row['accuracy']:.2f}% | "
            f"{100 * exact_row['accuracy']:.2f}% | "
            f"{100 * routed_row['accuracy']:.2f}% | "
            f"{100 * (routed_row['accuracy'] - exact_row['accuracy']):+.2f} pp | "
            f"{pair['sparse_regressions']} | {pair['sparse_improvements']} |"
        )
    overall = paired["all_samples"]
    lines.extend(
        [
            f"| **Task-balanced mean** | {len(payload['records'])} | "
            f"**{100 * dense['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * exact['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * routed['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * (routed['task_balanced_accuracy'] - exact['task_balanced_accuracy']):+.2f} pp** | "
            f"**{overall['sparse_regressions']}** | "
            f"**{overall['sparse_improvements']}** |",
            "",
            f"Logical exact-K traffic: `{exact_k_mib:.3f} MiB/decode token`; "
            f"physical selected-K fraction: `{runtime['selected_token_fraction']:.6f}`; "
            f"deployable joint-cache scalar ratio: `{metadata['joint_cache_scalar_ratio']:.4f}`.",
            "",
            "Prompt prefill and the first generated token use Store80 exact-K. "
            "Route32 is enabled for subsequent decode tokens. Exact K remains "
            "GPU-resident in this correctness oracle, so exact-K traffic is logical.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--s80-export", type=Path, required=True)
    parser.add_argument("--dense-baseline", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--assignment-rank-offset", type=int, default=0)
    parser.add_argument("--routing-rank", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--exact-token-budget", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--empty-cache-between-prefill-chunks", action="store_true")
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
    started = time.perf_counter()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    assert torch.cuda.is_available()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="gloo")

    model_path = args.model_path.expanduser().resolve()
    s80_export = args.s80_export.expanduser().resolve()
    dense_baseline_path = args.dense_baseline.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_tasks(args.tasks)
    task_names = [task.name for task in tasks]
    effective_model_config, rope_scaling = _qwen3_config(
        model_path,
        sequence_length=args.sequence_length,
        yarn_factor=None,
    )
    dataset_manifest, dataset_hash = _load_dataset_manifest(
        data_dir,
        sequence_length=args.sequence_length,
        samples=args.samples_per_task,
        tasks=tasks,
        tokenizer_path=model_path,
    )
    _, dense_records = _load_dense_baseline(
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

    s80_manifest_path = s80_export / "manifest.json"
    s80_manifest = json.loads(s80_manifest_path.read_text(encoding="utf-8"))
    geometry = s80_manifest["geometry"]
    assert int(geometry["joint_rank"]) == 80
    assert int(geometry["routing_rank"]) == args.routing_rank == 32
    assert s80_manifest["routing_layout"] == {
        "stored_coordinates": 80,
        "routed_coordinates": 32,
        "routed_coordinate_start": 0,
        "selector_materialized": False,
    }
    layers = int(effective_model_config.num_hidden_layers)
    head_dim = int(effective_model_config.head_dim)
    assert len(s80_manifest["layer_coverage"]) == layers

    arms = (DENSE_ARM, EXACT_ARM, ROUTED_ARM)
    protocol_identity = {
        "model_config_sha256": _sha256(model_path / "config.json"),
        "s80_manifest_sha256": _sha256(s80_manifest_path),
        "dense_baseline_sha256": _sha256(dense_baseline_path),
        "dataset_manifest_sha256": dataset_hash,
        "sequence_length": args.sequence_length,
        "tasks": task_names,
        "samples_per_task": args.samples_per_task,
        "arms": list(arms),
        "joint_rank": 80,
        "routing_rank": args.routing_rank,
        "page_size": args.page_size,
        "exact_token_budget": args.exact_token_budget,
        "prefill_chunk_size": args.prefill_chunk_size,
        "force_last_page": args.force_last_page,
        "dtype": args.dtype,
        "assignment_rank_offset": args.assignment_rank_offset,
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
        f"[Store80 RULER] rank={rank}/{world_size} assigned={len(assigned)} "
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
        replacements = install_qwen3_s80_factor_bank(
            model,
            s80_export,
            attention_backend="sdpa",
        )
        modules = _attention_modules(model)
        assert len(replacements) == len(modules) == layers
        rank_state["runtime"] = {
            "cuda_device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "prefill_attention_backend": "torch_sdpa_chunked_exact",
            "prefill_chunk_size": args.prefill_chunk_size,
            "route32_source": "first_32_coordinates_of_store80",
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
            assert prompt_tokens + task.tokens_to_generate <= args.sequence_length
            _set_sparse_policy(
                modules,
                token_budget=None,
                page_size=args.page_size,
                force_last_page=False,
            )
            torch.cuda.synchronize(device)
            prefill_started = time.perf_counter()
            first_token, cache = _chunked_exact_prefill(
                model,
                input_ids,
                chunk_size=args.prefill_chunk_size,
                empty_cuda_cache_between_chunks=(
                    args.empty_cache_between_prefill_chunks
                ),
                incremental_routing_sidecar=False,
            )
            torch.cuda.synchronize(device)
            prefill_seconds = time.perf_counter() - prefill_started
            del input_ids
            references = list(source["outputs"])
            exact_result = _evaluate_arm(
                model=model,
                modules=modules,
                cache=cache,
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
            _restore_prompt_cache(cache, prompt_tokens)
            routed_result = _evaluate_arm(
                model=model,
                modules=modules,
                cache=cache,
                first_token=first_token,
                task=task,
                references=references,
                tokenizer=tokenizer,
                sparse=True,
                page_size=args.page_size,
                token_budget=args.exact_token_budget,
                force_last_page=args.force_last_page,
                device=device,
            )
            _restore_prompt_cache(cache, prompt_tokens)
            dense_result = copy.deepcopy(dense_records[key]["arms"][DENSE_ARM])
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
                    "first_generated_token_from_store80_exact_prefill": first_token,
                    "arms": {
                        DENSE_ARM: dense_result,
                        EXACT_ARM: exact_result,
                        ROUTED_ARM: routed_result,
                    },
                }
            )
            rank_state["elapsed_seconds"] = time.perf_counter() - started
            rank_state["maximum_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            _atomic_json(rank_path, rank_state)
            print(
                f"[Store80 RULER] rank={rank} sample={key} prompt={prompt_tokens} "
                f"dense={dense_result['score']:.3f} exact={exact_result['score']:.3f} "
                f"route32={routed_result['score']:.3f} prefill={prefill_seconds:.2f}s "
                f"exact_decode={exact_result['elapsed_seconds']:.2f}s "
                f"routed_decode={routed_result['elapsed_seconds']:.2f}s",
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
        assert len(records) == len(work)
        runtime = _merge_runtime(
            [record["arms"][ROUTED_ARM]["runtime_logical"] for record in records]
        )
        payload = {
            "format": FORMAT,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "metadata": {
                **protocol_identity,
                "model_path": str(model_path),
                "s80_export": str(s80_export),
                "dense_baseline": str(dense_baseline_path),
                "data_dir": str(data_dir),
                "ruler_revision": dataset_manifest["ruler"]["revision"],
                "official_base_prompt": "input + answer_prefix",
                "greedy_decoding": True,
                "first_generated_token_from_store80_exact_prefill": True,
                "route32_source": "first_32_coordinates_of_store80",
                "separate_routing_sidecar": False,
                "exact_k_gpu_resident": True,
                "layers": layers,
                "joint_cache_scalar_ratio": 80 / (head_dim + head_dim),
                "rank_runtimes": [payload.get("runtime") for payload in rank_payloads],
            },
            "summary": {
                "arms": [summarize_arm(records, arm, tasks) for arm in arms],
                "paired_exact_vs_dense": paired_summary(
                    records,
                    DENSE_ARM,
                    EXACT_ARM,
                    tasks,
                ),
                "paired_routed_vs_exact": paired_summary(
                    records,
                    EXACT_ARM,
                    ROUTED_ARM,
                    tasks,
                ),
                "paired_routed_vs_dense": paired_summary(
                    records,
                    DENSE_ARM,
                    ROUTED_ARM,
                    tasks,
                ),
                "runtime_logical": runtime,
            },
            "records": records,
            "elapsed_seconds": time.perf_counter() - started,
            "limitations": [
                "Qwen3-8B-Base is not the post-trained Qwen3-8B in the public RULER table",
                "the first generated token is computed by exact-K Store80 prompt prefill",
                "exact K remains GPU resident in this correctness oracle",
                "logical exact-K traffic does not measure PCIe latency or page-cache hits",
                "the Python reference sparse path is not a serving-throughput benchmark",
            ],
        }
        _atomic_json(output_dir / "result.json", payload)
        _atomic_text(output_dir / "summary.md", _markdown(payload))
        print(f"[Store80 RULER] result={output_dir / 'result.json'}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
