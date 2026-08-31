#!/usr/bin/env python3
"""Block-scheduled RULER-v1 for pairwise KQ-SVD + QUEST + C1-V64."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.pairwise_k_qwen3 import (  # noqa: E402
    PairwiseKRuntime,
    PairwiseQuestConfig,
    install_qwen3_pairwise_k_runtime,
)
from evaluation import eval_qwen3_32b_c1_wikitext as c1_evaluator  # noqa: E402
from evaluation.eval_qwen3_8b_pairwise_k_dense_v_ppl import (  # noqa: E402
    _dtype,
    _load_factors,
    _sha256,
)
from evaluation.eval_qwen3_8b_post_rope_kqsvd_c1_wikitext import (  # noqa: E402
    _replace_attention,
)
from evaluation.eval_qwen3_c1_quest_ruler import (  # noqa: E402
    _atomic_json,
    _atomic_text,
    _build_work,
    _eos_ids,
    _fingerprint,
    _greedy_continuation,
    _load_dataset_manifest,
)
from evaluation.ruler_v1 import (  # noqa: E402
    RulerTask,
    paired_summary,
    parse_tasks,
    ruler_prompt,
    sample_score,
    summarize_arm,
)


FORMAT = "basisserve.qwen3_8b.pairwise_k_c1_v64_quest.ruler_v1.v1"
RANK_FORMAT = "basisserve.qwen3_8b.pairwise_k_c1_v64_quest.ruler_v1.rank.v1"
BASELINE_FORMAT = "basisserve.qwen3_8b.c1_quest.ruler_v1.v1"
C1_FORMAT = "basisserve.qwen3_8b.gqa_c1_joint.v1"
DENSE_ARM = "dense_k_c1_v64"
FULL_ARM = "pairwise_full_support"
SPARSE_ARM = "pairwise_quest_b512"
ARMS = (DENSE_ARM, FULL_ARM, SPARSE_ARM)


def _parse_layers(raw: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("full layers must be comma-separated integers") from error
    if not layers or min(layers) < 0 or len(layers) != len(set(layers)):
        raise ValueError("full layers must be unique nonnegative integers")
    return layers


def _load_baseline(
    path: Path,
    *,
    model_path: Path,
    c1_result_path: Path,
    dataset_manifest_hash: str,
    sequence_length: int,
    samples_per_task: int,
    tasks: Sequence[RulerTask],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != BASELINE_FORMAT or payload.get("status") != "complete":
        raise ValueError("RULER baseline is not a complete C1-V64 result")
    metadata = payload["metadata"]
    expected = {
        "model_config_sha256": _sha256(model_path / "config.json"),
        "c1_results_sha256": _sha256(c1_result_path),
        "dataset_manifest_sha256": dataset_manifest_hash,
        "sequence_length": sequence_length,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(
                f"RULER baseline {name}={metadata.get(name)!r}, expected {value!r}"
            )
    requested_tasks = [task.name for task in tasks]
    if any(task not in metadata["tasks"] for task in requested_tasks):
        raise ValueError("RULER baseline does not contain every requested task")
    if int(metadata["samples_per_task"]) < samples_per_task:
        raise ValueError("RULER baseline has fewer samples than requested")
    records = {
        str(record["key"]): record
        for record in payload["records"]
        if record["task"] in requested_tasks
        and int(record["sample_ordinal"]) < samples_per_task
    }
    expected_records = samples_per_task * len(tasks)
    if len(records) != expected_records:
        raise ValueError(
            f"RULER baseline yielded {len(records)} rows, expected {expected_records}"
        )
    if any(DENSE_ARM not in record["arms"] for record in records.values()):
        raise ValueError(f"RULER baseline is missing arm {DENSE_ARM!r}")
    return payload, records


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
    expected = (RANK_FORMAT, fingerprint, rank, world_size)
    observed = (
        payload.get("format"),
        payload.get("fingerprint"),
        payload.get("rank"),
        payload.get("world_size"),
    )
    if observed != expected:
        raise ValueError(f"rank resume state is incompatible: {path}")
    keys = [record["key"] for record in payload["records"]]
    if len(keys) != len(set(keys)):
        raise ValueError(f"rank resume state contains duplicate samples: {path}")
    return payload


def _block_prefill(
    model: torch.nn.Module,
    runtime: PairwiseKRuntime,
    input_ids: torch.Tensor,
    *,
    block_length: int,
) -> tuple[int, Any]:
    runtime.reset()
    cache = DynamicCache()
    final_logits: torch.Tensor | None = None
    prompt_tokens = int(input_ids.shape[1])
    for start in range(0, prompt_tokens, block_length):
        stop = min(start + block_length, prompt_tokens)
        output = model(
            input_ids=input_ids[:, start:stop],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        final_logits = output.logits[:, -1]
    if final_logits is None:
        raise ValueError("RULER prompt is empty")
    return int(final_logits[0].argmax().item()), cache


def _evaluate_candidate_arm(
    *,
    model: torch.nn.Module,
    runtime: PairwiseKRuntime,
    input_ids: torch.Tensor,
    task: RulerTask,
    references: Sequence[str],
    tokenizer: Any,
    arm: str,
    block_length: int,
    page_size: int,
    token_budget: int,
    full_layers: tuple[int, ...],
    landmark_dtype: str,
    device: torch.device,
) -> tuple[dict[str, Any], int]:
    runtime.set_mode("pairwise")
    runtime.set_sparse_policy(
        None
        if arm == FULL_ARM
        else PairwiseQuestConfig(
            page_size=page_size,
            historical_token_budget=token_budget,
            landmark_dtype=landmark_dtype,
        ),
        full_layer_indices=full_layers,
    )
    runtime.reset_sparse_statistics()
    torch.cuda.synchronize(device)
    prefill_started = time.perf_counter()
    first_token, cache = _block_prefill(
        model,
        runtime,
        input_ids,
        block_length=block_length,
    )
    torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - prefill_started
    torch.cuda.synchronize(device)
    decode_started = time.perf_counter()
    generated_ids, cache = _greedy_continuation(
        model,
        cache,
        first_token,
        maximum_tokens=task.tokens_to_generate,
        eos_ids=_eos_ids(tokenizer, model),
        device=device,
    )
    torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
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
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
    }
    if arm == SPARSE_ARM:
        result["runtime_logical"] = runtime.sparse_statistics()["totals"]
    del cache
    return result, first_token


def _markdown(payload: Mapping[str, Any]) -> str:
    summaries = {row["arm"]: row for row in payload["summary"]["arms"]}
    dense = summaries[DENSE_ARM]
    full = summaries[FULL_ARM]
    sparse = summaries[SPARSE_ARM]
    paired = payload["summary"]["paired_sparse_vs_full"]
    lines = [
        "# Qwen3-8B-Base Pairwise-KQ-QUEST + C1-V64 RULER-v1 4K",
        "",
        "Prompt prefill and generation both use the 128-token transactional block schedule.",
        "",
        "| Task | Samples | Dense-K+C1 | Pairwise full | Pairwise B512 | "
        "B512 vs full | Regressions | Improvements |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in payload["metadata"]["tasks"]:
        dense_row = dense["tasks"][task]
        full_row = full["tasks"][task]
        sparse_row = sparse["tasks"][task]
        pair = paired["tasks"][task]
        lines.append(
            f"| {task} | {full_row['samples']} | "
            f"{100 * dense_row['accuracy']:.2f}% | "
            f"{100 * full_row['accuracy']:.2f}% | "
            f"{100 * sparse_row['accuracy']:.2f}% | "
            f"{100 * (sparse_row['accuracy'] - full_row['accuracy']):+.2f} pp | "
            f"{pair['sparse_regressions']} | {pair['sparse_improvements']} |"
        )
    overall = paired["all_samples"]
    lines.extend(
        [
            f"| **Task-balanced mean** | {len(payload['records'])} | "
            f"**{100 * dense['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * full['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * sparse['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * (sparse['task_balanced_accuracy'] - full['task_balanced_accuracy']):+.2f} pp** | "
            f"**{overall['sparse_regressions']}** | "
            f"**{overall['sparse_improvements']}** |",
            "",
            "Protocol: official RULER-v1 base prompts and substring scorers; "
            "greedy decoding; block length 128; page size 16; Pairwise-QUEST "
            "historical budget 512; layers 0 and 1 use full compressed-K support; "
            "resident C1-V64.",
            "",
            "Dense-K+C1 is reused from the frozen matching RULER baseline. "
            "This remains a quality oracle: exact K is retained by DynamicCache "
            "and eager gather time is not serving latency.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--pair-factor-dir", type=Path, required=True)
    parser.add_argument("--c1-factor-dir", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "results/datasets/qwen3_8b_base_ruler_v1_4k",
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--samples-per-task", type=int, default=100)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--historical-token-budget", type=int, default=512)
    parser.add_argument("--full-layers", default="0,1")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


@torch.inference_mode()
def main() -> None:
    args = _parser().parse_args()
    started = time.perf_counter()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not torch.cuda.is_available():
        raise RuntimeError("pairwise-K RULER evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    model_path = args.model_path.expanduser().resolve()
    pair_factor_dir = args.pair_factor_dir.expanduser().resolve()
    c1_factor_dir = args.c1_factor_dir.expanduser().resolve()
    baseline_path = args.baseline_result.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_tasks(args.tasks)
    full_layers = _parse_layers(args.full_layers)
    if min(
        args.sequence_length,
        args.samples_per_task,
        args.block_length,
        args.page_size,
        args.historical_token_budget,
    ) <= 0:
        raise ValueError("RULER and sparse-attention sizes must be positive")
    if args.historical_token_budget % args.page_size:
        raise ValueError("historical token budget must align to complete pages")

    dataset_manifest, dataset_manifest_hash = _load_dataset_manifest(
        data_dir,
        sequence_length=args.sequence_length,
        samples=args.samples_per_task,
        tasks=tasks,
        tokenizer_path=model_path,
    )
    c1_evaluator.activate_model_profile("qwen3_8b")
    c1_result = c1_evaluator._load_results(c1_factor_dir, model_path)
    if c1_result.get("format") != C1_FORMAT:
        raise ValueError("C1 factors are not the uniform Qwen3-8B checkpoint")
    if int(c1_result["fit_config"]["cache_rank_per_head"]) != 64:
        raise ValueError("RULER protocol requires C1-V64")
    c1_result_path = c1_factor_dir / "results.json"
    baseline, baseline_records = _load_baseline(
        baseline_path,
        model_path=model_path,
        c1_result_path=c1_result_path,
        dataset_manifest_hash=dataset_manifest_hash,
        sequence_length=args.sequence_length,
        samples_per_task=args.samples_per_task,
        tasks=tasks,
    )
    pair_result, pair_factors, pair_result_path = _load_factors(
        pair_factor_dir,
        model_path,
    )
    work = _build_work(data_dir, tasks, args.samples_per_task)
    assigned = [row for row in work if row[0] % world_size == rank]
    protocol_identity = {
        "model_config_sha256": _sha256(model_path / "config.json"),
        "pair_results_sha256": _sha256(pair_result_path),
        "c1_results_sha256": _sha256(c1_result_path),
        "baseline_result_sha256": _sha256(baseline_path),
        "dataset_manifest_sha256": dataset_manifest_hash,
        "sequence_length": args.sequence_length,
        "tasks": [task.name for task in tasks],
        "samples_per_task": args.samples_per_task,
        "arms": list(ARMS),
        "block_length": args.block_length,
        "page_size": args.page_size,
        "historical_token_budget": args.historical_token_budget,
        "full_layers": list(full_layers),
        "landmark_dtype": args.landmark_dtype,
        "dtype": args.dtype,
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
        f"[Pairwise RULER] rank={rank}/{world_size} assigned={len(assigned)} "
        f"completed={len(completed)} pending={len(pending)} device={device}",
        flush=True,
    )

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=_dtype(args.dtype),
            attn_implementation="eager",
            local_files_only=True,
        ).to(device).eval()
        c1_installation = _replace_attention(model, c1_factor_dir, c1_result)
        runtime = install_qwen3_pairwise_k_runtime(
            model,
            independent_key_projector=pair_factors["independent_key_projector"],
            independent_query_projector=pair_factors["independent_query_projector"],
            pair_key_projector=pair_factors["pair_key_projector"],
            pair_query_projector=pair_factors["pair_query_projector"],
        )
        if any(layer >= len(runtime.modules) for layer in full_layers):
            raise ValueError("full-support layer is outside the model")
        rank_state["runtime"] = {
            "cuda_device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "c1_layers": len(c1_installation),
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
            references = list(source["outputs"])
            full_result, full_first_token = _evaluate_candidate_arm(
                model=model,
                runtime=runtime,
                input_ids=input_ids,
                task=task,
                references=references,
                tokenizer=tokenizer,
                arm=FULL_ARM,
                block_length=args.block_length,
                page_size=args.page_size,
                token_budget=args.historical_token_budget,
                full_layers=full_layers,
                landmark_dtype=args.landmark_dtype,
                device=device,
            )
            sparse_result, sparse_first_token = _evaluate_candidate_arm(
                model=model,
                runtime=runtime,
                input_ids=input_ids,
                task=task,
                references=references,
                tokenizer=tokenizer,
                arm=SPARSE_ARM,
                block_length=args.block_length,
                page_size=args.page_size,
                token_budget=args.historical_token_budget,
                full_layers=full_layers,
                landmark_dtype=args.landmark_dtype,
                device=device,
            )
            dense_result = copy.deepcopy(baseline_records[key]["arms"][DENSE_ARM])
            dense_ids = dense_result["generated_token_ids"]
            row = {
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
                "full_first_token_matches_dense_c1": bool(
                    dense_ids and int(dense_ids[0]) == full_first_token
                ),
                "sparse_first_token_matches_dense_c1": bool(
                    dense_ids and int(dense_ids[0]) == sparse_first_token
                ),
                "sparse_first_token_matches_full": (
                    sparse_first_token == full_first_token
                ),
                "arms": {
                    DENSE_ARM: dense_result,
                    FULL_ARM: full_result,
                    SPARSE_ARM: sparse_result,
                },
            }
            del input_ids
            rank_state["records"].append(row)
            rank_state["elapsed_seconds"] = time.perf_counter() - started
            rank_state["maximum_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            _atomic_json(rank_path, rank_state)
            print(
                f"[Pairwise RULER] rank={rank} sample={key} prompt={prompt_tokens} "
                f"dense={dense_result['score']:.3f} full={full_result['score']:.3f} "
                f"b512={sparse_result['score']:.3f} "
                f"full_prefill={full_result['prefill_seconds']:.2f}s "
                f"b512_prefill={sparse_result['prefill_seconds']:.2f}s",
                flush=True,
            )

    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
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
                f"pairwise RULER merge has {len(records)} rows, expected {len(work)}"
            )
        payload = {
            "format": FORMAT,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "metadata": {
                **protocol_identity,
                "model_path": str(model_path),
                "pair_factor_result": str(pair_result_path),
                "pair_factor_format": pair_result["format"],
                "c1_factor_result": str(c1_result_path),
                "baseline_result": str(baseline_path),
                "data_dir": str(data_dir),
                "ruler_revision": dataset_manifest["ruler"]["revision"],
                "official_base_prompt": "input + answer_prefix",
                "greedy_decoding": True,
                "block_scheduled_prompt_prefill": True,
                "exact_current_block": True,
                "exact_k_gpu_resident": True,
                "rank_runtimes": [payload.get("runtime") for payload in rank_payloads],
            },
            "summary": {
                "arms": [summarize_arm(records, arm, tasks) for arm in ARMS],
                "paired_sparse_vs_full": paired_summary(
                    records,
                    FULL_ARM,
                    SPARSE_ARM,
                    tasks,
                ),
                "paired_full_vs_dense": paired_summary(
                    records,
                    DENSE_ARM,
                    FULL_ARM,
                    tasks,
                ),
                "paired_sparse_vs_dense": paired_summary(
                    records,
                    DENSE_ARM,
                    SPARSE_ARM,
                    tasks,
                ),
            },
            "records": records,
            "elapsed_seconds": time.perf_counter() - started,
            "limitations": [
                "Qwen3-8B-Base is not the post-trained Qwen3-8B in the public RULER table",
                "dense-K C1-V64 scores are reused from the frozen matching baseline",
                "native DynamicCache physically retains exact historical K",
                "latent historical K is not physically CPU-offloaded",
                "eager reference elapsed time is not serving latency",
            ],
        }
        _atomic_json(output_dir / "result.json", payload)
        _atomic_text(output_dir / "summary.md", _markdown(payload))
        print(f"[Pairwise RULER] result={output_dir / 'result.json'}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
