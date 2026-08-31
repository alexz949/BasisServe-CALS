#!/usr/bin/env python3
"""RULER control for one-shot versus block-scheduled Dense-K+C1-V64."""

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
from typing import Any, Mapping

import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import eval_qwen3_32b_c1_wikitext as c1_evaluator  # noqa: E402
from evaluation.eval_qwen3_8b_pairwise_k_c1_v64_quest_ruler import (  # noqa: E402
    _load_baseline,
)
from evaluation.eval_qwen3_8b_pairwise_k_dense_v_ppl import (  # noqa: E402
    _dtype,
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
from evaluation.eval_qwen3_c1_k_reverse_shadow_ppl import (  # noqa: E402
    _attention_modules,
    _runtime_statistics,
    _set_policy,
)
from evaluation.ruler_v1 import (  # noqa: E402
    paired_summary,
    parse_tasks,
    ruler_prompt,
    sample_score,
    summarize_arm,
)


FORMAT = "basisserve.qwen3_8b.c1_dense_block.ruler_v1.v1"
RANK_FORMAT = "basisserve.qwen3_8b.c1_dense_block.ruler_v1.rank.v1"
C1_FORMAT = "basisserve.qwen3_8b.gqa_c1_joint.v1"
ONESHOT_ARM = "dense_k_c1_v64"
BLOCK_ARM = "dense_k_c1_v64_block128"
QUEST_ARM = "exact_k_quest_b512_block128"
ARMS = (ONESHOT_ARM, BLOCK_ARM)


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


def _block_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    block_length: int,
) -> tuple[int, Any]:
    cache = DynamicCache()
    final_logits: torch.Tensor | None = None
    for start in range(0, int(input_ids.shape[1]), block_length):
        stop = min(start + block_length, int(input_ids.shape[1]))
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


def _evaluate_block(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    task: Any,
    references: list[str],
    tokenizer: Any,
    block_length: int,
    device: torch.device,
) -> tuple[dict[str, Any], int]:
    torch.cuda.synchronize(device)
    prefill_started = time.perf_counter()
    first_token, cache = _block_prefill(
        model,
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
    del cache
    return result, first_token


def _markdown(payload: Mapping[str, Any]) -> str:
    summaries = {row["arm"]: row for row in payload["summary"]["arms"]}
    oneshot = summaries[ONESHOT_ARM]
    block = summaries[BLOCK_ARM]
    paired = payload["summary"]["paired"]
    lines = [
        "# Dense-K + C1-V64 RULER block-schedule control",
        "",
        "The frozen one-shot baseline is compared with exact Dense-K+C1-V64 "
        "using 128-token prompt blocks and the same greedy decode.",
        "",
        "| Task | Samples | One-shot | Block-128 | Delta | Regressions | Improvements |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in payload["metadata"]["tasks"]:
        one = oneshot["tasks"][task]
        blk = block["tasks"][task]
        pair = paired["tasks"][task]
        lines.append(
            f"| {task} | {blk['samples']} | {100 * one['accuracy']:.2f}% | "
            f"{100 * blk['accuracy']:.2f}% | "
            f"{100 * (blk['accuracy'] - one['accuracy']):+.2f} pp | "
            f"{pair['sparse_regressions']} | {pair['sparse_improvements']} |"
        )
    overall = paired["all_samples"]
    lines.extend(
        [
            f"| **Task-balanced mean** | {len(payload['records'])} | "
            f"**{100 * oneshot['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * block['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * (block['task_balanced_accuracy'] - oneshot['task_balanced_accuracy']):+.2f} pp** | "
            f"**{overall['sparse_regressions']}** | "
            f"**{overall['sparse_improvements']}** |",
            "",
            "This isolates transaction scheduling: both arms use exact K and the "
            "same resident C1-V64 checkpoint.",
        ]
    )
    if QUEST_ARM in summaries:
        quest = summaries[QUEST_ARM]
        quest_pair = payload["summary"]["paired_quest_vs_block"]
        lines.extend(
            [
                "",
                "## Exact-K QUEST-B512 control",
                "",
                "| Task | Dense block-128 | Exact-K QUEST-B512 | Delta |",
                "|:---|---:|---:|---:|",
            ]
        )
        for task in payload["metadata"]["tasks"]:
            blk = block["tasks"][task]
            sparse = quest["tasks"][task]
            lines.append(
                f"| {task} | {100 * blk['accuracy']:.2f}% | "
                f"{100 * sparse['accuracy']:.2f}% | "
                f"{100 * (sparse['accuracy'] - blk['accuracy']):+.2f} pp |"
            )
        overall = quest_pair["all_samples"]
        lines.extend(
            [
                f"| **Task-balanced mean** | "
                f"**{100 * block['task_balanced_accuracy']:.2f}%** | "
                f"**{100 * quest['task_balanced_accuracy']:.2f}%** | "
                f"**{100 * (quest['task_balanced_accuracy'] - block['task_balanced_accuracy']):+.2f} pp** |",
                "",
                f"Paired samples: {overall['samples']}; regressions: "
                f"{overall['sparse_regressions']}; improvements: "
                f"{overall['sparse_improvements']}.",
            ]
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--c1-factor-dir", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument(
        "--exact-quest-budget",
        type=int,
        help="enable strict exact-K physical-shared QUEST with this token budget",
    )
    parser.add_argument("--page-size", type=int, default=16)
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
        raise RuntimeError("dense block RULER control requires CUDA")
    if min(
        args.sequence_length,
        args.samples_per_task,
        args.block_length,
        args.page_size,
    ) <= 0:
        raise ValueError("RULER sizes must be positive")
    if args.exact_quest_budget is not None:
        if args.exact_quest_budget != 512:
            raise ValueError("the controlled exact-K QUEST experiment requires B512")
        if args.exact_quest_budget % args.page_size:
            raise ValueError("exact QUEST budget must align to complete pages")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    model_path = args.model_path.expanduser().resolve()
    c1_factor_dir = args.c1_factor_dir.expanduser().resolve()
    c1_result_path = c1_factor_dir / "results.json"
    baseline_path = args.baseline_result.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_tasks(args.tasks)
    full_exact_layers = {
        int(item.strip())
        for item in args.full_exact_layers.split(",")
        if item.strip()
    }
    if not full_exact_layers or min(full_exact_layers) < 0:
        raise ValueError("full-exact layers must be nonnegative")
    arms = ARMS if args.exact_quest_budget is None else (*ARMS, QUEST_ARM)
    dataset_manifest, dataset_hash = _load_dataset_manifest(
        data_dir,
        sequence_length=args.sequence_length,
        samples=args.samples_per_task,
        tasks=tasks,
        tokenizer_path=model_path,
    )
    baseline, baseline_records = _load_baseline(
        baseline_path,
        model_path=model_path,
        c1_result_path=c1_result_path,
        dataset_manifest_hash=dataset_hash,
        sequence_length=args.sequence_length,
        samples_per_task=args.samples_per_task,
        tasks=tasks,
    )
    c1_evaluator.activate_model_profile("qwen3_8b")
    c1_result = c1_evaluator._load_results(c1_factor_dir, model_path)
    if c1_result.get("format") != C1_FORMAT:
        raise ValueError("C1 factors are not the uniform Qwen3-8B checkpoint")
    if int(c1_result["fit_config"]["cache_rank_per_head"]) != 64:
        raise ValueError("RULER control requires C1-V64")

    work = _build_work(data_dir, tasks, args.samples_per_task)
    assigned = [row for row in work if row[0] % world_size == rank]
    protocol = {
        "model_config_sha256": _sha256(model_path / "config.json"),
        "c1_results_sha256": _sha256(c1_result_path),
        "baseline_result_sha256": _sha256(baseline_path),
        "dataset_manifest_sha256": dataset_hash,
        "sequence_length": args.sequence_length,
        "tasks": [task.name for task in tasks],
        "samples_per_task": args.samples_per_task,
        "arms": list(arms),
        "block_length": args.block_length,
        "dtype": args.dtype,
        "world_size": world_size,
    }
    if args.exact_quest_budget is not None:
        protocol.update(
            {
                "exact_quest_budget": args.exact_quest_budget,
                "page_size": args.page_size,
                "full_exact_layers": sorted(full_exact_layers),
                "landmark_dtype": args.landmark_dtype,
                "quest_support": "physical_shared",
                "quest_applied_during_prompt_prefill": True,
            }
        )
    fingerprint = _fingerprint(protocol)
    rank_path = output_dir / f"rank_{rank:02d}.json"
    state = _load_rank_state(
        rank_path,
        fingerprint=fingerprint,
        rank=rank,
        world_size=world_size,
    )
    completed = {record["key"] for record in state["records"]}
    pending = [row for row in assigned if f"{row[1].name}:{row[2]}" not in completed]
    print(
        f"[Dense block RULER] rank={rank}/{world_size} assigned={len(assigned)} "
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
        installation = _replace_attention(model, c1_factor_dir, c1_result)
        modules = _attention_modules(model)
        if any(layer >= len(modules) for layer in full_exact_layers):
            raise ValueError("full-exact layer is outside the model")
        state["runtime"] = {
            "cuda_device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "c1_layers": len(installation),
        }
        _atomic_json(rank_path, state)

        for _, task, ordinal, source in pending:
            key = f"{task.name}:{ordinal}"
            input_ids = tokenizer(
                ruler_prompt(source),
                add_special_tokens=True,
                return_tensors="pt",
            )["input_ids"].to(device)
            prompt_tokens = int(input_ids.shape[1])
            if prompt_tokens + task.tokens_to_generate > args.sequence_length:
                raise ValueError(f"RULER prompt {key} exceeds total sequence length")
            references = list(source["outputs"])
            _set_policy(
                modules,
                budget=None,
                full_exact_layers=full_exact_layers,
                page_size=args.page_size,
                landmark_dtype=args.landmark_dtype,
                quest_support="physical_shared",
                force_last_page=False,
            )
            block_result, block_first = _evaluate_block(
                model=model,
                input_ids=input_ids,
                task=task,
                references=references,
                tokenizer=tokenizer,
                block_length=args.block_length,
                device=device,
            )
            oneshot_result = copy.deepcopy(baseline_records[key]["arms"][ONESHOT_ARM])
            oneshot_ids = oneshot_result["generated_token_ids"]
            arm_results = {
                ONESHOT_ARM: oneshot_result,
                BLOCK_ARM: block_result,
            }
            quest_result = None
            quest_first = None
            if args.exact_quest_budget is not None:
                _set_policy(
                    modules,
                    budget=args.exact_quest_budget,
                    full_exact_layers=full_exact_layers,
                    page_size=args.page_size,
                    landmark_dtype=args.landmark_dtype,
                    quest_support="physical_shared",
                    force_last_page=False,
                )
                quest_result, quest_first = _evaluate_block(
                    model=model,
                    input_ids=input_ids,
                    task=task,
                    references=references,
                    tokenizer=tokenizer,
                    block_length=args.block_length,
                    device=device,
                )
                quest_result["runtime_logical"] = _runtime_statistics(
                    modules,
                    full_exact_layers,
                )
                arm_results[QUEST_ARM] = quest_result
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
                "block_first_token_matches_oneshot": bool(
                    oneshot_ids and int(oneshot_ids[0]) == block_first
                ),
                "block_generation_matches_oneshot": (
                    list(oneshot_ids) == block_result["generated_token_ids"]
                ),
                "quest_first_token_matches_dense_block": (
                    None if quest_first is None else quest_first == block_first
                ),
                "arms": arm_results,
            }
            del input_ids
            state["records"].append(row)
            state["elapsed_seconds"] = time.perf_counter() - started
            state["maximum_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            _atomic_json(rank_path, state)
            quest_label = "n/a" if quest_result is None else f"{quest_result['score']:.3f}"
            print(
                f"[Dense block RULER] rank={rank} sample={key} prompt={prompt_tokens} "
                f"oneshot={oneshot_result['score']:.3f} block={block_result['score']:.3f} "
                f"quest={quest_label} "
                f"tokens_match={row['block_generation_matches_oneshot']}",
                flush=True,
            )

    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
    if rank == 0:
        rank_states = [
            _load_rank_state(
                output_dir / f"rank_{other:02d}.json",
                fingerprint=fingerprint,
                rank=other,
                world_size=world_size,
            )
            for other in range(world_size)
        ]
        task_order = {task.name: index for index, task in enumerate(tasks)}
        records = sorted(
            (record for item in rank_states for record in item["records"]),
            key=lambda row: (task_order[row["task"]], row["sample_ordinal"]),
        )
        if len(records) != len(work):
            raise RuntimeError(f"dense block merge has {len(records)} rows, expected {len(work)}")
        summary = {
            "arms": [summarize_arm(records, arm, tasks) for arm in arms],
            "paired": paired_summary(records, ONESHOT_ARM, BLOCK_ARM, tasks),
            "first_token_matches": sum(
                bool(record["block_first_token_matches_oneshot"])
                for record in records
            ),
            "full_generation_matches": sum(
                bool(record["block_generation_matches_oneshot"])
                for record in records
            ),
        }
        if QUEST_ARM in arms:
            summary["paired_quest_vs_block"] = paired_summary(
                records,
                BLOCK_ARM,
                QUEST_ARM,
                tasks,
            )
            summary["quest_first_token_matches_dense_block"] = sum(
                bool(record["quest_first_token_matches_dense_block"])
                for record in records
            )
        payload = {
            "format": FORMAT,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "metadata": {
                **protocol,
                "model_path": str(model_path),
                "c1_factor_result": str(c1_result_path),
                "baseline_result": str(baseline_path),
                "data_dir": str(data_dir),
                "ruler_revision": dataset_manifest["ruler"]["revision"],
                "official_base_prompt": "input + answer_prefix",
                "greedy_decoding": True,
                "block_scheduled_prompt_prefill": True,
                "exact_k_gpu_resident": True,
                "rank_runtimes": [item.get("runtime") for item in rank_states],
            },
            "summary": summary,
            "records": records,
            "elapsed_seconds": time.perf_counter() - started,
            "limitations": [
                "the one-shot arm is reused from the frozen matching baseline",
                "this is a quality control rather than a serving latency benchmark",
            ],
        }
        _atomic_json(output_dir / "result.json", payload)
        _atomic_text(output_dir / "summary.md", _markdown(payload))
        print(f"[Dense block RULER] result={output_dir / 'result.json'}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
