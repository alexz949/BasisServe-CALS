#!/usr/bin/env python3
"""Paired RULER-v1 generation for dense K and physical-shared QUEST K."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
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

from basisserve.checkpoint.gqa_vo_qwen3 import (  # noqa: E402
    GQATiedVOQwen3Attention,
    install_qwen3_gqa_vo_als_export,
)
from evaluation.eval_qwen3_c1_k_reverse_shadow_ppl import (  # noqa: E402
    _attention_modules,
    _runtime_statistics,
    _set_policy,
)
from evaluation.eval_c1_block_scheduled_ppl import _dtype  # noqa: E402
from evaluation.ruler_v1 import (  # noqa: E402
    RulerTask,
    load_task_records,
    paired_summary,
    parse_tasks,
    ruler_prompt,
    sample_score,
    summarize_arm,
)


FORMAT = "basisserve.qwen3_8b.c1_quest.ruler_v1.v1"
RANK_FORMAT = "basisserve.qwen3_8b.c1_quest.ruler_v1.rank.v1"
DENSE_ARM = "dense_k_c1_v64"
SPARSE_ARM = "physical_shared_1024"
EXPECTED_ARMS = (DENSE_ARM, SPARSE_ARM)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _parse_layers(raw: str) -> tuple[int, ...]:
    layers = tuple(int(item) for item in raw.split(",") if item)
    if any(layer < 0 for layer in layers) or len(set(layers)) != len(layers):
        raise ValueError("full-exact layers must be unique nonnegative integers")
    return layers


def _parse_arms(raw: str) -> tuple[str, ...]:
    arms = tuple(item.strip() for item in raw.split(",") if item.strip())
    if arms != EXPECTED_ARMS:
        raise ValueError(
            "this paired protocol requires --arms " + ",".join(EXPECTED_ARMS)
        )
    return arms


def _eos_ids(tokenizer: Any, model: torch.nn.Module) -> set[int]:
    values = []
    for value in (tokenizer.eos_token_id, model.generation_config.eos_token_id):
        if value is None:
            continue
        values.extend(value if isinstance(value, (list, tuple)) else [value])
    return {int(value) for value in values}


def _greedy_continuation(
    model: torch.nn.Module,
    cache: Any,
    first_token: int,
    *,
    maximum_tokens: int,
    eos_ids: set[int],
    device: torch.device,
) -> tuple[list[int], Any]:
    generated = [first_token]
    if first_token in eos_ids:
        return generated, cache
    while len(generated) < maximum_tokens:
        token = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        output = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        next_token = int(output.logits[0, -1].argmax().item())
        generated.append(next_token)
        if next_token in eos_ids:
            break
    return generated, cache


def _evaluate_arm(
    *,
    model: torch.nn.Module,
    modules: list[GQATiedVOQwen3Attention],
    cache: Any,
    first_token: int,
    task: RulerTask,
    references: Sequence[str],
    tokenizer: Any,
    arm: str,
    page_size: int,
    full_exact_layers: set[int],
    force_last_page: bool,
    landmark_dtype: str,
    device: torch.device,
) -> dict[str, Any]:
    budget = None if arm == DENSE_ARM else 1024
    _set_policy(
        modules,
        budget=budget,
        full_exact_layers=full_exact_layers,
        page_size=page_size,
        landmark_dtype=landmark_dtype,
        quest_support="physical_shared",
        force_last_page=force_last_page,
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
    if arm == SPARSE_ARM:
        result["runtime_logical"] = _runtime_statistics(
            modules,
            full_exact_layers,
        )
    del cache
    return result


def _build_work(
    data_dir: Path,
    tasks: Sequence[RulerTask],
    samples: int,
) -> list[tuple[int, RulerTask, int, dict[str, Any]]]:
    work = []
    global_index = 0
    for task in tasks:
        for ordinal, record in enumerate(load_task_records(data_dir, task, samples)):
            work.append((global_index, task, ordinal, record))
            global_index += 1
    return work


def _load_dataset_manifest(
    data_dir: Path,
    *,
    sequence_length: int,
    samples: int,
    tasks: Sequence[RulerTask],
    tokenizer_path: Path,
) -> tuple[dict[str, Any], str]:
    path = data_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("RULER dataset manifest is incomplete")
    protocol = payload["protocol"]
    if int(protocol["sequence_length"]) != sequence_length:
        raise ValueError("RULER dataset sequence length differs from evaluation")
    if int(protocol["samples_per_task"]) < samples:
        raise ValueError("RULER dataset has fewer samples than requested")
    if payload["tokenizer_config_sha256"] != _sha256(
        tokenizer_path / "tokenizer_config.json"
    ):
        raise ValueError("RULER data was generated with another tokenizer")
    for task in tasks:
        artifact = payload["artifacts"][task.name]
        task_path = data_dir / task.name / "validation.jsonl"
        if _sha256(task_path) != artifact["sha256"]:
            raise ValueError(f"RULER data hash mismatch for {task.name}")
    return payload, _sha256(path)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def _markdown(payload: Mapping[str, Any]) -> str:
    arms = {row["arm"]: row for row in payload["summary"]["arms"]}
    dense = arms[DENSE_ARM]
    sparse = arms[SPARSE_ARM]
    paired = payload["summary"]["paired"]
    lines = [
        "# Qwen3-8B-Base C1-V64 + QUEST RULER-v1 4K",
        "",
        "Dense-K and physical-shared QUEST use the same C1-V64 ALS5 Value "
        "checkpoint and identical greedy prompts.",
        "",
        "| Task | Samples | Dense-K + C1-V64 | Physical-shared-1024 | Delta | "
        "Regressions | Improvements |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in payload["metadata"]["tasks"]:
        dense_row = dense["tasks"][task]
        sparse_row = sparse["tasks"][task]
        pair = paired["tasks"][task]
        lines.append(
            f"| {task} | {dense_row['samples']} | "
            f"{100 * dense_row['accuracy']:.2f}% | "
            f"{100 * sparse_row['accuracy']:.2f}% | "
            f"{100 * (sparse_row['accuracy'] - dense_row['accuracy']):+.2f} pp | "
            f"{pair['sparse_regressions']} | {pair['sparse_improvements']} |"
        )
    overall_pair = paired["all_samples"]
    lines.extend(
        [
            f"| **Task-balanced mean** | {len(payload['records'])} | "
            f"**{100 * dense['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * sparse['task_balanced_accuracy']:.2f}%** | "
            f"**{100 * (sparse['task_balanced_accuracy'] - dense['task_balanced_accuracy']):+.2f} pp** | "
            f"**{overall_pair['sparse_regressions']}** | "
            f"**{overall_pair['sparse_improvements']}** |",
            "",
            "Protocol: official RULER-v1 base completion prompts and substring "
            "scorers; 100 examples per task; greedy decoding; dense prompt "
            "prefill; QUEST enabled for decode after the first generated token; "
            "page size 16; fixed 1024-token budget; layers 0 and 1 exact.",
            "",
            "This is a quality oracle. Exact K remains GPU resident and QUEST "
            "metadata is rebuilt in Python, so elapsed time is not an offload "
            "or serving-throughput result.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "results/datasets/qwen3_8b_base_ruler_v1_4k",
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--samples-per-task", type=int, default=100)
    parser.add_argument(
        "--arms",
        default=",".join(EXPECTED_ARMS),
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--full-exact-layers", default="0,1")
    parser.add_argument("--force-last-page", action="store_true")
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
        raise RuntimeError("RULER C1/QUEST evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    model_path = args.model_path.expanduser().resolve()
    c1_export = args.c1_export.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_tasks(args.tasks)
    _parse_arms(args.arms)
    full_exact_layers = set(_parse_layers(args.full_exact_layers))
    if min(args.sequence_length, args.samples_per_task, args.page_size) <= 0:
        raise ValueError("evaluation lengths and samples must be positive")
    if 1024 % args.page_size:
        raise ValueError("the fixed 1024-token QUEST budget must align to pages")

    dataset_manifest, dataset_manifest_hash = _load_dataset_manifest(
        data_dir,
        sequence_length=args.sequence_length,
        samples=args.samples_per_task,
        tasks=tasks,
        tokenizer_path=model_path,
    )
    work = _build_work(data_dir, tasks, args.samples_per_task)
    assigned = [row for row in work if row[0] % world_size == rank]
    protocol_identity = {
        "model_config_sha256": _sha256(model_path / "config.json"),
        "c1_results_sha256": _sha256(c1_export / "results.json"),
        "dataset_manifest_sha256": dataset_manifest_hash,
        "sequence_length": args.sequence_length,
        "tasks": [task.name for task in tasks],
        "samples_per_task": args.samples_per_task,
        "arms": list(EXPECTED_ARMS),
        "page_size": args.page_size,
        "exact_token_budget": 1024,
        "full_exact_layers": sorted(full_exact_layers),
        "force_last_page": args.force_last_page,
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
        f"[RULER] rank={rank}/{world_size} assigned={len(assigned)} "
        f"completed={len(completed)} pending={len(pending)} device={device}",
        flush=True,
    )

    if pending:
        dtype = _dtype(args.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation="eager",
            local_files_only=True,
        ).to(device)
        model.eval()
        replacements = install_qwen3_gqa_vo_als_export(model, c1_export)
        modules = _attention_modules(model)
        if len(replacements) != len(modules):
            raise RuntimeError("C1 replacement count differs from attention layers")
        if any(layer >= len(modules) for layer in full_exact_layers):
            raise ValueError("full-exact layer is outside the model")
        rank_state["runtime"] = {
            "cuda_device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        }
        _atomic_json(rank_path, rank_state)

        for _, task, ordinal, source in pending:
            key = f"{task.name}:{ordinal}"
            prompt = ruler_prompt(source)
            tokenized = tokenizer(
                prompt,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(device)
            prompt_tokens = int(input_ids.shape[1])
            if prompt_tokens + task.tokens_to_generate > args.sequence_length:
                raise ValueError(
                    f"RULER prompt {key} exceeds total length: {prompt_tokens} + "
                    f"{task.tokens_to_generate} > {args.sequence_length}"
                )
            _set_policy(
                modules,
                budget=None,
                full_exact_layers=full_exact_layers,
                page_size=args.page_size,
                landmark_dtype=args.landmark_dtype,
                quest_support="physical_shared",
                force_last_page=False,
            )
            torch.cuda.synchronize(device)
            prefill_started = time.perf_counter()
            prefill = model(
                input_ids=input_ids,
                use_cache=True,
                logits_to_keep=1,
            )
            torch.cuda.synchronize(device)
            prefill_seconds = time.perf_counter() - prefill_started
            first_token = int(prefill.logits[0, -1].argmax().item())
            dense_cache = copy.deepcopy(prefill.past_key_values)
            sparse_cache = prefill.past_key_values
            del prefill, input_ids
            references = list(source["outputs"])
            arm_results = {
                DENSE_ARM: _evaluate_arm(
                    model=model,
                    modules=modules,
                    cache=dense_cache,
                    first_token=first_token,
                    task=task,
                    references=references,
                    tokenizer=tokenizer,
                    arm=DENSE_ARM,
                    page_size=args.page_size,
                    full_exact_layers=full_exact_layers,
                    force_last_page=args.force_last_page,
                    landmark_dtype=args.landmark_dtype,
                    device=device,
                ),
                SPARSE_ARM: _evaluate_arm(
                    model=model,
                    modules=modules,
                    cache=sparse_cache,
                    first_token=first_token,
                    task=task,
                    references=references,
                    tokenizer=tokenizer,
                    arm=SPARSE_ARM,
                    page_size=args.page_size,
                    full_exact_layers=full_exact_layers,
                    force_last_page=args.force_last_page,
                    landmark_dtype=args.landmark_dtype,
                    device=device,
                ),
            }
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
                "prefill_seconds": prefill_seconds,
                "arms": arm_results,
            }
            rank_state["records"].append(row)
            rank_state["elapsed_seconds"] = time.perf_counter() - started
            rank_state["maximum_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            _atomic_json(rank_path, rank_state)
            print(
                f"[RULER] rank={rank} sample={key} prompt={prompt_tokens} "
                f"dense={arm_results[DENSE_ARM]['score']:.3f} "
                f"quest={arm_results[SPARSE_ARM]['score']:.3f} "
                f"prefill={prefill_seconds:.2f}s "
                f"dense_decode={arm_results[DENSE_ARM]['elapsed_seconds']:.2f}s "
                f"quest_decode={arm_results[SPARSE_ARM]['elapsed_seconds']:.2f}s",
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
        records = sorted(
            (record for payload in rank_payloads for record in payload["records"]),
            key=lambda row: (
                [task.name for task in tasks].index(row["task"]),
                row["sample_ordinal"],
            ),
        )
        if len(records) != len(work):
            raise RuntimeError(
                f"RULER merge has {len(records)} rows, expected {len(work)}"
            )
        payload = {
            "format": FORMAT,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "metadata": {
                **protocol_identity,
                "model_path": str(model_path),
                "c1_export": str(c1_export),
                "data_dir": str(data_dir),
                "ruler_revision": dataset_manifest["ruler"]["revision"],
                "official_base_prompt": "input + answer_prefix",
                "greedy_decoding": True,
                "dense_prompt_prefill": True,
                "first_generated_token_from_dense_prefill": True,
                "quest_enabled_during_decode": True,
                "exact_k_gpu_resident": True,
                "rank_runtimes": [payload.get("runtime") for payload in rank_payloads],
            },
            "summary": {
                "arms": [
                    summarize_arm(records, arm, tasks) for arm in EXPECTED_ARMS
                ],
                "paired": paired_summary(
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
                "exact K remains GPU resident in this quality oracle",
                "QUEST metadata is rebuilt in Python for every decode query",
                "elapsed time is not a serving or CPU-offload latency result",
            ],
        }
        _atomic_json(output_dir / "result.json", payload)
        _atomic_text(output_dir / "summary.md", _markdown(payload))
        print(f"[RULER] result={output_dir / 'result.json'}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
