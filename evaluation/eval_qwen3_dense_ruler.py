#!/usr/bin/env python3
"""Official RULER-v1 generation for dense Qwen3-8B-Base."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

import torch
import torch.distributed as dist
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_c1_block_scheduled_ppl import _dtype  # noqa: E402
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
from evaluation.ruler_v1 import (  # noqa: E402
    parse_tasks,
    ruler_prompt,
    sample_score,
    summarize_arm,
)


FORMAT = "basisserve.qwen3_8b.dense.ruler_v1.v1"
RANK_FORMAT = "basisserve.qwen3_8b.dense.ruler_v1.rank.v1"
ARM = "dense_k_dense_v"


def _yarn_override(
    *,
    sequence_length: int,
    original_max_position_embeddings: int,
    yarn_factor: float | None,
) -> tuple[int, dict[str, float | int | str] | None]:
    """Return an explicit Qwen-recommended static-YaRN configuration."""

    if min(sequence_length, original_max_position_embeddings) <= 0:
        raise ValueError("sequence and original context lengths must be positive")
    if yarn_factor is None:
        if sequence_length > original_max_position_embeddings:
            raise ValueError(
                "evaluation exceeds the native context; set --yarn-factor"
            )
        return original_max_position_embeddings, None
    if not math.isfinite(yarn_factor) or yarn_factor <= 1.0:
        raise ValueError("YaRN factor must be finite and greater than one")
    maximum = int(round(original_max_position_embeddings * yarn_factor))
    if sequence_length > maximum:
        raise ValueError("YaRN-scaled maximum is shorter than the evaluation")
    return maximum, {
        "rope_type": "yarn",
        "factor": float(yarn_factor),
        "original_max_position_embeddings": original_max_position_embeddings,
    }


def _qwen3_config(
    model_path: Path,
    *,
    sequence_length: int,
    yarn_factor: float | None,
) -> tuple[Any, dict[str, float | int | str] | None]:
    native = AutoConfig.from_pretrained(model_path, local_files_only=True)
    maximum, rope_scaling = _yarn_override(
        sequence_length=sequence_length,
        original_max_position_embeddings=int(native.max_position_embeddings),
        yarn_factor=yarn_factor,
    )
    if rope_scaling is None:
        return native, None
    rope_parameters = {
        **dict(native.rope_parameters),
        **rope_scaling,
    }
    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        max_position_embeddings=maximum,
        rope_parameters=rope_parameters,
    )
    return config, rope_parameters


def _chunked_exact_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    chunk_size: int,
    empty_cuda_cache_between_chunks: bool = False,
) -> tuple[torch.Tensor, Any]:
    """Run mathematically dense causal prefill with bounded activations."""

    if chunk_size <= 0:
        raise ValueError("prefill chunk size must be positive")
    cache = DynamicCache()
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
    return final_logits, cache


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
    arm = payload["summary"]["arm"]
    metadata = payload["metadata"]
    lines = [
        f"# Qwen3-8B-Base dense RULER-v1 {metadata['sequence_length'] // 1024}K",
        "",
        "Unmodified dense K and dense V with greedy decoding.",
        "",
        "| Task | Samples | Dense accuracy |",
        "|:---|---:|---:|",
    ]
    for task in payload["metadata"]["tasks"]:
        row = arm["tasks"][task]
        lines.append(
            f"| {task} | {row['samples']} | {100 * row['accuracy']:.2f}% |"
        )
    lines.extend(
        [
            f"| **Task-balanced mean** | {len(payload['records'])} | "
            f"**{100 * arm['task_balanced_accuracy']:.2f}%** |",
            "",
            "Protocol: official RULER-v1 base completion prompts and substring "
            f"scorers; {metadata['samples_per_task']} examples per task; greedy "
            "decoding; BF16 dense SDPA.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--samples-per-task", type=int, default=100)
    parser.add_argument("--sample-spec", default="")
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--empty-cache-between-prefill-chunks",
        action="store_true",
        help="release allocator fragments between long-context prefill chunks",
    )
    parser.add_argument(
        "--yarn-factor",
        type=float,
        help="static YaRN factor; required when sequence length exceeds native context",
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
        raise RuntimeError("dense RULER evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="gloo")

    model_path = args.model_path.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_tasks(args.tasks)
    if min(
        args.sequence_length, args.samples_per_task, args.prefill_chunk_size
    ) <= 0:
        raise ValueError("evaluation length, samples, and prefill chunk must be positive")
    model_config, rope_scaling = _qwen3_config(
        model_path,
        sequence_length=args.sequence_length,
        yarn_factor=args.yarn_factor,
    )
    dataset_manifest, dataset_manifest_hash = _load_dataset_manifest(
        data_dir,
        sequence_length=args.sequence_length,
        samples=args.samples_per_task,
        tasks=tasks,
        tokenizer_path=model_path,
    )
    work = _build_work(data_dir, tasks, args.samples_per_task)
    if args.sample_spec.strip():
        requested = {
            (task.strip(), int(ordinal))
            for item in args.sample_spec.split(",")
            for task, ordinal in (item.rsplit(":", 1),)
        }
        work = [
            row
            for row in work
            if (str(row[1].name), int(row[2])) in requested
        ]
        work = [(index, *row[1:]) for index, row in enumerate(work)]
    assigned = [row for row in work if row[0] % world_size == rank]
    protocol_identity = {
        "model_config_sha256": _sha256(model_path / "config.json"),
        "dataset_manifest_sha256": dataset_manifest_hash,
        "sequence_length": args.sequence_length,
        "tasks": [task.name for task in tasks],
        "samples_per_task": args.samples_per_task,
        "sample_spec": args.sample_spec,
        "arm": ARM,
        "dtype": args.dtype,
        "rope_scaling": rope_scaling,
        "effective_max_position_embeddings": int(
            model_config.max_position_embeddings
        ),
        "prefill_chunk_size": args.prefill_chunk_size,
        "empty_cache_between_prefill_chunks": (
            args.empty_cache_between_prefill_chunks
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
        f"[Dense RULER] rank={rank}/{world_size} assigned={len(assigned)} "
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
            config=model_config,
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(device)
        model.eval()
        rank_state["runtime"] = {
            "cuda_device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "attention_backend": "sdpa",
            "rope_scaling": rope_scaling,
            "prefill_chunk_size": args.prefill_chunk_size,
            "empty_cache_between_prefill_chunks": (
                args.empty_cache_between_prefill_chunks
            ),
        }
        _atomic_json(rank_path, rank_state)

        for _, task, ordinal, source in pending:
            key = f"{task.name}:{ordinal}"
            prompt = ruler_prompt(source)
            input_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                return_tensors="pt",
            )["input_ids"].to(device)
            prompt_tokens = int(input_ids.shape[1])
            if prompt_tokens + task.tokens_to_generate > args.sequence_length:
                raise ValueError(
                    f"RULER prompt {key} exceeds total length: {prompt_tokens} + "
                    f"{task.tokens_to_generate} > {args.sequence_length}"
                )
            torch.cuda.synchronize(device)
            prefill_started = time.perf_counter()
            final_logits, cache = _chunked_exact_prefill(
                model,
                input_ids,
                chunk_size=args.prefill_chunk_size,
                empty_cuda_cache_between_chunks=(
                    args.empty_cache_between_prefill_chunks
                ),
            )
            torch.cuda.synchronize(device)
            prefill_seconds = time.perf_counter() - prefill_started
            first_token = int(final_logits[0].argmax().item())
            del final_logits, input_ids
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
            references = list(source["outputs"])
            result = {
                "prediction": prediction,
                "generated_token_ids": generated_ids,
                "generated_tokens": len(generated_ids),
                "stopped_on_eos": generated_ids[-1]
                in _eos_ids(tokenizer, model),
                "score": sample_score(prediction, references, task.match_type),
                "elapsed_seconds": decode_seconds,
            }
            del cache
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
                    "arms": {ARM: result},
                }
            )
            rank_state["elapsed_seconds"] = time.perf_counter() - started
            rank_state["maximum_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            _atomic_json(rank_path, rank_state)
            print(
                f"[Dense RULER] rank={rank} sample={key} prompt={prompt_tokens} "
                f"score={result['score']:.3f} prefill={prefill_seconds:.2f}s "
                f"decode={decode_seconds:.2f}s",
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
                f"dense RULER merge has {len(records)} rows, expected {len(work)}"
            )
        payload = {
            "format": FORMAT,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "metadata": {
                **protocol_identity,
                "model_path": str(model_path),
                "data_dir": str(data_dir),
                "ruler_revision": dataset_manifest["ruler"]["revision"],
                "official_base_prompt": "input + answer_prefix",
                "greedy_decoding": True,
                "rank_runtimes": [payload.get("runtime") for payload in rank_payloads],
            },
            "summary": {"arm": summarize_arm(records, ARM, tasks)},
            "records": records,
            "elapsed_seconds": time.perf_counter() - started,
            "limitations": [
                "Qwen3-8B-Base is not the post-trained Qwen3-8B in the public RULER table",
                "prompt prefill is chunk-scheduled but remains mathematically dense causal attention",
                *(
                    [
                        "static YaRN changes RoPE at every position, including positions inside the native context"
                    ]
                    if rope_scaling is not None
                    else []
                ),
            ],
        }
        _atomic_json(output_dir / "result.json", payload)
        _atomic_text(output_dir / "summary.md", _markdown(payload))
        print(f"[Dense RULER] result={output_dir / 'result.json'}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
