#!/usr/bin/env python3
"""Evaluate Qwen3-8B C1-V80 sparse-Key RULER with native vLLM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.vllm import (  # noqa: E402
    QWEN3_8B_SPARSE_C1_MODEL_ARCHITECTURE,
    register as register_basisserve_vllm,
)
from basisserve.vllm.qwen3_8b_sparse_c1 import (  # noqa: E402
    router_checkpoint_fingerprint,
)
from evaluation.ruler_v1 import (  # noqa: E402
    RulerTask,
    load_task_records,
    parse_tasks,
    ruler_prompt,
    sample_score,
    summarize_arm,
)


FORMAT = "basisserve.qwen3_8b.sparse_c1.ruler_vllm.v1"
SHARD_FORMAT = "basisserve.qwen3_8b.sparse_c1.ruler_vllm.shard.v1"
PINNED_VLLM_VERSION = "0.18.1.dev0+gbcf2be961.d20260828.cu128"
DEFAULT_TASKS = (
    "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,"
    "niah_multikey_2,niah_multiquery,niah_multivalue,vt,fwe,qa_1,qa_2"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.get("status") == "complete"
    assert int(payload["protocol"]["sequence_length"]) == sequence_length
    assert int(payload["protocol"]["samples_per_task"]) >= samples
    assert payload["tokenizer_config_sha256"] == _sha256(
        tokenizer_path / "tokenizer_config.json"
    )
    for task in tasks:
        task_path = data_dir / task.name / "validation.jsonl"
        assert _sha256(task_path) == payload["artifacts"][task.name]["sha256"]
    return payload, _sha256(path)


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _arm_name(mode: str) -> str:
    return (
        "c1_v80_base16_r8_page32_group_b2048"
        if mode == "conditional"
        else "c1_v80_loki_r32_token_b2048"
    )


def _rank_state(
    path: Path,
    *,
    fingerprint: str,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return {
            "format": SHARD_FORMAT,
            "status": "running",
            "fingerprint": fingerprint,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "records": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = (SHARD_FORMAT, fingerprint, shard_index, num_shards)
    observed = (
        payload.get("format"),
        payload.get("fingerprint"),
        payload.get("shard_index"),
        payload.get("num_shards"),
    )
    if not _check(observed == expected, f"incompatible resume file: {path}"):
        return None
    keys = [str(record["key"]) for record in payload["records"]]
    if not _check(len(keys) == len(set(keys)), f"duplicate rows in {path}"):
        return None
    return payload


def _merge_statistics(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    layer_count = len(shards[0]["runtime"]["routing_statistics"]["layers"])
    per_layer = []
    for layer in range(layer_count):
        per_layer.append(
            {
                key: sum(
                    int(shard["runtime"]["routing_statistics"]["layers"][layer][key])
                    for shard in shards
                )
                for key in ("queries", "physical_tokens", "logical_tokens")
            }
        )
    totals = {
        key: sum(record[key] for record in per_layer)
        for key in ("queries", "physical_tokens", "logical_tokens")
    }
    totals["physical_tokens_per_layer_query"] = (
        totals["physical_tokens"] / totals["queries"] if totals["queries"] else 0.0
    )
    totals["logical_tokens_per_layer_query"] = (
        totals["logical_tokens"] / totals["queries"] if totals["queries"] else 0.0
    )
    totals["physical_exact_k_mib_per_layer_query"] = (
        totals["physical_tokens_per_layer_query"] * 128 * 2 / (1 << 20)
    )
    return {"totals": totals, "layers": per_layer}


def _sum_routing_statistics(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    if previous is None:
        return dict(current)
    per_layer = [
        {
            key: int(left[key]) + int(right[key])
            for key in ("queries", "physical_tokens", "logical_tokens")
        }
        for left, right in zip(previous["layers"], current["layers"], strict=True)
    ]
    totals = {
        key: sum(record[key] for record in per_layer)
        for key in ("queries", "physical_tokens", "logical_tokens")
    }
    totals["physical_tokens_per_layer_query"] = (
        totals["physical_tokens"] / totals["queries"] if totals["queries"] else 0.0
    )
    totals["logical_tokens_per_layer_query"] = (
        totals["logical_tokens"] / totals["queries"] if totals["queries"] else 0.0
    )
    return {"totals": totals, "layers": per_layer}


def _markdown(payload: Mapping[str, Any]) -> str:
    metadata = payload["metadata"]
    summary = payload["summary"]["arm"]
    routing = payload["summary"]["routing_statistics"]["totals"]
    lines = [
        f"# Qwen3-8B C1-V80 {metadata['router_mode']} RULER-v1 32K",
        "",
        "| Task | Samples | Accuracy |",
        "|:---|---:|---:|",
    ]
    for task in metadata["tasks"]:
        row = summary["tasks"][task]
        lines.append(f"| {task} | {row['samples']} | {100.0 * row['accuracy']:.2f}% |")
    lines.extend(
        [
            f"| **Task-balanced mean** | {len(payload['records'])} | "
            f"**{100.0 * summary['task_balanced_accuracy']:.2f}%** |",
            "",
            f"- Mean physical exact-K tokens per layer/query: "
            f"`{routing['physical_tokens_per_layer_query']:.2f}`.",
            f"- Mean physical exact-K traffic per layer/query: "
            f"`{routing['physical_exact_k_mib_per_layer_query']:.3f} MiB` "
            "at BF16 K128.",
            f"- Mean nominal per-Q-head token visits per layer/query: "
            f"`{routing['logical_tokens_per_layer_query']:.2f}`.",
            "",
            "Protocol: BF16 Qwen3-8B-Base, uniform C1-V80 calibrated on "
            "32 C4 documents x 32768 positions, dense exact-K FlashAttention "
            "prefill, then native vLLM paged sparse exact-K decode. The first "
            "generated token comes from dense prefill. Exact K remains "
            "GPU-resident in this quality runtime.",
        ]
    )
    return "\n".join(lines) + "\n"


def _merge_if_complete(
    output_dir: Path,
    *,
    fingerprint: str,
    protocol: Mapping[str, Any],
    tasks: Sequence[RulerTask],
    arm: str,
    command: str,
) -> bool:
    paths = [
        output_dir / f"shard_{index:02d}.json"
        for index in range(int(protocol["num_shards"]))
    ]
    if not all(path.is_file() for path in paths):
        return False
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    compatible = all(
        shard.get("format") == SHARD_FORMAT
        and shard.get("fingerprint") == fingerprint
        and int(shard.get("shard_index", -1)) == index
        for index, shard in enumerate(shards)
    )
    if not _check(compatible, "one or more shards are incompatible"):
        return False
    if not all(shard.get("status") == "complete" for shard in shards):
        return False
    records = sorted(
        [record for shard in shards for record in shard["records"]],
        key=lambda record: int(record["global_index"]),
    )
    expected_rows = len(tasks) * int(protocol["samples_per_task"])
    if not _check(len(records) == expected_rows, "merged row count is incomplete"):
        return False
    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "fingerprint": fingerprint,
        "metadata": dict(protocol),
        "summary": {
            "arm": summarize_arm(records, arm, tasks),
            "routing_statistics": _merge_statistics(shards),
            "elapsed_seconds_sum": sum(
                float(shard["runtime"]["elapsed_seconds"]) for shard in shards
            ),
            "peak_cuda_allocated_bytes_max": max(
                int(shard["runtime"]["peak_cuda_allocated_bytes"]) for shard in shards
            ),
        },
        "records": records,
        "shards": [{"file": path.name, "sha256": _sha256(path)} for path in paths],
    }
    _atomic_json(output_dir / "result.json", result)
    _atomic_text(output_dir / "summary.md", _markdown(result))
    print(f"[Result] {output_dir / 'result.json'}", flush=True)
    return True


def _protocol(
    args: argparse.Namespace,
    *,
    model: Path,
    data_dir: Path,
    value_dir: Path,
    router_dir: Path,
    tasks: Sequence[RulerTask],
    dataset_sha256: str,
    value_sha256: str,
    router_fingerprint: str,
) -> dict[str, Any]:
    return {
        "model": str(model),
        "model_config_sha256": _sha256(model / "config.json"),
        "data_dir": str(data_dir),
        "dataset_manifest_sha256": dataset_sha256,
        "value_factor_dir": str(value_dir),
        "value_result_sha256": value_sha256,
        "router_factor_dir": str(router_dir),
        "router_fingerprint": router_fingerprint,
        "router_mode": args.router_mode,
        "arm": _arm_name(args.router_mode),
        "tasks": [task.name for task in tasks],
        "samples_per_task": args.samples_per_task,
        "sequence_length": args.sequence_length,
        "page_size": 32 if args.router_mode == "conditional" else 1,
        "physical_page_size": 32,
        "nominal_token_budget": 2048,
        "pinned_prefix_pages": 1 if args.router_mode == "conditional" else 0,
        "selection": (
            "normalized per-query-head Page32 LSE, GQA group-max, fixed "
            "B2048 physical tokens with Page0 pinned"
            if args.router_mode == "conditional"
            else "Loki R32 per-query-head token Top-2048 followed by GQA union"
        ),
        "value_calibration": "32 independent C4 documents x 32768 positions",
        "router_calibration": "64 independent C4 documents x 32768 tokens",
        "prefill": "dense exact-K FlashAttention over C1-V80",
        "decode": "native vLLM paged sparse exact-K attention over C1-V80",
        "exact_k_residency": "GPU",
        "cache_layout": "exact K128 + V80 + routing R32 + 16 zero ABI padding",
        "semantic_cache_width": 240,
        "physical_cache_width": 256,
        "dtype": "bfloat16",
        "vllm_version": PINNED_VLLM_VERSION,
        "max_num_seqs": 1,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "num_shards": args.num_shards,
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> int:
    model = args.model.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    value_dir = args.value_factor_dir.expanduser().resolve()
    router_dir = args.router_factor_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_tasks(args.tasks)
    valid = all(
        (
            _check(args.sequence_length == 32768, "fair protocol requires 32768"),
            _check(0 < args.samples_per_task <= 8, "samples/task must lie in [1, 8]"),
            _check(0 <= args.shard_index < args.num_shards, "invalid shard index"),
            _check(args.max_num_batched_tokens > 0, "invalid batched-token limit"),
            _check(0.0 < args.gpu_memory_utilization < 1.0, "invalid GPU utilization"),
            _check((value_dir / "results.json").is_file(), "missing V80 result"),
            _check(router_dir.is_dir(), "missing router directory"),
            _check(
                importlib.metadata.version("vllm") == PINNED_VLLM_VERSION,
                f"vLLM must be {PINNED_VLLM_VERSION}",
            ),
        )
    )
    if not valid:
        return 2
    _, dataset_sha256 = _load_dataset_manifest(
        data_dir,
        sequence_length=args.sequence_length,
        samples=args.samples_per_task,
        tasks=tasks,
        tokenizer_path=model,
    )
    value_sha256 = _sha256(value_dir / "results.json")
    router_fingerprint = router_checkpoint_fingerprint(
        router_dir,
        args.router_mode,
    )
    protocol = _protocol(
        args,
        model=model,
        data_dir=data_dir,
        value_dir=value_dir,
        router_dir=router_dir,
        tasks=tasks,
        dataset_sha256=dataset_sha256,
        value_sha256=value_sha256,
        router_fingerprint=router_fingerprint,
    )
    fingerprint = _fingerprint(protocol)
    command = shlex.join(sys.argv)
    if args.merge_only:
        return (
            0
            if _merge_if_complete(
                output_dir,
                fingerprint=fingerprint,
                protocol=protocol,
                tasks=tasks,
                arm=protocol["arm"],
                command=command,
            )
            else 2
        )

    valid = all(
        (
            _check(torch.cuda.is_available(), "CUDA is required"),
            _check(torch.cuda.device_count() == 1, "expose exactly one GPU per shard"),
        )
    )
    if not valid:
        return 2
    torch.set_num_threads(args.torch_num_threads)
    rank_path = output_dir / f"shard_{args.shard_index:02d}.json"
    state = _rank_state(
        rank_path,
        fingerprint=fingerprint,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if state is None:
        return 2
    if state.get("status") == "complete":
        print(f"[Resume] {rank_path} is complete", flush=True)
        _merge_if_complete(
            output_dir,
            fingerprint=fingerprint,
            protocol=protocol,
            tasks=tasks,
            arm=protocol["arm"],
            command=command,
        )
        return 0

    work = _build_work(data_dir, tasks, args.samples_per_task)
    assigned = [
        row for row in work if int(row[0]) % args.num_shards == args.shard_index
    ]
    completed = {str(record["key"]) for record in state["records"]}
    pending = [row for row in assigned if f"{row[1].name}:{row[2]}" not in completed]
    print(
        f"[RULER vLLM] mode={args.router_mode} "
        f"shard={args.shard_index}/{args.num_shards} pending={len(pending)}",
        flush=True,
    )

    register_basisserve_vllm()
    from vllm import LLM, SamplingParams, TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=True,
        use_fast=True,
    )
    hf_overrides = {
        "architectures": [QWEN3_8B_SPARSE_C1_MODEL_ARCHITECTURE],
        "basisserve_value_factor_dir": str(value_dir),
        "basisserve_value_result_sha256": value_sha256,
        "basisserve_router_mode": args.router_mode,
        "basisserve_router_factor_dir": str(router_dir),
        "basisserve_router_fingerprint": router_fingerprint,
        "basisserve_token_budget": 2048,
        "basisserve_pinned_prefix_pages": (
            1 if args.router_mode == "conditional" else 0
        ),
    }
    started = time.perf_counter()
    previous_runtime = state.get("runtime", {})
    previous_statistics = previous_runtime.get("routing_statistics")
    previous_elapsed = float(previous_runtime.get("elapsed_seconds", 0.0))
    engine = LLM(
        model=str(model),
        tensor_parallel_size=1,
        data_parallel_size=1,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        block_size=32,
        max_model_len=args.sequence_length,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        trust_remote_code=False,
        hf_overrides=hf_overrides,
        disable_log_stats=False,
        worker_extension_cls="basisserve.vllm.worker_extension.BasisServeWorkerExtension",
    )

    for global_index, task, ordinal, source in pending:
        key = f"{task.name}:{ordinal}"
        token_ids = tokenizer.encode(
            ruler_prompt(source),
            add_special_tokens=True,
        )
        if not _check(
            len(token_ids) + task.tokens_to_generate <= args.sequence_length,
            f"sample {key} exceeds total context length",
        ):
            return 2
        sample_started = time.perf_counter()
        outputs = engine.generate(
            [TokensPrompt(prompt_token_ids=token_ids)],
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=task.tokens_to_generate,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            ),
            use_tqdm=False,
        )
        generated = list(map(int, outputs[0].outputs[0].token_ids))
        prediction = tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        row = {
            "global_index": int(global_index),
            "key": key,
            "task": task.name,
            "ordinal": int(ordinal),
            "prompt_tokens": len(token_ids),
            "references": list(source["outputs"]),
            "arms": {
                protocol["arm"]: {
                    "prediction": prediction,
                    "generated_token_ids": generated,
                    "generated_tokens": len(generated),
                    "score": sample_score(
                        prediction,
                        source["outputs"],
                        task.match_type,
                    ),
                    "elapsed_seconds": time.perf_counter() - sample_started,
                }
            },
        }
        state["records"].append(row)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        runtime_snapshot = engine.collective_rpc("basisserve_runtime_statistics")[0]
        state["runtime"] = {
            "routing_statistics": _sum_routing_statistics(
                previous_statistics,
                runtime_snapshot["routing_statistics"],
            ),
            "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
            "peak_cuda_allocated_bytes": runtime_snapshot["peak_cuda_allocated_bytes"],
            "cuda_device": runtime_snapshot["cuda_device"],
        }
        _atomic_json(rank_path, state)
        print(
            f"[{args.shard_index}] {key} prompt={len(token_ids)} "
            f"generated={len(generated)} score="
            f"{row['arms'][protocol['arm']]['score']:.3f}",
            flush=True,
        )

    runtime_snapshot = engine.collective_rpc("basisserve_runtime_statistics")[0]
    statistics = _sum_routing_statistics(
        previous_statistics,
        runtime_snapshot["routing_statistics"],
    )
    state.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol,
            "runtime": {
                "routing_statistics": statistics,
                "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                "cuda_device": runtime_snapshot["cuda_device"],
                "peak_cuda_allocated_bytes": runtime_snapshot[
                    "peak_cuda_allocated_bytes"
                ],
                "python": sys.version,
                "python_executable": sys.executable,
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "torch": torch.__version__,
                "vllm": importlib.metadata.version("vllm"),
            },
        }
    )
    _atomic_json(rank_path, state)
    _merge_if_complete(
        output_dir,
        fingerprint=fingerprint,
        protocol=protocol,
        tasks=tasks,
        arm=protocol["arm"],
        command=command,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--value-factor-dir", type=Path, required=True)
    parser.add_argument("--router-factor-dir", type=Path, required=True)
    parser.add_argument("--router-mode", choices=("conditional", "loki"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--merge-only", action="store_true")
    return parser


if __name__ == "__main__":
    sys.exit(evaluate(_parser().parse_args()))
