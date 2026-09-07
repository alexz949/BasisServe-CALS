#!/usr/bin/env python3
"""Evaluate dense-K Qwen3-8B V-side controls with native vLLM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    QWEN3_8B_FOLDED_C1_MODEL_ARCHITECTURE,
    QWEN3_8B_FOLDED_PALU_MODEL_ARCHITECTURE,
    register as register_basisserve_vllm,
)
from evaluation.eval_qwen3_8b_sparse_c1_ruler_vllm import (  # noqa: E402
    DEFAULT_TASKS,
    PINNED_VLLM_VERSION,
    _atomic_json,
    _atomic_text,
    _build_work,
    _check,
    _fingerprint,
    _load_dataset_manifest,
    _sha256,
)
from evaluation.ruler_v1 import (  # noqa: E402
    RulerTask,
    parse_tasks,
    ruler_prompt,
    sample_score,
    summarize_arm,
)


FORMAT = "basisserve.qwen3_8b.dense_controls.ruler_vllm.v1"
SHARD_FORMAT = "basisserve.qwen3_8b.dense_controls.ruler_vllm.shard.v1"
CHECKPOINT_FORMAT = "basisserve.qwen3_8b.iclr_v_factors.v1"
C1_MODES = {
    "c1_uniform_r80": {
        "arm": "dense_k_c1_uniform_r80_c4_32x32k",
        "label": "Dense-K + C1 uniform R80 (C4 32x32K)",
        "run_id": "Q3-8B-C1U-R80",
        "method": "c1-uniform",
        "retained_ratio": 0.625,
    },
    "c1_twosided_r80": {
        "arm": "dense_k_c1_twosided_kl_r80_c4_32x32k",
        "label": "Dense-K + C1 two-sided-KL R80 (C4 32x32K)",
        "run_id": "Q3-8B-C1-R80",
        "method": "c1-two-sided-kl",
        "retained_ratio": 0.625,
    },
}
PALU_MODES = {
    "palu_m_r80": {
        "arm": "dense_k_palu_m_fisher_r80_c4_32x32k",
        "label": "Dense-K + PaLU M-LRD Fisher R80 (C4 32x32K)",
        "run_id": "Q3-8B-PALUM-R80",
        "retained_ratio": 0.6458333333333334,
    },
    "palu_g2_r80": {
        "arm": "dense_k_palu_g2_fisher_r80_c4_32x32k",
        "label": "Dense-K + PaLU G2-LRD Fisher R80 (C4 32x32K)",
        "run_id": "Q3-8B-PALUG2-R80",
        "retained_ratio": 0.625,
    },
    "palu_g4_r80": {
        "arm": "dense_k_palu_g4_fisher_r80_c4_32x32k",
        "label": "Dense-K + PaLU G4-LRD Fisher R80 (C4 32x32K)",
        "run_id": "Q3-8B-PALUG4-R80",
        "retained_ratio": 0.625,
    },
}


def _arm_name(mode: str) -> str:
    if mode in C1_MODES:
        return str(C1_MODES[mode]["arm"])
    if mode in PALU_MODES:
        return str(PALU_MODES[mode]["arm"])
    return "dense_k_dense_v"


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


def _markdown(payload: Mapping[str, Any]) -> str:
    metadata = payload["metadata"]
    summary = payload["summary"]["arm"]
    mode = str(metadata["mode"])
    label = (
        str(C1_MODES[mode]["label"])
        if mode in C1_MODES
        else str(PALU_MODES[mode]["label"])
        if mode in PALU_MODES
        else "Dense-K + Dense-V"
    )
    lines = [
        f"# Qwen3-8B {label} RULER-v1 32K",
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
            "Protocol: BF16 Qwen3-8B-Base, dense exact K, native vLLM "
            "FlashAttention, greedy decoding, and the same 11-task x 8-sample "
            "RULER-32K dataset used by the sparse-Key arms.",
        ]
    )
    if mode in C1_MODES:
        lines.extend(
            [
                "",
                "C1-V80 is folded into standard dense V/O slots for a quality-only "
                "control; this run does not use a physically compact V cache.",
            ]
        )
    elif metadata["mode"] in PALU_MODES:
        lines.extend(
            [
                "",
                "PaLU is reconstructed into standard dense V projection slots for "
                "a quality-only control; this run does not use a compact V cache.",
            ]
        )
    return "\n".join(lines) + "\n"


def _merge_if_complete(
    output_dir: Path,
    *,
    fingerprint: str,
    protocol: Mapping[str, Any],
    tasks: Sequence[RulerTask],
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
    arm = str(protocol["arm"])
    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "fingerprint": fingerprint,
        "metadata": dict(protocol),
        "summary": {
            "arm": summarize_arm(records, arm, tasks),
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
    checkpoint_dir: Path | None,
    checkpoint_sha256: str | None,
    tasks: Sequence[RulerTask],
    dataset_sha256: str,
) -> dict[str, Any]:
    return {
        "model": str(model),
        "model_config_sha256": _sha256(model / "config.json"),
        "data_dir": str(data_dir),
        "dataset_manifest_sha256": dataset_sha256,
        "mode": args.mode,
        "arm": _arm_name(args.mode),
        "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "tasks": [task.name for task in tasks],
        "samples_per_task": args.samples_per_task,
        "sequence_length": args.sequence_length,
        "key_cache": "dense K128",
        "value_cache": (
            f"{C1_MODES[args.mode]['label']} folded into dense V128 slots"
            if args.mode in C1_MODES
            else f"{PALU_MODES[args.mode]['label']} reconstructed into dense V128 slots"
            if args.mode in PALU_MODES
            else "dense V128"
        ),
        "value_retained_ratio": (
            float(C1_MODES[args.mode]["retained_ratio"])
            if args.mode in C1_MODES
            else float(PALU_MODES[args.mode]["retained_ratio"])
            if args.mode in PALU_MODES
            else 1.0
        ),
        "attention": "native vLLM FlashAttention",
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
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir is not None
        else None
    )
    tasks = parse_tasks(args.tasks)
    valid = all(
        (
            _check(args.sequence_length == 32768, "fair protocol requires 32768"),
            _check(0 < args.samples_per_task <= 8, "samples/task must lie in [1, 8]"),
            _check(0 <= args.shard_index < args.num_shards, "invalid shard index"),
            _check(args.max_num_batched_tokens > 0, "invalid batched-token limit"),
            _check(0.0 < args.gpu_memory_utilization < 1.0, "invalid GPU utilization"),
            _check(
                (args.mode == "dense_v") == (checkpoint_dir is None),
                "compressed V modes require --checkpoint-dir",
            ),
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
    checkpoint_sha256 = None
    if checkpoint_dir is not None:
        manifest_path = checkpoint_dir / "manifest.json"
        if not _check(manifest_path.is_file(), f"missing manifest: {manifest_path}"):
            return 2
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_sha256 = _sha256(manifest_path)
        expected_run_id = str(
            (C1_MODES | PALU_MODES)[args.mode]["run_id"]
        )
        if not all(
            (
                _check(
                    manifest.get("format") == CHECKPOINT_FORMAT,
                    "checkpoint format mismatch",
                ),
                _check(
                    manifest.get("status") == "complete", "checkpoint is incomplete"
                ),
                _check(
                    manifest.get("run_id") == expected_run_id,
                    "checkpoint run ID mismatch",
                ),
                _check(
                    args.mode not in C1_MODES
                    or manifest.get("compression", {}).get("method")
                    == C1_MODES[args.mode]["method"],
                    "C1 checkpoint method mismatch",
                ),
                _check(
                    manifest.get("model", {}).get("config_sha256")
                    == _sha256(model / "config.json"),
                    "checkpoint model mismatch",
                ),
            )
        ):
            return 2
    protocol = _protocol(
        args,
        model=model,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_sha256=checkpoint_sha256,
        tasks=tasks,
        dataset_sha256=dataset_sha256,
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
                command=command,
            )
            else 2
        )
    if not all(
        (
            _check(torch.cuda.is_available(), "CUDA is required"),
            _check(torch.cuda.device_count() == 1, "expose exactly one GPU per shard"),
        )
    ):
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
        f"[RULER vLLM] mode={args.mode} "
        f"shard={args.shard_index}/{args.num_shards} pending={len(pending)}",
        flush=True,
    )

    register_basisserve_vllm()
    from vllm import LLM, SamplingParams, TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(
        model, local_files_only=True, use_fast=True
    )
    hf_overrides = None
    if checkpoint_dir is not None:
        if args.mode in C1_MODES:
            hf_overrides = {
                "architectures": [QWEN3_8B_FOLDED_C1_MODEL_ARCHITECTURE],
                "basisserve_c1_checkpoint_dir": str(checkpoint_dir),
                "basisserve_c1_manifest_sha256": checkpoint_sha256,
            }
        else:
            hf_overrides = {
                "architectures": [QWEN3_8B_FOLDED_PALU_MODEL_ARCHITECTURE],
                "basisserve_palu_checkpoint_dir": str(checkpoint_dir),
                "basisserve_palu_manifest_sha256": checkpoint_sha256,
            }
    started = time.perf_counter()
    previous_runtime = state.get("runtime", {})
    previous_elapsed = float(previous_runtime.get("elapsed_seconds", 0.0))
    previous_peak = int(previous_runtime.get("peak_cuda_allocated_bytes", 0))
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
        token_ids = tokenizer.encode(ruler_prompt(source), add_special_tokens=True)
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
        state["records"].append(
            {
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
        )
        snapshot = engine.collective_rpc("basisserve_cuda_statistics")[0]
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["runtime"] = {
            "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
            "peak_cuda_allocated_bytes": max(
                previous_peak,
                int(snapshot["peak_cuda_allocated_bytes"]),
            ),
            "cuda_device": snapshot["cuda_device"],
        }
        _atomic_json(rank_path, state)
        score = state["records"][-1]["arms"][protocol["arm"]]["score"]
        print(
            f"[{args.shard_index}] {key} prompt={len(token_ids)} "
            f"generated={len(generated)} score={score:.3f}",
            flush=True,
        )
    snapshot = engine.collective_rpc("basisserve_cuda_statistics")[0]
    state.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol,
            "runtime": {
                "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                "cuda_device": snapshot["cuda_device"],
                "peak_cuda_allocated_bytes": max(
                    previous_peak,
                    int(snapshot["peak_cuda_allocated_bytes"]),
                ),
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
        command=command,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(*C1_MODES, *PALU_MODES, "dense_v"),
        required=True,
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--merge-only", action="store_true")
    return parser


if __name__ == "__main__":
    sys.exit(evaluate(_parser().parse_args()))
