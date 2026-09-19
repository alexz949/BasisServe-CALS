#!/usr/bin/env python3
"""Evaluate exact dense-attention 128K PPL with position-bucket NLL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

from safetensors.torch import load_file
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, DynamicCache


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import (  # noqa: E402
    install_qwen3_gqa_vo_als_export,
)
from evaluation.eval_qwen3_c1_quest_ruler import _sha256  # noqa: E402
from evaluation.eval_qwen3_dense_ruler import _qwen3_config  # noqa: E402


FORMAT = "basisserve.qwen3_8b.section3.long_context_ppl.v1"
BUCKETS = (
    ("0_8k", 0, 8192),
    ("8_32k", 8192, 32768),
    ("32_64k", 32768, 65536),
    ("64_128k", 65536, 131072),
)


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--method", choices=("dense", "c1"), required=True)
    parser.add_argument("--factor-dir", type=Path)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yarn-factor", type=float, default=4.0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument(
        "--token-limit",
        type=int,
        help="Evaluate only this many leading tokens per 128K document (smoke only).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _token_nll(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_target = target.reshape(-1)
    return F.cross_entropy(flat_logits, flat_target, reduction="none")


def _accumulate(
    destination: dict[str, dict[str, float | int]],
    nll: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    for name, start, stop in BUCKETS:
        selected = (positions >= start) & (positions < stop)
        if bool(selected.any()):
            destination[name]["nll_sum"] = float(destination[name]["nll_sum"]) + float(
                nll[selected].double().sum()
            )
            destination[name]["tokens"] = int(destination[name]["tokens"]) + int(
                selected.sum()
            )


@torch.inference_mode()
def _evaluate_document(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    cache = DynamicCache()
    buckets = {name: {"nll_sum": 0.0, "tokens": 0} for name, _, _ in BUCKETS}
    previous_last_logits = None
    length = int(tokens.shape[1])
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        output = model(
            input_ids=tokens[:, start:stop],
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        logits = output.logits
        if previous_last_logits is not None:
            boundary_nll = _token_nll(
                previous_last_logits,
                tokens[:, start : start + 1],
            )
            _accumulate(
                buckets,
                boundary_nll,
                torch.tensor([start], device=boundary_nll.device),
            )
        if stop - start > 1:
            within_nll = _token_nll(
                logits[:, :-1],
                tokens[:, start + 1 : stop],
            )
            _accumulate(
                buckets,
                within_nll,
                torch.arange(start + 1, stop, device=within_nll.device),
            )
        previous_last_logits = logits[:, -1:].clone()
        del output, logits
        print(f"[128K PPL] tokens={stop}/{length}", flush=True)
    result = {}
    for name, _, _ in BUCKETS:
        row = buckets[name]
        token_count = int(row["tokens"])
        mean_nll = float(row["nll_sum"]) / token_count if token_count else None
        result[name] = {
            **row,
            "mean_nll": mean_nll,
            "ppl": math.exp(mean_nll) if mean_nll is not None else None,
        }
    total_nll = sum(float(row["nll_sum"]) for row in buckets.values())
    total_tokens = sum(int(row["tokens"]) for row in buckets.values())
    del cache, previous_last_logits
    torch.cuda.empty_cache()
    return {
        "nll_sum": total_nll,
        "tokens": total_tokens,
        "mean_nll": total_nll / total_tokens,
        "ppl": math.exp(total_nll / total_tokens),
        "position_buckets": result,
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> int:
    factor_required = args.method == "c1"
    valid = all(
        (
            _check(torch.cuda.is_available(), "CUDA is required"),
            _check(
                factor_required == (args.factor_dir is not None),
                "factor-dir is required exactly for C1",
            ),
            _check(args.chunk_size > 0, "chunk size must be positive"),
            _check(
                args.token_limit is None or 2 <= args.token_limit <= 131072,
                "token limit must be in [2, 131072]",
            ),
        )
    )
    if not valid:
        return 2
    model_path = args.model.expanduser().resolve()
    windows_path = args.windows.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    result_path = output_dir / "results.json"
    if not _check(not result_path.exists(), f"result exists: {result_path}"):
        return 2
    manifest_path = windows_path.parent / "manifest.json"
    if not _check(windows_path.is_file() and manifest_path.is_file(), "windows missing"):
        return 2
    windows_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    windows = load_file(str(windows_path), device="cpu")["input_ids"].to(torch.long)
    if args.max_documents is not None:
        windows = windows[: args.max_documents]
    if not _check(
        windows.ndim == 2 and int(windows.shape[1]) == 131072 and len(windows) > 0,
        f"expected nonempty 128K windows, found {tuple(windows.shape)}",
    ):
        return 2
    source_sequence_length = int(windows.shape[1])
    if args.token_limit is not None:
        windows = windows[:, : args.token_limit]
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    config, rope_parameters = _qwen3_config(
        model_path,
        sequence_length=131072,
        yarn_factor=args.yarn_factor,
    )
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": str(device)},
    ).eval()
    model.config.use_cache = True
    installation = []
    factor_record = None
    if factor_required:
        factor_dir = args.factor_dir.expanduser().resolve()
        installation = [
            record.__dict__
            for record in install_qwen3_gqa_vo_als_export(
                model,
                factor_dir,
                attention_backend="sdpa",
            )
        ]
        factor_record = {
            "directory": str(factor_dir),
            "results_sha256": _sha256(factor_dir / "results.json"),
        }
    document_results = []
    aggregate = {name: {"nll_sum": 0.0, "tokens": 0} for name, _, _ in BUCKETS}
    for index, row in enumerate(windows):
        document_started = time.perf_counter()
        metrics = _evaluate_document(
            model,
            row.unsqueeze(0).to(device),
            chunk_size=args.chunk_size,
        )
        document_results.append(
            {
                "document": index,
                **metrics,
                "elapsed_seconds": time.perf_counter() - document_started,
            }
        )
        for name, _, _ in BUCKETS:
            aggregate[name]["nll_sum"] += metrics["position_buckets"][name]["nll_sum"]
            aggregate[name]["tokens"] += metrics["position_buckets"][name]["tokens"]
        print(
            f"[128K PPL] document={index + 1}/{len(windows)} ppl={metrics['ppl']:.9f}",
            flush=True,
        )
    buckets = {}
    for name, _, _ in BUCKETS:
        row = aggregate[name]
        token_count = int(row["tokens"])
        mean_nll = float(row["nll_sum"]) / token_count if token_count else None
        buckets[name] = {
            **row,
            "mean_nll": mean_nll,
            "ppl": math.exp(mean_nll) if mean_nll is not None else None,
        }
    total_nll = sum(float(row["nll_sum"]) for row in aggregate.values())
    total_tokens = sum(int(row["tokens"]) for row in aggregate.values())
    payload = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "method": args.method,
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "rope_parameters": rope_parameters,
            "effective_max_position_embeddings": int(model.config.max_position_embeddings),
        },
        "factor_bank": factor_record,
        "windows": {
            "path": str(windows_path),
            "sha256": _sha256(windows_path),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "documents": len(windows),
            "source_sequence_length": source_sequence_length,
            "evaluated_sequence_length": int(windows.shape[1]),
        },
        "protocol": {
            "dense_k": True,
            "full_dense_attention": True,
            "routing": False,
            "page_fisher": False,
            "sparse_selection": False,
            "chunk_size": args.chunk_size,
            "loss_dtype": "float32",
            "position_bucket_target_indexing": True,
            "smoke_token_limit": args.token_limit,
        },
        "installation": installation,
        "metrics": {
            "nll_sum": total_nll,
            "tokens": total_tokens,
            "mean_nll": total_nll / total_tokens,
            "ppl": math.exp(total_nll / total_tokens),
            "position_buckets": buckets,
            "documents": document_results,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name,
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(result_path, payload)
    print(f"[Complete] {result_path} ppl={payload['metrics']['ppl']:.9f}", flush=True)
    return 0


def main() -> None:
    status = evaluate(parse_args())
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()
