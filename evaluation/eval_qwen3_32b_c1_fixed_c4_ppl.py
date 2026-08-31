#!/usr/bin/env python3
"""Evaluate one fixed uniform Qwen3-32B C1 checkpoint on C4 validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import sys
import time

import torch
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_qwen3_32b_c1_c4_ppl_shard import (  # noqa: E402
    _evaluate_document_ppl,
    _load_windows,
)
from evaluation.eval_qwen3_32b_c1_wikitext import (  # noqa: E402
    _atomic_json,
    _load_results,
    _sha256,
    install_c1_factors,
)


FORMAT = "basisserve.qwen3_32b.gqa_c1.fixed_c4_validation_ppl.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.torch_num_threads <= 0:
        raise ValueError("batch size and thread count must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("C1 C4 PPL evaluation requires CUDA")
    torch.set_num_threads(args.torch_num_threads)
    torch.cuda.set_device(0)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)

    started = time.perf_counter()
    model_path = Path(args.model).expanduser().resolve()
    factor_dir = args.factor_dir.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = _load_results(factor_dir, model_path)
    sequences, windows = _load_windows(args.windows, model_path=model_path)
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        },
    ).eval()
    model.config.use_cache = False

    install_started = time.perf_counter()
    installation = install_c1_factors(model, factor_dir, result)
    installation_seconds = time.perf_counter() - install_started
    rank = int(result["fit_config"]["cache_rank_per_head"])
    solver = result["fit_config"].get("decoder_solver", "normal_equation_als")
    metrics = _evaluate_document_ppl(
        model,
        sequences,
        batch_size=args.batch_size,
        label=f"c1_rank_{rank}_{solver}",
    )

    result_path = factor_dir / "results.json"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "arm": f"c1_rank_{rank}_{solver}",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "factor_result": {
            "path": str(result_path),
            "sha256": _sha256(result_path),
            "format": result["format"],
            "aggregate": result["aggregate"],
        },
        "compression": {
            "target": "value_cache_only",
            "key_cache": "dense",
            "num_query_heads": 64,
            "num_physical_kv_heads": 8,
            "head_dim": 128,
            "rank_per_physical_kv_head": rank,
            "rank_sum": 8 * rank,
            "retained_v_ratio": rank / 128,
            "v_cache_compression_ratio": 1.0 - rank / 128,
        },
        "windows": windows,
        "protocol": {
            "split": "validation",
            "documents": 128,
            "sequence_length": 2048,
            "batch_size": args.batch_size,
            "model_dtype": args.model_dtype,
            "attn_implementation": args.attn_implementation,
            "loss_dtype": "float32",
            "cross_document_transitions": False,
        },
        "quality_reference_runtime": {
            "description": (
                "C1 coordinates zero-padded into Hugging Face V/O slots; "
                "function-equivalent quality path, not a cache-performance benchmark"
            ),
            "installation_seconds": installation_seconds,
            "layers": installation,
        },
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(torch.cuda.device_count())
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_path, payload)
    print(f"[C4 PPL] rank={rank} solver={solver} ppl={metrics['ppl']:.9f}", flush=True)
    print(f"[C4 PPL] wrote {output_path}", flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
