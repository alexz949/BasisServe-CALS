#!/usr/bin/env python3
"""Evaluate a GQA PaLU-M checkpoint on fixed C4-validation documents."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_gqa_palu_m_wikitext import (  # noqa: E402
    _atomic_json,
    _decoder_layers,
    _load_checkpoint,
    _sha256,
    install_palu_m_factors,
)
from evaluation.eval_qwen3_32b_c1_c4_ppl_shard import (  # noqa: E402
    _evaluate_document_ppl,
    _load_windows,
)


FORMAT = "basisserve.gqa.palu_m.c4_validation_ppl.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
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


def _layer_ranks(
    manifest: Mapping[str, Any], *, num_layers: int
) -> list[list[int]]:
    compression = manifest["compression"]
    if "layer_ranks" in compression:
        ranks = [[int(rank) for rank in row] for row in compression["layer_ranks"]]
    else:
        uniform = [int(rank) for rank in compression["ranks"]]
        ranks = [uniform] * num_layers
    if len(ranks) != num_layers:
        raise ValueError("PaLU rank schedule does not cover every decoder layer")
    return ranks


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.torch_num_threads <= 0:
        raise ValueError("batch size and thread count must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("PaLU C4 PPL evaluation requires CUDA")
    torch.set_num_threads(args.torch_num_threads)
    torch.cuda.set_device(0)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)

    started = time.perf_counter()
    model_path = Path(args.model).expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest, factors = _load_checkpoint(checkpoint_dir, model_path)
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
    )
    model.eval()
    model.config.use_cache = False

    layers = _decoder_layers(model)
    compression = manifest["compression"]
    geometry = {
        "num_query_heads": int(model.config.num_attention_heads),
        "num_physical_kv_heads": int(model.config.num_key_value_heads),
        "head_dim": int(
            getattr(model.config, "head_dim", 0)
            or model.config.hidden_size // model.config.num_attention_heads
        ),
    }
    for key, value in geometry.items():
        if value != int(compression[key]):
            raise ValueError(f"model/checkpoint geometry mismatch for {key}")
    if len(layers) != len(manifest["layers"]):
        raise ValueError("model/checkpoint layer count mismatch")

    install_started = time.perf_counter()
    installation = install_palu_m_factors(
        model,
        factors,
        layer_ranks=_layer_ranks(manifest, num_layers=len(layers)),
        head_dim=geometry["head_dim"],
        require_cuda_resident=True,
    )
    installation_seconds = time.perf_counter() - install_started
    del factors
    metrics = _evaluate_document_ppl(
        model,
        sequences,
        batch_size=args.batch_size,
        label="palu_m_fisher",
    )

    manifest_path = checkpoint_dir / "manifest.json"
    artifact_path = checkpoint_dir / manifest["artifact"]["file"]
    result = {
        "format": FORMAT,
        "status": "complete",
        "arm": "palu_m",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": manifest["model"],
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "manifest_sha256": _sha256(manifest_path),
            "artifact_sha256": _sha256(artifact_path),
            "format": manifest["format"],
        },
        "compression": compression,
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
                "stored latent V factors reconstructed immediately before attention; "
                "mathematically equivalent quality path, not a cache-performance benchmark"
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
    _atomic_json(output_path, result)
    print(f"[C4 PPL] palu_m_fisher ppl={metrics['ppl']:.9f}", flush=True)
    print(f"[C4 PPL] wrote {output_path}", flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
