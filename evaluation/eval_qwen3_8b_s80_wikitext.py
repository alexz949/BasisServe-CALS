#!/usr/bin/env python3
"""Evaluate Qwen3-8B Dense-K + Store80 latent on full WikiText-2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_joint_routing_payload_s80_qwen3 import (  # noqa: E402
    install_qwen3_s80_factor_bank,
    load_s80_factor_bank,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)


FORMAT = "basisserve.qwen3_8b.s80_dense_k_wikitext2_ppl.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _cuda_device_indices() -> tuple[int, ...]:
    return tuple(range(torch.cuda.device_count()))


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    model_path = Path(args.model).expanduser().resolve()
    factor_dir = Path(args.factor_dir).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest, bank = load_s80_factor_bank(factor_dir)
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    torch.cuda.init()
    cuda_indices = _cuda_device_indices()
    for index in cuda_indices:
        torch.cuda.reset_peak_memory_stats(index)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB" for index in cuda_indices
        },
    ).eval()
    model.config.use_cache = False

    install_started = time.perf_counter()
    records = install_qwen3_s80_factor_bank(
        model,
        factor_dir,
        attention_backend=args.attn_implementation,
    )
    install_seconds = time.perf_counter() - install_started
    ppl = _eval_ppl_fp32_loss(
        model,
        tokenizer,
        dataset="wikitext2",
        split="test",
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        max_samples=None,
        max_tokens=None,
    )

    geometry = manifest["geometry"]
    payload = {
        "format": FORMAT,
        "status": "complete",
        "arm": "dense_k_store80_route32",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "factor_bank": {
            "path": str(factor_dir),
            "manifest_sha256": _sha256(factor_dir / "manifest.json"),
            "format": manifest["format"],
            "layer_coverage": manifest["layer_coverage"],
        },
        "compression": {
            "key_cache": "dense_exact",
            "value_payload": "joint_store80",
            "stored_rank_per_physical_kv_head": int(geometry["joint_rank"]),
            "routing_rank": int(geometry["routing_rank"]),
            "dense_value_head_dim": int(geometry["value_dim"]),
            "retained_value_width_ratio": (
                int(geometry["joint_rank"]) / int(geometry["value_dim"])
            ),
        },
        "quality_reference_runtime": {
            "description": (
                "exact dense QK attention with the joint Store80 payload; "
                "Route32 proxy selection is not used by full-attention PPL"
            ),
            "attention_backend": args.attn_implementation,
            "installation_seconds": install_seconds,
            "installed_layers": [record.layer_index for record in records],
            "bank_layers": sorted(bank),
        },
        "ppl": ppl,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index) for index in cuda_indices
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in cuda_indices
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_path, payload)
    print(f"[Result] arm={payload['arm']} ppl={ppl['ppl']:.9f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--model-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=72)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
