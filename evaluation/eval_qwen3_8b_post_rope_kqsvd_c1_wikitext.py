#!/usr/bin/env python3
"""Evaluate K-SVD/KQ-SVD K64 combined with uniform C1 V64 on WikiText-2."""

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
from typing import Any

from safetensors.torch import load_file
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import (  # noqa: E402
    GQATiedVOQwen3Attention,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)
from evaluation import eval_qwen3_32b_c1_wikitext as c1_evaluator  # noqa: E402


FORMAT = "basisserve.qwen3_8b.post_rope_kqsvd_c1_wikitext.v1"
KQ_FORMAT = "basisserve.qwen3_8b.post_rope_kqsvd.v1"
NUM_LAYERS = 36
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 4096
ARMS = ("dense_k", "key_svd", "kq_svd")


def _replace_attention(
    model: nn.Module,
    factor_dir: Path,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    rank = int(result["fit_config"]["cache_rank_per_head"])
    if rank != 64:
        raise ValueError(f"controlled KQ-SVD experiment requires C1 V64, got {rank}")
    records = []
    for layer_index, layer in enumerate(model.model.layers):
        base_attention = layer.self_attn
        artifact = result["artifacts"][str(layer_index)]
        path = factor_dir / artifact["file"]
        if c1_evaluator._sha256(path) != artifact["sha256"]:
            raise ValueError(f"C1 factor hash mismatch at layer {layer_index}")
        factors = load_file(str(path), device="cpu")
        if set(factors) != {
            "value_coordinate_encoders",
            "head_output_decoders",
        }:
            raise ValueError(f"unexpected C1 factor tensors at layer {layer_index}")
        encoders = factors["value_coordinate_encoders"]
        decoders = factors["head_output_decoders"]
        if tuple(encoders.shape) != (NUM_KV_HEADS, HEAD_DIM, rank):
            raise ValueError(f"invalid C1 encoder geometry at layer {layer_index}")
        if tuple(decoders.shape) != (NUM_QUERY_HEADS, rank, HIDDEN_SIZE):
            raise ValueError(f"invalid C1 decoder geometry at layer {layer_index}")
        if (
            not isinstance(base_attention.v_proj, nn.Linear)
            or not isinstance(base_attention.o_proj, nn.Linear)
            or base_attention.v_proj.bias is not None
            or base_attention.o_proj.bias is not None
        ):
            raise TypeError(f"unsupported Qwen attention at layer {layer_index}")

        device = base_attention.v_proj.weight.device
        dense_v = base_attention.v_proj.weight.detach().float().reshape(
            NUM_KV_HEADS,
            HEAD_DIM,
            HIDDEN_SIZE,
        )
        compressed_v = torch.bmm(
            encoders.to(device=device, dtype=torch.float32).mT,
            dense_v,
        ).reshape(NUM_KV_HEADS * rank, HIDDEN_SIZE)
        decoder_weight = (
            decoders.to(device=device, dtype=torch.float32)
            .permute(2, 0, 1)
            .reshape(HIDDEN_SIZE, NUM_QUERY_HEADS * rank)
        )
        layer.self_attn = GQATiedVOQwen3Attention(
            base_attention,
            v_proj_compressed_weight=compressed_v,
            o_decoder_weight=decoder_weight,
            attention_backend="native",
        )
        records.append(
            {
                "layer": layer_index,
                "factor_file": artifact["file"],
                "factor_sha256": artifact["sha256"],
                "value_rank": rank,
            }
        )
    return records


def _set_qk_arm(
    model: nn.Module,
    factors: dict[str, Tensor],
    arm: str,
) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown KQ-SVD arm: {arm}")
    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        if not isinstance(attention, GQATiedVOQwen3Attention):
            raise TypeError(f"layer {layer_index} is not compact C1 attention")
        if arm == "dense_k":
            attention.set_qk_projectors(None, None)
        elif arm == "key_svd":
            projector = factors["key_svd_projector"][layer_index]
            attention.set_qk_projectors(projector, projector)
        else:
            attention.set_qk_projectors(
                factors["kq_svd_key_projector"][layer_index],
                factors["kq_svd_query_projector"][layer_index],
            )


def _load_kq_result(
    factor_dir: Path,
    model_path: Path,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    result_path = factor_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("format") != KQ_FORMAT or result.get("status") != "complete":
        raise ValueError("post-RoPE KQ-SVD result is incomplete or incompatible")
    if result["model"]["config_sha256"] != c1_evaluator._sha256(
        model_path / "config.json"
    ):
        raise ValueError("KQ-SVD factors belong to another model config")
    geometry = result["geometry"]
    expected = {
        "layers": NUM_LAYERS,
        "query_heads": NUM_QUERY_HEADS,
        "physical_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "rank": 64,
    }
    for name, value in expected.items():
        if int(geometry[name]) != value:
            raise ValueError(f"KQ-SVD {name} mismatch: {geometry[name]} vs {value}")
    artifact = result["artifacts"]["factors"]
    factor_path = factor_dir / artifact["file"]
    if c1_evaluator._sha256(factor_path) != artifact["sha256"]:
        raise ValueError("KQ-SVD factor artifact hash mismatch")
    factors = load_file(str(factor_path), device="cpu")
    expected_tensors = {
        "key_svd_projector",
        "key_svd_spectrum",
        "kq_svd_key_projector",
        "kq_svd_query_projector",
        "kq_svd_spectrum",
    }
    if set(factors) != expected_tensors:
        raise ValueError("KQ-SVD factor artifact has unexpected tensors")
    projector_shape = (NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, 64)
    for name in (
        "key_svd_projector",
        "kq_svd_key_projector",
        "kq_svd_query_projector",
    ):
        if tuple(factors[name].shape) != projector_shape:
            raise ValueError(f"KQ-SVD tensor {name} has incompatible geometry")
    return result, factors


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("KQ-SVD C1 PPL evaluation requires CUDA")
    if min(args.seqlen, args.batch_size, args.torch_num_threads) <= 0:
        raise ValueError("PPL compute arguments must be positive")
    arms = tuple(name.strip() for name in args.arms.split(",") if name.strip())
    if not arms or len(arms) != len(set(arms)) or any(name not in ARMS for name in arms):
        raise ValueError(f"arms must be unique members of {ARMS}")
    torch.set_num_threads(args.torch_num_threads)
    torch.cuda.set_device(0)
    started = time.perf_counter()
    model_path = Path(args.model).expanduser().resolve()
    c1_factor_dir = Path(args.c1_factor_dir).expanduser().resolve()
    kq_factor_dir = Path(args.kq_factor_dir).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    c1_evaluator.activate_model_profile("qwen3_8b")
    c1_result = c1_evaluator._load_results(c1_factor_dir, model_path)
    kq_result, kq_factors = _load_kq_result(kq_factor_dir, model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=True,
    )
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        },
    ).eval()
    model.config.use_cache = False
    installation_started = time.perf_counter()
    installation = _replace_attention(model, c1_factor_dir, c1_result)
    installation_seconds = time.perf_counter() - installation_started

    arm_results = {}
    for arm in arms:
        _set_qk_arm(model, kq_factors, arm)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        arm_started = time.perf_counter()
        ppl = _eval_ppl_fp32_loss(
            model,
            tokenizer,
            dataset=args.dataset,
            split=args.split,
            seqlen=args.seqlen,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
        )
        qk_rank = HEAD_DIM if arm == "dense_k" else 64
        retained_kv_ratio = (qk_rank + 64) / (2 * HEAD_DIM)
        arm_results[arm] = {
            "ppl": ppl,
            "compression": {
                "key_rank_per_physical_head": qk_rank,
                "value_rank_per_physical_head": 64,
                "retained_kv_ratio": retained_kv_ratio,
                "kv_cache_reduction": 1.0 - retained_kv_ratio,
                "qk_dot_dimension": qk_rank,
                "pv_accumulation_dimension": 64,
            },
            "elapsed_seconds": time.perf_counter() - arm_started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        }
        print(f"[KQ-SVD C1 PPL] arm={arm} ppl={ppl['ppl']:.9f}", flush=True)

    c1_result_path = c1_factor_dir / "results.json"
    kq_result_path = kq_factor_dir / "result.json"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": c1_evaluator._sha256(model_path / "config.json"),
        },
        "c1_value_factors": {
            "path": str(c1_result_path),
            "sha256": c1_evaluator._sha256(c1_result_path),
            "format": c1_result["format"],
            "rank": 64,
        },
        "key_query_factors": {
            "path": str(kq_result_path),
            "sha256": c1_evaluator._sha256(kq_result_path),
            "format": kq_result["format"],
            "rank": 64,
            "calibration": kq_result["calibration"],
            "score_frobenius_metrics": kq_result["score_frobenius_metrics"],
        },
        "protocol": {
            "dataset": args.dataset,
            "split": args.split,
            "sequence_length": args.seqlen,
            "batch_size": args.batch_size,
            "model_dtype": args.model_dtype,
            "attention_implementation": "eager for all arms",
            "attention_scaling": f"1/sqrt({HEAD_DIM}); unchanged after K compression",
            "loss_dtype": "float32",
        },
        "arms": arm_results,
        "quality_reference_runtime": {
            "description": (
                "actual compact post-RoPE K/Q and compact C1 V/O coordinates; "
                "eager PyTorch quality path, not a fused performance benchmark"
            ),
            "installation_seconds": installation_seconds,
            "layers": installation,
        },
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
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    c1_evaluator._atomic_json(output_path, payload)
    print(f"[KQ-SVD C1 PPL] wrote {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--c1-factor-dir", required=True)
    parser.add_argument("--kq-factor-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--model-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
