#!/usr/bin/env python3
"""Evaluate matched-storage independent/pairwise K with fixed Qwen3 C1-V64."""

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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.pairwise_k_qwen3 import (  # noqa: E402
    install_qwen3_pairwise_k_runtime,
)
from evaluation import eval_qwen3_32b_c1_wikitext as c1_evaluator  # noqa: E402
from evaluation.eval_qwen3_8b_pairwise_k_dense_v_ppl import (  # noqa: E402
    _aggregate,
    _dtype,
    _load_factors,
    _score_sequence,
    _sha256,
)
from evaluation.eval_qwen3_8b_post_rope_kqsvd_c1_wikitext import (  # noqa: E402
    _replace_attention,
)
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids  # noqa: E402


FORMAT = "basisserve.qwen3_8b.pairwise_k_c1_v64_ppl.v1"
C1_FORMAT = "basisserve.qwen3_8b.gqa_c1_joint.v1"
ARMS = ("dense", "independent", "pairwise")
HEAD_DIM = 128
VALUE_RANK = 64


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("pairwise K plus C1-V64 PPL requires CUDA")
    if min(
        args.sequence_length,
        args.prefill_length,
        args.block_length,
        args.torch_num_threads,
    ) <= 0:
        raise ValueError("PPL sizes must be positive")
    if args.prefill_length >= args.sequence_length:
        raise ValueError("prefill length must be shorter than the sequence")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive when specified")
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if not arms or len(arms) != len(set(arms)) or any(arm not in ARMS for arm in arms):
        raise ValueError(f"arms must be unique members of {ARMS}")

    torch.set_num_threads(args.torch_num_threads)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    device = torch.device("cuda:0")
    model_path = Path(args.model).expanduser().resolve()
    pair_factor_dir = Path(args.pair_factor_dir).expanduser().resolve()
    c1_factor_dir = Path(args.c1_factor_dir).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    pair_result, pair_factors, pair_result_path = _load_factors(
        pair_factor_dir,
        model_path,
    )
    c1_evaluator.activate_model_profile("qwen3_8b")
    c1_result = c1_evaluator._load_results(c1_factor_dir, model_path)
    if c1_result.get("format") != C1_FORMAT:
        raise ValueError("C1 factors are not the uniform Qwen3-8B checkpoint")
    value_rank = int(c1_result["fit_config"]["cache_rank_per_head"])
    if value_rank != VALUE_RANK:
        raise ValueError(f"expected C1-V64, found C1-V{value_rank}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=True,
    )
    stream = _token_ids(tokenizer, args.dataset, args.split, None)
    complete_samples = int(stream.numel() // args.sequence_length)
    sample_count = (
        complete_samples
        if args.max_samples is None
        else min(complete_samples, args.max_samples)
    )
    if sample_count <= 0:
        raise ValueError("dataset contains no complete PPL sequence")
    samples = [
        stream[
            :,
            sample_index * args.sequence_length : (sample_index + 1) * args.sequence_length,
        ].to(device=device, dtype=torch.long)
        for sample_index in range(sample_count)
    ]
    del stream

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=_dtype(args.model_dtype),
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
        device_map={"": 0},
    ).eval()
    installation_started = time.perf_counter()
    c1_installation = _replace_attention(model, c1_factor_dir, c1_result)
    c1_installation_seconds = time.perf_counter() - installation_started

    arm_samples: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    if "dense" in arms:
        for sample_index, sample in enumerate(samples):
            result = _score_sequence(
                model,
                sample,
                prefill_length=args.prefill_length,
                block_length=args.block_length,
                runtime=None,
            )
            arm_samples["dense"].append(result)
            print(
                f"[pair-K C1-V64 PPL] arm=dense sample={sample_index + 1}/"
                f"{sample_count} ppl={result['ppl']:.6f}",
                flush=True,
            )

    runtime = install_qwen3_pairwise_k_runtime(
        model,
        independent_key_projector=pair_factors["independent_key_projector"],
        independent_query_projector=pair_factors["independent_query_projector"],
        pair_key_projector=pair_factors["pair_key_projector"],
        pair_query_projector=pair_factors["pair_query_projector"],
    )
    for arm in ("independent", "pairwise"):
        if arm not in arms:
            continue
        runtime.set_mode(arm)
        for sample_index, sample in enumerate(samples):
            result = _score_sequence(
                model,
                sample,
                prefill_length=args.prefill_length,
                block_length=args.block_length,
                runtime=runtime,
            )
            arm_samples[arm].append(result)
            print(
                f"[pair-K C1-V64 PPL] arm={arm} sample={sample_index + 1}/"
                f"{sample_count} ppl={result['ppl']:.6f}",
                flush=True,
            )

    aggregate = {arm: _aggregate(arm_samples[arm]) for arm in arms}
    if "dense" in aggregate:
        dense_nll = float(aggregate["dense"]["mean_nll"])
        for arm in arms:
            delta = float(aggregate[arm]["mean_nll"]) - dense_nll
            aggregate[arm]["mean_nll_delta_vs_dense"] = delta
            aggregate[arm]["ppl_ratio_vs_dense"] = math.exp(delta)
    if "independent" in aggregate and "pairwise" in aggregate:
        delta = (
            float(aggregate["pairwise"]["mean_nll"])
            - float(aggregate["independent"]["mean_nll"])
        )
        aggregate["pairwise"]["mean_nll_delta_vs_independent"] = delta
        aggregate["pairwise"]["ppl_ratio_vs_independent"] = math.exp(delta)

    c1_result_path = c1_factor_dir / "results.json"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "pairwise_key_factors": {
            "path": str(pair_result_path),
            "sha256": _sha256(pair_result_path),
            "format": pair_result["format"],
            "geometry": pair_result["geometry"],
            "calibration": pair_result["calibration"],
            "method": pair_result["method"],
            "score_frobenius_metrics": pair_result["score_frobenius_metrics"],
        },
        "c1_value_factors": {
            "path": str(c1_result_path),
            "sha256": _sha256(c1_result_path),
            "format": c1_result["format"],
            "value_rank": value_rank,
            "fit_config": c1_result["fit_config"],
        },
        "protocol": {
            "dataset": args.dataset,
            "split": args.split,
            "sequence_length": args.sequence_length,
            "prefill_length": args.prefill_length,
            "block_length": args.block_length,
            "samples": sample_count,
            "complete_dataset_chunks": complete_samples,
            "predictions_per_chunk": args.sequence_length - 1,
            "all_complete_chunks_evaluated": sample_count == complete_samples,
            "arms": list(arms),
            "value_path": "fixed resident C1-V64 and folded C1 output decoder in every arm",
            "candidate_key_path": (
                "compressed completed history plus exact layer-local current block"
            ),
            "attention_implementation": "single-GPU eager reference",
        },
        "logical_cache": {
            "dense": {
                "key_rank_per_layer": HEAD_DIM,
                "value_rank_per_layer": value_rank,
                "retained_kv_ratio": (HEAD_DIM + value_rank) / (2 * HEAD_DIM),
            },
            "independent": {
                "key_rank_per_layer": 64,
                "value_rank_per_layer": value_rank,
                "retained_kv_ratio": (64 + value_rank) / (2 * HEAD_DIM),
            },
            "pairwise": {
                "key_rank_per_layer_pair": 128,
                "average_key_rank_per_layer": 64,
                "value_rank_per_layer": value_rank,
                "retained_kv_ratio": (64 + value_rank) / (2 * HEAD_DIM),
            },
        },
        "aggregate": aggregate,
        "samples": arm_samples,
        "c1_installation": {
            "elapsed_seconds": c1_installation_seconds,
            "layers": c1_installation,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "torch_num_threads": torch.get_num_threads(),
        },
        "limitations": [
            "teacher-forced block-scheduled PPL rather than free generation",
            "native DynamicCache physically retains exact historical K for reference bookkeeping",
            "logical cache-quality comparison only; not a physical-memory or latency measurement",
            "the current block uses exact layer-local K before transactional pair-code commit",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    print(f"[pair-K C1-V64 PPL] wrote {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pair-factor-dir", required=True)
    parser.add_argument("--c1-factor-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--prefill-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument(
        "--model-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
