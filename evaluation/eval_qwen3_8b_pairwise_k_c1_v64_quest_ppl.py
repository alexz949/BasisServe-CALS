#!/usr/bin/env python3
"""Evaluate latent-QUEST pairwise K with fixed resident Qwen3 C1-V64."""

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
    PairwiseQuestConfig,
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


FORMAT = "basisserve.qwen3_8b.pairwise_k_c1_v64_quest_ppl.v1"
BASELINE_FORMAT = "basisserve.qwen3_8b.pairwise_k_c1_v64_ppl.v1"
C1_FORMAT = "basisserve.qwen3_8b.gqa_c1_joint.v1"
VALUE_RANK = 64


def _parse_ints(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise ValueError(f"{name} must contain unique nonnegative integers")
    return values


def _validate_baseline(
    baseline: dict[str, Any],
    *,
    args: argparse.Namespace,
    model_path: Path,
    pair_result_path: Path,
    c1_result_path: Path,
) -> None:
    if baseline.get("format") != BASELINE_FORMAT or baseline.get("status") != "complete":
        raise ValueError("baseline is not a complete pairwise-K C1-V64 PPL result")
    if baseline["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("baseline model config differs from the requested model")
    if baseline["pairwise_key_factors"]["sha256"] != _sha256(pair_result_path):
        raise ValueError("baseline pairwise-K factors differ from the requested factors")
    if baseline["c1_value_factors"]["sha256"] != _sha256(c1_result_path):
        raise ValueError("baseline C1-V factors differ from the requested factors")
    expected_protocol = {
        "dataset": args.dataset,
        "split": args.split,
        "sequence_length": args.sequence_length,
        "prefill_length": args.prefill_length,
        "block_length": args.block_length,
    }
    observed = baseline["protocol"]
    for name, expected in expected_protocol.items():
        if observed.get(name) != expected:
            raise ValueError(
                f"baseline protocol {name}={observed.get(name)!r}, expected {expected!r}"
            )
    for mode in ("dense", "independent", "pairwise"):
        if mode not in baseline["samples"]:
            raise ValueError(f"baseline is missing {mode!r} samples")


def _comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float]:
    delta = float(candidate["mean_nll"]) - float(baseline["mean_nll"])
    return {
        "mean_nll_delta": delta,
        "ppl_ratio": math.exp(delta),
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("pairwise-K QUEST plus C1-V64 PPL requires CUDA")
    if min(
        args.sequence_length,
        args.prefill_length,
        args.block_length,
        args.page_size,
        args.independent_control_budget,
        args.torch_num_threads,
    ) <= 0:
        raise ValueError("PPL and QUEST sizes must be positive")
    if args.prefill_length >= args.sequence_length:
        raise ValueError("prefill length must be shorter than the sequence")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive when specified")
    pairwise_budgets = _parse_ints(args.pairwise_budgets, name="pairwise budgets")
    if min(pairwise_budgets) <= 0:
        raise ValueError("pairwise budgets must be positive")
    full_layers = _parse_ints(args.full_layers, name="full layers")

    torch.set_num_threads(args.torch_num_threads)
    torch.cuda.set_device(0)
    started = time.perf_counter()
    device = torch.device("cuda:0")
    model_path = Path(args.model).expanduser().resolve()
    pair_factor_dir = Path(args.pair_factor_dir).expanduser().resolve()
    c1_factor_dir = Path(args.c1_factor_dir).expanduser().resolve()
    baseline_path = Path(args.baseline_result).expanduser().resolve()
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
    c1_result_path = c1_factor_dir / "results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    _validate_baseline(
        baseline,
        args=args,
        model_path=model_path,
        pair_result_path=pair_result_path,
        c1_result_path=c1_result_path,
    )

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
    if any(len(baseline["samples"][mode]) < sample_count for mode in baseline["samples"]):
        raise ValueError("baseline does not cover every requested sample")
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
    runtime = install_qwen3_pairwise_k_runtime(
        model,
        independent_key_projector=pair_factors["independent_key_projector"],
        independent_query_projector=pair_factors["independent_query_projector"],
        pair_key_projector=pair_factors["pair_key_projector"],
        pair_query_projector=pair_factors["pair_query_projector"],
    )
    installation_seconds = time.perf_counter() - installation_started
    installed_layers = {record.layer_index for record in runtime.records}
    if not set(full_layers) <= installed_layers:
        raise ValueError("full-support layer schedule exceeds installed model layers")

    configurations = [
        {
            "name": f"pairwise_b{budget}",
            "mode": "pairwise",
            "historical_token_budget": budget,
        }
        for budget in pairwise_budgets
    ]
    configurations.append(
        {
            "name": f"independent_b{args.independent_control_budget}",
            "mode": "independent",
            "historical_token_budget": args.independent_control_budget,
        }
    )
    samples_by_arm: dict[str, list[dict[str, Any]]] = {}
    aggregate: dict[str, dict[str, Any]] = {}
    sparse_statistics: dict[str, dict[str, Any]] = {}
    arm_peak_cuda_bytes: dict[str, int] = {}
    for arm in configurations:
        name = str(arm["name"])
        mode = str(arm["mode"])
        budget = int(arm["historical_token_budget"])
        runtime.set_mode(mode)  # type: ignore[arg-type]
        runtime.set_sparse_policy(
            PairwiseQuestConfig(
                page_size=args.page_size,
                historical_token_budget=budget,
                landmark_dtype=args.landmark_dtype,
            ),
            full_layer_indices=full_layers,
        )
        runtime.reset_sparse_statistics()
        torch.cuda.reset_peak_memory_stats(0)
        arm_samples: list[dict[str, Any]] = []
        for sample_index, sample in enumerate(samples):
            result = _score_sequence(
                model,
                sample,
                prefill_length=args.prefill_length,
                block_length=args.block_length,
                runtime=runtime,
            )
            arm_samples.append(result)
            print(
                f"[pair-K QUEST C1-V64] arm={name} sample={sample_index + 1}/"
                f"{sample_count} ppl={result['ppl']:.6f}",
                flush=True,
            )
        samples_by_arm[name] = arm_samples
        candidate_aggregate = _aggregate(arm_samples)
        compressed_baseline = _aggregate(
            baseline["samples"][mode][:sample_count]
        )
        dense_baseline = _aggregate(
            baseline["samples"]["dense"][:sample_count]
        )
        candidate_aggregate["comparison_vs_full_support_compressed_k"] = _comparison(
            candidate_aggregate,
            compressed_baseline,
        )
        candidate_aggregate["comparison_vs_dense_k"] = _comparison(
            candidate_aggregate,
            dense_baseline,
        )
        aggregate[name] = candidate_aggregate
        sparse_statistics[name] = runtime.sparse_statistics()
        arm_peak_cuda_bytes[name] = int(torch.cuda.max_memory_allocated(0))

    baseline_prefix = {
        mode: _aggregate(baseline["samples"][mode][:sample_count])
        for mode in ("dense", "independent", "pairwise")
    }
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
        },
        "c1_value_factors": {
            "path": str(c1_result_path),
            "sha256": _sha256(c1_result_path),
            "format": c1_result["format"],
            "value_rank": value_rank,
            "fit_config": c1_result["fit_config"],
        },
        "baseline": {
            "path": str(baseline_path),
            "sha256": _sha256(baseline_path),
            "prefix_aggregate": baseline_prefix,
        },
        "protocol": {
            "dataset": args.dataset,
            "split": args.split,
            "sequence_length": args.sequence_length,
            "prefill_length": args.prefill_length,
            "block_length": args.block_length,
            "samples": sample_count,
            "complete_dataset_chunks": complete_samples,
            "all_complete_chunks_evaluated": sample_count == complete_samples,
            "page_size": args.page_size,
            "pairwise_historical_token_budgets": list(pairwise_budgets),
            "independent_control_budget": args.independent_control_budget,
            "full_support_layers": list(full_layers),
            "quest_layers": sorted(installed_layers - set(full_layers)),
            "landmark_dtype": args.landmark_dtype,
            "gqa_page_policy": (
                "maximum QUEST bound across all Query heads sharing a physical KV head"
            ),
            "historical_path": (
                "QUEST page selection and sparse exact attention in pairwise/independent "
                "latent K coordinates"
            ),
            "current_block_path": "complete exact post-RoPE QK with causal masking",
            "normalization": (
                "one softmax over selected historical scores and complete current-block scores"
            ),
            "value_path": "fixed resident C1-V64 for selected history and current block",
            "attention_implementation": "single-GPU eager reference",
        },
        "aggregate": aggregate,
        "samples": samples_by_arm,
        "sparse_work_accounting": sparse_statistics,
        "installation": {
            "elapsed_seconds": installation_seconds,
            "c1_layers": c1_installation,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "arm_peak_cuda_allocated_bytes": arm_peak_cuda_bytes,
            "torch_num_threads": torch.get_num_threads(),
        },
        "limitations": [
            "teacher-forced block-scheduled PPL rather than free generation",
            "reference implementation recomputes QUEST bounds and uses eager gather operations",
            "native DynamicCache physically retains exact historical K for bookkeeping",
            "latent historical K is not physically CPU-offloaded in this quality oracle",
            "reported sparse work is logical accounting, not measured PCIe traffic or latency",
            "layers 0-1 use full compressed-K history, not exact dense historical K",
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
    print(f"[pair-K QUEST C1-V64] wrote {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pair-factor-dir", required=True)
    parser.add_argument("--c1-factor-dir", required=True)
    parser.add_argument("--baseline-result", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--prefill-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--pairwise-budgets", default="256,512,1024")
    parser.add_argument("--independent-control-budget", type=int, default=512)
    parser.add_argument("--full-layers", default="0,1")
    parser.add_argument(
        "--landmark-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
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
