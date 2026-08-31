#!/usr/bin/env python3
"""Evaluate dense V with dense, independent-K64, and pairwise-K128 history."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.pairwise_k_qwen3 import (  # noqa: E402
    PairwiseKRuntime,
    install_qwen3_pairwise_k_runtime,
)
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids  # noqa: E402


FORMAT = "basisserve.qwen3_8b.pairwise_k_dense_v_ppl.v1"
FACTOR_FORMAT = "basisserve.qwen3_8b.pairwise_kq_svd.v1"
ARMS = ("dense", "independent", "pairwise")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load_factors(
    factor_dir: Path,
    model_path: Path,
) -> tuple[dict[str, Any], dict[str, Tensor], Path]:
    result_path = factor_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("format") != FACTOR_FORMAT or result.get("status") != "complete":
        raise ValueError("pairwise K factor result is incomplete or incompatible")
    if result["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("pairwise K factors belong to another model config")
    artifact = result["artifacts"]["factors"]
    factor_path = factor_dir / artifact["file"]
    if _sha256(factor_path) != artifact["sha256"]:
        raise ValueError("pairwise K factor artifact hash mismatch")
    factors = load_file(str(factor_path), device="cpu")
    expected_names = {
        "independent_key_projector",
        "independent_query_projector",
        "pair_key_projector",
        "pair_query_projector",
    }
    if set(factors) != expected_names:
        raise ValueError("pairwise K artifact contains unexpected tensors")
    expected_shapes = result["artifacts"]["factors"]["tensors"]
    for name in expected_names:
        if list(factors[name].shape) != expected_shapes[name]:
            raise ValueError(f"pairwise K tensor {name} has incompatible geometry")
    return result, factors, result_path


@torch.inference_mode()
def _score_sequence(
    model: nn.Module,
    input_ids: Tensor,
    *,
    prefill_length: int,
    block_length: int,
    runtime: PairwiseKRuntime | None,
) -> dict[str, Any]:
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("pairwise K PPL supports batch size one")
    sequence_length = int(input_ids.shape[1])
    if not 0 < prefill_length < sequence_length:
        raise ValueError("prefill length must leave scored tokens")
    if block_length <= 0:
        raise ValueError("block length must be positive")
    if runtime is not None:
        runtime.reset()
    cache = DynamicCache()
    prefill = model(
        input_ids=input_ids[:, :prefill_length],
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=0,
    )
    prefill_labels = input_ids[:, 1:prefill_length].to(prefill.logits.device)
    prefill_losses = F.cross_entropy(
        prefill.logits[:, :-1].float().reshape(-1, prefill.logits.shape[-1]),
        prefill_labels.reshape(-1),
        reduction="sum",
    )
    if not bool(torch.isfinite(prefill_losses)):
        raise FloatingPointError("pairwise K prefill PPL produced non-finite loss")
    next_logits = prefill.logits[:, -1, :]
    cursor = prefill_length
    nll_sum = float(prefill_losses)
    tokens = int(prefill_labels.numel())
    top1_correct = int(
        (prefill.logits[:, :-1].argmax(dim=-1) == prefill_labels).sum()
    )
    while cursor < sequence_length:
        stop = min(cursor + block_length, sequence_length)
        block_tokens = input_ids[:, cursor:stop]
        output = model(
            input_ids=block_tokens,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=0,
        )
        aligned = torch.cat((next_logits.unsqueeze(1), output.logits[:, :-1]), dim=1)
        labels = block_tokens.to(aligned.device)
        losses = F.cross_entropy(
            aligned.float().reshape(-1, aligned.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        )
        if not bool(torch.isfinite(losses)):
            raise FloatingPointError("pairwise K PPL produced non-finite loss")
        nll_sum += float(losses)
        tokens += int(labels.numel())
        top1_correct += int((aligned.argmax(dim=-1) == labels).sum())
        next_logits = output.logits[:, -1, :]
        cursor = stop
    if tokens != sequence_length - 1:
        raise RuntimeError("pairwise K PPL token accounting is inconsistent")
    if runtime is not None:
        for state in runtime.states:
            if state.history_length("pairwise", 0) != sequence_length:
                raise RuntimeError("pairwise K state did not commit the full sequence")
    return {
        "tokens": tokens,
        "nll_sum": nll_sum,
        "mean_nll": nll_sum / tokens,
        "ppl": math.exp(nll_sum / tokens),
        "label_top1_rate": top1_correct / tokens,
    }


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = sum(int(item["tokens"]) for item in samples)
    nll_sum = sum(float(item["nll_sum"]) for item in samples)
    return {
        "samples": len(samples),
        "tokens": tokens,
        "nll_sum": nll_sum,
        "mean_nll": nll_sum / tokens,
        "ppl": math.exp(nll_sum / tokens),
    }


def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("pairwise K dense-V PPL requires CUDA")
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
    factor_dir = Path(args.factor_dir).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    factor_result, factors, factor_result_path = _load_factors(factor_dir, model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
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
                f"[pair-K PPL] arm=dense sample={sample_index + 1}/{sample_count} "
                f"ppl={result['ppl']:.6f}",
                flush=True,
            )

    runtime = install_qwen3_pairwise_k_runtime(
        model,
        independent_key_projector=factors["independent_key_projector"],
        independent_query_projector=factors["independent_query_projector"],
        pair_key_projector=factors["pair_key_projector"],
        pair_query_projector=factors["pair_query_projector"],
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
                f"[pair-K PPL] arm={arm} sample={sample_index + 1}/{sample_count} "
                f"ppl={result['ppl']:.6f}",
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

    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "factor_result": {
            "path": str(factor_result_path),
            "sha256": _sha256(factor_result_path),
            "format": factor_result["format"],
            "geometry": factor_result["geometry"],
            "calibration": factor_result["calibration"],
            "method": factor_result["method"],
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
            "value_path": "original dense V and o_proj in every arm",
            "candidate_key_path": (
                "compressed completed history plus exact layer-local current block"
            ),
            "dense_prefill": True,
        },
        "aggregate": aggregate,
        "samples": arm_samples,
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
            "teacher-forced fixed-token PPL rather than free generation",
            "reference runtime retains exact historical K inside DynamicCache but ignores it for candidate attention",
            "logical cache comparison only; not a physical-memory or latency measurement",
            "single-GPU eager reference",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    print(f"[pair-K PPL] wrote {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--factor-dir", required=True)
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
