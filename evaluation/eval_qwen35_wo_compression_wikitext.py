#!/usr/bin/env python3
"""Evaluate Qwen3.5 Wo compression checkpoints on WikiText-2 test PPL."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import sys
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen35_full_attention_private_ag_runtime import (  # noqa: E402
    Qwen35FullAttentionPrivateAGRuntime,
    load_qwen35_full_attention_private_ag_factors,
)
from basisserve.core.qwen35_gdn_private_ag_runtime import (  # noqa: E402
    Qwen35PrivateAGRuntime,
    load_qwen35_gdn_private_ag_factors,
)
from basisserve.core.qwen35_global_aa_svd import (  # noqa: E402
    Qwen35GlobalAASVDRuntime,
    load_qwen35_global_aa_svd_factors,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)
from evaluation.eval_qwen35_hybrid_private_ag_ppl import (  # noqa: E402
    _private_metadata,
)
from evaluation.eval_qwen35_global_aa_svd_ppl import (  # noqa: E402
    _runtime_metadata as _global_runtime_metadata,
    _validate_factors as _validate_global_factors,
)
from scripts.eval_qwen35_projected_gdn_nll import _dtype  # noqa: E402
from scripts.eval_svdllm_safetensors_ppl_accelerate import (  # noqa: E402
    WIKITEXT_REPO,
    WIKITEXT_REVISION,
)


FORMAT = "basisserve.qwen35.wo_compression_wikitext2_ppl.v1"
NUM_LAYERS = 32
EXPECTED_GDN = tuple(layer for layer in range(NUM_LAYERS) if layer % 4 != 3)
EXPECTED_FULL = tuple(layer for layer in range(NUM_LAYERS) if layer % 4 == 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--uniform-gdn-factors", required=True)
    parser.add_argument("--uniform-full-factors", required=True)
    parser.add_argument("--ragged-gdn-factors", required=True)
    parser.add_argument("--ragged-full-factors", required=True)
    parser.add_argument("--global-aa-svd-factors", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_private_factors(
    factors: Mapping[str, Any],
    *,
    model_path: Path,
    tp_size: int,
    expected_layers: tuple[int, ...],
) -> None:
    if Path(factors.get("model_path", "")).resolve() != model_path:
        raise ValueError("Private-AG factors belong to another model")
    if int(factors.get("tp_size", -1)) != tp_size:
        raise ValueError("Private-AG factors use another TP size")
    layers = tuple(int(layer["layer_index"]) for layer in factors.get("layers", ()))
    if layers != expected_layers:
        raise ValueError(
            f"Private-AG factor coverage {layers} differs from {expected_layers}"
        )


def _private_rank_schedule(
    gdn_factors: Mapping[str, Any],
    full_factors: Mapping[str, Any],
) -> dict[int, int]:
    schedule = {
        int(layer["layer_index"]): int(layer["private_encoders"].shape[-1])
        for factors in (gdn_factors, full_factors)
        for layer in factors["layers"]
    }
    if tuple(sorted(schedule)) != tuple(range(NUM_LAYERS)):
        raise ValueError("combined Private-AG factors do not cover all decoder layers")
    return schedule


def _private_variant_metadata(
    gdn_runtime: Qwen35PrivateAGRuntime,
    full_runtime: Qwen35FullAttentionPrivateAGRuntime,
) -> dict[str, Any]:
    gdn = _private_metadata(gdn_runtime.records)
    full = _private_metadata(full_runtime.records)
    schedule = {
        **gdn["local_rank_schedule"],
        **full["local_rank_schedule"],
    }
    ranks = tuple(int(schedule[str(layer)]) for layer in range(NUM_LAYERS))
    local_widths = {int(gdn["local_width"]), int(full["local_width"])}
    if len(local_widths) != 1:
        raise ValueError("GDN and full-attention Private-AG local widths differ")
    local_width = next(iter(local_widths))
    return {
        "collective": "source_private_allgather",
        "gdn": gdn,
        "full_attention": full,
        "local_rank_schedule": list(ranks),
        "average_local_rank": statistics.fmean(ranks),
        "communication_fraction_of_dense_allreduce": (
            sum(ranks) / (len(ranks) * 2 * local_width)
        ),
        "communication_reduction_fraction": (
            1.0 - sum(ranks) / (len(ranks) * 2 * local_width)
        ),
        "single_gpu_tp_math_simulation": True,
        "gdn_recurrent_state_compressed": False,
        "attention_cache_compressed": False,
    }


def _result(
    metrics: Mapping[str, Any],
    *,
    variant: str,
    dense: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    result = {"variant": variant, **dict(metrics), **dict(metadata)}
    if dense is not None:
        result["delta_mean_nll"] = (
            float(metrics["nll_sum"]) / int(metrics["tokens"])
            - float(dense["nll_sum"]) / int(dense["tokens"])
        )
        result["perplexity_ratio"] = float(metrics["ppl"]) / float(dense["ppl"])
    return result


def _cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if min(args.tp, args.seqlen, args.batch_size) <= 0:
        raise ValueError("TP size, sequence length, and batch size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max samples must be positive")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("max tokens must be positive")

    model_path = Path(args.model_path).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite WikiText result: {output_path}")
    factor_paths = {
        "uniform_gdn": Path(args.uniform_gdn_factors).expanduser().resolve(),
        "uniform_full": Path(args.uniform_full_factors).expanduser().resolve(),
        "ragged_gdn": Path(args.ragged_gdn_factors).expanduser().resolve(),
        "ragged_full": Path(args.ragged_full_factors).expanduser().resolve(),
        "global_aa_svd": Path(args.global_aa_svd_factors).expanduser().resolve(),
    }
    uniform_gdn = load_qwen35_gdn_private_ag_factors(factor_paths["uniform_gdn"])
    uniform_full = load_qwen35_full_attention_private_ag_factors(
        factor_paths["uniform_full"]
    )
    ragged_gdn = load_qwen35_gdn_private_ag_factors(factor_paths["ragged_gdn"])
    ragged_full = load_qwen35_full_attention_private_ag_factors(
        factor_paths["ragged_full"]
    )
    global_factors = load_qwen35_global_aa_svd_factors(
        factor_paths["global_aa_svd"]
    )
    for factors in (uniform_gdn, ragged_gdn):
        _validate_private_factors(
            factors,
            model_path=model_path,
            tp_size=args.tp,
            expected_layers=EXPECTED_GDN,
        )
    for factors in (uniform_full, ragged_full):
        _validate_private_factors(
            factors,
            model_path=model_path,
            tp_size=args.tp,
            expected_layers=EXPECTED_FULL,
        )
    uniform_schedule = _private_rank_schedule(uniform_gdn, uniform_full)
    ragged_schedule = _private_rank_schedule(ragged_gdn, ragged_full)
    if set(uniform_schedule.values()) != {192}:
        raise ValueError("uniform checkpoint must use local rank 192 on every layer")
    if sum(ragged_schedule.values()) != NUM_LAYERS * 192:
        raise ValueError("ragged checkpoint must preserve the average local rank 192 budget")
    _validate_global_factors(
        global_factors,
        model_path=model_path,
        tp_size=args.tp,
    )

    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_path),
        dtype=_dtype(args.dtype),
        device_map={"": args.device},
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).eval()
    eval_kwargs = {
        "dataset": "wikitext2",
        "split": "test",
        "seqlen": args.seqlen,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "max_tokens": args.max_tokens,
    }

    dense = _eval_ppl_fp32_loss(model, tokenizer, **eval_kwargs)
    results = [
        _result(
            dense,
            variant="dense",
            dense=None,
            metadata={
                "attention_output_projection_compressed": False,
                "gdn_recurrent_state_compressed": False,
                "attention_cache_compressed": False,
            },
        )
    ]
    private_variants = (
        ("c1_uniform_local_r192_als10", uniform_gdn, uniform_full),
        ("c1_global_kl_ragged_avg_local_r192", ragged_gdn, ragged_full),
    )
    for label, gdn_factors, full_factors in private_variants:
        gdn_runtime = Qwen35PrivateAGRuntime(model, gdn_factors)
        full_runtime = Qwen35FullAttentionPrivateAGRuntime(model, full_factors)
        with ExitStack() as stack:
            stack.enter_context(gdn_runtime)
            stack.enter_context(full_runtime)
            metadata = _private_variant_metadata(gdn_runtime, full_runtime)
            metrics = _eval_ppl_fp32_loss(model, tokenizer, **eval_kwargs)
        results.append(
            _result(metrics, variant=label, dense=dense, metadata=metadata)
        )
        del metrics, gdn_runtime, full_runtime
        _cleanup()

    global_runtime = Qwen35GlobalAASVDRuntime(model, global_factors)
    with global_runtime:
        global_metadata = _global_runtime_metadata(
            global_runtime.records,
            global_factors,
        )
        metrics = _eval_ppl_fp32_loss(model, tokenizer, **eval_kwargs)
    results.append(
        _result(
            metrics,
            variant="global_aa_svd_allreduce_r768",
            dense=dense,
            metadata={
                "global_aa_svd": global_metadata,
                "gdn_recurrent_state_compressed": False,
                "attention_cache_compressed": False,
            },
        )
    )

    device = torch.device(args.device)
    cuda_index = device.index if device.index is not None else 0
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "factor_artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in factor_paths.items()
        },
        "dataset": {
            "repo": WIKITEXT_REPO,
            "config": "wikitext-2-raw-v1",
            "revision": WIKITEXT_REVISION,
            "split": "test",
            "protocol": "concatenate_then_nonoverlapping_chunks",
            "seqlen": args.seqlen,
            "dropped_incomplete_tail": True,
        },
        "dtype": args.dtype,
        "tp_size": args.tp,
        "results": results,
        "environment": {
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": str(torch.__version__),
            "cuda_device_name": torch.cuda.get_device_name(cuda_index),
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(cuda_index)
            ),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(f"[Saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
