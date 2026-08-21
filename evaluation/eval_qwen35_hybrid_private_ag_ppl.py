#!/usr/bin/env python3
"""Evaluate Qwen3.5 GDN/full-attention Private-AG and sparse-gate hybrids."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
from typing import Any

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
from basisserve.core.qwen35_postgate_topk import (  # noqa: E402
    Qwen35PostGateTopKRuntime,
)
from scripts.collect_qwen35_gdn_moments import _fixed_length_samples  # noqa: E402
from scripts.eval_qwen35_projected_gdn_nll import _dtype, _evaluate  # noqa: E402


FORMAT = "basisserve.qwen35.hybrid_private_ag_ppl.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gdn-private-factors", required=True)
    parser.add_argument("--full-private-factors", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--full-topk-keep-ratios",
        default="0.5,0.75",
        help="source-local post-gate sparse alternatives; empty disables",
    )
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--sample-offset", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _ratios(raw: str) -> tuple[float, ...]:
    values = tuple(
        sorted({float(piece.strip()) for piece in raw.split(",") if piece.strip()})
    )
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in values):
        raise ValueError("Top-K keep ratios must lie in (0,1]")
    return values


def _serializable(
    result: dict[str, Any],
    *,
    variant: str,
    baseline: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"token_nlls", "top1_tokens"}
    }
    payload["variant"] = variant
    payload.update(metadata)
    if baseline is not None:
        differences = result["token_nlls"] - baseline["token_nlls"]
        window_differences = [
            candidate - teacher
            for teacher, candidate in zip(
                baseline["window_nlls"], result["window_nlls"], strict=True
            )
        ]
        payload.update(
            {
                "delta_mean_nll": result["mean_nll"] - baseline["mean_nll"],
                "perplexity_ratio": result["perplexity"] / baseline["perplexity"],
                "paired_window_standard_error": (
                    statistics.stdev(window_differences)
                    / math.sqrt(len(window_differences))
                    if len(window_differences) > 1
                    else 0.0
                ),
                "target_nll_mae": float(differences.abs().mean()),
                "target_nll_rmse": float(differences.square().mean().sqrt()),
                "target_nll_max_abs": float(differences.abs().max()),
                "top1_agreement": float(
                    (result["top1_tokens"] == baseline["top1_tokens"])
                    .float()
                    .mean()
                ),
            }
        )
    return payload


def _cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _uniform_private_metadata(records: tuple[Any, ...]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("Private AG runtime installed no layers")
    first = records[0]
    geometry = (
        first.tp_size,
        first.local_width,
        first.local_rank,
        first.total_private_rank,
    )
    if any(
        (
            record.tp_size,
            record.local_width,
            record.local_rank,
            record.total_private_rank,
        )
        != geometry
        for record in records
    ):
        raise ValueError("Private AG runtime uses inconsistent layer geometry")
    return {
        "layers_installed": len(records),
        "layer_indices": [record.layer_index for record in records],
        "tp_size": first.tp_size,
        "local_width": first.local_width,
        "local_rank": first.local_rank,
        "total_private_rank": first.total_private_rank,
        "ring_units": first.total_private_rank,
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if min(args.num_samples, args.sequence_length, args.batch_size, args.tp) <= 0:
        raise ValueError("sample, sequence, batch, and TP sizes must be positive")
    if args.sample_offset < 0:
        raise ValueError("sample offset must be nonnegative")
    topk_ratios = _ratios(args.full_topk_keep_ratios)
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite hybrid PPL result: {output_path}")
    dataset_path = Path(args.dataset_jsonl).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    gdn_path = Path(args.gdn_private_factors).expanduser().resolve()
    full_path = Path(args.full_private_factors).expanduser().resolve()
    gdn_factors = load_qwen35_gdn_private_ag_factors(gdn_path)
    full_factors = load_qwen35_full_attention_private_ag_factors(full_path)

    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    model_path = Path(args.model_path).expanduser().resolve()
    model_source = str(model_path) if model_path.exists() else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, local_files_only=args.local_files_only
    )
    candidate_samples, candidate_records = _fixed_length_samples(
        dataset_path,
        tokenizer,
        text_field=args.text_field,
        sequence_length=args.sequence_length,
        num_samples=args.num_samples + args.sample_offset,
    )
    samples = candidate_samples[args.sample_offset :]
    records = candidate_records[args.sample_offset :]
    if len(samples) != args.num_samples:
        raise RuntimeError(
            f"requested {args.num_samples} held-out samples, found {len(samples)}"
        )
    model = AutoModelForMultimodalLM.from_pretrained(
        model_source,
        dtype=_dtype(args.dtype),
        device_map={"": args.device},
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).eval()

    baseline = _evaluate(
        model, samples, batch_size=args.batch_size, device=args.device, label="dense"
    )
    results = [
        _serializable(
            baseline,
            variant="dense",
            baseline=None,
            metadata={
                "gdn_recurrent_state_compressed": False,
                "attention_cache_compressed": False,
                "gate_approximated": False,
            },
        )
    ]

    gdn_runtime = Qwen35PrivateAGRuntime(model, gdn_factors)
    with gdn_runtime:
        candidate = _evaluate(
            model,
            samples,
            batch_size=args.batch_size,
            device=args.device,
            label="gdn_private_ag",
        )
        gdn_metadata = _uniform_private_metadata(gdn_runtime.records)
    results.append(
        _serializable(
            candidate,
            variant="gdn_private_allgather_joint_decoder",
            baseline=baseline,
            metadata={
                "gdn": gdn_metadata,
                "gdn_recurrent_state_compressed": False,
                "attention_cache_compressed": False,
                "gate_approximated": False,
            },
        )
    )
    del candidate, gdn_runtime
    _cleanup()

    full_runtime = Qwen35FullAttentionPrivateAGRuntime(model, full_factors)
    with full_runtime:
        candidate = _evaluate(
            model,
            samples,
            batch_size=args.batch_size,
            device=args.device,
            label="full_private_ag",
        )
        full_metadata = _uniform_private_metadata(full_runtime.records)
    results.append(
        _serializable(
            candidate,
            variant="full_attention_private_allgather_joint_decoder",
            baseline=baseline,
            metadata={
                "full_attention": full_metadata,
                "gdn_recurrent_state_compressed": False,
                "attention_cache_compressed": False,
                "gate_approximated": False,
            },
        )
    )
    del candidate, full_runtime
    _cleanup()

    gdn_runtime = Qwen35PrivateAGRuntime(model, gdn_factors)
    full_runtime = Qwen35FullAttentionPrivateAGRuntime(model, full_factors)
    with gdn_runtime, full_runtime:
        candidate = _evaluate(
            model,
            samples,
            batch_size=args.batch_size,
            device=args.device,
            label="hybrid_private_ag",
        )
        gdn_metadata = _uniform_private_metadata(gdn_runtime.records)
        full_metadata = _uniform_private_metadata(full_runtime.records)
    results.append(
        _serializable(
            candidate,
            variant="all_attention_private_allgather_joint_decoder",
            baseline=baseline,
            metadata={
                "gdn": gdn_metadata,
                "full_attention": full_metadata,
                "attention_layers_installed": (
                    gdn_metadata["layers_installed"]
                    + full_metadata["layers_installed"]
                ),
                "gdn_recurrent_state_compressed": False,
                "attention_cache_compressed": False,
                "gate_approximated": False,
            },
        )
    )
    del candidate, gdn_runtime, full_runtime
    _cleanup()

    for keep_ratio in topk_ratios:
        gdn_runtime = Qwen35PrivateAGRuntime(model, gdn_factors)
        topk_runtime = Qwen35PostGateTopKRuntime(
            model,
            intervention="full",
            keep_ratio=keep_ratio,
            selection_scope="source_local",
            tp_size=args.tp,
            profile=True,
        )
        with gdn_runtime, topk_runtime:
            candidate = _evaluate(
                model,
                samples,
                batch_size=args.batch_size,
                device=args.device,
                label=f"gdn_private_full_topk_{keep_ratio:g}",
            )
            gdn_metadata = _uniform_private_metadata(gdn_runtime.records)
            topk_records = topk_runtime.records
            profile_summaries, _ = topk_runtime.profile_snapshot()
        if not topk_records:
            raise RuntimeError("full-attention Top-K runtime installed no layers")
        local_width = topk_records[0].width // args.tp
        kept_per_source = topk_records[0].kept_per_source
        if kept_per_source is None or any(
            record.kept_per_source != kept_per_source for record in topk_records
        ):
            raise ValueError("full-attention Top-K uses inconsistent source packets")
        mask_bytes = math.ceil(local_width / 8)
        packet_bytes = mask_bytes + 2 * kept_per_source
        results.append(
            _serializable(
                candidate,
                variant="gdn_private_ag_plus_full_postgate_topk",
                baseline=baseline,
                metadata={
                    "gdn": gdn_metadata,
                    "full_attention_topk": {
                        "layers_installed": len(topk_records),
                        "layer_indices": [record.layer_index for record in topk_records],
                        "keep_ratio": keep_ratio,
                        "selection_scope": "source_local",
                        "local_width": local_width,
                        "kept_per_source": kept_per_source,
                        "bitmask_bytes_per_source": mask_bytes,
                        "selected_bf16_value_bytes_per_source": 2 * kept_per_source,
                        "bitmask_packet_bytes_per_source": packet_bytes,
                        "profile": profile_summaries,
                        "quality_oracle_dense_zero_fill": True,
                    },
                    "gdn_recurrent_state_compressed": False,
                    "attention_cache_compressed": False,
                    "gate_approximated": False,
                },
            )
        )
        del candidate, gdn_runtime, topk_runtime
        _cleanup()

    device = torch.device(args.device)
    cuda_index = device.index if device.index is not None else 0
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_source,
        "gdn_private_factors": str(gdn_path),
        "full_private_factors": str(full_path),
        "dataset_jsonl": str(dataset_path),
        "records": list(records),
        "num_samples": args.num_samples,
        "sample_offset": args.sample_offset,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "evaluated_tokens_per_variant": args.num_samples * (args.sequence_length - 1),
        "dtype": args.dtype,
        "device": args.device,
        "results": results,
        "environment": {
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": str(torch.__version__),
            "cuda_device_name": torch.cuda.get_device_name(cuda_index),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_path)
    print(f"[Saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
