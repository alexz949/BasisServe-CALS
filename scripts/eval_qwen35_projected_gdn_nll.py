#!/usr/bin/env python3
"""Evaluate Qwen3.5 projected GDN state ranks on frozen JSONL windows."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen35_gdn_runtime import (
    Qwen35ProjectedGDNRuntime,
    load_qwen35_gdn_spectra,
)
from scripts.collect_qwen35_gdn_moments import _batches, _fixed_length_samples


FORMAT = "basisserve.qwen35.projected_gdn_nll.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ranks", default="32,64,96,128")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="skip this many eligible fixed-length windows before evaluation",
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _ranks(raw: str, maximum: int) -> tuple[int, ...]:
    values = tuple(sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()}))
    if not values or values[0] <= 0 or values[-1] > maximum:
        raise ValueError(f"ranks must lie in [1,{maximum}]")
    return values


def _chunked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    maximum_rows: int = 512,
) -> Tensor:
    """Compute FP32 token NLL without materializing all FP32 logits at once."""

    if logits.shape[:-1] != targets.shape:
        raise ValueError("logit and target token shapes do not match")
    if maximum_rows <= 0:
        raise ValueError("maximum loss rows must be positive")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    pieces = [
        F.cross_entropy(
            flat_logits[start : start + maximum_rows].float(),
            flat_targets[start : start + maximum_rows],
            reduction="none",
        )
        for start in range(0, flat_targets.numel(), maximum_rows)
    ]
    return torch.cat(pieces).reshape_as(targets)


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    samples: Sequence[Tensor],
    *,
    batch_size: int,
    device: str,
    label: str,
) -> dict[str, Any]:
    window_nlls: list[float] = []
    token_nlls: list[Tensor] = []
    top1_tokens: list[Tensor] = []
    started = time.perf_counter()
    for batch_index, input_ids in enumerate(_batches(samples, batch_size), start=1):
        input_ids = input_ids.to(device)
        logits = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        ).logits[:, :-1]
        targets = input_ids[:, 1:]
        losses = _chunked_cross_entropy(logits, targets)
        if not torch.isfinite(losses).all():
            raise FloatingPointError(f"{label} produced non-finite token NLL")
        window_nlls.extend(float(value) for value in losses.mean(dim=1).cpu())
        token_nlls.append(losses.cpu())
        top1_tokens.append(logits.argmax(dim=-1).cpu())
        print(
            f"[NLL] variant={label} batch={batch_index} "
            f"samples={min(batch_index * batch_size, len(samples))}/{len(samples)} "
            f"mean={losses.mean().item():.8f}",
            flush=True,
        )
        del logits, losses
    elapsed = time.perf_counter() - started
    tokens = torch.cat([value.reshape(-1) for value in token_nlls])
    top1 = torch.cat([value.reshape(-1) for value in top1_tokens])
    mean_nll = float(tokens.mean())
    return {
        "label": label,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "window_nlls": window_nlls,
        "token_nlls": tokens,
        "top1_tokens": top1,
        "elapsed_seconds": elapsed,
        "evaluated_tokens": int(tokens.numel()),
    }


def _serializable_result(
    result: dict[str, Any],
    *,
    baseline: dict[str, Any] | None,
    rank: int | None,
    state_fraction: float,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"token_nlls", "top1_tokens"}
    }
    payload["rank"] = rank
    payload["state_fraction"] = state_fraction
    payload["state_reduction_fraction"] = 1.0 - state_fraction
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
                "paired_window_standard_error": (
                    statistics.stdev(window_differences) / math.sqrt(len(window_differences))
                    if len(window_differences) > 1
                    else 0.0
                ),
                "target_nll_mae": float(differences.abs().mean()),
                "target_nll_rmse": float(differences.square().mean().sqrt()),
                "target_nll_max_abs": float(differences.abs().max()),
                "top1_agreement": float(
                    (result["top1_tokens"] == baseline["top1_tokens"]).float().mean()
                ),
            }
        )
    return payload


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if args.sample_offset < 0:
        raise ValueError("sample offset must be nonnegative")
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite projected GDN result: {output_path}")
    dataset_path = Path(args.dataset_jsonl).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    spectra_path = Path(args.spectra).expanduser().resolve()
    spectra = load_qwen35_gdn_spectra(spectra_path)
    value_dim = int(spectra["geometry"]["value_head_dim"])
    ranks = _ranks(args.ranks, value_dim)

    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    model_path = Path(args.model_path).expanduser().resolve()
    model_source = str(model_path) if model_path.exists() else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
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
    model = AutoModelForMultimodalLM.from_pretrained(
        model_source,
        dtype=_dtype(args.dtype),
        device_map={"": args.device},
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).eval()

    baseline = _evaluate(
        model,
        samples,
        batch_size=args.batch_size,
        device=args.device,
        label="baseline",
    )
    results = [
        _serializable_result(
            baseline,
            baseline=None,
            rank=None,
            state_fraction=1.0,
        )
    ]
    for rank in ranks:
        runtime = Qwen35ProjectedGDNRuntime(
            model,
            spectra,
            rank=rank,
            signal="core",
        )
        with runtime:
            print(
                f"[Install] rank={rank} layers={len(runtime.records)} "
                f"state_fraction={rank / value_dim:.4f}",
                flush=True,
            )
            candidate = _evaluate(
                model,
                samples,
                batch_size=args.batch_size,
                device=args.device,
                label=f"rank_{rank}",
            )
        summary = _serializable_result(
            candidate,
            baseline=baseline,
            rank=rank,
            state_fraction=rank / value_dim,
        )
        results.append(summary)
        print(
            f"[RankDone] rank={rank} nll={summary['mean_nll']:.9f} "
            f"delta={summary['delta_mean_nll']:+.9f} "
            f"top1={summary['top1_agreement']:.6f}",
            flush=True,
        )

    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "model": model_source,
        "spectra": str(spectra_path),
        "dataset_jsonl": str(dataset_path),
        "records": list(records),
        "num_samples": args.num_samples,
        "sample_offset": args.sample_offset,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "evaluated_tokens_per_variant": args.num_samples * (args.sequence_length - 1),
        "dtype": args.dtype,
        "device": args.device,
        "ranks": list(ranks),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(f"[Saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
