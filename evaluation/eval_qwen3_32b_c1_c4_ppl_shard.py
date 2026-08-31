#!/usr/bin/env python3
"""Evaluate a subset of dense/C1 Qwen3-32B arms on C4-validation PPL."""

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
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
from torch import Tensor, nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import allocate_qwen3_32b_c1_tp_source_global_kl as common  # noqa: E402
from evaluation import run_qwen3_32b_c1_tp_source_global_kl_sharded as runtime  # noqa: E402
from evaluation.eval_qwen3_32b_c1_wikitext import (  # noqa: E402
    _atomic_json,
    _decoder_layers,
    _load_layer_allocation_results,
    _load_results,
    _sha256,
    install_c1_factors,
    install_layer_allocation_schedule,
    load_layer_allocation_reconstruction_state,
)


FORMAT = "basisserve.qwen3_32b.gqa_c1.c4_validation_ppl_shard.v1"
WINDOWS_FORMAT = "basisserve.calibration.c4_document_windows.v1"
ALL_ARMS = ("dense", "uniform_anchor", "mean_dp", "ucb_dp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--allocation-dir", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--arm", action="append", choices=ALL_ARMS, required=True)
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


def _load_windows(path: Path, *, model_path: Path) -> tuple[Tensor, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    manifest_path = resolved.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != WINDOWS_FORMAT:
        raise ValueError("incompatible C4 window manifest")
    if _sha256(resolved) != manifest["artifact"]["sha256"]:
        raise ValueError("C4-validation window artifact hash mismatch")
    if manifest["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("C4-validation windows belong to another model config")
    if manifest["dataset"]["split"] != "validation":
        raise ValueError("formal C4 PPL requires the validation split")
    if manifest["sampling"]["samples"] != 128:
        raise ValueError("formal C4 PPL requires exactly 128 documents")
    if manifest["sampling"]["sequence_length"] != 2048:
        raise ValueError("formal C4 PPL requires complete 2048-token windows")
    records = manifest.get("records", ())
    document_ids = [str(row["document_id"]) for row in records]
    if len(records) != 128 or len(document_ids) != len(set(document_ids)):
        raise ValueError("C4 PPL documents are missing or duplicated")
    payload = load_file(str(resolved), device="cpu")
    if set(payload) != {"input_ids"}:
        raise ValueError("C4 PPL bank must contain only input_ids")
    input_ids = payload["input_ids"].to(torch.long)
    if tuple(input_ids.shape) != (128, 2048):
        raise ValueError("C4 PPL tensor has an incompatible shape")
    provenance = {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "dataset": manifest["dataset"],
        "sampling": manifest["sampling"],
        "document_ids": document_ids,
    }
    return input_ids.contiguous(), provenance


def _document_nll_from_logits(logits: Tensor, input_ids: Tensor) -> Tensor:
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("C4 PPL expects batched logits and token ids")
    shift_logits = logits[:, :-1].float().contiguous()
    shift_labels = input_ids[:, 1:].to(shift_logits.device).contiguous()
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(len(input_ids), -1)
    return losses.mean(dim=1)


@torch.inference_mode()
def _evaluate_document_ppl(
    model: nn.Module,
    sequences: Tensor,
    *,
    batch_size: int,
    label: str,
) -> dict[str, Any]:
    input_device = model.get_input_embeddings().weight.device
    document_nll = []
    for start in range(0, len(sequences), batch_size):
        input_ids = sequences[start : start + batch_size].to(input_device)
        logits = model(input_ids=input_ids, use_cache=False).logits
        values = _document_nll_from_logits(logits, input_ids)
        if not bool(torch.isfinite(values).all().cpu()):
            raise FloatingPointError(f"non-finite C4 NLL for {label} at document {start}")
        document_nll.extend(map(float, values.cpu().tolist()))
        stop = min(start + batch_size, len(sequences))
        if stop % 16 == 0 or stop == len(sequences):
            print(f"[C4 PPL] {label} documents={stop}/{len(sequences)}", flush=True)
        del input_ids, logits, values
    paired = common._paired(document_nll)
    token_count = len(sequences) * (int(sequences.shape[1]) - 1)
    nll_sum = sum(document_nll) * (int(sequences.shape[1]) - 1)
    return {
        "documents": len(sequences),
        "tokens": token_count,
        "nll_sum": nll_sum,
        "mean_nll": nll_sum / token_count,
        "ppl": math.exp(nll_sum / token_count),
        "document_nll": paired,
        "loss_dtype": "float32",
    }


def _validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    arms = tuple(args.arm)
    if len(arms) != len(set(arms)):
        raise ValueError("C4 PPL arms must be unique")
    if arms != tuple(arm for arm in ALL_ARMS if arm in arms):
        raise ValueError("C4 PPL arms must follow dense/uniform/mean/UCB order")
    if args.batch_size <= 0 or args.torch_num_threads <= 0:
        raise ValueError("batch size and thread count must be positive")
    return arms


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    arms = _validate_args(args)
    torch.set_num_threads(args.torch_num_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("C4 PPL evaluation requires CUDA")
    torch.cuda.set_device(0)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    started = time.perf_counter()
    model_path = Path(args.model).expanduser().resolve()
    allocation_dir = args.allocation_dir.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allocation = _load_layer_allocation_results(allocation_dir, model_path)
    sequences, provenance = _load_windows(args.windows, model_path=model_path)
    model = runtime._load_model(args, model_path)
    dense_v_weights = tuple(
        layer.self_attn.v_proj.weight.detach().cpu().clone()
        for layer in _decoder_layers(model)
    )
    reconstruction_state = None
    if "mean_dp" in arms:
        reconstruction_state = load_layer_allocation_reconstruction_state(
            allocation,
            model_path=model_path,
        )

    arm_rows = {}
    for arm in arms:
        arm_started = time.perf_counter()
        if arm == "dense":
            installation: Sequence[Mapping[str, Any]] = ()
        elif arm == "uniform_anchor":
            source = allocation["factor_sources"]["64"]
            factor_dir = Path(source["path"]).expanduser().resolve()
            if _sha256(factor_dir / "results.json") != source["results_sha256"]:
                raise ValueError("uniform rank-64 factor result hash mismatch")
            factor_result = _load_results(factor_dir, model_path)
            installation = install_c1_factors(model, factor_dir, factor_result)
        else:
            installation = install_layer_allocation_schedule(
                model,
                allocation_dir,
                allocation,
                schedule_name=arm,
                dense_v_weights=dense_v_weights,
                reconstruction_state=reconstruction_state,
            )
        metrics = _evaluate_document_ppl(
            model,
            sequences,
            batch_size=args.batch_size,
            label=arm,
        )
        arm_rows[arm] = {
            "metrics": metrics,
            "installation": list(installation),
            "elapsed_seconds": time.perf_counter() - arm_started,
        }
        print(f"[C4 PPL] {arm} ppl={metrics['ppl']:.9f}", flush=True)
        torch.cuda.empty_cache()

    allocation_path = allocation_dir / "result.json"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "allocation": {
            "path": str(allocation_path),
            "sha256": _sha256(allocation_path),
            "factor_stage": allocation["profile"]["factor_stage"],
        },
        "windows": provenance,
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
        "arms": arm_rows,
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
        },
    }
    _atomic_json(output_path, payload)
    print(f"[C4 PPL] wrote {output_path}", flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
