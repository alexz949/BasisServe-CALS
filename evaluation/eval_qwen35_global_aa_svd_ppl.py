#!/usr/bin/env python3
"""Evaluate Qwen3.5 global AA-SVD AllReduce factors on frozen PPL windows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen35_global_aa_svd import (  # noqa: E402
    Qwen35GlobalAASVDRecord,
    Qwen35GlobalAASVDRuntime,
    load_qwen35_global_aa_svd_factors,
)
from evaluation.eval_qwen35_hybrid_private_ag_ppl import (  # noqa: E402
    _serializable,
)
from scripts.collect_qwen35_gdn_wo_activations import (  # noqa: E402
    _load_window_split,
)
from scripts.eval_qwen35_projected_gdn_nll import (  # noqa: E402
    _dtype,
    _evaluate,
)


FORMAT = "basisserve.qwen35.global_aa_svd_allreduce_ppl.v1"
NUM_LAYERS = 32
EXPECTED_GDN = tuple(layer for layer in range(NUM_LAYERS) if layer % 4 != 3)
EXPECTED_FULL = tuple(layer for layer in range(NUM_LAYERS) if layer % 4 == 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--factors", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sample-offset", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _validate_factors(
    factors: Mapping[str, Any],
    *,
    model_path: Path,
    tp_size: int,
) -> None:
    if Path(factors.get("model_path", "")).resolve() != model_path:
        raise ValueError("global AA-SVD factors belong to another model")
    if int(factors.get("tp_size", -1)) != tp_size:
        raise ValueError("global AA-SVD factors use another TP size")
    layers = factors.get("layers", ())
    indices = tuple(int(layer["layer_index"]) for layer in layers)
    if indices != tuple(range(NUM_LAYERS)):
        raise ValueError("global AA-SVD factors do not cover all 32 layers")
    gdn = tuple(
        int(layer["layer_index"])
        for layer in layers
        if layer["block_type"] == "linear_attention"
    )
    full = tuple(
        int(layer["layer_index"])
        for layer in layers
        if layer["block_type"] == "full_attention"
    )
    if gdn != EXPECTED_GDN or full != EXPECTED_FULL:
        raise ValueError("global AA-SVD block-type schedule is malformed")
    ranks = {int(layer["input_factor"].shape[1]) for layer in layers}
    if ranks != {int(factors.get("global_rank", -1))}:
        raise ValueError("global AA-SVD layers use inconsistent ranks")


def _subset(
    factors: Mapping[str, Any],
    block_types: Sequence[str],
) -> dict[str, Any]:
    allowed = set(block_types)
    selected = [
        layer for layer in factors["layers"] if str(layer["block_type"]) in allowed
    ]
    if not selected:
        raise ValueError("global AA-SVD subset contains no layers")
    return {**factors, "layers": selected}


def _runtime_metadata(
    records: tuple[Qwen35GlobalAASVDRecord, ...],
    factors: Mapping[str, Any],
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("global AA-SVD runtime installed no layers")
    ranks = {record.global_rank for record in records}
    tp_sizes = {record.tp_size for record in records}
    input_widths = {record.input_width for record in records}
    if len(ranks) != 1 or len(tp_sizes) != 1 or len(input_widths) != 1:
        raise ValueError("global AA-SVD runtime geometry is inconsistent")
    rank = next(iter(ranks))
    width = next(iter(input_widths))
    return {
        "layers_installed": len(records),
        "layer_indices": [record.layer_index for record in records],
        "block_types": [record.block_type for record in records],
        "tp_size": next(iter(tp_sizes)),
        "input_width": width,
        "global_rank": rank,
        "collective": "common_code_allreduce",
        "single_gpu_tp_math_simulation": True,
        "communication_fraction_of_dense_allreduce": rank / width,
        "communication_reduction_fraction": 1.0 - rank / width,
        "checkpoint_communication": dict(factors["communication"]),
    }


def _cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if min(args.tp, args.num_samples, args.batch_size) <= 0:
        raise ValueError("TP size, sample count, and batch size must be positive")
    model_path = Path(args.model_path).expanduser().resolve()
    factor_path = Path(args.factors).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite PPL result: {output_path}")
    factors = load_qwen35_global_aa_svd_factors(factor_path)
    _validate_factors(factors, model_path=model_path, tp_size=args.tp)
    samples, window_records, windows_manifest = _load_window_split(
        windows_path,
        sample_offset=args.sample_offset,
        num_samples=args.num_samples,
    )
    samples = samples.long()

    from transformers import AutoModelForMultimodalLM

    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_path),
        dtype=_dtype(args.dtype),
        device_map={"": args.device},
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).eval()
    dense = _evaluate(
        model,
        samples,
        batch_size=args.batch_size,
        device=args.device,
        label="dense",
    )
    results = [
        _serializable(
            dense,
            variant="dense",
            baseline=None,
            metadata={
                "attention_output_projection_compressed": False,
                "attention_cache_compressed": False,
                "gdn_recurrent_state_compressed": False,
            },
        )
    ]
    variants = (
        ("gdn_global_aa_svd_allreduce", ("linear_attention",)),
        ("full_attention_global_aa_svd_allreduce", ("full_attention",)),
        (
            "all_attention_global_aa_svd_allreduce",
            ("linear_attention", "full_attention"),
        ),
    )
    for label, block_types in variants:
        runtime = Qwen35GlobalAASVDRuntime(model, _subset(factors, block_types))
        with runtime:
            candidate = _evaluate(
                model,
                samples,
                batch_size=args.batch_size,
                device=args.device,
                label=label,
            )
            metadata = _runtime_metadata(runtime.records, factors)
        results.append(
            _serializable(
                candidate,
                variant=label,
                baseline=dense,
                metadata={
                    "global_aa_svd": metadata,
                    "attention_cache_compressed": False,
                    "gdn_recurrent_state_compressed": False,
                },
            )
        )
        del candidate, runtime
        _cleanup()

    device = torch.device(args.device)
    cuda_index = device.index if device.index is not None else 0
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "factors": str(factor_path),
        "windows": str(
            windows_path / "windows.safetensors"
            if windows_path.is_dir()
            else windows_path
        ),
        "windows_sha256": windows_manifest["artifact"]["sha256"],
        "records": list(window_records),
        "num_samples": args.num_samples,
        "sample_offset": args.sample_offset,
        "sequence_length": int(samples.shape[1]),
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "tp_size": args.tp,
        "results": results,
        "environment": {
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": str(torch.__version__),
            "cuda_device": torch.cuda.get_device_name(cuda_index),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_index)),
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
