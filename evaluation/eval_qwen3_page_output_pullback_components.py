#!/usr/bin/env python3
"""Report output-pullback and Page-Fisher components without mixing them."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors.torch import load_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fit_qwen3_page_output_pullback_router import (  # noqa: E402
    _load_metric,
    _metrics,
    _parse_layers,
)


FORMAT = "basisserve.qwen3.page_output_pullback_component_evaluation.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kq-r32-init", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument(
        "--work-dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"layers": len(records)}
    for split in ("fit", "validation"):
        result[split] = {}
        for component in ("output", "page_fisher"):
            initial = [
                float(row[split][component]["initial_relative_loss"])
                for row in records
            ]
            fitted = [
                float(row[split][component]["fitted_relative_loss"])
                for row in records
            ]
            result[split][component] = {
                "initial_mean": sum(initial) / len(initial),
                "fitted_mean": sum(fitted) / len(fitted),
                "fitted_over_initial_mean": sum(
                    final / start for start, final in zip(initial, fitted)
                )
                / len(initial),
                "layers_improved": sum(
                    final < start for start, final in zip(initial, fitted)
                ),
            }
    return result


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    started = time.monotonic()
    statistics_root = Path(args.statistics_dir).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint).expanduser().resolve()
    init_root = Path(args.kq_r32_init).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    checkpoint_manifest = json.loads(
        (checkpoint_root / "result.json").read_text(encoding="utf-8")
    )
    init_manifest = json.loads(
        (init_root / "result.json").read_text(encoding="utf-8")
    )
    init_tensors = load_file(
        str(init_root / init_manifest["artifacts"]["factors"]["file"]),
        device="cpu",
    )
    key_bank = init_tensors["kq_svd_key_projector"]
    query_bank = init_tensors["kq_svd_query_projector"]
    layers = _parse_layers(
        args.layers,
        int(checkpoint_manifest["geometry"]["layers"]),
    )
    device = torch.device(args.work_device)
    dtype = torch.float64 if args.work_dtype == "float64" else torch.float32
    records: list[dict[str, Any]] = []

    for ordinal, layer in enumerate(layers, start=1):
        print(
            f"[Page output components] layer={layer} ({ordinal}/{len(layers)})",
            flush=True,
        )
        checkpoint_artifact = checkpoint_manifest["artifacts"][str(layer)]
        factors = load_file(
            str(checkpoint_root / checkpoint_artifact["file"]),
            device="cpu",
        )
        fitted_encoder = factors["key_routing_encoders"].to(
            device=device,
            dtype=dtype,
        )
        fitted_query = factors["query_routing_factors"].to(
            device=device,
            dtype=dtype,
        )
        rank = int(fitted_encoder.shape[-1])
        initial_encoder = key_bank[layer, :, :, :rank].to(
            device=device,
            dtype=dtype,
        )
        initial_query = query_bank[layer, :, :, :rank].to(
            device=device,
            dtype=dtype,
        )
        if initial_query.shape[0] != fitted_query.shape[0]:
            initial_query = initial_query.repeat_interleave(
                fitted_query.shape[0] // initial_query.shape[0],
                dim=0,
            )
        record: dict[str, Any] = {"layer": layer}
        for split in ("fit", "validation"):
            split_root = statistics_root / split
            manifest = json.loads(
                (split_root / "manifest.json").read_text(encoding="utf-8")
            )
            record[split] = {}
            for component, weights in {
                "output": (1.0, 0.0),
                "page_fisher": (0.0, 1.0),
            }.items():
                statistics = _load_metric(
                    split_root,
                    manifest,
                    layer,
                    output_weight=weights[0],
                    page_fisher_weight=weights[1],
                    component_normalization="teacher_energy",
                    device=device,
                    dtype=dtype,
                )
                initial = _metrics(
                    statistics,
                    encoders=initial_encoder,
                    query_factors=initial_query,
                )
                fitted = _metrics(
                    statistics,
                    encoders=fitted_encoder,
                    query_factors=fitted_query,
                )
                record[split][component] = {
                    "initial_relative_loss": initial["relative_loss"],
                    "fitted_relative_loss": fitted["relative_loss"],
                    "fitted_over_initial": (
                        float(fitted["relative_loss"])
                        / float(initial["relative_loss"])
                    ),
                }
                del statistics
        records.append(record)

    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(checkpoint_root),
        "statistics": str(statistics_root),
        "layer_coverage": list(layers),
        "aggregate": _aggregate(records),
        "records": records,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(f"[Page output components] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
