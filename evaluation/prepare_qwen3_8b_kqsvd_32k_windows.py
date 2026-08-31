#!/usr/bin/env python3
"""Pack document-disjoint C4 calibration windows without changing tokens."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor


FORMAT = "basisserve.calibration.c4_packed_windows.v1"
SOURCE_SAMPLES = 128
SOURCE_SEQUENCE_LENGTH = 2048
TARGET_SAMPLES = 8
TARGET_SEQUENCE_LENGTH = 32768


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--source-samples", type=int, default=SOURCE_SAMPLES)
    parser.add_argument("--source-sequence-length", type=int, default=SOURCE_SEQUENCE_LENGTH)
    parser.add_argument("--target-samples", type=int, default=TARGET_SAMPLES)
    parser.add_argument("--target-sequence-length", type=int, default=TARGET_SEQUENCE_LENGTH)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.source_start < 0 or min(
        args.source_samples,
        args.source_sequence_length,
        args.target_samples,
        args.target_sequence_length,
    ) <= 0:
        raise ValueError("packing offsets and dimensions must be valid")
    if args.source_samples % args.target_samples:
        raise ValueError("source samples must divide evenly across target windows")
    if (
        args.source_samples * args.source_sequence_length
        != args.target_samples * args.target_sequence_length
    ):
        raise RuntimeError("source and target token counts differ")

    model_path = args.model.expanduser().resolve()
    source_path = args.source_windows.expanduser().resolve()
    source_manifest_path = source_path.parent / "manifest.json"
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    config_path = model_path / "config.json"
    if source_manifest["model"]["config_sha256"] != _sha256(config_path):
        raise ValueError("source windows belong to another model config")
    if source_manifest["artifact"]["sha256"] != _sha256(source_path):
        raise ValueError("source window hash does not match its manifest")
    if (
        source_manifest["dataset"]["repo"] != "allenai/c4"
        or source_manifest["dataset"]["split"] != "train"
        or source_manifest["sampling"]["sequence_length"]
        != args.source_sequence_length
        or not source_manifest["sampling"]["document_disjoint"]
    ):
        raise ValueError("source is not the expected document-disjoint C4-train bank")

    source = load_file(str(source_path), device="cpu")["input_ids"]
    source_stop = args.source_start + args.source_samples
    if (
        tuple(source.shape[1:]) != (args.source_sequence_length,)
        or source_stop > len(source)
    ):
        raise ValueError("source window tensor has incompatible geometry")
    selected = source[args.source_start:source_stop].to(torch.int32).contiguous()
    packed = selected.reshape(
        args.target_samples,
        args.target_sequence_length,
    ).contiguous()

    source_records = source_manifest["records"][args.source_start:source_stop]
    document_ids = [str(record["document_id"]) for record in source_records]
    if len(document_ids) != len(set(document_ids)):
        raise RuntimeError("selected source windows are not document-disjoint")
    records = []
    documents_per_target = args.source_samples // args.target_samples
    for target_index in range(args.target_samples):
        first = target_index * documents_per_target
        stop = first + documents_per_target
        records.append(
            {
                "sample_index": target_index,
                "source_sample_indices": list(
                    range(args.source_start + first, args.source_start + stop)
                ),
                "source_document_ids": document_ids[first:stop],
                "input_ids_sha256": _tensor_sha256(packed[target_index]),
            }
        )

    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "windows.safetensors"
    _atomic_safetensors(artifact_path, {"input_ids": packed})
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(config_path),
        },
        "dataset": source_manifest["dataset"],
        "packing": {
            "policy": "row-major concatenation without inserted separator tokens",
            "source_file": str(source_path),
            "source_sha256": _sha256(source_path),
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "source_start": args.source_start,
            "source_samples": args.source_samples,
            "source_sequence_length": args.source_sequence_length,
            "source_documents_per_target": documents_per_target,
            "target_samples": args.target_samples,
            "target_sequence_length": args.target_sequence_length,
            "token_content_preserved": bool(
                torch.equal(selected.reshape(-1), packed.reshape(-1))
            ),
        },
        "records": records,
        "artifact": {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "tensor": "input_ids",
            "dtype": str(packed.dtype),
            "shape": list(packed.shape),
        },
    }
    _atomic_json(output_dir / "manifest.json", payload)
    print(
        f"[packed calibration windows] wrote {artifact_path} "
        f"shape={tuple(packed.shape)} from "
        f"{args.source_samples}x{args.source_sequence_length} source tokens",
        flush=True,
    )


if __name__ == "__main__":
    main()
