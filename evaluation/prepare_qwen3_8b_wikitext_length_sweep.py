#!/usr/bin/env python3
"""Prepare disjoint WikiText-2 windows for C1+K-routing evaluation."""

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

from datasets import load_dataset
from safetensors.torch import save_file
import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FORMAT = "basisserve.qwen3_8b.c1_k_routing_sidecar.wikitext_windows.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"format": FORMAT})
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> None:
    if min(args.samples, args.sequence_length) <= 0 or args.start_token < 0:
        raise ValueError("samples/length must be positive and start nonnegative")
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=True,
    )
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split=args.split,
        cache_dir=str(args.dataset_cache.expanduser().resolve()),
        download_mode="reuse_dataset_if_exists",
    )
    text = "\n\n".join(str(row["text"]) for row in dataset)
    input_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]
    required = args.start_token + args.samples * args.sequence_length
    if int(input_ids.numel()) < required:
        raise ValueError(
            f"WikiText {args.split} has {input_ids.numel()} tokens, requires {required}"
        )
    rows = []
    records = []
    for sample in range(args.samples):
        start = args.start_token + sample * args.sequence_length
        stop = start + args.sequence_length
        row = input_ids[start:stop].to(dtype=torch.int32).contiguous()
        rows.append(row)
        records.append(
            {
                "sample_index": sample,
                "token_start": start,
                "token_stop": stop,
                "input_ids_sha256": _tensor_sha256(row),
            }
        )
    windows = torch.stack(rows)
    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "windows.safetensors"
    _atomic_safetensors(artifact_path, {"input_ids": windows})
    manifest = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "dataset": {
            "repo": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": args.split,
            "cache_dir": str(args.dataset_cache.expanduser().resolve()),
            "tokenized_total": int(input_ids.numel()),
        },
        "sampling": {
            "samples": args.samples,
            "sequence_length": args.sequence_length,
            "start_token": args.start_token,
            "disjoint": True,
            "nested_prefix_lengths": [
                length
                for length in (1024, 2048, 4096, 8192, 16384, 32768)
                if length <= args.sequence_length
            ],
        },
        "records": records,
        "artifact": {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "tensor": "input_ids",
            "shape": list(windows.shape),
            "dtype": str(windows.dtype),
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    print(f"[C1+R windows] wrote {artifact_path} shape={tuple(windows.shape)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=REPO_ROOT / "results/cache/huggingface/datasets",
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--start-token", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    prepare(_parser().parse_args())
