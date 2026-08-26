#!/usr/bin/env python3
"""Prepare document-disjoint C4 windows for DeepSeek-V2-Lite C1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import sys
import time
from typing import Any, Mapping

from datasets import load_dataset
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file
import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer


FORMAT = "basisserve.deepseek_v2_lite.c1_c4_document_windows.v1"
MODEL_REPO = "deepseek-ai/DeepSeek-V2-Lite"
MODEL_TYPE = "deepseek_v2"
NUM_LAYERS = 27
HIDDEN_SIZE = 2048
NUM_ATTENTION_HEADS = 16
VALUE_HEAD_DIM = 128
FIT_WINDOWS = 256
VALIDATION_WINDOWS = 64
SEQUENCE_LENGTH = 2048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _document_id(row: Mapping[str, Any], stream_index: int) -> str:
    for key in ("id", "url", "timestamp"):
        value = row.get(key)
        if value:
            return f"{key}:{value}"
    digest = hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:24]}:stream:{stream_index}"


def _snapshot_revision(model_path: Path) -> str | None:
    if model_path.parent.name != "snapshots":
        return None
    revision = model_path.name.lower()
    if len(revision) == 40 and all(character in "0123456789abcdef" for character in revision):
        return revision
    return None


def _validate_config(config: Any) -> None:
    observed = (
        str(config.model_type),
        int(config.num_hidden_layers),
        int(config.hidden_size),
        int(config.num_attention_heads),
        int(config.v_head_dim),
    )
    expected = (
        MODEL_TYPE,
        NUM_LAYERS,
        HIDDEN_SIZE,
        NUM_ATTENTION_HEADS,
        VALUE_HEAD_DIM,
    )
    if observed != expected:
        raise ValueError(f"unexpected DeepSeek-V2-Lite geometry: {observed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-windows", type=int, default=FIT_WINDOWS)
    parser.add_argument(
        "--validation-windows", type=int, default=VALIDATION_WINDOWS
    )
    parser.add_argument("--profile-windows", type=int, default=0)
    parser.add_argument("--confirmation-windows", type=int, default=0)
    parser.add_argument(
        "--prefix-windows",
        help="Optional existing input_ids bank that the generated prefix must match",
    )
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument(
        "--dataset-revision",
        default="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--dataset-cache", default="results/cache/huggingface/datasets"
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        min(args.fit_windows, args.validation_windows) <= 0
        or min(args.profile_windows, args.confirmation_windows) < 0
        or bool(args.profile_windows) != bool(args.confirmation_windows)
        or args.sequence_length != SEQUENCE_LENGTH
        or args.shuffle_buffer
        < args.fit_windows
        + args.validation_windows
        + args.profile_windows
        + args.confirmation_windows
    ):
        raise ValueError("invalid DeepSeek C1 calibration-window configuration")
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("DeepSeek model snapshot is incomplete")
    config = AutoConfig.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    _validate_config(config)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )

    resolved_dataset_revision = HfApi().dataset_info(
        "allenai/c4", revision=args.dataset_revision
    ).sha
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split=args.dataset_split,
        streaming=True,
        revision=resolved_dataset_revision,
        cache_dir=args.dataset_cache,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    total = (
        args.fit_windows
        + args.validation_windows
        + args.profile_windows
        + args.confirmation_windows
    )
    generator = random.Random(args.seed)
    seen_ids: set[str] = set()
    windows: list[Tensor] = []
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for stream_index, row in enumerate(dataset):
        document_id = _document_id(row, stream_index)
        if document_id in seen_ids:
            continue
        input_ids = tokenizer(
            str(row["text"]), return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        if input_ids.numel() < args.sequence_length:
            continue
        token_start = generator.randint(0, input_ids.numel() - args.sequence_length)
        window = input_ids[token_start : token_start + args.sequence_length].to(
            torch.int32
        )
        sample_index = len(windows)
        windows.append(window.cpu().contiguous())
        if sample_index < args.fit_windows:
            split = "fit"
        elif sample_index < args.fit_windows + args.validation_windows:
            split = "validation"
        elif sample_index < total - args.confirmation_windows:
            split = "global_kl_profile"
        else:
            split = "global_kl_confirmation"
        records.append(
            {
                "sample_index": sample_index,
                "split": split,
                "document_id": document_id,
                "stream_index": stream_index,
                "token_start": token_start,
                "document_token_count": int(input_ids.numel()),
                "input_ids_sha256": _tensor_sha256(window),
            }
        )
        seen_ids.add(document_id)
        if len(windows) == total:
            break
        if len(windows) % 32 == 0:
            print(f"[DeepSeek windows] collected={len(windows)}/{total}", flush=True)
    if len(windows) != total:
        raise RuntimeError(f"C4 stream yielded only {len(windows)} usable documents")
    if args.prefix_windows:
        prefix_path = Path(args.prefix_windows).expanduser().resolve()
        prefix = load_file(str(prefix_path), device="cpu")
        if set(prefix) != {"input_ids"}:
            raise ValueError("prefix window bank must contain only input_ids")
        prefix_ids = prefix["input_ids"].to(torch.int32)
        if (
            prefix_ids.ndim != 2
            or int(prefix_ids.shape[1]) != args.sequence_length
            or len(prefix_ids) > len(windows)
        ):
            raise ValueError("prefix window geometry is incompatible")
        if not torch.equal(torch.stack(windows[: len(prefix_ids)]), prefix_ids):
            raise RuntimeError("generated C4 prefix differs from the existing bank")
        print(
            f"[DeepSeek windows] verified prefix={len(prefix_ids)} documents",
            flush=True,
        )

    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "windows.safetensors"
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    save_file({"input_ids": torch.stack(windows)}, str(temporary))
    os.replace(temporary, artifact_path)
    manifest = {
        "format": FORMAT,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model": {
            "path": str(model_path),
            "huggingface_repo": MODEL_REPO,
            "revision": _snapshot_revision(model_path),
            "config_sha256": _sha256(config_path),
            "safetensors_index_sha256": _sha256(index_path),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
        },
        "dataset": {
            "repo": "allenai/c4",
            "config": "en",
            "split": args.dataset_split,
            "requested_revision": args.dataset_revision,
            "resolved_revision": resolved_dataset_revision,
            "streaming": True,
            "shuffle_buffer": args.shuffle_buffer,
        },
        "sampling": {
            "seed": args.seed,
            "fit_windows": args.fit_windows,
            "validation_windows": args.validation_windows,
            "profile_windows": args.profile_windows,
            "confirmation_windows": args.confirmation_windows,
            "sequence_length": args.sequence_length,
            "document_disjoint": True,
            "all_positions_used": True,
        },
        "artifact": {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "shape": list((total, args.sequence_length)),
            "dtype": str(windows[0].dtype),
        },
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(f"[DeepSeek windows] complete output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
