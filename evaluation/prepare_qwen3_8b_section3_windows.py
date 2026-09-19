#!/usr/bin/env python3
"""Prepare audited WikiText-2 and equal-token long-context Section-3 windows."""

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
from typing import Any, Mapping

from datasets import load_dataset
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file
import torch
from torch import Tensor
from transformers import AutoTokenizer


WT2_FORMAT = "basisserve.section3.wikitext2_calibration_windows.v1"
LONG_FORMAT = "basisserve.section3.long_context_windows.v1"
QWEN_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
QWEN_CONFIG_SHA256 = "3bd01d7ad7a2e203ecbbe84e24087a51c6d2a108ee4bcc42d0016bf49564983a"
LONG_CONTEXTS = (2048, 8192, 32768, 131072)


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: Mapping[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(tensors), str(temporary))
    os.replace(temporary, path)


def _model_record(model: Path) -> dict[str, Any] | None:
    config = model / "config.json"
    if not _check(config.is_file(), f"missing model config: {config}"):
        return None
    revision = model.name if model.parent.name == "snapshots" else None
    if not _check(revision == QWEN_REVISION, f"unexpected Qwen revision: {revision}"):
        return None
    digest = _sha256(config)
    if not _check(digest == QWEN_CONFIG_SHA256, f"unexpected Qwen config hash: {digest}"):
        return None
    return {
        "path": str(model),
        "huggingface_repo": "Qwen/Qwen3-8B-Base",
        "revision": revision,
        "config_sha256": digest,
    }


def _wikitext_text(dataset_repo: str, revision: str, split: str, cache: str | None) -> str:
    dataset = load_dataset(
        dataset_repo,
        "wikitext-2-raw-v1",
        split=split,
        revision=revision,
        cache_dir=cache,
    )
    return "\n\n".join(str(text) for text in dataset["text"])


def _official_wt2_windows(
    tokenizer: Any,
    text: str,
    *,
    samples: int,
    sequence_length: int,
    seed: int,
    split: str,
) -> tuple[list[Tensor], list[dict[str, Any]]]:
    generator = random.Random(seed)
    windows = []
    records = []
    for sample_index in range(samples):
        character_start = generator.randint(0, len(text) - sequence_length - 1)
        character_stop = min(len(text), character_start + sequence_length * 10)
        encoded = tokenizer(
            text[character_start:character_stop],
            return_tensors="pt",
        ).input_ids[0, :sequence_length]
        if not _check(
            len(encoded) == sequence_length,
            f"{split} sample {sample_index} produced only {len(encoded)} tokens",
        ):
            return [], []
        window = encoded.to(torch.int32).cpu().contiguous()
        windows.append(window)
        records.append(
            {
                "sample_index": sample_index,
                "split": split,
                "character_start": character_start,
                "character_stop": character_stop,
                "selected_token_start": 0,
                "selected_token_stop": sequence_length,
                "input_ids_sha256": _tensor_sha256(window),
            }
        )
    return windows, records


def prepare_wikitext2(args: argparse.Namespace) -> int:
    model = Path(args.model).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    model_record = _model_record(model)
    if model_record is None or not _check(not output.exists(), f"output exists: {output}"):
        return 2
    resolved_revision = HfApi().dataset_info(
        args.dataset_repo, revision=args.dataset_revision
    ).sha
    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, use_fast=True
    )
    train_text = _wikitext_text(
        args.dataset_repo, resolved_revision, "train", args.dataset_cache
    )
    heldout_text = _wikitext_text(
        args.dataset_repo, resolved_revision, "validation", args.dataset_cache
    )
    fit, fit_records = _official_wt2_windows(
        tokenizer,
        train_text,
        samples=args.fit_windows,
        sequence_length=args.sequence_length,
        seed=args.seed,
        split="train",
    )
    heldout, heldout_records = _official_wt2_windows(
        tokenizer,
        heldout_text,
        samples=args.heldout_windows,
        sequence_length=args.sequence_length,
        seed=args.seed + 1,
        split="validation",
    )
    if not _check(
        len(fit) == args.fit_windows and len(heldout) == args.heldout_windows,
        "WikiText-2 window construction was incomplete",
    ):
        return 2
    stacked = torch.stack((*fit, *heldout))
    output.mkdir(parents=True)
    artifact = output / "windows.safetensors"
    _atomic_safetensors(artifact, {"input_ids": stacked})
    manifest = {
        "format": WT2_FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_record,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
        },
        "dataset": {
            "repo": args.dataset_repo,
            "config": "wikitext-2-raw-v1",
            "requested_revision": args.dataset_revision,
            "resolved_revision": resolved_revision,
            "fit_split": "train",
            "heldout_split": "validation",
        },
        "sampling": {
            "protocol": "PaLU-compatible random character start then first N tokens",
            "seed": args.seed,
            "heldout_seed": args.seed + 1,
            "fit_windows": args.fit_windows,
            "heldout_windows": args.heldout_windows,
            "sequence_length": args.sequence_length,
            "calibration_tokens": args.fit_windows * args.sequence_length,
            "fit_heldout_split_disjoint": True,
        },
        "records": {
            "fit": fit_records,
            "heldout": heldout_records,
        },
        "artifact": {
            "file": artifact.name,
            "sha256": _sha256(artifact),
            "tensor": "input_ids",
            "dtype": str(stacked.dtype),
            "shape": list(stacked.shape),
            "ordered_partitions": [
                {"name": "fit", "start": 0, "stop": args.fit_windows},
                {
                    "name": "heldout",
                    "start": args.fit_windows,
                    "stop": args.fit_windows + args.heldout_windows,
                },
            ],
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    print(f"[WikiText-2] wrote {artifact} shape={tuple(stacked.shape)}", flush=True)
    return 0


def _source_records_for_range(
    records: list[Mapping[str, Any]],
    *,
    source_start: int,
    source_sequence_length: int,
    flat_start: int,
    flat_stop: int,
) -> tuple[list[int], list[str]]:
    first = flat_start // source_sequence_length
    last = (flat_stop - 1) // source_sequence_length
    indices = list(range(source_start + first, source_start + last + 1))
    documents = [str(records[index]["document_id"]) for index in indices]
    return indices, documents


def _repartition(
    flat: Tensor,
    *,
    sequence_length: int,
    source_records: list[Mapping[str, Any]],
    source_start: int,
    source_sequence_length: int,
) -> tuple[Tensor, list[dict[str, Any]]]:
    windows = flat.reshape(-1, sequence_length).to(torch.int32).contiguous()
    records = []
    for index, window in enumerate(windows):
        flat_start = index * sequence_length
        flat_stop = flat_start + sequence_length
        source_indices, documents = _source_records_for_range(
            source_records,
            source_start=source_start,
            source_sequence_length=source_sequence_length,
            flat_start=flat_start,
            flat_stop=flat_stop,
        )
        records.append(
            {
                "sample_index": index,
                "source_sample_indices": source_indices,
                "source_document_ids": documents,
                "flat_token_start": flat_start,
                "flat_token_stop": flat_stop,
                "input_ids_sha256": _tensor_sha256(window),
            }
        )
    return windows, records


def prepare_long_context(args: argparse.Namespace) -> int:
    model = Path(args.model).expanduser().resolve()
    source_path = Path(args.source_windows).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    model_record = _model_record(model)
    source_manifest_path = source_path.parent / "manifest.json"
    if (
        model_record is None
        or not _check(not output.exists(), f"output exists: {output}")
        or not _check(source_path.is_file(), f"missing source: {source_path}")
        or not _check(source_manifest_path.is_file(), "missing source manifest")
    ):
        return 2
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source = load_file(str(source_path), device="cpu")["input_ids"].to(torch.int32)
    source_sequence_length = int(source.shape[1])
    required_tokens = (
        args.calibration_tokens + args.heldout_tokens + args.evaluation_tokens
    )
    required_source_windows = required_tokens // source_sequence_length
    if not _check(
        required_tokens % source_sequence_length == 0,
        "token partitions must align with the source geometry",
    ):
        return 2
    source_stop = args.source_start + required_source_windows
    if not _check(source_stop <= len(source), "source window bank is too small"):
        return 2
    selected = source[args.source_start:source_stop].reshape(-1).contiguous()
    calibration_stop = args.calibration_tokens
    heldout_stop = calibration_stop + args.heldout_tokens
    calibration_flat = selected[:calibration_stop]
    heldout_flat = selected[calibration_stop:heldout_stop]
    evaluation_flat = selected[heldout_stop:]
    source_records = list(source_manifest["records"])
    condition_records = []
    output.mkdir(parents=True)
    for context in LONG_CONTEXTS:
        if not _check(
            args.calibration_tokens % context == 0 and args.heldout_tokens % context == 0,
            f"token budgets do not divide context {context}",
        ):
            return 2
        fit, fit_records = _repartition(
            calibration_flat,
            sequence_length=context,
            source_records=source_records,
            source_start=args.source_start,
            source_sequence_length=source_sequence_length,
        )
        heldout_source_start = args.source_start + args.calibration_tokens // source_sequence_length
        heldout, heldout_records = _repartition(
            heldout_flat,
            sequence_length=context,
            source_records=source_records,
            source_start=heldout_source_start,
            source_sequence_length=source_sequence_length,
        )
        combined = torch.cat((fit, heldout), dim=0).contiguous()
        directory = output / "calibration" / f"{context // 1024}k"
        directory.mkdir(parents=True)
        artifact = directory / "windows.safetensors"
        _atomic_safetensors(artifact, {"input_ids": combined})
        manifest = {
            "format": LONG_FORMAT,
            "status": "complete",
            "command": shlex.join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_record,
            "dataset": source_manifest["dataset"],
            "rope": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
                "effective_max_position_embeddings": 131072,
            },
            "packing": {
                "policy": "row-major token-preserving repartition without separators",
                "source_file": str(source_path),
                "source_sha256": _sha256(source_path),
                "source_manifest": str(source_manifest_path),
                "source_manifest_sha256": _sha256(source_manifest_path),
                "context_length": context,
                "calibration_tokens": args.calibration_tokens,
                "heldout_tokens": args.heldout_tokens,
                "fit_windows": len(fit),
                "heldout_windows": len(heldout),
                "token_content_preserved": bool(
                    torch.equal(fit.reshape(-1), calibration_flat)
                    and torch.equal(heldout.reshape(-1), heldout_flat)
                ),
            },
            "records": {"fit": fit_records, "heldout": heldout_records},
            "artifact": {
                "file": artifact.name,
                "sha256": _sha256(artifact),
                "tensor": "input_ids",
                "dtype": str(combined.dtype),
                "shape": list(combined.shape),
            },
        }
        _atomic_json(directory / "manifest.json", manifest)
        condition_records.append(
            {
                "context_length": context,
                "directory": str(directory),
                "fit_windows": len(fit),
                "heldout_windows": len(heldout),
                "artifact_sha256": manifest["artifact"]["sha256"],
            }
        )

    evaluation_source_start = (
        args.source_start
        + (args.calibration_tokens + args.heldout_tokens) // source_sequence_length
    )
    evaluation, evaluation_records = _repartition(
        evaluation_flat,
        sequence_length=131072,
        source_records=source_records,
        source_start=evaluation_source_start,
        source_sequence_length=source_sequence_length,
    )
    evaluation_dir = output / "evaluation" / "128k"
    evaluation_dir.mkdir(parents=True)
    evaluation_artifact = evaluation_dir / "windows.safetensors"
    _atomic_safetensors(evaluation_artifact, {"input_ids": evaluation})
    evaluation_manifest = {
        "format": LONG_FORMAT,
        "status": "complete",
        "model": model_record,
        "dataset": source_manifest["dataset"],
        "rope": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "effective_max_position_embeddings": 131072,
        },
        "packing": {
            "policy": "row-major token-preserving repartition without separators",
            "source_file": str(source_path),
            "source_start": evaluation_source_start,
            "evaluation_tokens": args.evaluation_tokens,
            "context_length": 131072,
            "documents_disjoint_from_calibration_and_fit_heldout": True,
            "token_content_preserved": bool(
                torch.equal(evaluation.reshape(-1), evaluation_flat)
            ),
        },
        "records": evaluation_records,
        "artifact": {
            "file": evaluation_artifact.name,
            "sha256": _sha256(evaluation_artifact),
            "tensor": "input_ids",
            "dtype": str(evaluation.dtype),
            "shape": list(evaluation.shape),
        },
    }
    _atomic_json(evaluation_dir / "manifest.json", evaluation_manifest)
    master = {
        "format": LONG_FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_record,
        "source": {
            "file": str(source_path),
            "sha256": _sha256(source_path),
            "manifest": str(source_manifest_path),
            "manifest_sha256": _sha256(source_manifest_path),
            "source_start": args.source_start,
            "source_stop": source_stop,
        },
        "token_budgets": {
            "calibration": args.calibration_tokens,
            "fit_heldout": args.heldout_tokens,
            "evaluation": args.evaluation_tokens,
        },
        "conditions": condition_records,
        "evaluation": {
            "directory": str(evaluation_dir),
            "windows": len(evaluation),
            "artifact_sha256": evaluation_manifest["artifact"]["sha256"],
        },
    }
    _atomic_json(output / "manifest.json", master)
    print(f"[Long context] wrote controlled windows under {output}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    wt2 = subparsers.add_parser("wikitext2")
    wt2.add_argument("--model", required=True)
    wt2.add_argument("--output-dir", required=True)
    wt2.add_argument("--fit-windows", type=int, default=128)
    wt2.add_argument("--heldout-windows", type=int, default=32)
    wt2.add_argument("--sequence-length", type=int, default=2048)
    wt2.add_argument("--seed", type=int, default=20260918)
    wt2.add_argument("--dataset-repo", default="Salesforce/wikitext")
    wt2.add_argument("--dataset-revision", default="main")
    wt2.add_argument("--dataset-cache", default="results/cache/huggingface/datasets")
    long = subparsers.add_parser("long-context")
    long.add_argument("--model", required=True)
    long.add_argument("--source-windows", required=True)
    long.add_argument("--output-dir", required=True)
    long.add_argument("--source-start", type=int, default=0)
    long.add_argument("--calibration-tokens", type=int, default=1_048_576)
    long.add_argument("--heldout-tokens", type=int, default=262_144)
    long.add_argument("--evaluation-tokens", type=int, default=1_048_576)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = prepare_wikitext2(args) if args.command == "wikitext2" else prepare_long_context(args)
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()
