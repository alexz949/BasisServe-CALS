#!/usr/bin/env python3
"""Prepare 32x128K C4 fit windows plus 16x128K held-out windows."""

import argparse
import hashlib
from pathlib import Path
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from evaluation.v96kl_common import read_json, save_tensors, sha256, tensor_hash, write_json


FIT_WINDOWS = 32
HELDOUT_WINDOWS = 16
WINDOW_LENGTH = 131072
PIECE_LENGTH = 4096
PIECES_PER_WINDOW = WINDOW_LENGTH // PIECE_LENGTH
TOTAL_WINDOWS = FIT_WINDOWS + HELDOUT_WINDOWS
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
SEED = 20260828


def prepare(args):
    output = args.calibration
    manifest_path = output / "manifest.json"
    windows_path = output / "windows.safetensors"
    expected_fit = list(range(FIT_WINDOWS))
    expected_heldout = list(range(FIT_WINDOWS, TOTAL_WINDOWS))
    if manifest_path.exists():
        record = read_json(manifest_path)
        assert record["status"] == "complete"
        assert record["sha256"] == sha256(windows_path)
        assert record["model_config_sha256"] == sha256(args.model / "config.json")
        assert record["fit_ids"] == expected_fit
        assert record["validation_ids"] == expected_heldout
        assert record["shape"] == [TOTAL_WINDOWS, WINDOW_LENGTH]
        assert record["source_sha256"] == sha256(Path(__file__))
        print("C4 32x128K + 16x128K windows already complete", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        revision=C4_REVISION,
    ).shuffle(seed=SEED, buffer_size=10000)
    rng = random.Random(SEED)
    required_pieces = TOTAL_WINDOWS * PIECES_PER_WINDOW
    pieces = []
    records = []
    seen = set()
    for stream_index, row in enumerate(dataset):
        document = hashlib.sha256(row["text"].encode()).hexdigest()
        if document in seen:
            continue
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        if len(ids) < PIECE_LENGTH:
            continue
        start = rng.randint(0, len(ids) - PIECE_LENGTH)
        piece = torch.tensor(ids[start : start + PIECE_LENGTH], dtype=torch.int32)
        pieces.append(piece)
        seen.add(document)
        records.append(
            {
                "document_sha256": document,
                "stream_index": stream_index,
                "start": start,
                "token_sha256": tensor_hash(piece),
            }
        )
        if len(pieces) % PIECES_PER_WINDOW == 0:
            print(
                "prepared C4 packed windows",
                len(pieces) // PIECES_PER_WINDOW,
                "/",
                TOTAL_WINDOWS,
                flush=True,
            )
        if len(pieces) == required_pieces:
            break

    assert len(pieces) == len(seen) == required_pieces
    packed = torch.stack(pieces).reshape(TOTAL_WINDOWS, WINDOW_LENGTH)
    save_tensors(windows_path, {"input_ids": packed})
    write_json(
        manifest_path,
        {
            "format": "basisserve.c4_packed_windows.v1",
            "status": "complete",
            "sha256": sha256(windows_path),
            "model_config_sha256": sha256(args.model / "config.json"),
            "dataset": "allenai/c4",
            "revision": C4_REVISION,
            "seed": SEED,
            "shuffle_buffer": 10000,
            "fit_ids": expected_fit,
            "validation_ids": expected_heldout,
            "shape": [TOTAL_WINDOWS, WINDOW_LENGTH],
            "packing": "32 document-disjoint 4096-token excerpts per 128K window; no separators",
            "records": records,
            "source_sha256": sha256(Path(__file__)),
        },
    )
    print("Prepared C4 32x128K fit + 16x128K held-out windows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    prepare(parser.parse_args())
