"""Freeze the V2Q8 TP4 64K prompt cohorts: cohort c holds 16 distinct prompts, the first 65536 tokens of
C4 calibration windows 16c..16c+15 (c4-48x128k, 48 windows). A batch of B uses rows 0..B-1 of a cohort,
identically for every method. Timing inputs, not a held-out quality set."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=65536)
    args = parser.parse_args()
    windows = load_file(str(args.windows))["input_ids"]
    assert windows.shape[0] == 48 and windows.shape[1] >= args.length
    cohorts = {f"cohort_{c}": windows[16 * c:16 * c + 16, :args.length].to(torch.int32).contiguous() for c in range(3)}
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(cohorts, str(args.output / "prompts.safetensors"))
    manifest = dict(source=str(args.windows), source_sha256=hashlib.sha256(args.windows.read_bytes()).hexdigest(),
        length=args.length, cohorts={name: dict(windows=list(range(16 * c, 16 * c + 16)),
            row_sha256=[hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in tensor])
            for c, (name, tensor) in enumerate(cohorts.items())},
        protocol="Batch B uses rows 0..B-1 of a cohort for every method; fixed-length greedy generation")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "cohorts"}))


if __name__ == "__main__":
    main()
