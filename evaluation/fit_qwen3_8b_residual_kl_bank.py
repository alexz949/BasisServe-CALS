#!/usr/bin/env python3
"""Complete the fixed-Base16 R4/R8/R16 bank for two-sided terminal KL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (
    _build_residual_statistics, _discover_capture, _fit_residual_grid,
    _load_direct, _rotary_embeddings,
)
from evaluation.fit_qwen3_8b_v80_base16_r8_nonsink_page32 import _load_base_maps


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--calibration-root", type=Path, default=ROOT / "results/calibration")
    p.add_argument("--c1-checkpoint", type=Path, required=True)
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--reuse-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--torch-num-threads", type=int, default=2)
    return p


@torch.inference_mode()
def main() -> None:
    args = parser().parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(args.torch_num_threads)
    # Match the existing three-layer RRR + Page-Fisher BCD experiment.
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cos, sin = _rotary_embeddings(args.model, sequence=32768, device=device)
    c1 = json.loads((args.c1_checkpoint / "results.json").read_text())
    protocol = {
        "format": "basisserve.residual_kl_bank.v1", "ranks": [4, 8, 16],
        "base_rank": 16, "fit_windows": 64, "diagnostic_windows": 16,
        "sequence_length": 32768, "fit_query": "last token of each window",
        "page_size": 32, "excluded_prefix_pages": 1, "sweeps": 40,
        "base_root": str(args.base_root.resolve()),
        "c1_checkpoint": str(args.c1_checkpoint.resolve()),
        "c1_manifest_sha256": sha256(args.c1_checkpoint / "results.json"),
        "windows_sha256": sha256(args.calibration_root / "qwen3_8b_c4_64f16h_s32768/windows.safetensors"),
        "validation_selects_factors": False,
    }
    rows = []
    for layer in range(args.shard_index, 36, args.num_shards):
        started = time.monotonic()
        output = args.output_dir / f"layer_{layer:03d}.safetensors"
        record_path = output.with_suffix(".json")
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert record["protocol"] == protocol and record["status"] == "complete"
            assert sha256(output) == record["sha256"]
            rows.append(record)
            print(f"[bank] reuse completed layer={layer}", flush=True)
            continue
        print(f"[bank] layer={layer}", flush=True)
        maps = _load_base_maps(args.base_root, layer)
        tensors = {
            f"base_{name}_b16": torch.stack([getattr(x, name) for x in maps]).float()
            for name in ("left", "right", "bias")
        }
        reused = args.reuse_root / f"layer_{layer}" / f"layer_{layer:03d}.safetensors"
        diagnostic = {}
        if reused.exists():
            source = load_file(str(reused))
            for name, value in tensors.items():
                assert torch.equal(source[name], value)
            for rank in (4, 8, 16):
                for kind in ("encoder", "query"):
                    name = f"residual_{kind}_b16_r{rank}"
                    tensors[name] = source[name]
            diagnostic["reused_from"] = str(reused.resolve())
            diagnostic["source_sha256"] = sha256(reused)
        else:
            c1_path = args.c1_checkpoint / c1["artifacts"][str(layer)]["file"]
            encoder = load_file(str(c1_path))["value_coordinate_encoders"]
            statistics = {}
            for split in ("fit", "validation"):
                root, manifest = _discover_capture(args.calibration_root, split=split, layer=layer)
                assert manifest["calibration"]["windows_sha256"] == protocol["windows_sha256"]
                assert manifest["calibration"]["fit_start"] == (0 if split == "fit" else 64)
                queries, activations = _load_direct(root, manifest, layer)
                assert queries.shape[0] == (64 if split == "fit" else 16)
                assert activations.shape[1] == 32768
                statistics[split], _ = _build_residual_statistics(
                    queries, activations, value_encoder=encoder, base_maps={16: maps},
                    cos=cos, sin=sin, page_size=32, excluded_prefix_pages=1, device=device,
                )
                del queries, activations
            bank, diagnostic = _fit_residual_grid(
                statistics["fit"], statistics["validation"], residual_ranks=(4, 8, 16),
                sweeps=40, relative_damping=1e-5, iterative_tolerance=1e-5,
                iterative_max_iterations=100, device=device,
            )
            for rank in (4, 8, 16):
                tensors[f"residual_encoder_b16_r{rank}"] = bank[(16, rank)][0]
                tensors[f"residual_query_b16_r{rank}"] = bank[(16, rank)][1]
            del statistics, bank
        assert all(bool(torch.isfinite(x).all()) for x in tensors.values())
        temporary = output.with_suffix(".tmp")
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(temporary))
        temporary.replace(output)
        record = {
            "status": "complete", "layer": layer, "protocol": protocol,
            "sha256": sha256(output), "diagnostic": diagnostic,
            "wall_seconds": time.monotonic() - started,
            "command": shlex.join(sys.argv), "python": sys.executable,
        }
        write_json(record_path, record)
        rows.append(record)
        print(f"[bank] layer={layer} complete seconds={record['wall_seconds']:.1f}", flush=True)
    write_json(args.output_dir / f"shard_{args.shard_index}.json", {
        "status": "complete", "layers": [r["layer"] for r in rows], "protocol": protocol,
    })


if __name__ == "__main__":
    main()
