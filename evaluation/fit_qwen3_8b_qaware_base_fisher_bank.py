#!/usr/bin/env python3
"""Fit per-query Q-aware Base16, then refit its Page-Fisher residual bank."""

from __future__ import annotations

import argparse
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

from basisserve.core.c1_v_conditional_k_router import AffineReducedRankMap
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (
    _build_residual_statistics, _discover_capture, _fit_residual_grid,
    _load_direct, _rotary_embeddings,
)
from evaluation.eval_qwen3_8b_v80_exact_query_weighted_rrr import (
    METRIC_NAME, _query_weighted_document_loss,
)
from evaluation.eval_qwen3_8b_v80_fisher_base import (
    _check_query_alignment, _discover_query_statistics, _load_query_observations,
)
from evaluation.eval_qwen3_8b_v80_pre_rope_fisher_base import _train_base
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json

QUERY_POSITIONS = list(range(25599, 32768, 1024))
SOURCE_FILES = (
    "evaluation/fit_qwen3_8b_qaware_base_fisher_bank.py",
    "evaluation/eval_qwen3_8b_v80_exact_query_weighted_rrr.py",
    "evaluation/eval_qwen3_8b_v80_pre_rope_fisher_base.py",
    "evaluation/eval_qwen3_8b_v80_conditional_residual_router.py",
    "evaluation/eval_qwen3_8b_v80_fisher_base.py",
    "basisserve/core/c1_v_conditional_k_router.py",
    "basisserve/core/gqa_joint_routing_payload_s80_fisher.py",
    "basisserve/core/gqa_joint_routing_payload_s80_ablation.py",
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--c1-checkpoint", type=Path, required=True)
    p.add_argument("--calibration-root", type=Path, default=ROOT / "results/calibration")
    p.add_argument("--initial-bank", type=Path, default=ROOT / "results/checkpoints/q8_residual_kl_bank")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/checkpoints/q8_qbase_fisher_bank")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--torch-num-threads", type=int, default=2)
    p.add_argument("--preflight-only", action="store_true")
    return p


def base_maps(tensors):
    return tuple(AffineReducedRankMap(
        left=tensors["base_left_b16"][g], right=tensors["base_right_b16"][g],
        bias=tensors["base_bias_b16"][g],
    ) for g in range(tensors["base_left_b16"].shape[0]))


def pack_bank(maps, residuals):
    result = {f"base_{name}_b16": torch.stack([getattr(m, name) for m in maps]).float()
              for name in ("left", "right", "bias")}
    for rank in (4, 8, 16):
        result[f"residual_encoder_b16_r{rank}"] = residuals[(16, rank)][0].float()
        result[f"residual_query_b16_r{rank}"] = residuals[(16, rank)][1].float()
    assert all(bool(torch.isfinite(value).all()) for value in result.values())
    return {name: value.detach().cpu().contiguous() for name, value in result.items()}


def protocol(args):
    windows = args.calibration_root / "qwen3_8b_c4_64f16h_s32768/windows.safetensors"
    result = {
        "format": "basisserve.qaware_base_fisher_bank.v1",
        "model_config_sha256": sha256(args.model / "config.json"),
        "c1_checkpoint": str(args.c1_checkpoint.resolve()),
        "c1_manifest_sha256": sha256(args.c1_checkpoint / "results.json"),
        "windows_sha256": sha256(windows), "sequence_length": 32768,
        "fit_windows": 64, "validation_windows": 16,
        "base_rank": 16, "ranks": [4, 8, 16], "page_size": 32,
        "excluded_prefix_pages": 1, "base_query_positions": QUERY_POSITIONS,
        "base_objective": "per-query/per-position causal non-sink raw-QK squared error",
        "base_optimizer": {"epochs": 12, "documents_per_step": 4, "factor_lr": .002,
                           "bias_lr": .0005, "gradient_clip": 1., "patience": 4, "seed": 73},
        "base_selection": "lowest validation raw-QK NMSE, including epoch zero",
        "residual_objective": "exact-teacher non-sink Page-Fisher on NEW post-RoPE residual",
        "residual_query_positions": [32767], "residual_sweeps": 40,
        "residual_relative_damping": 1e-5, "residual_iterative_tolerance": 1e-5,
        "residual_iterative_max_iterations": 100,
        "residual_selection": "fixed final sweep; validation diagnostics only",
        "capture_trajectory": "dense teacher; current C1 encoder applied offline to raw V",
        "code_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "inputs": {},
    }
    c1 = json.loads((args.c1_checkpoint / "results.json").read_text())
    for layer in range(36):
        source = args.initial_bank / f"layer_{layer:03d}.safetensors"
        record = json.loads(source.with_suffix(".json").read_text())
        assert record["status"] == "complete" and sha256(source) == record["sha256"]
        assert record["protocol"]["c1_manifest_sha256"] == result["c1_manifest_sha256"]
        assert record["protocol"]["windows_sha256"] == result["windows_sha256"]
        inputs = {"initial_bank_sha256": record["sha256"],
                  "c1_layer_sha256": sha256(args.c1_checkpoint / c1["artifacts"][str(layer)]["file"])}
        for split, start, count in (("fit", 0, 64), ("validation", 64, 16)):
            dr, dm = _discover_capture(args.calibration_root, split=split, layer=layer)
            qr, qm = _discover_query_statistics(args.calibration_root, split=split, layer=layer)
            _check_query_alignment(dm, qm)
            assert dm["calibration"]["windows_sha256"] == result["windows_sha256"]
            assert dm["calibration"]["fit_start"] == qm["calibration"]["start"] == start
            assert dm["calibration"]["routing_fit_windows"] == qm["calibration"]["documents"] == count
            assert qm["calibration"]["query_positions"] == QUERY_POSITIONS
            assert qm["source"]["model_config_sha256"] == result["model_config_sha256"]
            assert (qr / qm["artifacts"][str(layer)]["file"]).is_file()
            for item in dm["artifacts"][str(layer)].values():
                if isinstance(item, dict) and "file" in item:
                    assert (dr / item["file"]).is_file()
            inputs[split] = {"direct_manifest_sha256": sha256(dr / "manifest.json"),
                             "query_manifest_sha256": sha256(qr / "manifest.json")}
        result["inputs"][str(layer)] = inputs
    return result


def fit_layer(args, layer, cos, sin, device):
    c1 = json.loads((args.c1_checkpoint / "results.json").read_text())
    encoder = load_file(str(args.c1_checkpoint / c1["artifacts"][str(layer)]["file"]))[
        "value_coordinate_encoders"]
    initial = base_maps(load_file(str(args.initial_bank / f"layer_{layer:03d}.safetensors")))
    data = {}
    for split, count in (("fit", 64), ("validation", 16)):
        dr, dm = _discover_capture(args.calibration_root, split=split, layer=layer)
        qr, qm = _discover_query_statistics(args.calibration_root, split=split, layer=layer)
        terminal, rows = _load_direct(dr, dm, layer)
        queries, positions = _load_query_observations(qr, qm, layer=layer)
        assert positions.tolist() == QUERY_POSITIONS
        assert queries.shape == (count, 8, 32, 128)
        assert terminal.shape == (count, 32, 128) and rows.shape == (count, 32768, 8, 256)
        assert torch.isfinite(queries).all() and torch.isfinite(terminal).all()
        data[split] = (terminal, rows, queries)
    maps, history, best_epoch = _train_base(
        data["fit"][2], data["fit"][1], data["validation"][2], data["validation"][1],
        query_positions=positions, value_encoder=encoder, initial_maps=initial,
        cos=cos, sin=sin, page_size=32, pinned_prefix_pages=1,
        epochs=12, documents_per_step=4, factor_learning_rate=.002, bias_learning_rate=.0005,
        gradient_clip=1., patience=4, seed=73, device=device,
        document_loss=_query_weighted_document_loss, metric_name=METRIC_NAME,
    )
    # Freeze the newly fitted Base. Never reuse old residual statistics or factors.
    statistics, reconstruction = {}, {}
    with torch.inference_mode():
        for split in ("fit", "validation"):
            statistics[split], reconstruction[split] = _build_residual_statistics(
                data[split][0], data[split][1], value_encoder=encoder, base_maps={16: maps},
                cos=cos, sin=sin, page_size=32, excluded_prefix_pages=1, device=device,
            )
        residuals, diagnostics = _fit_residual_grid(
            statistics["fit"], statistics["validation"], residual_ranks=(4, 8, 16),
            sweeps=40, relative_damping=1e-5, iterative_tolerance=1e-5,
            iterative_max_iterations=100, device=device,
        )
    return pack_bank(maps, residuals), {
        "base": {"history": history, "best_epoch": best_epoch},
        "residual": diagnostics, "reconstruction": reconstruction,
    }


def main():
    args = parser().parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(args.torch_num_threads)
    specification = protocol(args)
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "layers": 36,
                          "base_query_positions": QUERY_POSITIONS, "protocol": specification}, indent=2))
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    cos, sin = _rotary_embeddings(args.model, sequence=32768, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete = []
    for layer in range(args.shard_index, 36, args.num_shards):
        output = args.output_dir / f"layer_{layer:03d}.safetensors"
        record_path = output.with_suffix(".json")
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert record["status"] == "complete" and record["protocol"] == specification
            assert record["sha256"] == sha256(output)
        else:
            started = time.monotonic()
            print(f"[Q-aware Base + Fisher] layer={layer} start", flush=True)
            tensors, diagnostics = fit_layer(args, layer, cos, sin, device)
            temporary = output.with_suffix(".tmp")
            save_file(tensors, str(temporary))
            temporary.replace(output)
            record = {"status": "complete", "layer": layer, "protocol": specification,
                      "sha256": sha256(output), "diagnostic": diagnostics,
                      "wall_seconds": time.monotonic() - started, "command": shlex.join(sys.argv),
                      "python": sys.executable, "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(device)}
            write_json(record_path, record)
            print(f"[Q-aware Base + Fisher] layer={layer} done seconds={record['wall_seconds']:.1f}", flush=True)
        complete.append(layer)
    write_json(args.output_dir / f"shard_{args.shard_index}.json",
               {"status": "complete", "layers": complete, "protocol": specification})


if __name__ == "__main__":
    main()
