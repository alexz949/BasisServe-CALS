#!/usr/bin/env python3
"""Refit Base16 and Page-Fisher R8 with Q16 spanning the full 32K context."""

from __future__ import annotations

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
from evaluation.eval_qwen3_8b_v80_exact_query_weighted_rrr import (
    METRIC_NAME, _query_weighted_document_loss,
)
from evaluation.eval_qwen3_8b_v80_pre_rope_fisher_base import _train_base
from evaluation.fit_qwen3_8b_qaware_base_fisher_bank import (
    base_maps, parser as base_parser, protocol as source_protocol,
)
from evaluation.fit_qwen3_8b_q8_fisher_residual import (
    build_multi_query_statistics, expanded_queries,
)
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from scripts.capture_qwen3_8b_q16 import (
    FORMAT as QUERY_FORMAT, UNIFORM_Q16_POSITIONS, load_sources, verify_document,
)

FORMAT = "basisserve.qaware_base_fisher_bank.v1"


def parser():
    p = base_parser()
    p.description = __doc__
    p.add_argument("--query-capture", type=Path, required=True)
    p.set_defaults(initial_bank=ROOT / "results/checkpoints/q8_qbase_fisher_bank",
                   output_dir=ROOT / "results/checkpoints/uniform32k_q16_base_fisher_r8")
    return p


def protocol(args):
    specification = source_protocol(args)
    frozen = json.loads((args.initial_bank / "layer_000.json").read_text())["protocol"]
    assert frozen["format"] == FORMAT
    for layer in range(36):
        record = json.loads((args.initial_bank / f"layer_{layer:03d}.json").read_text())
        assert record["status"] == "complete" and record["protocol"] == frozen
        assert sha256(args.initial_bank / f"layer_{layer:03d}.safetensors") == record["sha256"]
    capture_path = args.query_capture / "manifest.json"
    capture = json.loads(capture_path.read_text())
    expected, _ = load_sources(args.model, args.calibration_root, layout="uniform32k")
    assert capture["status"] == "complete" and capture["overlap_bitwise_equal"]
    assert capture["protocol"] == expected and expected["format"] == QUERY_FORMAT
    assert expected["query_positions"] == UNIFORM_Q16_POSITIONS
    assert set(capture["artifacts"]) == {str(index) for index in range(80)}
    for index in range(80):
        record, _ = verify_document(args.query_capture, index, expected)
        assert capture["artifacts"][str(index)] == {
            "file": f"window_{index:03d}.safetensors", "sha256": record["sha256"]}
    specification.update({
        "format": FORMAT, "ranks": [8],
        "initial_base_bank": str(args.initial_bank.resolve()),
        "initial_base_protocol": frozen,
        "base_query_positions": UNIFORM_Q16_POSITIONS,
        "base_query_layout": "16 equal-block endpoints spanning 32768 tokens",
        "base_objective": "per-query/per-position causal non-sink raw-QK squared error",
        "base_optimizer": {"epochs": 12, "documents_per_step": 4, "factor_lr": .002,
                           "bias_lr": .0005, "gradient_clip": 1., "patience": 4, "seed": 73},
        "base_selection": "lowest validation raw-QK NMSE, including epoch zero",
        "residual_query_positions": UNIFORM_Q16_POSITIONS,
        "residual_objective": "sum of separate causal exact-teacher non-sink Page-Fisher losses on NEW residual",
        "residual_example_order": "query position major, document minor; equal example weights",
        "residual_sweeps": 40, "residual_relative_damping": 1e-5,
        "residual_iterative_tolerance": 1e-5, "residual_iterative_max_iterations": 100,
        "residual_selection": "fixed final sweep; validation diagnostics only",
        "base_fit_examples": 64 * 16, "base_validation_examples": 16 * 16,
        "residual_fit_examples": 64 * 16, "residual_validation_examples": 16 * 16,
        "query_capture": str(args.query_capture.resolve()),
        "query_capture_manifest_sha256": sha256(capture_path),
        "query_capture_protocol": expected,
        "rank_allocation": "none; uniform Base16 plus uniform residual R8",
    })
    specification["code_sha256"].update({
        "evaluation/fit_qwen3_8b_uniform_q16_base_fisher.py": sha256(Path(__file__)),
        "evaluation/fit_qwen3_8b_q8_fisher_residual.py": sha256(ROOT / "evaluation/fit_qwen3_8b_q8_fisher_residual.py"),
        "scripts/capture_qwen3_8b_q16.py": expected["source_sha256"],
    })
    return specification


def pack(maps, residuals, source):
    tensors = {
        "base_left_b16": torch.stack([item.left for item in maps]).float(),
        "base_right_b16": torch.stack([item.right for item in maps]).float(),
        "base_bias_b16": torch.stack([item.bias for item in maps]).float(),
        "residual_encoder_b16_r8": residuals[(16, 8)][0].float(),
        "residual_query_b16_r8": residuals[(16, 8)][1].float(),
    }
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in tensors.values())
    return {name: value.detach().cpu().contiguous() for name, value in tensors.items()}, {
        name: bool(torch.equal(tensors[name], source[name]))
        for name in ("base_left_b16", "base_right_b16", "base_bias_b16")
    }


def fit_layer(args, layer, cos, sin, device):
    c1 = json.loads((args.c1_checkpoint / "results.json").read_text())
    encoder = load_file(str(args.c1_checkpoint / c1["artifacts"][str(layer)]["file"]))[
        "value_coordinate_encoders"]
    source = load_file(str(args.initial_bank / f"layer_{layer:03d}.safetensors"))
    initial = base_maps(source)
    data = {}
    for split, count in (("fit", 64), ("validation", 16)):
        direct_root, direct_manifest = _discover_capture(args.calibration_root, split=split, layer=layer)
        _, rows = _load_direct(direct_root, direct_manifest, layer)
        queries, positions = expanded_queries(
            args.query_capture, split, layer, positions=UNIFORM_Q16_POSITIONS)
        assert positions.tolist() == UNIFORM_Q16_POSITIONS
        assert queries.shape == (count, 16, 32, 128) and torch.isfinite(queries).all()
        assert rows.shape == (count, 32768, 8, 256)
        data[split] = (queries, rows)
    maps, history, best_epoch = _train_base(
        data["fit"][0], data["fit"][1], data["validation"][0], data["validation"][1],
        query_positions=positions, value_encoder=encoder, initial_maps=initial,
        cos=cos, sin=sin, page_size=32, pinned_prefix_pages=1,
        epochs=12, documents_per_step=4, factor_learning_rate=.002, bias_learning_rate=.0005,
        gradient_clip=1., patience=4, seed=73, device=device,
        document_loss=_query_weighted_document_loss, metric_name=METRIC_NAME,
    )
    statistics, reconstruction = {}, {}
    with torch.inference_mode():
        for split in ("fit", "validation"):
            statistics[split], reconstruction[split] = build_multi_query_statistics(
                data[split][0], data[split][1], query_positions=positions,
                value_encoder=encoder, base_maps={16: maps}, cos=cos, sin=sin,
                page_size=32, excluded_prefix_pages=1, device=device,
            )
        residuals, diagnostics = _fit_residual_grid(
            statistics["fit"], statistics["validation"], residual_ranks=(8,),
            sweeps=40, relative_damping=1e-5, iterative_tolerance=1e-5,
            iterative_max_iterations=100, device=device,
        )
    tensors, source_equal = pack(maps, residuals, source)
    return tensors, {
        "base": {"history": history, "best_epoch": best_epoch,
                 "bitwise_equal_to_initial": source_equal},
        "residual": diagnostics, "reconstruction_by_query_position": reconstruction,
    }


def main():
    args = parser().parse_args()
    assert 0 <= args.shard_index < args.num_shards
    assert args.initial_bank.resolve() != args.output_dir.resolve()
    torch.set_num_threads(args.torch_num_threads)
    specification = protocol(args)
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "protocol": specification}, indent=2))
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
            print(f"[uniform32K Q16 Base+residual] layer={layer} start", flush=True)
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
            print(f"[uniform32K Q16 Base+residual] layer={layer} done seconds={record['wall_seconds']:.1f}", flush=True)
        complete.append(layer)
    write_json(args.output_dir / f"shard_{args.shard_index}.json",
               {"status": "complete", "layers": complete, "protocol": specification})


if __name__ == "__main__":
    main()
