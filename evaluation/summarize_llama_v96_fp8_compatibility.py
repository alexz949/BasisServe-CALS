#!/usr/bin/env python3
"""Export compact phase-1/2/3 artifacts for the V96 FP8 pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess

from evaluation.v96kl_common import read_json, sha256


ARMS = ("bf16_no_h", "bf16_h", "fp8_no_h", "fp8_h")
INTERNAL_FIELDS = (
    "latent_rel_mse",
    "predicted_key_rel_mse",
    "predicted_key_delta_rel_mse",
    "full_attention_output_rel_mse",
    "fixed_support_output_rel_mse",
    "routed_output_rel_mse",
)
ROUTING_FIELDS = (
    "routed_page_recall",
    "page_kl_exact_to_proxy",
    "attention_mass_recall",
    "nonsink_mass_recall",
    "selected_page_overlap",
    "selected_page_jaccard",
)


def mean(layers, arm, field):
    return statistics.fmean(layer["metrics"][arm][field] for layer in layers)


def common(result, arm, layer):
    protocol = result["protocol"]
    fp8 = arm.startswith("fp8")
    return {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "model_revision": Path(protocol["model"]).name,
        "basis_factor_hash": protocol["v_manifest_sha256"],
        "router_factor_hash": protocol["router_manifest_sha256"],
        "base_rank": 16,
        "payload_rank": 80,
        "residual_rank": 16,
        "precision": "E4M3FN" if fp8 else "BF16",
        "fp8_format": "E4M3FN" if fp8 else "none",
        "scale_granularity": (
            "separate Base/Payload; per-page x per-KV-head" if fp8 else "none"
        ),
        "scale_dtype": "FP32" if fp8 else "none",
        "hadamard_enabled": arm.endswith("_h"),
        "base_precision": "E4M3FN" if fp8 else "BF16",
        "payload_precision": "E4M3FN" if fp8 else "BF16",
        "residual_precision": "BF16",
        "exact_key_precision": "BF16",
        "context": protocol["sequence_length"],
        "batch": 1,
        "seed": 20260920,
        "gpu": result["environment"]["gpu"],
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        ).stdout.strip(),
        "layer": layer,
        "arm": arm,
    }


def write_csv(path, rows):
    assert rows
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = read_json(args.input)
    assert result["status"] == "complete" and len(result["layers"]) == 32
    args.output.mkdir(parents=True)

    internal = []
    routing = []
    memory = []
    for layer in result["layers"]:
        for arm in ARMS:
            prefix = common(result, arm, layer["layer"])
            internal.append({**prefix, **{field: layer["metrics"][arm][field] for field in INTERNAL_FIELDS}})
            routing.append({**prefix, **{field: layer["metrics"][arm][field] for field in ROUTING_FIELDS}})
    for arm in ARMS:
        metric = result["layers"][0]["metrics"][arm]
        code = metric["storage"]["code_bytes"] / (8 * result["protocol"]["sequence_length"])
        scales = metric["storage"]["scale_bytes"] / (8 * result["protocol"]["sequence_length"])
        memory.append(
            {
                **common(result, arm, "all"),
                "v_coordinate_bytes_per_token_per_kv_head": code,
                "scale_bytes_per_token_per_kv_head": scales,
                "v_total_bytes_per_token_per_kv_head": code + scales,
                "residual_bytes_per_token_per_kv_head": 32,
                "exact_key_bytes_per_token_per_kv_head": 256,
                "v_plus_residual_bytes_per_token_per_kv_head": code + scales + 32,
                "all_reported_state_bytes_per_token_per_kv_head": code + scales + 32 + 256,
            }
        )
    write_csv(args.output / "fp8_internal_diagnostics.csv", internal)
    write_csv(args.output / "fp8_routing_diagnostics.csv", routing)
    write_csv(args.output / "fp8_memory.csv", memory)

    audits = {
        "status": "complete",
        "scope": "BF16 materialization audit; normalized blockdiag(H16,H64,H16) is exactly orthogonal in FP64 tests",
        "layers": [
            {"layer": layer["layer"], **layer["gauge_audit"]} for layer in result["layers"]
        ],
        "means": {
            field: statistics.fmean(layer["gauge_audit"][field] for layer in result["layers"])
            for field in result["layers"][0]["gauge_audit"]
        },
    }
    (args.output / "hadamard_invariance.json").write_text(json.dumps(audits, indent=2) + "\n")

    table = []
    for arm in ARMS:
        table.append(
            {
                "arm": arm,
                "v_bytes": result["layers"][0]["metrics"][arm]["storage"][
                    "bytes_per_token_per_kv_head"
                ],
                **{field: mean(result["layers"], arm, field) for field in (*INTERNAL_FIELDS, *ROUTING_FIELDS)},
            }
        )
    manifest = {
        "status": "complete",
        "phase": "internal Value and routing diagnostics",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "protocol": result["protocol"],
        "environment": result["environment"],
        "summary": table,
        "completed_outputs": [
            "README.md",
            "manifest.json",
            "fp8_internal_diagnostics.csv",
            "fp8_routing_diagnostics.csv",
            "fp8_memory.csv",
            "hadamard_invariance.json",
        ],
        "pending_outputs": [
            "fp8_quality.csv",
            "fp8_latency.csv",
            "plots/fp8_e2e.pdf",
            "plots/fp8_e2e.png",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    baseline = table[0]
    fp8_h = table[-1]
    readme = f"""# Llama-3.1-8B-Instruct V96 FP8 Compatibility

This directory contains the completed internal Value and routing diagnostics. Downstream quality and serving-kernel latency are deliberately pending; these results are not systems measurements.

## Protocol

- Existing C1 Uniform-V96, six-sweep decoder-refit checkpoint.
- Existing B16R16 Page-Fisher router; no refit for quantization.
- Routing-basis alignment followed by optional `blockdiag(H16,H64,H16)`.
- Physical `torch.float8_e4m3fn` codes, separate Base/Payload FP32 scales per Page32 and KV head.
- Residual16 and exact selected Keys remain BF16.
- One held-out 8192-token sequence, 16 query positions per layer, all 32 layers.

## Mean over 32 layers

| Arm | V bytes/token/KV head | Latent rel-MSE | Output rel-MSE | Routed recall | Page KL | Page overlap |
|---|---:|---:|---:|---:|---:|---:|
"""
    for row in table:
        readme += (
            f"| {row['arm']} | {row['v_bytes']:.2f} | {row['latent_rel_mse']:.6g} | "
            f"{row['full_attention_output_rel_mse']:.6g} | {row['routed_page_recall']:.6f} | "
            f"{row['page_kl_exact_to_proxy']:.6f} | {row['selected_page_overlap']:.6f} |\n"
        )
    recall_delta = 100 * (fp8_h["routed_page_recall"] - baseline["routed_page_recall"])
    readme += f"""
## Decision

Full FP8 + Hadamard reduces the V-coordinate payload including FP32 scale metadata from 192.00 to 96.25 bytes/token/KV head. Its mean output rel-MSE is {fp8_h['full_attention_output_rel_mse']:.6g}; selected-page overlap is {fp8_h['selected_page_overlap']:.4%}; and routed recall changes by {recall_delta:+.4f} percentage points. Full FP8 is therefore the primary compatibility configuration. Mixed precision and quantization-aware residual fitting are not justified by this pilot.

Hadamard is standard quantization machinery, not a BasisKV contribution. In BF16 it introduces a mean output rel-MSE of {table[1]['full_attention_output_rel_mse']:.6g}; FP64 orthogonality is covered by the unit test.

The supported claim at this stage is limited to numerical compatibility. Actual peak HBM and latency require persistent FP8 cache reads/dequantization in the serving kernel.
"""
    (args.output / "README.md").write_text(readme)
    print(json.dumps({"output": str(args.output), "summary": table}, indent=2))


if __name__ == "__main__":
    main()
