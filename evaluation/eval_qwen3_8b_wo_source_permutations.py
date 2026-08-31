#!/usr/bin/env python3
"""Exhaustively permute fitted Qwen3-8B Wo-C1 TP-source factor pairs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FORMAT = "basisserve.qwen3_8b.wo_c1_source_permutations.v1"
PHASE1_FORMAT = "basisserve.qwen3_8b.wo_c1_lr_ar_phase1.v2"
COVARIANCE_FORMAT = "basisserve.attention_o_proj_covariances.v1"


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() or None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_layers(raw: str, *, num_layers: int) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return tuple(range(num_layers))
    selected = tuple(
        sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()})
    )
    if not selected or min(selected) < 0 or max(selected) >= num_layers:
        raise ValueError("selected layers are outside the checkpoint")
    return selected


def _validated_inputs(
    phase1_dir: Path,
    covariance_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase1_path = phase1_dir / "results.json"
    covariance_path = covariance_dir / "manifest.json"
    phase1 = _load_json(phase1_path)
    covariance = _load_json(covariance_path)
    if phase1.get("format") != PHASE1_FORMAT or phase1.get("status") != "complete":
        raise ValueError("Phase-1 input is not complete")
    if covariance.get("format") != COVARIANCE_FORMAT:
        raise ValueError("unexpected covariance manifest format")
    if _sha256(covariance_path) != phase1["source"]["covariance_manifest_sha256"]:
        raise ValueError("covariance manifest does not match Phase-1")
    if int(phase1["method"]["run_signature"]["tp_size"]) != 4:
        raise ValueError("source permutation test requires the TP4 checkpoint")
    if int(phase1["method"]["run_signature"]["source_rank"]) != 512:
        raise ValueError("source permutation test requires source rank 512")
    layers = sorted(phase1["layers"], key=lambda row: int(row["layer"]))
    if [int(row["layer"]) for row in layers] != list(range(36)):
        raise ValueError("Phase-1 must contain all 36 ordered layers")
    phase1["layers"] = layers
    return phase1, covariance


def _relative_error(
    weight: Tensor,
    approximation: Tensor,
    covariance: Tensor,
    target_energy: Tensor,
) -> float:
    residual = weight - approximation
    loss = ((residual @ covariance) * residual).sum().clamp_min(0)
    return float(loss / target_energy)


def _summarize(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def _aggregate(layers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    original = [float(row["original_relative_mse"]) for row in layers]
    all_nonidentity = [
        permutation
        for row in layers
        for permutation in row["nonidentity_permutations"]
    ]
    gaps = [float(row["absolute_mse_gap_vs_original"]) for row in all_nonidentity]
    ratios = [float(row["mse_ratio_vs_original"]) for row in all_nonidentity]
    by_moved_sources = {}
    for moved in (2, 3, 4):
        selected = [
            row
            for row in all_nonidentity
            if int(row["moved_sources"]) == moved
        ]
        by_moved_sources[str(moved)] = {
            "comparisons": len(selected),
            "relative_mse": _summarize(
                [float(row["relative_mse"]) for row in selected]
            ),
            "absolute_mse_gap_vs_original": _summarize(
                [float(row["absolute_mse_gap_vs_original"]) for row in selected]
            ),
            "mse_ratio_vs_original": _summarize(
                [float(row["mse_ratio_vs_original"]) for row in selected]
            ),
        }
    return {
        "layers": len(layers),
        "nonidentity_permutations_per_layer": 23,
        "nonidentity_comparisons": len(all_nonidentity),
        "original_relative_mse": _summarize(original),
        "nonidentity_relative_mse": _summarize(
            [float(row["relative_mse"]) for row in all_nonidentity]
        ),
        "absolute_mse_gap_vs_original": _summarize(gaps),
        "mse_ratio_vs_original": _summarize(ratios),
        "comparisons_strictly_worse_than_original": sum(gap > 0 for gap in gaps),
        "layers_where_original_is_strict_best": sum(
            bool(row["original_is_strict_best"]) for row in layers
        ),
        "by_moved_sources": by_moved_sources,
    }


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# Qwen3-8B Wo-C1 source-permutation sanity test",
        "",
        (
            "The four stored BF16 `(encoder, decoder)` pairs are kept intact "
            "and exhaustively reassigned to the four logical TP input sources. "
            "No factor is refitted."
        ),
        "",
        "## Aggregate",
        "",
        f"- Layers: `{aggregate['layers']}`",
        (
            "- Non-identity comparisons: "
            f"`{aggregate['nonidentity_comparisons']}`"
        ),
        (
            "- Comparisons strictly worse than the original mapping: "
            f"`{aggregate['comparisons_strictly_worse_than_original']}`"
        ),
        (
            "- Layers where the original mapping is strictly best: "
            f"`{aggregate['layers_where_original_is_strict_best']}`"
        ),
        (
            "- Mean original heldout relative MSE: "
            f"`{aggregate['original_relative_mse']['mean']:.8g}`"
        ),
        (
            "- Mean permuted heldout relative MSE: "
            f"`{aggregate['nonidentity_relative_mse']['mean']:.8g}`"
        ),
        (
            "- Median permutation/original MSE ratio: "
            f"`{aggregate['mse_ratio_vs_original']['median']:.6g}x`"
        ),
        "",
        "| Moved sources | Comparisons | Mean MSE | Mean gap | Median ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for moved in (2, 3, 4):
        row = aggregate["by_moved_sources"][str(moved)]
        lines.append(
            f"| {moved} | {row['comparisons']} | "
            f"{row['relative_mse']['mean']:.8g} | "
            f"{row['absolute_mse_gap_vs_original']['mean']:.8g} | "
            f"{row['mse_ratio_vs_original']['median']:.6g}x |"
        )
    lines.extend(
        [
            "",
            "## Per-layer",
            "",
            "| Layer | Original MSE | Best wrong MSE | Mean wrong MSE | Worst wrong MSE | Best wrong ratio | Original best |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for layer in payload["layers"]:
        wrong = layer["nonidentity_summary"]
        lines.append(
            f"| {layer['layer']} | {layer['original_relative_mse']:.8g} | "
            f"{wrong['relative_mse']['minimum']:.8g} | "
            f"{wrong['relative_mse']['mean']:.8g} | "
            f"{wrong['relative_mse']['maximum']:.8g} | "
            f"{wrong['mse_ratio_vs_original']['minimum']:.6g}x | "
            f"{'yes' if layer['original_is_strict_best'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "These are heldout attention-layer output errors, not terminal-logit KL or PPL.",
            (
                "The checkpoint sweep was selected using this heldout split; "
                "the source mapping itself was fixed a priori and was not "
                "selected from the permutation family."
            ),
            "",
            "## Command",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", required=True)
    parser.add_argument("--covariance-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    torch.set_float32_matmul_precision("highest")
    phase1_dir = Path(args.phase1_dir).expanduser().resolve()
    covariance_dir = Path(args.covariance_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    phase1, covariance_manifest = _validated_inputs(phase1_dir, covariance_dir)
    selected_layers = _parse_layers(args.layers, num_layers=36)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real-model permutation test requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    work_dtype = {"float32": torch.float32, "float64": torch.float64}[
        args.work_dtype
    ]
    permutations = tuple(itertools.permutations(range(4)))
    identity = tuple(range(4))
    phase1_by_layer = {int(row["layer"]): row for row in phase1["layers"]}
    covariance_artifacts = covariance_manifest["artifacts"]
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    layer_records = []
    maximum_identity_delta = 0.0
    for ordinal, layer in enumerate(selected_layers, start=1):
        phase_record = phase1_by_layer[layer]
        factor_path = phase1_dir / phase_record["artifact"]["file"]
        if _sha256(factor_path) != phase_record["artifact"]["sha256"]:
            raise RuntimeError(f"factor hash mismatch at layer {layer}")
        covariance_record = covariance_artifacts[str(layer)]
        covariance_path = covariance_dir / covariance_record["file"]
        if _sha256(covariance_path) != covariance_record["sha256"]:
            raise RuntimeError(f"covariance hash mismatch at layer {layer}")

        factors = load_file(str(factor_path), device="cpu")
        statistics_payload = load_file(str(covariance_path), device="cpu")
        encoders = factors["c1_source_encoders"].to(
            device=device, dtype=work_dtype
        )
        decoders = factors["c1_source_decoders"].to(
            device=device, dtype=work_dtype
        )
        source_products = torch.bmm(encoders, decoders)
        weight = statistics_payload["weight"].to(device=device, dtype=work_dtype)
        covariance = statistics_payload["heldout_covariance"].to(
            device=device, dtype=work_dtype
        )
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        target_energy = ((weight @ covariance) * weight).sum().clamp_min(
            torch.finfo(work_dtype).tiny
        )

        permutation_rows = []
        original_mse = None
        for permutation in permutations:
            order = torch.tensor(permutation, device=device, dtype=torch.long)
            approximation = (
                source_products.index_select(0, order)
                .reshape(4096, 4096)
                .transpose(0, 1)
                .contiguous()
            )
            relative_mse = _relative_error(
                weight, approximation, covariance, target_energy
            )
            moved_sources = sum(
                fitted_source != logical_source
                for logical_source, fitted_source in enumerate(permutation)
            )
            row = {
                "permutation": list(permutation),
                "mapping": "logical_source_p_uses_fitted_pair_permutation_p",
                "moved_sources": moved_sources,
                "relative_mse": relative_mse,
            }
            permutation_rows.append(row)
            if permutation == identity:
                original_mse = relative_mse
            del approximation, order
        assert original_mse is not None
        recorded = float(
            phase_record["c1_allgather"]["factor_dtype_heldout_relative_mse"]
        )
        identity_delta = abs(original_mse - recorded)
        maximum_identity_delta = max(maximum_identity_delta, identity_delta)
        if identity_delta > 5.0e-6:
            raise RuntimeError(
                f"identity metric mismatch at layer {layer}: {identity_delta}"
            )
        nonidentity = []
        for row in permutation_rows:
            if row["permutation"] == list(identity):
                continue
            row["absolute_mse_gap_vs_original"] = (
                float(row["relative_mse"]) - original_mse
            )
            row["mse_ratio_vs_original"] = float(row["relative_mse"]) / max(
                original_mse, torch.finfo(work_dtype).tiny
            )
            nonidentity.append(row)
        wrong_mse = [float(row["relative_mse"]) for row in nonidentity]
        wrong_gaps = [
            float(row["absolute_mse_gap_vs_original"]) for row in nonidentity
        ]
        wrong_ratios = [float(row["mse_ratio_vs_original"]) for row in nonidentity]
        layer_record = {
            "layer": layer,
            "original_permutation": list(identity),
            "original_relative_mse": original_mse,
            "phase1_recorded_relative_mse": recorded,
            "identity_consistency_absolute_delta": identity_delta,
            "original_is_strict_best": min(wrong_gaps) > 0.0,
            "nonidentity_summary": {
                "relative_mse": _summarize(wrong_mse),
                "absolute_mse_gap_vs_original": _summarize(wrong_gaps),
                "mse_ratio_vs_original": _summarize(wrong_ratios),
            },
            "nonidentity_permutations": nonidentity,
        }
        layer_records.append(layer_record)
        print(
            json.dumps(
                {
                    "event": "layer_complete",
                    "layer": layer,
                    "progress": f"{ordinal}/{len(selected_layers)}",
                    "original_mse": original_mse,
                    "best_wrong_mse": min(wrong_mse),
                    "mean_wrong_mse": statistics.fmean(wrong_mse),
                    "worst_wrong_mse": max(wrong_mse),
                    "original_is_strict_best": min(wrong_gaps) > 0.0,
                }
            ),
            flush=True,
        )
        del factors, statistics_payload, encoders, decoders
        del source_products, weight, covariance, target_energy
        torch.cuda.empty_cache()

    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "status": "complete",
        "command": shlex.join((sys.executable, *sys.argv)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "work_dtype": args.work_dtype,
            "torch_num_threads": args.torch_num_threads,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "model": phase1["model"],
        "source": {
            "phase1_dir": str(phase1_dir),
            "phase1_results_sha256": _sha256(phase1_dir / "results.json"),
            "covariance_dir": str(covariance_dir),
            "covariance_manifest_sha256": _sha256(
                covariance_dir / "manifest.json"
            ),
        },
        "protocol": {
            "scope": "post-attention Wo only; dense V and dense KV cache",
            "metric": "heldout covariance-weighted attention-layer output relative MSE",
            "tp_size": 4,
            "source_rank": 512,
            "stored_factor_dtype": "bfloat16",
            "permutation_scope": "paired encoder and decoder factors",
            "refit_after_permutation": False,
            "permutations": 24,
            "nonidentity_permutations": 23,
            "layers": list(selected_layers),
        },
        "validation": {
            "maximum_identity_vs_phase1_absolute_delta": maximum_identity_delta,
            "identity_tolerance": 5.0e-6,
            "identity_consistency_passed": maximum_identity_delta <= 5.0e-6,
        },
        "aggregate": _aggregate(layer_records),
        "layers": layer_records,
    }
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.md"
    _atomic_text(results_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(summary_path, _summary_markdown(payload))
    print(
        json.dumps(
            {
                "event": "result_written",
                "results": str(results_path),
                "results_sha256": _sha256(results_path),
                "summary": str(summary_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
