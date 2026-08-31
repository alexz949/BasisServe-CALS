#!/usr/bin/env python3
"""Fit Wo-only C1-AllGather and strong LR-AllReduce controls for Qwen3-8B."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.analysis.c1_rigorous import (  # noqa: E402
    fit_strong_lr_allreduce_rank_bank,
    relative_output_mse,
    ring_allgather_bytes_per_rank,
    ring_allreduce_bytes_per_rank,
    trace_damped_covariance,
    wire_matched_allreduce_rank,
)
from basisserve.core.tp_source_wo_c1 import (  # noqa: E402
    TPSourceWOLayout,
    fold_factors_to_dense_weight,
)
from basisserve.core.tp_source_wo_fit import (  # noqa: E402
    TPSourceWOFitConfig,
    fit_tp_source_wo_c1,
)


FORMAT = "basisserve.qwen3_8b.wo_c1_lr_ar_phase1.v2"
LAYER_FORMAT = "basisserve.qwen3_8b.wo_c1_lr_ar_phase1.layer.v2"
COVARIANCE_FORMAT = "basisserve.attention_o_proj_covariances.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, payload: Mapping[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(payload), str(temporary))
    os.replace(temporary, path)


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
    selected: set[int] = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            first, last = map(int, piece.split("-", 1))
            if last < first:
                raise ValueError(f"descending layer range: {piece}")
            selected.update(range(first, last + 1))
        else:
            selected.add(int(piece))
    layers = tuple(sorted(selected))
    if not layers or min(layers) < 0 or max(layers) >= num_layers:
        raise ValueError("selected layers are outside the model")
    return layers


def _work_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _factor_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _add_relative_l2(metrics: dict[str, Any]) -> None:
    for name, value in tuple(metrics.items()):
        if name.endswith("_relative_mse"):
            metrics[name.removesuffix("_mse") + "_l2"] = math.sqrt(
                max(float(value), 0.0)
            )


def _validate_covariances(
    covariance_dir: Path,
    *,
    tp_size: int,
    source_rank: int,
) -> dict[str, Any]:
    manifest_path = covariance_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json(manifest_path)
    if manifest.get("format") != COVARIANCE_FORMAT:
        raise ValueError("unsupported covariance snapshot format")
    model = manifest["model"]
    hidden_size = int(model["hidden_size"])
    if model.get("model_type") != "qwen3" or hidden_size != 4096:
        raise ValueError("this entrypoint requires the audited Qwen3-8B geometry")
    if int(model["num_hidden_layers"]) != 36:
        raise ValueError("this entrypoint requires 36 Qwen3-8B layers")
    if tp_size <= 1 or hidden_size % tp_size:
        raise ValueError("TP size must divide the W_o input width")
    source_width = hidden_size // tp_size
    if not 0 < source_rank <= source_width:
        raise ValueError("source rank must lie within one TP source width")
    calibration = manifest["calibration"]
    if calibration.get("objective") != "attention_o_proj_output_mse":
        raise ValueError("covariances do not describe W_o inputs")
    if not calibration.get("activation_aware"):
        raise ValueError("Phase 1 requires activation-aware covariances")
    return manifest


def _c1_config(
    *,
    encoder_sweeps: int,
    minimum_encoder_sweeps: int,
) -> TPSourceWOFitConfig:
    return TPSourceWOFitConfig(
        encoder_sweeps=encoder_sweeps,
        minimum_encoder_sweeps=minimum_encoder_sweeps,
        covariance_damping=1.0e-5,
        encoder_relative_tolerance=1.0e-6,
        encoder_patience=2,
        maximum_backtracks=10,
    )


def _run_signature(
    *,
    tp_size: int,
    source_rank: int,
    work_dtype: torch.dtype,
    factor_dtype: torch.dtype,
    fit_config: TPSourceWOFitConfig,
) -> dict[str, Any]:
    return {
        "tp_size": tp_size,
        "source_rank": source_rank,
        "work_dtype": str(work_dtype).removeprefix("torch."),
        "factor_dtype": str(factor_dtype).removeprefix("torch."),
        "c1_fit_config": asdict(fit_config),
    }


def _layer_paths(output_dir: Path, layer: int) -> tuple[Path, Path]:
    return (
        output_dir / f"layer_{layer:03d}.safetensors",
        output_dir / f"layer_{layer:03d}.json",
    )


def _verified_record(
    output_dir: Path,
    layer: int,
    *,
    covariance_sha256: str,
    run_signature: Mapping[str, Any],
) -> dict[str, Any] | None:
    artifact_path, record_path = _layer_paths(output_dir, layer)
    if not artifact_path.exists() and not record_path.exists():
        return None
    if not artifact_path.is_file() or not record_path.is_file():
        raise RuntimeError(f"incomplete Phase-1 layer output at {layer}")
    record = _load_json(record_path)
    valid = (
        record.get("format") == LAYER_FORMAT
        and int(record.get("layer", -1)) == layer
        and record.get("source", {}).get("covariance_sha256")
        == covariance_sha256
        and record.get("run_signature") == dict(run_signature)
        and record.get("artifact", {}).get("sha256") == _sha256(artifact_path)
    )
    if not valid:
        raise RuntimeError(f"incompatible Phase-1 resume record at layer {layer}")
    return record


def _fit_layer(
    *,
    layer: int,
    covariance_dir: Path,
    covariance_manifest: Mapping[str, Any],
    output_dir: Path,
    tp_size: int,
    source_rank: int,
    fit_config: TPSourceWOFitConfig,
    work_dtype: torch.dtype,
    factor_dtype: torch.dtype,
    device: torch.device,
    resume: bool,
) -> dict[str, Any]:
    covariance_record = covariance_manifest["artifacts"][str(layer)]
    covariance_path = covariance_dir / covariance_record["file"]
    covariance_sha256 = _sha256(covariance_path)
    if covariance_sha256 != covariance_record["sha256"]:
        raise ValueError(f"covariance hash mismatch at layer {layer}")
    signature = _run_signature(
        tp_size=tp_size,
        source_rank=source_rank,
        work_dtype=work_dtype,
        factor_dtype=factor_dtype,
        fit_config=fit_config,
    )
    prior = _verified_record(
        output_dir,
        layer,
        covariance_sha256=covariance_sha256,
        run_signature=signature,
    )
    if prior is not None:
        if not resume:
            raise FileExistsError(f"Phase-1 layer output already exists: {layer}")
        print(f"[Wo Phase1] layer={layer} resume=verified", flush=True)
        return prior

    started = time.perf_counter()
    payload = load_file(str(covariance_path), device="cpu")
    fit_covariance = payload["fit_covariance"].to(
        device=device, dtype=work_dtype
    )
    heldout_covariance = payload["heldout_covariance"].to(
        device=device, dtype=work_dtype
    )
    weight = payload["weight"].to(device=device, dtype=work_dtype)
    hidden_size = int(covariance_manifest["model"]["hidden_size"])
    layout = TPSourceWOLayout(
        input_width=hidden_size,
        output_width=hidden_size,
        tp_size=tp_size,
        source_rank=source_rank,
        dtype_bytes=torch.empty((), dtype=factor_dtype).element_size(),
    )
    source_ranks = (source_rank,) * tp_size
    capacity_rank = layout.gathered_width
    wire_rank = wire_matched_allreduce_rank(source_ranks)

    print(
        f"[Wo Phase1] layer={layer} phase=fit_c1_ag "
        f"source_rank={source_rank} sweeps={fit_config.encoder_sweeps}",
        flush=True,
    )
    c1 = fit_tp_source_wo_c1(
        weight,
        fit_covariance,
        heldout_covariance,
        layout,
        config=fit_config,
        work_device=device,
        work_dtype=work_dtype,
        factor_dtype=factor_dtype,
        objective_name=f"qwen3_8b_layer_{layer}_wo_c1",
    )
    c1_approximation = fold_factors_to_dense_weight(
        c1.source_encoders.to(device=device, dtype=work_dtype),
        c1.source_decoders.to(device=device, dtype=work_dtype),
        layout,
    )
    _, absolute_damping = trace_damped_covariance(
        fit_covariance,
        relative_damping=fit_config.covariance_damping,
    )
    c1_metrics: dict[str, Any] = {
        "exact_fit_damped_relative_mse": c1.fit_relative_mse,
        "exact_heldout_relative_mse": c1.heldout_relative_mse,
        "factor_dtype_fit_damped_relative_mse": c1.quantized_fit_relative_mse,
        "factor_dtype_heldout_relative_mse": c1.quantized_heldout_relative_mse,
        "factor_dtype_fit_raw_relative_mse": relative_output_mse(
            weight, c1_approximation, fit_covariance
        ),
        "factor_dtype_heldout_direct_relative_mse": relative_output_mse(
            weight, c1_approximation, heldout_covariance
        ),
        "selected_boundary": c1.selected_boundary,
        "selected_sweep": c1.selected_sweep,
        "checkpoints": list(c1.checkpoints),
        "diagnostics": c1.diagnostics,
    }
    _add_relative_l2(c1_metrics)
    heldout_consistency = abs(
        float(c1_metrics["factor_dtype_heldout_direct_relative_mse"])
        - float(c1_metrics["factor_dtype_heldout_relative_mse"])
    )
    if heldout_consistency > 5.0e-6:
        raise RuntimeError(
            f"C1 folded-weight metric mismatch at layer {layer}: "
            f"{heldout_consistency}"
        )

    print(
        f"[Wo Phase1] layer={layer} phase=fit_lr_ar "
        f"wire_rank={wire_rank} capacity_rank={capacity_rank}",
        flush=True,
    )
    bank = fit_strong_lr_allreduce_rank_bank(
        weight,
        fit_covariance,
        heldout_covariance,
        ranks=(wire_rank, capacity_rank),
        covariance_damping=fit_config.covariance_damping,
        work_dtype=work_dtype,
        factor_dtype=factor_dtype,
    )
    by_rank = {factors.rank: factors for factors in bank}
    wire = by_rank[wire_rank]
    capacity = by_rank[capacity_rank]
    for name, factors in (("wire", wire), ("capacity", capacity)):
        if not bool(torch.isfinite(factors.input_factor).all()) or not bool(
            torch.isfinite(factors.decoder).all()
        ):
            raise RuntimeError(f"{name} LR factors contain non-finite values")
    input_prefix = capacity.input_factor[:, :wire_rank]
    decoder_prefix = capacity.decoder[:wire_rank]
    prefix_diagnostics = {
        "input_factor_bitwise_equal": torch.equal(
            wire.input_factor, input_prefix
        ),
        "decoder_bitwise_equal": torch.equal(wire.decoder, decoder_prefix),
        "input_factor_maximum_absolute_difference": float(
            (wire.input_factor.float() - input_prefix.float()).abs().max()
        ),
        "decoder_maximum_absolute_difference": float(
            (wire.decoder.float() - decoder_prefix.float()).abs().max()
        ),
        "note": (
            "Both ranks share one eigendecomposition, but GEMM shape can change "
            "last-bit factor rounding; both factor sets are stored explicitly."
        ),
    }

    exact_gap = float(capacity.metrics["fit_damped_relative_mse"]) - float(
        c1.fit_relative_mse
    )
    quantized_gap = float(
        capacity.metrics["factor_dtype_fit_raw_relative_mse"]
    ) - float(c1_metrics["factor_dtype_fit_raw_relative_mse"])
    tolerance = max(1.0e-7, 1.0e-5 * max(c1.fit_relative_mse, 1.0e-12))
    inclusion_passed = exact_gap <= tolerance

    artifact_path, record_path = _layer_paths(output_dir, layer)
    artifact_payload = {
        "c1_source_encoders": c1.source_encoders,
        "c1_source_decoders": c1.source_decoders,
        "lr_wire_input_factor": wire.input_factor,
        "lr_wire_shared_decoder": wire.decoder,
        "lr_wire_singular_values": wire.singular_values,
        "lr_capacity_input_factor": capacity.input_factor,
        "lr_capacity_shared_decoder": capacity.decoder,
        "lr_capacity_singular_values": capacity.singular_values,
    }
    _atomic_safetensors(artifact_path, artifact_payload)
    rows = 1
    record = {
        "format": LAYER_FORMAT,
        "layer": layer,
        "status": "passed" if inclusion_passed else "kill_gate_1_failed",
        "source": {
            "covariance_file": covariance_path.name,
            "covariance_sha256": covariance_sha256,
        },
        "run_signature": signature,
        "geometry": {
            "tp_size": tp_size,
            "input_width": hidden_size,
            "output_width": hidden_size,
            "source_width": layout.source_width,
            "c1_source_ranks": list(source_ranks),
            "c1_gathered_rank": capacity_rank,
            "wire_matched_lr_ar_rank": wire_rank,
            "capacity_matched_lr_ar_rank": capacity_rank,
            "source_order": "contiguous W_o input shards in TP-rank order",
            "dense_v": True,
            "kv_cache_compression": "none",
        },
        "c1_allgather": c1_metrics,
        "lr_allreduce": {
            "wire_matched": dict(wire.metrics),
            "capacity_matched": dict(capacity.metrics),
        },
        "inclusion_gate": {
            "contract": "capacity-matched exact LR-AR fit <= exact Wo-C1 fit",
            "passed": inclusion_passed,
            "tolerance": tolerance,
            "exact_fit_mse_gap_lr_minus_c1": exact_gap,
            "factor_dtype_raw_fit_mse_gap_lr_minus_c1": quantized_gap,
        },
        "communication": {
            "model": "ideal ring bytes per rank per activation row",
            "dtype_bytes": layout.dtype_bytes,
            "c1_allgather": ring_allgather_bytes_per_rank(
                rows=rows,
                source_ranks=source_ranks,
                dtype_bytes=layout.dtype_bytes,
            ),
            "lr_ar_wire_matched": ring_allreduce_bytes_per_rank(
                rows=rows,
                rank=wire_rank,
                tp_size=tp_size,
                dtype_bytes=layout.dtype_bytes,
            ),
            "lr_ar_capacity_matched": ring_allreduce_bytes_per_rank(
                rows=rows,
                rank=capacity_rank,
                tp_size=tp_size,
                dtype_bytes=layout.dtype_bytes,
            ),
            "dense_wo_allreduce": ring_allreduce_bytes_per_rank(
                rows=rows,
                rank=hidden_size,
                tp_size=tp_size,
                dtype_bytes=layout.dtype_bytes,
            ),
        },
        "artifact": {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "tensors": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in artifact_payload.items()
            },
        },
        "numerics": {
            "absolute_covariance_damping": absolute_damping,
            "c1_folded_heldout_consistency_absolute_error": heldout_consistency,
            "lr_rank_bank_prefix": prefix_diagnostics,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    _atomic_json(record_path, record)
    print(
        f"[Wo Phase1] layer={layer} status={record['status']} "
        f"c1={c1.quantized_heldout_relative_mse:.8g} "
        f"wire={wire.metrics['factor_dtype_heldout_relative_mse']:.8g} "
        f"capacity={capacity.metrics['factor_dtype_heldout_relative_mse']:.8g}",
        flush=True,
    )
    if not inclusion_passed:
        raise RuntimeError(
            f"capacity inclusion gate failed at layer {layer}: gap={exact_gap}"
        )
    return record


def _mean(records: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    values = []
    for record in records:
        current: Any = record
        for key in path:
            current = current[key]
        values.append(float(current))
    return sum(values) / len(values)


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Qwen3-8B Wo-only C1-AllGather vs LR-AllReduce: Phase 1",
        "",
        (
            "Both methods preserve dense V and compress only the post-attention "
            "`W_o` map. C1 uses private TP-source decoders; LR-AllReduce uses "
            "one globally optimal activation-aware shared decoder."
        ),
        "",
        "| Layer | Wo-C1 heldout MSE | Wire LR-AR heldout MSE | Capacity LR-AR heldout MSE | Inclusion |",
        "|---:|---:|---:|---:|---|",
    ]
    for record in payload["layers"]:
        lines.append(
            f"| {record['layer']} | "
            f"{record['c1_allgather']['factor_dtype_heldout_relative_mse']:.6g} | "
            f"{record['lr_allreduce']['wire_matched']['factor_dtype_heldout_relative_mse']:.6g} | "
            f"{record['lr_allreduce']['capacity_matched']['factor_dtype_heldout_relative_mse']:.6g} | "
            f"{'pass' if record['inclusion_gate']['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Layers completed: `{summary['layers_completed']}`.",
            f"Capacity inclusion gates passed: `{summary['inclusion_gates_passed']}`.",
            (
                "Mean BF16 heldout relative MSE: "
                f"Wo-C1 `{summary['mean_c1_heldout_relative_mse']:.6g}`, "
                f"wire-matched LR-AR `{summary['mean_wire_lr_ar_heldout_relative_mse']:.6g}`, "
                f"capacity-matched LR-AR "
                f"`{summary['mean_capacity_lr_ar_heldout_relative_mse']:.6g}`."
            ),
            "",
            "These are W_o/layer-output errors, not terminal-logit metrics or PPL.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covariance-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--source-rank", type=int, default=512)
    parser.add_argument("--c1-encoder-sweeps", type=int, default=5)
    parser.add_argument("--c1-minimum-encoder-sweeps", type=int, default=5)
    parser.add_argument(
        "--work-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--factor-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-final-summary",
        action="store_true",
        help="Write only per-layer records for a parallel layer worker.",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    covariance_dir = Path(args.covariance_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    covariance = _validate_covariances(
        covariance_dir,
        tp_size=args.tp_size,
        source_rank=args.source_rank,
    )
    layers = _parse_layers(
        args.layers,
        num_layers=int(covariance["model"]["num_hidden_layers"]),
    )
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    work_dtype = _work_dtype(args.work_dtype)
    factor_dtype = _factor_dtype(args.factor_dtype)
    fit_config = _c1_config(
        encoder_sweeps=args.c1_encoder_sweeps,
        minimum_encoder_sweeps=args.c1_minimum_encoder_sweeps,
    )

    started = time.perf_counter()
    records = []
    for ordinal, layer in enumerate(layers, start=1):
        print(
            f"[Wo Phase1] layer={layer} progress={ordinal}/{len(layers)}",
            flush=True,
        )
        records.append(
            _fit_layer(
                layer=layer,
                covariance_dir=covariance_dir,
                covariance_manifest=covariance,
                output_dir=output_dir,
                tp_size=args.tp_size,
                source_rank=args.source_rank,
                fit_config=fit_config,
                work_dtype=work_dtype,
                factor_dtype=factor_dtype,
                device=device,
                resume=args.resume,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    records.sort(key=lambda row: int(row["layer"]))
    if args.skip_final_summary:
        print(
            f"[Wo Phase1] worker_complete layers={len(records)} "
            f"output={output_dir}",
            flush=True,
        )
        return
    summary = {
        "layers_completed": len(records),
        "inclusion_gates_passed": sum(
            bool(record["inclusion_gate"]["passed"]) for record in records
        ),
        "mean_c1_heldout_relative_mse": _mean(
            records, ("c1_allgather", "factor_dtype_heldout_relative_mse")
        ),
        "mean_wire_lr_ar_heldout_relative_mse": _mean(
            records,
            (
                "lr_allreduce",
                "wire_matched",
                "factor_dtype_heldout_relative_mse",
            ),
        ),
        "mean_capacity_lr_ar_heldout_relative_mse": _mean(
            records,
            (
                "lr_allreduce",
                "capacity_matched",
                "factor_dtype_heldout_relative_mse",
            ),
        ),
    }
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - started,
        "model": covariance["model"],
        "calibration": covariance["calibration"],
        "source": {
            "covariance_dir": str(covariance_dir),
            "covariance_manifest_sha256": _sha256(
                covariance_dir / "manifest.json"
            ),
        },
        "method": {
            "scope": "post-attention W_o only; dense V and dense KV cache",
            "c1_allgather": (
                "TP-source activation-aware initialization plus full-layer "
                "decoder-closed ALS with exact two-sided encoder solves"
            ),
            "lr_allreduce": (
                "global activation-aware SVD with source-specific encoder shards "
                "and one shared decoder"
            ),
            "budget_modes": ["wire_matched", "capacity_matched"],
            "run_signature": _run_signature(
                tp_size=args.tp_size,
                source_rank=args.source_rank,
                work_dtype=work_dtype,
                factor_dtype=factor_dtype,
                fit_config=fit_config,
            ),
        },
        "environment": {
            "conda": os.environ.get("CONDA_DEFAULT_ENV")
            or Path(sys.prefix).name,
            "python": sys.version,
            "torch": str(torch.__version__),
            "device": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else str(device)
            ),
            "torch_num_threads": torch.get_num_threads(),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "summary": summary,
        "layers": records,
    }
    _atomic_json(output_dir / "results.json", payload)
    (output_dir / "summary.md").write_text(
        _summary_markdown(payload), encoding="utf-8"
    )
    print(
        f"[Wo Phase1] complete layers={len(records)} "
        f"output={output_dir / 'results.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
