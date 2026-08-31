#!/usr/bin/env python3
"""Capture layer-23 streaming TSQR checkpoints and run QR(0) sample scaling."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
from transformers import AutoModel


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.pairwise_qr import (  # noqa: E402
    StreamingTSQR,
    pack_upper_triangular,
    solve_square_root_least_squares,
    unpack_upper_triangular,
)
from evaluation.analyze_qwen3_o_proj_collective_endpoints import (  # noqa: E402
    _sample_positions,
)
from evaluation.capture_attention_o_proj_covariances import (  # noqa: E402
    _load_ordered_windows,
)
from evaluation.capture_attention_o_proj_ppl_snapshots import (  # noqa: E402
    _atomic_json,
    _decoder_layers,
    _git_commit,
    _installed_version,
    _sha256,
    _validate_attention,
)


FORMAT = "basisserve.qwen3_32b.c1_layer23_streaming_tsqr.v1"
FIT_CHECKPOINT_FORMAT = "basisserve.qwen3_32b.c1_layer23_streaming_tsqr_fit.v1"
RESULT_FORMAT = "basisserve.qwen3_32b.c1_layer23_qr0_sample_scaling.v2"
LAYER = 23
NUM_QUERY_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
QUERY_WIDTH = NUM_QUERY_HEADS * HEAD_DIM
HIDDEN_SIZE = 5120
DEFAULT_MILESTONES = (16_384, 32_768, 65_536)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--fit-windows", type=int, default=256)
    parser.add_argument("--heldout-windows", type=int, default=64)
    parser.add_argument("--fit-positions-per-window", type=int, default=256)
    parser.add_argument("--heldout-positions-per-window", type=int, default=128)
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=DEFAULT_MILESTONES,
    )
    parser.add_argument("--position-seed", type=int, default=20260901)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--model-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map",
        choices=("none", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=70)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output-column-chunk-size", type=int, default=256)
    parser.add_argument(
        "--solve-dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, *, sequence_length: int) -> tuple[int, ...]:
    positive = (
        args.fit_windows,
        args.heldout_windows,
        args.fit_positions_per_window,
        args.heldout_positions_per_window,
        args.batch_size,
        args.max_memory_per_gpu_gib,
        args.torch_num_threads,
        args.output_column_chunk_size,
    )
    if min(positive) <= 0:
        raise ValueError("all counts and chunk sizes must be positive")
    if max(
        args.fit_positions_per_window,
        args.heldout_positions_per_window,
    ) > sequence_length:
        raise ValueError("sampled positions exceed the sequence length")
    milestones = tuple(sorted(set(map(int, args.milestones))))
    if not milestones or min(milestones) < QUERY_WIDTH:
        raise ValueError("TSQR milestones must contain at least one full-width leaf")
    if any(value % QUERY_WIDTH for value in milestones):
        raise ValueError("TSQR milestones must be multiples of the query width")
    expected_fit_rows = args.fit_windows * args.fit_positions_per_window
    if milestones[-1] != expected_fit_rows:
        raise ValueError("largest milestone must equal the complete fit row count")
    expected_heldout_rows = (
        args.heldout_windows * args.heldout_positions_per_window
    )
    if expected_heldout_rows != QUERY_WIDTH:
        raise ValueError("held-out sampling must produce exactly one full TSQR leaf")
    return milestones


def _validate_document_disjointness(
    manifest: Mapping[str, Any],
    *,
    fit_windows: int,
    heldout_windows: int,
) -> None:
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != fit_windows + heldout_windows:
        raise ValueError("window manifest records do not match the requested split")
    fit_documents = {str(item["document_id"]) for item in records[:fit_windows]}
    heldout_documents = {
        str(item["document_id"]) for item in records[fit_windows:]
    }
    overlap = fit_documents & heldout_documents
    if overlap:
        raise ValueError(f"fit and held-out windows share {len(overlap)} documents")


class _LayerTSQRCapture:
    def __init__(
        self,
        module: nn.Linear,
        *,
        input_width: int,
        fit_milestones: Sequence[int],
        heldout_rows: int,
        accumulation_dtype: torch.dtype,
    ) -> None:
        self.input_width = int(input_width)
        self.fit_milestones = tuple(map(int, fit_milestones))
        self.heldout_rows = int(heldout_rows)
        self.accumulation_dtype = accumulation_dtype
        self.active_split: str | None = None
        self.active_positions: Tensor | None = None
        self.seen = False
        self.accumulators: dict[str, StreamingTSQR | None] = {
            "fit": None,
            "heldout": None,
        }
        self.fit_checkpoints: dict[int, Tensor] = {}
        self.fit_diagnostics: dict[int, dict[str, Any]] = {}
        self.heldout_r: Tensor | None = None
        self.heldout_diagnostics: dict[str, Any] | None = None
        self._handle = module.register_forward_pre_hook(self._hook)

    def _accumulator(self, split: str, device: torch.device) -> StreamingTSQR:
        accumulator = self.accumulators[split]
        if accumulator is None:
            accumulator = StreamingTSQR(
                columns=self.input_width,
                block_rows=self.input_width,
                device=device,
                dtype=self.accumulation_dtype,
            )
            self.accumulators[split] = accumulator
        elif accumulator.device != device:
            raise RuntimeError("layer-23 activation device changed during capture")
        return accumulator

    @torch.no_grad()
    def _checkpoint(self, split: str, accumulator: StreamingTSQR) -> None:
        rows = accumulator.total_rows
        if split == "fit" and rows in self.fit_milestones:
            if rows in self.fit_checkpoints:
                raise RuntimeError(f"duplicate fit TSQR checkpoint at {rows} rows")
            normalized = accumulator.snapshot_r().mul_(rows**-0.5)
            self.fit_checkpoints[rows] = pack_upper_triangular(
                normalized.cpu().contiguous()
            )
            self.fit_diagnostics[rows] = asdict(accumulator.diagnostics())
            del normalized
            print(f"[TSQR] checkpoint split=fit rows={rows}", flush=True)
        elif split == "heldout" and rows == self.heldout_rows:
            if self.heldout_r is not None:
                raise RuntimeError("duplicate held-out TSQR checkpoint")
            normalized = accumulator.snapshot_r().mul_(rows**-0.5)
            self.heldout_r = pack_upper_triangular(
                normalized.cpu().contiguous()
            )
            self.heldout_diagnostics = asdict(accumulator.diagnostics())
            del normalized
            print(f"[TSQR] checkpoint split=heldout rows={rows}", flush=True)

    def _hook(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if self.active_split is None or self.active_positions is None:
            raise RuntimeError("layer-23 TSQR hook fired outside an active batch")
        if self.seen or not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError("layer-23 TSQR hook fired an unexpected number of times")
        activation = inputs[0].detach()
        if (
            activation.ndim != 3
            or int(activation.shape[0]) != int(self.active_positions.shape[0])
            or int(activation.shape[-1]) != self.input_width
        ):
            raise ValueError(
                f"unexpected layer-23 activation shape: {tuple(activation.shape)}"
            )
        positions = self.active_positions.to(device=activation.device)
        batch_indices = torch.arange(
            int(activation.shape[0]),
            device=activation.device,
        ).unsqueeze(1)
        selected = activation[batch_indices, positions].reshape(-1, self.input_width)
        accumulator = self._accumulator(self.active_split, activation.device)

        source_start = 0
        while source_start < len(selected):
            if self.active_split == "fit":
                pending = [
                    value
                    for value in self.fit_milestones
                    if value > accumulator.total_rows
                ]
                if not pending:
                    raise RuntimeError("fit capture exceeded the final milestone")
                target = pending[0]
            else:
                target = self.heldout_rows
            count = min(target - accumulator.total_rows, len(selected) - source_start)
            accumulator.append(selected[source_start : source_start + count])
            source_start += count
            if accumulator.total_rows == target:
                self._checkpoint(self.active_split, accumulator)
        self.seen = True

    def begin(self, split: str, positions: Tensor) -> None:
        if self.active_split is not None or split not in self.accumulators:
            raise RuntimeError("invalid TSQR capture batch transition")
        if positions.ndim != 2 or not int(positions.numel()):
            raise ValueError("capture positions must be a nonempty matrix")
        self.active_split = split
        self.active_positions = positions
        self.seen = False

    def finish(self) -> None:
        if self.active_split is None or not self.seen:
            raise RuntimeError("layer-23 TSQR capture missed its hook")
        self.active_split = None
        self.active_positions = None
        self.seen = False

    def finish_split(self, split: str) -> None:
        accumulator = self.accumulators.get(split)
        expected = self.fit_milestones[-1] if split == "fit" else self.heldout_rows
        if (
            self.active_split is not None
            or accumulator is None
            or accumulator.total_rows != expected
            or accumulator.buffered_rows
        ):
            raise RuntimeError(f"cannot finish incomplete TSQR split: {split}")
        if split == "fit" and set(self.fit_checkpoints) != set(self.fit_milestones):
            raise RuntimeError("fit TSQR milestones are incomplete")
        if split == "heldout" and self.heldout_r is None:
            raise RuntimeError("held-out TSQR checkpoint is incomplete")
        self.accumulators[split] = None
        del accumulator

    def close(self) -> None:
        self._handle.remove()


def _load_fixed_factors(
    factor_dir: Path,
    *,
    model_config_sha256: str,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    record_path = factor_dir / f"layer_{LAYER:03d}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if int(record.get("layer", -1)) != LAYER:
        raise ValueError("fixed-factor record has the wrong layer")
    config = record["fit_config"]
    if config.get("model_config_sha256") != model_config_sha256:
        raise ValueError("fixed factors and capture model differ")
    rank = int(config["cache_rank_per_head"])
    artifact_path = factor_dir / record["artifact"]["file"]
    if _sha256(artifact_path) != record["artifact"]["sha256"]:
        raise ValueError("fixed-factor artifact hash mismatch")
    payload = load_file(str(artifact_path), device="cpu")
    fixed_a = payload["value_coordinate_encoders"].contiguous()
    checkpoint_decoder = payload["head_output_decoders"].contiguous()
    if tuple(fixed_a.shape) != (NUM_KV_HEADS, HEAD_DIM, rank):
        raise ValueError("fixed encoder has incompatible shape")
    if tuple(checkpoint_decoder.shape) != (NUM_QUERY_HEADS, rank, HIDDEN_SIZE):
        raise ValueError("checkpoint decoder has incompatible shape")
    source = {
        "record_file": str(record_path),
        "artifact_file": str(artifact_path),
        "artifact_sha256": record["artifact"]["sha256"],
        "rank_per_head": rank,
        "covariance_damping": float(config["covariance_damping"]),
    }
    return fixed_a, checkpoint_decoder, source


@torch.no_grad()
def _dense_block_basis(fixed_a: Tensor, *, device: torch.device) -> Tensor:
    rank = int(fixed_a.shape[2])
    mapping = torch.arange(NUM_QUERY_HEADS, device=device) // (
        NUM_QUERY_HEADS // NUM_KV_HEADS
    )
    by_head = fixed_a.index_select(0, mapping)
    basis = torch.zeros(
        QUERY_WIDTH,
        NUM_QUERY_HEADS * rank,
        device=device,
        dtype=fixed_a.dtype,
    )
    for head in range(NUM_QUERY_HEADS):
        basis[
            head * HEAD_DIM : (head + 1) * HEAD_DIM,
            head * rank : (head + 1) * rank,
        ] = by_head[head]
    return basis


@torch.no_grad()
def _factor_output_energy(
    left_factor: Tensor,
    matrix: Tensor,
    *,
    output_chunk_size: int,
) -> float:
    if left_factor.shape[1] != matrix.shape[0]:
        raise ValueError("left factor and weight matrix have incompatible shapes")
    energy = 0.0
    for start in range(0, int(matrix.shape[1]), output_chunk_size):
        stop = min(start + output_chunk_size, int(matrix.shape[1]))
        output = left_factor @ matrix[:, start:stop]
        energy += float(output.square().sum(dtype=torch.float64))
    return energy


def _load_packed_r(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    payload = load_file(str(path), device="cpu")
    if set(payload) != {"r_upper_packed"}:
        raise ValueError(f"unexpected TSQR artifact tensors: {path}")
    return unpack_upper_triangular(
        payload["r_upper_packed"],
        dimension=QUERY_WIDTH,
    ).to(device=device, dtype=dtype)


def _persist_fit_checkpoint(
    *,
    partial_dir: Path,
    capture: _LayerTSQRCapture,
    milestones: Sequence[int],
    fit_positions: Tensor,
    heldout_positions: Tensor,
    weight: Tensor,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Durably write fit state before starting held-out model forwards."""

    positions_path = partial_dir / "positions.safetensors"
    save_file(
        {
            "fit_positions": fit_positions.contiguous(),
            "heldout_positions": heldout_positions.contiguous(),
        },
        str(positions_path),
    )
    artifacts: dict[str, Any] = {}
    for rows in milestones:
        path = partial_dir / f"fit_r_{rows:06d}.safetensors"
        value = capture.fit_checkpoints[rows]
        save_file({"r_upper_packed": value}, str(path))
        artifacts[str(rows)] = {
            "file": path.name,
            "sha256": _sha256(path),
            "matrix_shape": [QUERY_WIDTH, QUERY_WIDTH],
            "packed_shape": list(value.shape),
            "dtype": str(value.dtype),
            "storage": "rowwise_packed_upper_triangle",
            "normalization": "R.T @ R = X.T @ X / rows",
            "tsqr": capture.fit_diagnostics[rows],
        }
    weight_path = partial_dir / "weight.safetensors"
    save_file({"weight": weight}, str(weight_path))
    checkpoint_path = partial_dir / "fit_checkpoint.json"
    _atomic_json(
        checkpoint_path,
        {
            "format": FIT_CHECKPOINT_FORMAT,
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "fit_milestones": list(map(int, milestones)),
            "positions": {
                "file": positions_path.name,
                "sha256": _sha256(positions_path),
            },
            "fit": artifacts,
            "weight": {
                "file": weight_path.name,
                "sha256": _sha256(weight_path),
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
            },
        },
    )
    return artifacts, {
        "file": checkpoint_path.name,
        "sha256": _sha256(checkpoint_path),
    }


@torch.no_grad()
def _decoder_metrics(
    *,
    fixed_a: Tensor,
    decoder: Tensor,
    target_weight: Tensor,
    fit_r: Tensor,
    heldout_r: Tensor,
    output_chunk_size: int,
    fit_target_energy: float,
    heldout_target_energy: float,
) -> dict[str, float]:
    mapping = torch.arange(NUM_QUERY_HEADS, device=fixed_a.device) // (
        NUM_QUERY_HEADS // NUM_KV_HEADS
    )
    approximation = torch.einsum(
        "hdr,hro->hdo",
        fixed_a.index_select(0, mapping),
        decoder,
    )
    residual_weight = approximation.reshape(QUERY_WIDTH, HIDDEN_SIZE) - target_weight
    fit_error = _factor_output_energy(
        fit_r,
        residual_weight,
        output_chunk_size=output_chunk_size,
    )
    heldout_error = _factor_output_energy(
        heldout_r,
        residual_weight,
        output_chunk_size=output_chunk_size,
    )
    return {
        "fit_relative_mse": fit_error / fit_target_energy,
        "heldout_relative_mse": heldout_error / heldout_target_energy,
        "decoder_frobenius_norm": math.sqrt(
            float(decoder.square().sum(dtype=torch.float64))
        ),
        "decoder_maximum_absolute_value": float(decoder.abs().max()),
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3-32B layer-23 QR(0) sample scaling with streaming TSQR",
        "",
        "Raw layer activations are never written to disk. Fit TSQR checkpoints use "
        "64, 128, and 256 independent C4 windows with 256 sampled positions per "
        "window. Every checkpoint is evaluated on the same 64 document-disjoint "
        "held-out windows with 128 sampled positions per window.",
        f"Decoder solves and metric evaluation use "
        f"`{payload['configuration']['work_dtype']}`.",
        "",
        "## Command",
        "",
        "```bash",
        str(payload["command"]),
        "```",
        "",
        "## Results",
        "",
        "| Fit rows | Fit documents | Status | Fit relMSE | Held-out relMSE | "
        "Decoder norm | Min abs R diagonal | R diagonal ratio |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in payload["checkpoints"]:
        if checkpoint["status"] != "complete":
            lines.append(
                f"| {checkpoint['fit_rows']} | {checkpoint['fit_windows']} | "
                "failed | — | — | — | — | — |"
            )
            continue
        metrics = checkpoint["metrics"]
        solve = checkpoint["solve"]
        lines.append(
            f"| {checkpoint['fit_rows']} | {checkpoint['fit_windows']} | "
            "complete | "
            f"{metrics['fit_relative_mse']:.9e} | "
            f"{metrics['heldout_relative_mse']:.9e} | "
            f"{metrics['decoder_frobenius_norm']:.6e} | "
            f"{solve['minimum_abs_r_diagonal']:.6e} | "
            f"{solve['r_diagonal_ratio']:.3f} |"
        )
    baseline = payload["checkpoint_decoder_control"]
    trend = payload["trend"]
    lines.extend(
        [
            "",
            "The stored damped-checkpoint decoder, evaluated on the same new held-out "
            f"split, has relative MSE `{baseline['heldout_relative_mse']:.9e}`.",
            "",
            "## Trend",
            "",
        ]
    )
    if trend["available"]:
        lines.extend(
            [
                f"- From {trend['first_fit_rows']} to {trend['last_fit_rows']} rows, "
                f"QR(0) held-out MSE changes by "
                f"{trend['heldout_relative_change']:.3%}.",
                f"- Decoder norm changes by "
                f"{trend['decoder_norm_relative_change']:.3%}.",
                f"- Minimum abs R diagonal changes by "
                f"{trend['minimum_r_diagonal_relative_change']:.3%}.",
            ]
        )
    lines.append(f"- Interpretation: {trend['interpretation']}")
    failures = [
        checkpoint
        for checkpoint in payload["checkpoints"]
        if checkpoint["status"] != "complete"
    ]
    if failures:
        lines.extend(["", "## Failed checkpoints", ""])
        for checkpoint in failures:
            lines.append(
                f"- `{checkpoint['fit_rows']}` rows: `{checkpoint['error']}`"
            )
    lines.append("")
    return "\n".join(lines)


def _validated_artifact_path(
    capture_dir: Path,
    record: Mapping[str, Any],
) -> Path:
    path = capture_dir / str(record["file"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise ValueError(f"capture artifact hash mismatch: {path}")
    return path


@torch.no_grad()
def _evaluate_saved_capture(
    *,
    capture_dir: Path,
    factor_dir: Path,
    output_json: Path,
    output_markdown: Path,
    device: torch.device,
    solve_dtype: torch.dtype,
    output_column_chunk_size: int,
    command: str,
) -> None:
    for output in (output_json, output_markdown):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")
    manifest_path = capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("incompatible streaming TSQR capture format")
    milestones = tuple(map(int, manifest["calibration"]["fit_milestones"]))
    if not milestones:
        raise ValueError("capture has no fit milestones")

    print("[Evaluate] loading fixed A and checkpoint decoder", flush=True)
    fixed_a_cpu, checkpoint_decoder_cpu, factor_source = _load_fixed_factors(
        factor_dir,
        model_config_sha256=str(manifest["model"]["config_sha256"]),
    )
    weight_record = manifest["artifacts"]["weight"]
    weight_path = _validated_artifact_path(capture_dir, weight_record)
    weight_payload = load_file(str(weight_path), device="cpu")
    if set(weight_payload) != {"weight"}:
        raise ValueError("unexpected target-weight artifact tensors")
    weight = weight_payload["weight"]
    if tuple(weight.shape) != (HIDDEN_SIZE, QUERY_WIDTH):
        raise ValueError("target weight has incompatible shape")

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    fixed_a = fixed_a_cpu.to(device=device, dtype=solve_dtype)
    checkpoint_decoder = checkpoint_decoder_cpu.to(
        device=device,
        dtype=solve_dtype,
    )
    target_weight = weight.to(
        device=device,
        dtype=solve_dtype,
    ).transpose(0, 1).contiguous()
    basis = _dense_block_basis(fixed_a, device=device)
    heldout_record = manifest["artifacts"]["heldout"]
    heldout_r = _load_packed_r(
        _validated_artifact_path(capture_dir, heldout_record),
        device=device,
        dtype=solve_dtype,
    )
    heldout_target_energy = _factor_output_energy(
        heldout_r,
        target_weight,
        output_chunk_size=output_column_chunk_size,
    )
    checkpoint_control = _decoder_metrics(
        fixed_a=fixed_a,
        decoder=checkpoint_decoder,
        target_weight=target_weight,
        fit_r=heldout_r,
        heldout_r=heldout_r,
        output_chunk_size=output_column_chunk_size,
        fit_target_energy=heldout_target_energy,
        heldout_target_energy=heldout_target_energy,
    )

    checkpoint_results: list[dict[str, Any]] = []
    fit_positions_per_window = int(
        manifest["calibration"]["fit_positions_per_window"]
    )
    for rows in milestones:
        print(
            f"[Evaluate] QR(0) fit_rows={rows} dtype={solve_dtype}",
            flush=True,
        )
        fit_record = manifest["artifacts"]["fit"][str(rows)]
        fit_r = _load_packed_r(
            _validated_artifact_path(capture_dir, fit_record),
            device=device,
            dtype=solve_dtype,
        )
        common = {
            "fit_rows": rows,
            "fit_windows": rows // fit_positions_per_window,
            "tsqr": fit_record["tsqr"],
        }
        try:
            decoder_flat, solve_diagnostics = solve_square_root_least_squares(
                left_factor=fit_r,
                basis=basis,
                target_weight=target_weight,
                absolute_product_damping=0.0,
                output_chunk_size=output_column_chunk_size,
            )
        except torch.linalg.LinAlgError as error:
            checkpoint_results.append(
                {
                    **common,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(f"[Evaluate] fit_rows={rows} failed: {error}", flush=True)
            del fit_r
            continue

        rank = int(fixed_a.shape[2])
        decoder = decoder_flat.reshape(NUM_QUERY_HEADS, rank, HIDDEN_SIZE)
        fit_target_energy = _factor_output_energy(
            fit_r,
            target_weight,
            output_chunk_size=output_column_chunk_size,
        )
        metrics = _decoder_metrics(
            fixed_a=fixed_a,
            decoder=decoder,
            target_weight=target_weight,
            fit_r=fit_r,
            heldout_r=heldout_r,
            output_chunk_size=output_column_chunk_size,
            fit_target_energy=fit_target_energy,
            heldout_target_energy=heldout_target_energy,
        )
        checkpoint_results.append(
            {
                **common,
                "status": "complete",
                "metrics": metrics,
                "solve": asdict(solve_diagnostics),
            }
        )
        print(
            f"[Evaluate] fit_rows={rows} heldout_rel_mse="
            f"{metrics['heldout_relative_mse']:.9e} decoder_norm="
            f"{metrics['decoder_frobenius_norm']:.6e}",
            flush=True,
        )
        del fit_r, decoder_flat, decoder

    successful = [
        checkpoint
        for checkpoint in checkpoint_results
        if checkpoint["status"] == "complete"
    ]
    if len(successful) >= 2:
        first = successful[0]
        last = successful[-1]
        heldout_change = (
            last["metrics"]["heldout_relative_mse"]
            / first["metrics"]["heldout_relative_mse"]
            - 1.0
        )
        norm_change = (
            last["metrics"]["decoder_frobenius_norm"]
            / first["metrics"]["decoder_frobenius_norm"]
            - 1.0
        )
        minimum_diagonal_change = (
            last["solve"]["minimum_abs_r_diagonal"]
            / first["solve"]["minimum_abs_r_diagonal"]
            - 1.0
        )
        if heldout_change <= -0.1 and norm_change <= -0.1:
            interpretation = (
                "strong sample-limited evidence: more independent calibration windows "
                "substantially reduce both decoder norm and held-out error"
            )
        elif abs(heldout_change) <= 0.05 and abs(norm_change) <= 0.05:
            interpretation = (
                "little sample-limited evidence over this range: the pathology is "
                "mostly structural or distributional"
            )
        else:
            interpretation = (
                "mixed sample-limited evidence: larger calibration changes the "
                "solution, but does not cleanly remove the pathology"
            )
        trend: dict[str, Any] = {
            "available": True,
            "first_fit_rows": first["fit_rows"],
            "last_fit_rows": last["fit_rows"],
            "heldout_relative_change": heldout_change,
            "decoder_norm_relative_change": norm_change,
            "minimum_r_diagonal_relative_change": minimum_diagonal_change,
            "interpretation": interpretation,
        }
    else:
        trend = {
            "available": False,
            "interpretation": (
                "fewer than two QR(0) checkpoints produced finite decoders, so a "
                "sample-scaling trend cannot be estimated"
            ),
        }

    torch.cuda.synchronize(device)
    solve_dtype_name = str(solve_dtype).removeprefix("torch.")
    environment = dict(manifest["environment"])
    environment.update(
        {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV")
            or Path(sys.prefix).name,
            "python_executable": sys.executable,
            "solve_dtype": solve_dtype_name,
            "evaluation_cuda_device": torch.cuda.get_device_name(device),
            "evaluation_peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
        }
    )
    result = {
        "format": RESULT_FORMAT,
        "status": "complete",
        "command": command,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "capture_manifest": str(manifest_path),
        "capture_manifest_sha256": _sha256(manifest_path),
        "factor_source": factor_source,
        "configuration": {
            "layer": LAYER,
            "fit_positions_per_window": fit_positions_per_window,
            "heldout_positions_per_window": int(
                manifest["calibration"]["heldout_positions_per_window"]
            ),
            "heldout_windows": int(manifest["calibration"]["heldout_windows"]),
            "heldout_rows": int(manifest["calibration"]["heldout_rows"]),
            "milestones": list(milestones),
            "fixed_a": True,
            "decoder_solver": "pairwise_tsqr_square_root_qr_lambda_0",
            "capture_dtype": "float32",
            "work_dtype": solve_dtype_name,
            "output_column_chunk_size": output_column_chunk_size,
        },
        "checkpoints": checkpoint_results,
        "failed_checkpoint_count": len(checkpoint_results) - len(successful),
        "checkpoint_decoder_control": checkpoint_control,
        "trend": trend,
        "timing": {
            "capture_seconds": float(manifest["elapsed_seconds"]),
            "evaluation_seconds": time.perf_counter() - started,
        },
        "environment": environment,
    }
    _atomic_json(output_json, result)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    temporary_markdown = output_markdown.with_suffix(output_markdown.suffix + ".tmp")
    temporary_markdown.write_text(_render_markdown(result), encoding="utf-8")
    os.replace(temporary_markdown, output_markdown)
    print(f"[Evaluate] wrote {output_json}", flush=True)
    print(f"[Evaluate] wrote {output_markdown}", flush=True)


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("streaming TSQR capture requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be CUDA")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)

    model_path = Path(args.model_path).expanduser().resolve()
    windows_path = args.windows.expanduser().resolve()
    factor_dir = args.factor_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    solve_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[args.solve_dtype]
    if args.evaluate_only:
        if not (output_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"completed capture is missing: {output_dir}")
        _evaluate_saved_capture(
            capture_dir=output_dir,
            factor_dir=factor_dir,
            output_json=output_json,
            output_markdown=output_markdown,
            device=device,
            solve_dtype=solve_dtype,
            output_column_chunk_size=args.output_column_chunk_size,
            command=shlex.join([sys.executable, *sys.argv]),
        )
        return
    for path in (output_dir, partial_dir, output_json, output_markdown):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)

    expected_windows = args.fit_windows + args.heldout_windows
    windows, windows_manifest = _load_ordered_windows(
        windows_path,
        expected=expected_windows,
    )
    sequence_length = int(windows.shape[1])
    milestones = _validate_args(args, sequence_length=sequence_length)
    _validate_document_disjointness(
        windows_manifest,
        fit_windows=args.fit_windows,
        heldout_windows=args.heldout_windows,
    )
    maximum_positions = max(
        args.fit_positions_per_window,
        args.heldout_positions_per_window,
    )
    position_pool = _sample_positions(
        windows=expected_windows,
        sequence_length=sequence_length,
        positions_per_window=maximum_positions,
        seed=args.position_seed,
    )
    fit_positions = position_pool[: args.fit_windows, : args.fit_positions_per_window]
    heldout_positions = position_pool[
        args.fit_windows :, : args.heldout_positions_per_window
    ]

    partial_dir.mkdir(parents=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    model_kwargs: dict[str, Any] = {
        "dtype": model_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
        "local_files_only": True,
    }
    if args.device_map == "none":
        model_kwargs["device_map"] = {"": str(device)}
    else:
        model_kwargs["device_map"] = args.device_map
        model_kwargs["max_memory"] = {
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        }
    print(
        f"[Capture] loading model={model_path} device_map={args.device_map}",
        flush=True,
    )
    model = AutoModel.from_pretrained(str(model_path), **model_kwargs).eval()
    model.config.use_cache = False
    placement = {
        str(key): str(value)
        for key, value in getattr(model, "hf_device_map", {}).items()
    }
    if any(value in {"cpu", "disk"} for value in placement.values()):
        raise RuntimeError(f"streaming TSQR capture offloaded parameters: {placement}")
    geometry = _validate_attention(model.config, "gqa")
    observed_geometry = {
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": NUM_QUERY_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
    }
    for key, expected in observed_geometry.items():
        if int(geometry[key]) != expected:
            raise ValueError(f"unexpected Qwen3-32B geometry for {key}")
    decoder_layers = _decoder_layers(model)
    module = decoder_layers[LAYER].self_attn.o_proj
    if not isinstance(module, nn.Linear) or tuple(module.weight.shape) != (
        HIDDEN_SIZE,
        QUERY_WIDTH,
    ):
        raise TypeError("layer-23 o_proj is unsupported")
    weight = module.weight.detach().cpu().contiguous()
    capture = _LayerTSQRCapture(
        module,
        input_width=QUERY_WIDTH,
        fit_milestones=milestones,
        heldout_rows=args.heldout_windows * args.heldout_positions_per_window,
        accumulation_dtype=torch.float32,
    )
    input_device = model.get_input_embeddings().weight.device
    if input_device.type != "cuda":
        raise RuntimeError("model input embeddings are not on CUDA")

    artifacts: dict[str, Any] | None = None
    fit_checkpoint_record: dict[str, str] | None = None
    try:
        total = len(windows)
        for batch_index, start in enumerate(
            range(0, total, args.batch_size),
            start=1,
        ):
            boundary = args.fit_windows if start < args.fit_windows else total
            stop = min(start + args.batch_size, boundary)
            split = "fit" if start < args.fit_windows else "heldout"
            positions = (
                fit_positions[start:stop]
                if split == "fit"
                else heldout_positions[
                    start - args.fit_windows : stop - args.fit_windows
                ]
            )
            input_ids = windows[start:stop].to(input_device)
            capture.begin(split, positions)
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            )
            del outputs, input_ids
            capture.finish()
            print(
                f"[Capture] batch={batch_index}/{total} split={split} "
                f"windows={stop}/{total}",
                flush=True,
            )
            if stop == args.fit_windows:
                artifacts, fit_checkpoint_record = _persist_fit_checkpoint(
                    partial_dir=partial_dir,
                    capture=capture,
                    milestones=milestones,
                    fit_positions=fit_positions,
                    heldout_positions=heldout_positions,
                    weight=weight,
                )
                print(
                    f"[Capture] persisted fit TSQR state to {partial_dir}",
                    flush=True,
                )
                capture.finish_split("fit")
                print("[Capture] released fit TSQR GPU state", flush=True)
        capture.finish_split("heldout")
        print("[Capture] released held-out TSQR GPU state", flush=True)
    finally:
        capture.close()

    assert capture.heldout_r is not None
    assert capture.heldout_diagnostics is not None
    assert artifacts is not None
    assert fit_checkpoint_record is not None
    heldout_path = partial_dir / "heldout_r_008192.safetensors"
    save_file({"r_upper_packed": capture.heldout_r}, str(heldout_path))
    weight_path = partial_dir / "weight.safetensors"

    capture_seconds = time.perf_counter() - started
    model_config_sha256 = _sha256(model_path / "config.json")
    manifest = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_started_utc": started_utc,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": capture_seconds,
        "model": {
            "path": str(model_path),
            "config_sha256": model_config_sha256,
            **geometry,
        },
        "layer": LAYER,
        "calibration": {
            "storage": "normalized_streaming_tsqr_r_only",
            "windows_file": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest_format": windows_manifest.get("format"),
            "fit_windows": args.fit_windows,
            "heldout_windows": args.heldout_windows,
            "sequence_length": sequence_length,
            "fit_positions_per_window": args.fit_positions_per_window,
            "heldout_positions_per_window": args.heldout_positions_per_window,
            "fit_milestones": list(milestones),
            "heldout_rows": args.heldout_windows
            * args.heldout_positions_per_window,
            "position_seed": args.position_seed,
            "positions_file": "positions.safetensors",
            "positions_sha256": _sha256(partial_dir / "positions.safetensors"),
            "fit_checkpoint": fit_checkpoint_record,
            "fit_and_heldout_document_disjoint": True,
        },
        "artifacts": {
            "fit": artifacts,
            "heldout": {
                "file": heldout_path.name,
                "sha256": _sha256(heldout_path),
                "matrix_shape": [QUERY_WIDTH, QUERY_WIDTH],
                "packed_shape": list(capture.heldout_r.shape),
                "dtype": str(capture.heldout_r.dtype),
                "storage": "rowwise_packed_upper_triangle",
                "normalization": "R.T @ R = X.T @ X / rows",
                "tsqr": capture.heldout_diagnostics,
            },
            "weight": {
                "file": weight_path.name,
                "sha256": _sha256(weight_path),
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
            },
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV")
            or Path(sys.prefix).name,
            "python_executable": sys.executable,
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(torch.cuda.device_count())
            },
            "device_map_strategy": args.device_map,
            "device_map": placement,
            "torch_num_threads": torch.get_num_threads(),
            "tsqr_dtype": "float32",
            "tf32_enabled": False,
        },
    }
    _atomic_json(partial_dir / "manifest.json", manifest)
    os.replace(partial_dir, output_dir)
    print(
        f"[Capture] complete output={output_dir} seconds={capture_seconds:.3f}",
        flush=True,
    )

    del capture, model, decoder_layers, module, windows, position_pool, weight
    gc.collect()
    _evaluate_saved_capture(
        capture_dir=output_dir,
        factor_dir=factor_dir,
        output_json=output_json,
        output_markdown=output_markdown,
        device=device,
        solve_dtype=solve_dtype,
        output_column_chunk_size=args.output_column_chunk_size,
        command=shlex.join([sys.executable, *sys.argv]),
    )


if __name__ == "__main__":
    main()
