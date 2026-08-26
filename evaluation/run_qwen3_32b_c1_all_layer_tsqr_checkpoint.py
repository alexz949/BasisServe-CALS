#!/usr/bin/env python3
"""Refit every Qwen3-32B uniform-C1 decoder with a 64k-row TSQR solve.

The Value encoders are fixed to an existing uniform-rank C1 checkpoint.
Raw activations are never persisted: every layer writes only normalized packed
fit and held-out TSQR factors.  Each decoder is then solved in float64 with
zero damping and saved in the repository's standard uniform-C1 checkpoint
layout so existing quality evaluators can consume it directly.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

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
from evaluation.eval_qwen3_32b_c1_wikitext import (  # noqa: E402
    FACTOR_FORMAT,
)
from evaluation.run_qwen3_32b_c1_layer23_tsqr_scaling import (  # noqa: E402
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_KV_HEADS,
    NUM_QUERY_HEADS,
    QUERY_WIDTH,
    _decoder_metrics,
    _dense_block_basis,
    _factor_output_energy,
    _validate_document_disjointness,
)


CAPTURE_FORMAT = "basisserve.qwen3_32b.c1_all_layer_streaming_tsqr.v1"
FIT_CHECKPOINT_FORMAT = (
    "basisserve.qwen3_32b.c1_all_layer_streaming_tsqr_fit.v1"
)
LAYER_FORMAT = "basisserve.qwen3_32b.gqa_c1_v96_joint.layer.v1"
NUM_LAYERS = 64
FIT_ROWS = 65_536
HELDOUT_ROWS = 8_192


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--solve-only", action="store_true")
    parser.add_argument("--fit-windows", type=int, default=256)
    parser.add_argument("--heldout-windows", type=int, default=64)
    parser.add_argument("--fit-positions-per-window", type=int, default=256)
    parser.add_argument("--heldout-positions-per-window", type=int, default=128)
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
    parser.add_argument(
        "--device-map",
        choices=("balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output-column-chunk-size", type=int, default=256)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, *, sequence_length: int) -> None:
    values = (
        args.fit_windows,
        args.heldout_windows,
        args.fit_positions_per_window,
        args.heldout_positions_per_window,
        args.batch_size,
        args.max_memory_per_gpu_gib,
        args.torch_num_threads,
        args.output_column_chunk_size,
    )
    if min(values) <= 0:
        raise ValueError("all counts and chunk sizes must be positive")
    if args.fit_windows * args.fit_positions_per_window != FIT_ROWS:
        raise ValueError("all-layer fit sampling must produce exactly 65536 rows")
    if args.heldout_windows * args.heldout_positions_per_window != HELDOUT_ROWS:
        raise ValueError("all-layer held-out sampling must produce exactly 8192 rows")
    if max(
        args.fit_positions_per_window,
        args.heldout_positions_per_window,
    ) > sequence_length:
        raise ValueError("sampled positions exceed the sequence length")


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _r_record(
    *,
    path: Path,
    packed: Tensor,
    rows: int,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "matrix_shape": [QUERY_WIDTH, QUERY_WIDTH],
        "packed_shape": list(packed.shape),
        "dtype": str(packed.dtype),
        "storage": "rowwise_packed_upper_triangle",
        "normalization": "R.T @ R = X.T @ X / rows",
        "rows": rows,
        "tsqr": dict(diagnostics),
    }


class _FinalLayerTSQRCapture:
    """Capture one final fit and held-out factor, writing each immediately."""

    def __init__(
        self,
        module: nn.Linear,
        *,
        layer: int,
        output_dir: Path,
        input_width: int,
        fit_rows: int,
        heldout_rows: int,
        accumulation_dtype: torch.dtype,
    ) -> None:
        self.layer = int(layer)
        self.output_dir = output_dir
        self.input_width = int(input_width)
        self.target_rows = {
            "fit": int(fit_rows),
            "heldout": int(heldout_rows),
        }
        self.accumulation_dtype = accumulation_dtype
        self.active_split: str | None = None
        self.active_positions: Tensor | None = None
        self.seen = False
        self.accumulators: dict[str, StreamingTSQR | None] = {
            "fit": None,
            "heldout": None,
        }
        self.records: dict[str, dict[str, Any]] = {}
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
            raise RuntimeError(
                f"layer {self.layer} activation device changed during capture"
            )
        return accumulator

    @torch.no_grad()
    def _write_checkpoint(
        self,
        split: str,
        accumulator: StreamingTSQR,
    ) -> None:
        if split in self.records:
            raise RuntimeError(
                f"duplicate layer {self.layer} {split} TSQR checkpoint"
            )
        rows = self.target_rows[split]
        normalized = accumulator.snapshot_r().mul_(rows**-0.5)
        full_cpu = normalized.cpu().contiguous()
        packed = pack_upper_triangular(full_cpu)
        path = self.output_dir / (
            f"layer_{self.layer:03d}_{split}_r_{rows:06d}.safetensors"
        )
        save_file({"r_upper_packed": packed}, str(path))
        self.records[split] = _r_record(
            path=path,
            packed=packed,
            rows=rows,
            diagnostics=asdict(accumulator.diagnostics()),
        )
        del normalized, full_cpu, packed
        print(
            f"[TSQR] layer={self.layer} split={split} rows={rows} wrote={path.name}",
            flush=True,
        )

    def _hook(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if self.active_split is None or self.active_positions is None:
            raise RuntimeError(
                f"layer {self.layer} TSQR hook fired outside an active batch"
            )
        if self.seen or not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError(
                f"layer {self.layer} TSQR hook fired an unexpected number of times"
            )
        activation = inputs[0].detach()
        if (
            activation.ndim != 3
            or int(activation.shape[0]) != int(self.active_positions.shape[0])
            or int(activation.shape[-1]) != self.input_width
        ):
            raise ValueError(
                f"unexpected layer {self.layer} activation shape: "
                f"{tuple(activation.shape)}"
            )
        positions = self.active_positions.to(device=activation.device)
        batch_indices = torch.arange(
            int(activation.shape[0]),
            device=activation.device,
        ).unsqueeze(1)
        selected = activation[batch_indices, positions].reshape(-1, self.input_width)
        accumulator = self._accumulator(self.active_split, activation.device)
        target = self.target_rows[self.active_split]
        if accumulator.total_rows + len(selected) > target:
            raise RuntimeError(
                f"layer {self.layer} {self.active_split} capture exceeded {target} rows"
            )
        accumulator.append(selected)
        if accumulator.total_rows == target:
            self._write_checkpoint(self.active_split, accumulator)
        self.seen = True

    def begin(self, split: str, positions: Tensor) -> None:
        if self.active_split is not None or split not in self.accumulators:
            raise RuntimeError(
                f"invalid layer {self.layer} TSQR capture batch transition"
            )
        if positions.ndim != 2 or not int(positions.numel()):
            raise ValueError("capture positions must be a nonempty matrix")
        self.active_split = split
        self.active_positions = positions
        self.seen = False

    def finish(self) -> None:
        if self.active_split is None or not self.seen:
            raise RuntimeError(f"layer {self.layer} TSQR capture missed its hook")
        self.active_split = None
        self.active_positions = None
        self.seen = False

    def finish_split(self, split: str) -> None:
        accumulator = self.accumulators.get(split)
        expected = self.target_rows[split]
        if (
            self.active_split is not None
            or accumulator is None
            or accumulator.total_rows != expected
            or accumulator.buffered_rows
            or split not in self.records
        ):
            raise RuntimeError(
                f"cannot finish incomplete layer {self.layer} TSQR split: {split}"
            )
        self.accumulators[split] = None
        del accumulator

    def close(self) -> None:
        self._handle.remove()


def _load_base_result(
    factor_dir: Path,
    *,
    model_config_sha256: str,
) -> dict[str, Any]:
    result_path = factor_dir / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("format") != FACTOR_FORMAT or result.get("status") != "complete":
        raise ValueError("base uniform C1 checkpoint is incomplete or incompatible")
    if result.get("fit_config", {}).get("model_config_sha256") != (
        model_config_sha256
    ):
        raise ValueError("base C1 checkpoint belongs to another model config")
    if tuple(map(int, result.get("layers", ()))) != tuple(range(NUM_LAYERS)):
        raise ValueError("base C1 checkpoint does not cover all layers")
    if set(map(int, result.get("artifacts", {}))) != set(range(NUM_LAYERS)):
        raise ValueError("base C1 checkpoint artifacts are incomplete")
    return result


def _load_base_layer(
    *,
    factor_dir: Path,
    base_result: Mapping[str, Any],
    layer: int,
) -> tuple[Tensor, Tensor, dict[str, Any], dict[str, Any]]:
    artifact = base_result["artifacts"][str(layer)]
    artifact_path = factor_dir / artifact["file"]
    if _sha256(artifact_path) != artifact["sha256"]:
        raise ValueError(f"base C1 artifact hash mismatch at layer {layer}")
    payload = load_file(str(artifact_path), device="cpu")
    if set(payload) != {"value_coordinate_encoders", "head_output_decoders"}:
        raise ValueError(f"unexpected base C1 tensors at layer {layer}")
    encoder = payload["value_coordinate_encoders"].contiguous()
    decoder = payload["head_output_decoders"].contiguous()
    rank = int(base_result["fit_config"]["cache_rank_per_head"])
    if tuple(encoder.shape) != (NUM_KV_HEADS, HEAD_DIM, rank):
        raise ValueError(f"base C1 encoder shape mismatch at layer {layer}")
    if tuple(decoder.shape) != (NUM_QUERY_HEADS, rank, HIDDEN_SIZE):
        raise ValueError(f"base C1 decoder shape mismatch at layer {layer}")
    records = {
        int(record["layer"]): record for record in base_result["records"]
    }
    if set(records) != set(range(NUM_LAYERS)):
        raise ValueError("base C1 layer records are incomplete")
    return encoder, decoder, records[layer], {
        "file": str(artifact_path),
        "sha256": artifact["sha256"],
    }


def _updated_fit_config(
    base_config: Mapping[str, Any],
    *,
    capture_dir: Path,
    capture_manifest_sha256: str,
    model_path: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    config.update(
        {
            "model": str(model_path),
            "covariance_damping": 0.0,
            "decoder_solver": "pairwise_tsqr_square_root_qr_lambda_0",
            "decoder_objective": "full_layer",
            "decoder_relative_jitter": 0.0,
            "decoder_solve_dtype": "float64",
            "factor_dtype": "bfloat16",
            "fit_rows": FIT_ROWS,
            "fit_windows": 256,
            "positions_per_window": 256,
            "validation_rows": HELDOUT_ROWS,
            "validation_windows": 64,
            "validation_positions_per_window": 128,
            "snapshot_dir": str(capture_dir),
            "snapshot_manifest_sha256": capture_manifest_sha256,
            "selection": "fixed all-layer QR(0) decoder refit at 65536 rows",
            "selection_boundaries": "single fixed 64k decoder closure",
            "work_dtype": "float64",
            "tsqr_dtype": "float32",
            "encoder_source_covariance_damping": float(
                base_config["covariance_damping"]
            ),
        }
    )
    return config


def _load_layer_r(
    *,
    capture_dir: Path,
    record: Mapping[str, Any],
    device: torch.device,
) -> Tensor:
    path = capture_dir / str(record["file"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise ValueError(f"TSQR artifact hash mismatch: {path}")
    payload = load_file(str(path), device="cpu")
    if set(payload) != {"r_upper_packed"}:
        raise ValueError(f"unexpected TSQR artifact tensors: {path}")
    return unpack_upper_triangular(
        payload["r_upper_packed"],
        dimension=QUERY_WIDTH,
    ).to(device=device, dtype=torch.float64)


def _artifact_record(path: Path, encoder: Tensor, decoder: Tensor) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "tensors": {
            "head_output_decoders": {
                "shape": list(decoder.shape),
                "dtype": str(decoder.dtype),
            },
            "value_coordinate_encoders": {
                "shape": list(encoder.shape),
                "dtype": str(encoder.dtype),
            },
        },
    }


def _load_model(
    *,
    model_path: Path,
    model_dtype: torch.dtype,
    attn_implementation: str,
    device_map: str,
    max_memory_per_gpu_gib: int,
) -> nn.Module:
    print(f"[Load] model={model_path} device_map={device_map}", flush=True)
    model = AutoModel.from_pretrained(
        str(model_path),
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        local_files_only=True,
        device_map=device_map,
        max_memory={
            index: f"{max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        },
    ).eval()
    model.config.use_cache = False
    placement = {
        str(key): str(value)
        for key, value in getattr(model, "hf_device_map", {}).items()
    }
    if any(value in {"cpu", "disk"} for value in placement.values()):
        raise RuntimeError(f"all-layer TSQR offloaded parameters: {placement}")
    return model


@torch.inference_mode()
def _capture_all_layers(
    *,
    model: nn.Module,
    model_path: Path,
    windows_path: Path,
    capture_dir: Path,
    fit_windows: int,
    heldout_windows: int,
    fit_positions_per_window: int,
    heldout_positions_per_window: int,
    position_seed: int,
    batch_size: int,
    command: str,
) -> dict[str, Any]:
    partial_dir = capture_dir.with_name(capture_dir.name + ".partial")
    for path in (capture_dir, partial_dir):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite capture output: {path}")
    expected_windows = fit_windows + heldout_windows
    windows, windows_manifest = _load_ordered_windows(
        windows_path,
        expected=expected_windows,
    )
    sequence_length = int(windows.shape[1])
    _validate_document_disjointness(
        windows_manifest,
        fit_windows=fit_windows,
        heldout_windows=heldout_windows,
    )
    maximum_positions = max(
        fit_positions_per_window,
        heldout_positions_per_window,
    )
    position_pool = _sample_positions(
        windows=expected_windows,
        sequence_length=sequence_length,
        positions_per_window=maximum_positions,
        seed=position_seed,
    )
    fit_positions = position_pool[:fit_windows, :fit_positions_per_window]
    heldout_positions = position_pool[
        fit_windows:, :heldout_positions_per_window
    ]
    partial_dir.mkdir(parents=True)
    positions_path = partial_dir / "positions.safetensors"
    save_file(
        {
            "fit_positions": fit_positions.contiguous(),
            "heldout_positions": heldout_positions.contiguous(),
        },
        str(positions_path),
    )

    geometry = _validate_attention(model.config, "gqa")
    expected_geometry = {
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": NUM_QUERY_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
    }
    if int(model.config.num_hidden_layers) != NUM_LAYERS:
        raise ValueError("unexpected Qwen3-32B layer count")
    for key, expected in expected_geometry.items():
        if int(geometry[key]) != expected:
            raise ValueError(f"unexpected Qwen3-32B geometry for {key}")
    layers = _decoder_layers(model)
    captures: list[_FinalLayerTSQRCapture] = []
    for layer, decoder_layer in enumerate(layers):
        module = decoder_layer.self_attn.o_proj
        if not isinstance(module, nn.Linear) or tuple(module.weight.shape) != (
            HIDDEN_SIZE,
            QUERY_WIDTH,
        ):
            raise TypeError(f"layer {layer} o_proj is unsupported")
        captures.append(
            _FinalLayerTSQRCapture(
                module,
                layer=layer,
                output_dir=partial_dir,
                input_width=QUERY_WIDTH,
                fit_rows=FIT_ROWS,
                heldout_rows=HELDOUT_ROWS,
                accumulation_dtype=torch.float32,
            )
        )
    input_device = model.get_input_embeddings().weight.device
    if input_device.type != "cuda":
        raise RuntimeError("model input embeddings are not on CUDA")

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        total = len(windows)
        for batch_index, start in enumerate(
            range(0, total, batch_size),
            start=1,
        ):
            boundary = fit_windows if start < fit_windows else total
            stop = min(start + batch_size, boundary)
            split = "fit" if start < fit_windows else "heldout"
            positions = (
                fit_positions[start:stop]
                if split == "fit"
                else heldout_positions[
                    start - fit_windows : stop - fit_windows
                ]
            )
            for capture in captures:
                capture.begin(split, positions)
            input_ids = windows[start:stop].to(input_device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            )
            del outputs, input_ids
            for capture in captures:
                capture.finish()
            print(
                f"[Capture] batch={batch_index}/{total} split={split} "
                f"windows={stop}/{total}",
                flush=True,
            )
            if stop == fit_windows:
                for capture in captures:
                    capture.finish_split("fit")
                fit_records = {
                    str(capture.layer): capture.records["fit"]
                    for capture in captures
                }
                fit_checkpoint_path = partial_dir / "fit_checkpoint.json"
                _atomic_json(
                    fit_checkpoint_path,
                    {
                        "format": FIT_CHECKPOINT_FORMAT,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "fit_rows": FIT_ROWS,
                        "fit": fit_records,
                        "positions": {
                            "file": positions_path.name,
                            "sha256": _sha256(positions_path),
                        },
                    },
                )
                print(
                    f"[Capture] persisted all fit factors to {partial_dir}",
                    flush=True,
                )
        for capture in captures:
            capture.finish_split("heldout")
        print("[Capture] released all held-out TSQR states", flush=True)
    finally:
        for capture in captures:
            capture.close()

    placement = {
        str(key): str(value)
        for key, value in getattr(model, "hf_device_map", {}).items()
    }
    fit_checkpoint_path = partial_dir / "fit_checkpoint.json"
    layer_artifacts = {
        str(capture.layer): dict(capture.records) for capture in captures
    }
    model_config_sha256 = _sha256(model_path / "config.json")
    manifest = {
        "format": CAPTURE_FORMAT,
        "schema_version": 1,
        "status": "complete",
        "command": command,
        "git_commit": _git_commit(),
        "timestamp_started_utc": started_utc,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model": {
            "path": str(model_path),
            "config_sha256": model_config_sha256,
            **geometry,
        },
        "layers": list(range(NUM_LAYERS)),
        "calibration": {
            "storage": "normalized_streaming_tsqr_r_only",
            "windows_file": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest_format": windows_manifest.get("format"),
            "fit_windows": fit_windows,
            "heldout_windows": heldout_windows,
            "sequence_length": sequence_length,
            "fit_positions_per_window": fit_positions_per_window,
            "heldout_positions_per_window": heldout_positions_per_window,
            "fit_rows": FIT_ROWS,
            "heldout_rows": HELDOUT_ROWS,
            "position_seed": position_seed,
            "positions_file": positions_path.name,
            "positions_sha256": _sha256(positions_path),
            "fit_checkpoint": {
                "file": fit_checkpoint_path.name,
                "sha256": _sha256(fit_checkpoint_path),
            },
            "fit_and_heldout_document_disjoint": True,
        },
        "artifacts": {"layers": layer_artifacts},
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
            "device_map": placement,
            "torch_num_threads": torch.get_num_threads(),
            "tsqr_dtype": "float32",
            "tf32_enabled": False,
        },
    }
    _atomic_json(partial_dir / "manifest.json", manifest)
    os.replace(partial_dir, capture_dir)
    print(
        f"[Capture] complete output={capture_dir} "
        f"seconds={manifest['elapsed_seconds']:.3f}",
        flush=True,
    )
    return manifest


@torch.no_grad()
def _solve_checkpoint(
    *,
    model: nn.Module,
    model_path: Path,
    factor_dir: Path,
    capture_dir: Path,
    checkpoint_dir: Path,
    output_markdown: Path,
    output_column_chunk_size: int,
    command: str,
) -> dict[str, Any]:
    partial_dir = checkpoint_dir.with_name(checkpoint_dir.name + ".partial")
    for path in (checkpoint_dir, partial_dir, output_markdown):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint output: {path}")
    manifest_path = capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CAPTURE_FORMAT or manifest.get("status") != (
        "complete"
    ):
        raise ValueError("all-layer TSQR capture is incomplete or incompatible")
    model_config_sha256 = _sha256(model_path / "config.json")
    if manifest.get("model", {}).get("config_sha256") != model_config_sha256:
        raise ValueError("TSQR capture belongs to another model config")
    base_result = _load_base_result(
        factor_dir,
        model_config_sha256=model_config_sha256,
    )
    capture_manifest_sha256 = _sha256(manifest_path)
    fit_config = _updated_fit_config(
        base_result["fit_config"],
        capture_dir=capture_dir,
        capture_manifest_sha256=capture_manifest_sha256,
        model_path=model_path,
    )
    partial_dir.mkdir(parents=True)
    decoder_layers = _decoder_layers(model)
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    started = time.perf_counter()
    for layer, decoder_layer in enumerate(decoder_layers):
        layer_started = time.perf_counter()
        encoder_cpu, base_decoder_cpu, base_record, base_artifact = _load_base_layer(
            factor_dir=factor_dir,
            base_result=base_result,
            layer=layer,
        )
        o_proj = decoder_layer.self_attn.o_proj
        if not isinstance(o_proj, nn.Linear) or tuple(o_proj.weight.shape) != (
            HIDDEN_SIZE,
            QUERY_WIDTH,
        ):
            raise TypeError(f"layer {layer} o_proj is unsupported")
        device = o_proj.weight.device
        if device.type != "cuda":
            raise RuntimeError(f"layer {layer} o_proj is not on CUDA")
        capture_record = manifest["artifacts"]["layers"][str(layer)]
        fit_r = _load_layer_r(
            capture_dir=capture_dir,
            record=capture_record["fit"],
            device=device,
        )
        heldout_r = _load_layer_r(
            capture_dir=capture_dir,
            record=capture_record["heldout"],
            device=device,
        )
        fixed_a = encoder_cpu.to(device=device, dtype=torch.float64)
        target_weight = (
            o_proj.weight.detach().to(dtype=torch.float64).T.contiguous()
        )
        basis = _dense_block_basis(fixed_a, device=device)
        decoder_flat, diagnostics = solve_square_root_least_squares(
            left_factor=fit_r,
            basis=basis,
            target_weight=target_weight,
            absolute_product_damping=0.0,
            output_chunk_size=output_column_chunk_size,
        )
        rank = int(fixed_a.shape[2])
        decoder = decoder_flat.reshape(NUM_QUERY_HEADS, rank, HIDDEN_SIZE)
        fit_target_energy = _factor_output_energy(
            fit_r,
            target_weight,
            output_chunk_size=output_column_chunk_size,
        )
        heldout_target_energy = _factor_output_energy(
            heldout_r,
            target_weight,
            output_chunk_size=output_column_chunk_size,
        )
        work_metrics = _decoder_metrics(
            fixed_a=fixed_a,
            decoder=decoder,
            target_weight=target_weight,
            fit_r=fit_r,
            heldout_r=heldout_r,
            output_chunk_size=output_column_chunk_size,
            fit_target_energy=fit_target_energy,
            heldout_target_energy=heldout_target_energy,
        )
        decoder_bf16 = decoder.to(device="cpu", dtype=torch.bfloat16).contiguous()
        factor_metrics = _decoder_metrics(
            fixed_a=fixed_a,
            decoder=decoder_bf16.to(device=device, dtype=torch.float64),
            target_weight=target_weight,
            fit_r=fit_r,
            heldout_r=heldout_r,
            output_chunk_size=output_column_chunk_size,
            fit_target_energy=fit_target_energy,
            heldout_target_energy=heldout_target_energy,
        )
        base_metrics = _decoder_metrics(
            fixed_a=fixed_a,
            decoder=base_decoder_cpu.to(device=device, dtype=torch.float64),
            target_weight=target_weight,
            fit_r=fit_r,
            heldout_r=heldout_r,
            output_chunk_size=output_column_chunk_size,
            fit_target_energy=fit_target_energy,
            heldout_target_energy=heldout_target_energy,
        )
        if not bool(torch.isfinite(decoder_bf16).all()):
            raise FloatingPointError(f"layer {layer} produced a non-finite decoder")

        encoder_bf16 = encoder_cpu.to(dtype=torch.bfloat16).contiguous()
        artifact_path = partial_dir / f"layer_{layer:03d}.safetensors"
        save_file(
            {
                "value_coordinate_encoders": encoder_bf16,
                "head_output_decoders": decoder_bf16,
            },
            str(artifact_path),
        )
        artifact = _artifact_record(
            artifact_path,
            encoder_bf16,
            decoder_bf16,
        )
        artifacts[str(layer)] = artifact
        record = copy.deepcopy(base_record)
        record.update(
            {
                "format": LAYER_FORMAT,
                "layer": layer,
                "fit_config": fit_config,
                "artifact": artifact,
                "covariance": {
                    "fit_absolute_trace_damping": 0.0,
                    "heldout_regularized": False,
                    "solver": "pairwise_tsqr_square_root_qr_lambda_0",
                },
                "fit": {
                    "relative_mse": work_metrics["fit_relative_mse"],
                    "factor_dtype_relative_mse": factor_metrics[
                        "fit_relative_mse"
                    ],
                },
                "heldout": {
                    "relative_mse": work_metrics["heldout_relative_mse"],
                    "factor_dtype_relative_mse": factor_metrics[
                        "heldout_relative_mse"
                    ],
                },
                "elapsed_seconds": time.perf_counter() - layer_started,
                "decoder_refit": {
                    "solver": "pairwise_tsqr_square_root_qr_lambda_0",
                    "solve_dtype": "float64",
                    "factor_dtype": "bfloat16",
                    "solve": asdict(diagnostics),
                    "decoder_frobenius_norm": work_metrics[
                        "decoder_frobenius_norm"
                    ],
                    "decoder_maximum_absolute_value": work_metrics[
                        "decoder_maximum_absolute_value"
                    ],
                    "base_encoder_artifact": base_artifact,
                    "base_fit_factor_dtype_relative_mse": base_metrics[
                        "fit_relative_mse"
                    ],
                    "base_heldout_factor_dtype_relative_mse": base_metrics[
                        "heldout_relative_mse"
                    ],
                    "fit_r": capture_record["fit"],
                    "heldout_r": capture_record["heldout"],
                },
            }
        )
        _atomic_json(partial_dir / f"layer_{layer:03d}.json", record)
        records.append(record)
        print(
            f"[Solve] layer={layer}/63 fit={factor_metrics['fit_relative_mse']:.9e} "
            f"heldout={factor_metrics['heldout_relative_mse']:.9e} "
            f"norm={work_metrics['decoder_frobenius_norm']:.6e}",
            flush=True,
        )
        del (
            encoder_cpu,
            base_decoder_cpu,
            encoder_bf16,
            fixed_a,
            target_weight,
            basis,
            fit_r,
            heldout_r,
            decoder_flat,
            decoder,
            decoder_bf16,
        )

    fit_values = [record["fit"]["factor_dtype_relative_mse"] for record in records]
    heldout_values = [
        record["heldout"]["factor_dtype_relative_mse"] for record in records
    ]
    base_heldout_values = [
        record["decoder_refit"]["base_heldout_factor_dtype_relative_mse"]
        for record in records
    ]
    aggregate = {
        "mean_fit_factor_dtype_relative_mse": sum(fit_values) / NUM_LAYERS,
        "mean_heldout_factor_dtype_relative_mse": sum(heldout_values) / NUM_LAYERS,
        "base_mean_heldout_factor_dtype_relative_mse": sum(base_heldout_values)
        / NUM_LAYERS,
        "layers_improved_on_heldout": sum(
            value < base
            for value, base in zip(heldout_values, base_heldout_values, strict=True)
        ),
        "maximum_heldout_factor_dtype_relative_mse": max(heldout_values),
    }
    result = copy.deepcopy(base_result)
    result.update(
        {
            "format": FACTOR_FORMAT,
            "status": "complete",
            "command": command,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "layers": list(range(NUM_LAYERS)),
            "fit_config": fit_config,
            "records": records,
            "artifacts": artifacts,
            "aggregate": aggregate,
            "decoder_refit_source": {
                "base_factor_dir": str(factor_dir),
                "base_results_sha256": _sha256(factor_dir / "results.json"),
                "capture_manifest": str(manifest_path),
                "capture_manifest_sha256": capture_manifest_sha256,
                "elapsed_seconds": time.perf_counter() - started,
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
                "solve_dtype": "float64",
                "factor_dtype": "bfloat16",
                "tf32_enabled": False,
            },
        }
    )
    _atomic_json(partial_dir / "results.json", result)
    os.replace(partial_dir, checkpoint_dir)
    _atomic_text(output_markdown, _render_markdown(result, checkpoint_dir))
    print(
        f"[Checkpoint] complete output={checkpoint_dir} "
        f"mean_heldout={aggregate['mean_heldout_factor_dtype_relative_mse']:.9e}",
        flush=True,
    )
    return result


def _render_markdown(
    result: Mapping[str, Any],
    checkpoint_dir: Path,
) -> str:
    aggregate = result["aggregate"]
    rank = int(result["fit_config"]["cache_rank_per_head"])
    lines = [
        f"# Qwen3-32B rank-{rank} all-layer 64k TSQR QR(0) checkpoint",
        "",
        f"All 64 Value encoders are fixed to the uniform rank-{rank} C1 "
        "checkpoint. Every full-layer decoder is refit from 65,536 C4 rows with "
        "float32 streaming TSQR and a float64 square-root solve at `lambda=0`. "
        "Artifacts are stored in BF16.",
        "",
        f"- Checkpoint: `{checkpoint_dir}`",
        f"- Mean fit relMSE: "
        f"`{aggregate['mean_fit_factor_dtype_relative_mse']:.9e}`",
        f"- Mean held-out relMSE: "
        f"`{aggregate['mean_heldout_factor_dtype_relative_mse']:.9e}`",
        f"- Base mean held-out relMSE: "
        f"`{aggregate['base_mean_heldout_factor_dtype_relative_mse']:.9e}`",
        f"- Layers improved on held-out: "
        f"`{aggregate['layers_improved_on_heldout']}/64`",
        "",
        "## Per-layer results",
        "",
        "| Layer | Fit relMSE | Held-out relMSE | Base held-out | Decoder norm | "
        "R diagonal ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in result["records"]:
        refit = record["decoder_refit"]
        lines.append(
            f"| {record['layer']} | "
            f"{record['fit']['factor_dtype_relative_mse']:.9e} | "
            f"{record['heldout']['factor_dtype_relative_mse']:.9e} | "
            f"{refit['base_heldout_factor_dtype_relative_mse']:.9e} | "
            f"{refit['decoder_frobenius_norm']:.6e} | "
            f"{refit['solve']['r_diagonal_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Command",
            "",
            "```bash",
            str(result["command"]),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("all-layer streaming TSQR requires CUDA")
    torch.cuda.set_device(torch.device("cuda:0"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)

    model_path = Path(args.model_path).expanduser().resolve()
    windows_path = args.windows.expanduser().resolve()
    factor_dir = args.factor_dir.expanduser().resolve()
    capture_dir = args.capture_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
    if not (factor_dir / "results.json").is_file():
        raise FileNotFoundError(factor_dir / "results.json")
    if output_markdown.exists():
        raise FileExistsError(output_markdown)
    checkpoint_partial = checkpoint_dir.with_name(checkpoint_dir.name + ".partial")
    for path in (checkpoint_dir, checkpoint_partial):
        if path.exists():
            raise FileExistsError(path)
    if not args.solve_only:
        capture_partial = capture_dir.with_name(capture_dir.name + ".partial")
        for path in (capture_dir, capture_partial):
            if path.exists():
                raise FileExistsError(path)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)

    expected_windows = args.fit_windows + args.heldout_windows
    windows, windows_manifest = _load_ordered_windows(
        windows_path,
        expected=expected_windows,
    )
    _validate_args(args, sequence_length=int(windows.shape[1]))
    _validate_document_disjointness(
        windows_manifest,
        fit_windows=args.fit_windows,
        heldout_windows=args.heldout_windows,
    )
    del windows

    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    model = _load_model(
        model_path=model_path,
        model_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory_per_gpu_gib=args.max_memory_per_gpu_gib,
    )
    command = shlex.join([sys.executable, *sys.argv])
    if args.solve_only:
        if not (capture_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"completed capture is missing: {capture_dir}")
    else:
        _capture_all_layers(
            model=model,
            model_path=model_path,
            windows_path=windows_path,
            capture_dir=capture_dir,
            fit_windows=args.fit_windows,
            heldout_windows=args.heldout_windows,
            fit_positions_per_window=args.fit_positions_per_window,
            heldout_positions_per_window=args.heldout_positions_per_window,
            position_seed=args.position_seed,
            batch_size=args.batch_size,
            command=command,
        )
    _solve_checkpoint(
        model=model,
        model_path=model_path,
        factor_dir=factor_dir,
        capture_dir=capture_dir,
        checkpoint_dir=checkpoint_dir,
        output_markdown=output_markdown,
        output_column_chunk_size=args.output_column_chunk_size,
        command=command,
    )


if __name__ == "__main__":
    main()
