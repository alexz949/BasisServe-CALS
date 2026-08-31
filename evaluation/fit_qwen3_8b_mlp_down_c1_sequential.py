#!/usr/bin/env python3
"""Fit Qwen3-8B MLP C1 factors by sequential dense-teacher closure.

At layer ``l``, the regression input is the post-SwiGLU activation produced by
the student after layers ``0..l-1`` have already been compressed.  The target
is the corresponding dense-teacher ``down_proj`` output.  The fitted layer is
closed immediately, so its compressed output becomes the student input to the
next decoder layer.
"""

from __future__ import annotations

import argparse
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
from torch.nn import functional as F
from transformers import AutoModel


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.mlp_down_c1 import (  # noqa: E402
    FACTOR_FORMAT,
    MLPDownC1Linear,
    fit_mlp_down_c1_cross_moments_from_cholesky,
    fit_mlp_down_c1_fixed_encoder_decoder,
    load_mlp_down_c1_manifest,
)
from evaluation.capture_attention_o_proj_ppl_snapshots import (  # noqa: E402
    _atomic_json,
    _decoder_layers,
    _git_commit,
    _installed_version,
    _sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fixed-encoder-dir",
        help="Keep these independent AASVD encoders and refit only decoders",
    )
    parser.add_argument("--fit-windows", type=int, default=256)
    parser.add_argument("--heldout-windows", type=int, default=64)
    parser.add_argument("--rank", type=int, default=2560)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tsqr-windows-per-block", type=int, default=32)
    parser.add_argument("--relative-damping", type=float, default=0.0)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--factor-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _move_nested_tensors(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested_tensors(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _move_nested_tensors(item, device) for key, item in value.items()
        }
    return value


def _detach_nested_tensors(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_nested_tensors(item) for item in value)
    if isinstance(value, list):
        return [_detach_nested_tensors(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_nested_tensors(item) for key, item in value.items()}
    return value


class _CaptureExit(Exception):
    pass


class _DownInputCaptureExit(Exception):
    pass


class _FirstLayerBatchCatcher(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        destination: Tensor,
        captured_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.layer = layer
        self.destination = destination
        self.captured_kwargs = captured_kwargs
        self.index = 0
        if hasattr(layer, "attention_type"):
            self.attention_type = layer.attention_type

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        batch = int(hidden_states.shape[0])
        stop = self.index + batch
        if stop > len(self.destination):
            raise RuntimeError("first-layer input capture overflowed")
        self.destination[self.index : stop].copy_(hidden_states.detach())
        if not self.captured_kwargs:
            self.captured_kwargs.update(_detach_nested_tensors(kwargs))
        self.index = stop
        raise _CaptureExit


def _load_windows(
    path: Path, *, expected: int
) -> tuple[Tensor, dict[str, Any]]:
    payload = load_file(str(path), device="cpu")
    if set(payload) != {"input_ids"}:
        raise ValueError("sequential C1 windows must contain only input_ids")
    windows = payload["input_ids"].long().contiguous()
    if windows.ndim != 2 or len(windows) != expected:
        raise ValueError(f"expected {expected} windows, found {tuple(windows.shape)}")
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest.get("artifact", {})
    if artifact.get("sha256") != _sha256(path):
        raise ValueError("calibration window hash differs from its manifest")
    if tuple(artifact.get("shape", ())) != tuple(windows.shape):
        raise ValueError("calibration window shape differs from its manifest")
    return windows, manifest


def _capture_first_layer_inputs(
    model: nn.Module,
    windows: Tensor,
    destination: Tensor,
    *,
    batch_size: int,
) -> dict[str, Any]:
    layers = _decoder_layers(model)
    first_layer = layers[0]
    captured_kwargs: dict[str, Any] = {}
    catcher = _FirstLayerBatchCatcher(
        first_layer, destination, captured_kwargs
    )
    layers[0] = catcher
    try:
        for start in range(0, len(windows), batch_size):
            stop = min(start + batch_size, len(windows))
            input_ids = windows[start:stop]
            try:
                model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    return_dict=True,
                )
            except _CaptureExit:
                pass
            else:
                raise RuntimeError("first decoder layer catcher did not fire")
    finally:
        layers[0] = first_layer
    if catcher.index != len(windows) or not captured_kwargs:
        raise RuntimeError(
            f"captured {catcher.index}/{len(windows)} first-layer inputs"
        )
    return captured_kwargs


def _tensor_energy(value: Tensor) -> float:
    work = value.float()
    return float(torch.sum(work * work).item())


def _tensor_squared_error(prediction: Tensor, target: Tensor) -> float:
    difference = prediction.float() - target.float()
    return float(torch.sum(difference * difference).item())


def _streaming_tsqr_input_cholesky(
    inputs: Tensor,
    *,
    fit_windows: int,
    windows_per_block: int,
    relative_damping: float,
) -> tuple[Tensor, Tensor, dict[str, float | int]]:
    """Return ``R.T / sqrt(N)`` for the fit rows without forming ``X.T X``."""

    if inputs.ndim != 3 or not 0 < fit_windows <= len(inputs):
        raise ValueError("invalid sequential TSQR input geometry")
    if relative_damping < 0:
        raise ValueError("relative TSQR damping must be nonnegative")
    sequence_length = int(inputs.shape[1])
    width = int(inputs.shape[2])
    if windows_per_block * sequence_length < width:
        raise ValueError("each TSQR block must contain at least input_width rows")
    triangular: Tensor | None = None
    block_count = 0
    for start in range(0, fit_windows, windows_per_block):
        stop = min(start + windows_per_block, fit_windows)
        work = inputs[start:stop].reshape(-1, width).float()
        if triangular is not None:
            work = torch.cat((triangular, work), dim=0)
        empty_q, triangular = torch.linalg.qr(work, mode="r")
        del empty_q, work
        triangular = triangular.contiguous()
        block_count += 1
        print(
            f"[Sequential MLP C1] TSQR block={block_count} "
            f"windows={start}:{stop}",
            flush=True,
        )
    if triangular is None or tuple(triangular.shape) != (width, width):
        raise RuntimeError("streaming TSQR did not produce a square R factor")
    raw_diagonal = triangular.diagonal().abs()
    raw_minimum = float(raw_diagonal.min().item())
    raw_maximum = float(raw_diagonal.max().item())
    if raw_minimum == 0.0 or not bool(torch.isfinite(raw_diagonal).all()):
        raise torch.linalg.LinAlgError("streaming TSQR input matrix is rank deficient")
    rows = fit_windows * sequence_length
    root_rows = float(rows) ** 0.5
    objective_cholesky = (triangular.T / root_rows).contiguous()
    mean_input_second_moment = float(
        torch.sum(triangular * triangular).item() / (rows * width)
    )
    absolute_damping = relative_damping * mean_input_second_moment
    if absolute_damping > 0:
        ridge_rows = torch.eye(
            width, dtype=triangular.dtype, device=triangular.device
        )
        ridge_rows.mul_((rows * absolute_damping) ** 0.5)
        augmented = torch.cat((triangular, ridge_rows), dim=0)
        empty_q, solve_r = torch.linalg.qr(augmented, mode="r")
        del empty_q, augmented, ridge_rows
        solve_cholesky = (solve_r.T / root_rows).contiguous()
        del solve_r
    else:
        solve_cholesky = objective_cholesky
    solve_diagonal = solve_cholesky.diagonal().abs()
    solve_minimum = float(solve_diagonal.min().item())
    solve_maximum = float(solve_diagonal.max().item())
    diagnostics: dict[str, float | int] = {
        "blocks": block_count,
        "windows_per_block": windows_per_block,
        "rows": rows,
        "raw_minimum_abs_r_diagonal": raw_minimum,
        "raw_maximum_abs_r_diagonal": raw_maximum,
        "raw_r_diagonal_ratio": raw_minimum / raw_maximum,
        "solve_minimum_abs_cholesky_diagonal": solve_minimum,
        "solve_maximum_abs_cholesky_diagonal": solve_maximum,
        "solve_cholesky_diagonal_ratio": solve_minimum / solve_maximum,
        "mean_input_second_moment": mean_input_second_moment,
        "relative_damping": relative_damping,
        "absolute_damping": absolute_damping,
    }
    return solve_cholesky, objective_cholesky, diagnostics


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if min(
        args.fit_windows,
        args.heldout_windows,
        args.rank,
        args.tp_size,
        args.batch_size,
        args.tsqr_windows_per_block,
        args.torch_num_threads,
    ) <= 0:
        raise ValueError("all counts, ranks, and thread settings must be positive")
    if args.relative_damping < 0:
        raise ValueError("--relative-damping must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("sequential MLP C1 fitting requires CUDA")
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    model_path = Path(args.model_path).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    fixed_encoder_dir = (
        None
        if args.fixed_encoder_dir is None
        else Path(args.fixed_encoder_dir).expanduser().resolve()
    )
    if fixed_encoder_dir is not None and args.relative_damping != 0:
        raise ValueError("fixed-encoder decoder refit uses zero damping")
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    total_windows = args.fit_windows + args.heldout_windows
    if total_windows % args.batch_size:
        raise ValueError("fit + heldout windows must divide evenly by batch size")
    if args.fit_windows % args.batch_size:
        raise ValueError("fit windows must divide evenly by batch size")
    windows, windows_manifest = _load_windows(
        windows_path, expected=total_windows
    )
    sequence_length = int(windows.shape[1])
    if sequence_length != 2048:
        raise ValueError("this controlled sequential experiment requires seq2048")

    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    factor_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.factor_dtype]
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    partial_dir.mkdir(parents=True)
    print(f"[Sequential MLP C1] loading CPU model={model_path}", flush=True)
    model = AutoModel.from_pretrained(
        str(model_path),
        dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    model.config.use_cache = False
    if str(model.config.model_type) != "qwen3":
        raise ValueError("sequential MLP C1 currently supports dense Qwen3")
    hidden_size = int(model.config.hidden_size)
    intermediate_size = int(model.config.intermediate_size)
    num_layers = int(model.config.num_hidden_layers)
    if (hidden_size, intermediate_size, num_layers) != (4096, 12288, 36):
        raise ValueError("unexpected Qwen3-8B geometry")
    if not 0 < args.rank <= hidden_size:
        raise ValueError("rank must not exceed the dense AllReduce width")
    if intermediate_size % args.tp_size:
        raise ValueError("TP size must divide the intermediate width")
    fixed_encoder_manifest: dict[str, Any] | None = None
    if fixed_encoder_dir is not None:
        fixed_encoder_manifest = load_mlp_down_c1_manifest(
            fixed_encoder_dir,
            model_config_path=model_path / "config.json",
        )
        source_fit = fixed_encoder_manifest.get("fit_config", {})
        if source_fit.get("algorithm") != "teacher_output_pca_exact_shared_decoder":
            raise ValueError("fixed encoders must come from independent AASVD C1")
        if int(source_fit.get("rank", -1)) != args.rank:
            raise ValueError("fixed encoder rank differs from requested rank")
        if int(source_fit.get("tp_size", -1)) != args.tp_size:
            raise ValueError("fixed encoder TP size differs from requested TP size")

    teacher_hidden = torch.empty(
        total_windows,
        sequence_length,
        hidden_size,
        dtype=model_dtype,
        device=device,
    )
    print("[Sequential MLP C1] capturing first-layer inputs", flush=True)
    captured_kwargs = _capture_first_layer_inputs(
        model,
        windows,
        teacher_hidden,
        batch_size=args.batch_size,
    )
    layer_kwargs = _move_nested_tensors(captured_kwargs, device)
    student_hidden = teacher_hidden.clone()
    del captured_kwargs
    layers = _decoder_layers(model)

    artifacts: dict[str, Any] = {}
    metrics: list[dict[str, Any]] = []
    fit_rows = args.fit_windows * sequence_length
    heldout_start = args.fit_windows
    for layer_index, layer in enumerate(layers):
        layer_started = time.perf_counter()
        print(
            f"[Sequential MLP C1] layer={layer_index}/{num_layers - 1} "
            "capture_student",
            flush=True,
        )
        layer.to(device=device)
        down_proj = layer.mlp.down_proj
        if not isinstance(down_proj, nn.Linear) or tuple(down_proj.weight.shape) != (
            hidden_size,
            intermediate_size,
        ):
            raise TypeError(f"layer {layer_index} has unsupported down_proj")
        fixed_input_factor: Tensor | None = None
        source_encoder_record: dict[str, Any] | None = None
        regression_width = intermediate_size
        if fixed_encoder_manifest is not None and fixed_encoder_dir is not None:
            source_artifact = fixed_encoder_manifest["artifacts"][str(layer_index)]
            source_path = fixed_encoder_dir / source_artifact["file"]
            if _sha256(source_path) != source_artifact["sha256"]:
                raise ValueError(f"fixed encoder hash mismatch at layer {layer_index}")
            source_tensors = load_file(str(source_path), device="cpu")
            fixed_input_factor = source_tensors["input_factor"].to(
                device=device, dtype=model_dtype
            )
            if tuple(fixed_input_factor.shape) != (intermediate_size, args.rank):
                raise ValueError(f"layer {layer_index} fixed encoder shape differs")
            regression_width = args.rank
            source_encoder_record = {
                "directory": str(fixed_encoder_dir),
                "file": source_artifact["file"],
                "sha256": source_artifact["sha256"],
            }
        student_regression_inputs = torch.empty(
            total_windows,
            sequence_length,
            regression_width,
            dtype=model_dtype,
            device=device,
        )
        capture_state = {"start": 0, "stop": 0, "residual": False, "input": False}

        def capture_residual(
            module: nn.Module, hook_inputs: tuple[Tensor, ...]
        ) -> None:
            del module
            start = int(capture_state["start"])
            stop = int(capture_state["stop"])
            student_hidden[start:stop].copy_(hook_inputs[0].detach())
            capture_state["residual"] = True

        def capture_down_input(
            module: nn.Module, hook_inputs: tuple[Tensor, ...]
        ) -> None:
            del module
            start = int(capture_state["start"])
            stop = int(capture_state["stop"])
            down_input = hook_inputs[0].detach()
            if fixed_input_factor is None:
                student_regression_inputs[start:stop].copy_(down_input)
            else:
                student_regression_inputs[start:stop].copy_(
                    down_input @ fixed_input_factor
                )
            capture_state["input"] = True
            raise _DownInputCaptureExit

        residual_handle = layer.post_attention_layernorm.register_forward_pre_hook(
            capture_residual
        )
        input_handle = down_proj.register_forward_pre_hook(capture_down_input)
        try:
            for start in range(0, total_windows, args.batch_size):
                stop = start + args.batch_size
                capture_state.update(
                    {"start": start, "stop": stop, "residual": False, "input": False}
                )
                try:
                    layer(student_hidden[start:stop], **layer_kwargs)
                except _DownInputCaptureExit:
                    pass
                else:
                    raise RuntimeError("student down_proj catcher did not fire")
                if not capture_state["residual"] or not capture_state["input"]:
                    raise RuntimeError("student layer capture was incomplete")
        finally:
            residual_handle.remove()
            input_handle.remove()

        print(
            f"[Sequential MLP C1] layer={layer_index}/{num_layers - 1} "
            "streaming_tsqr",
            flush=True,
        )
        input_cholesky, objective_input_cholesky, tsqr_diagnostics = (
            _streaming_tsqr_input_cholesky(
                student_regression_inputs,
                fit_windows=args.fit_windows,
                windows_per_block=args.tsqr_windows_per_block,
                relative_damping=args.relative_damping,
            )
        )
        c_yx = torch.zeros(
            hidden_size,
            regression_width,
            dtype=torch.float32,
            device=device,
        )
        c_yy = torch.zeros(
            hidden_size,
            hidden_size,
            dtype=torch.float32,
            device=device,
        )
        heldout_targets = torch.empty(
            args.heldout_windows,
            sequence_length,
            hidden_size,
            dtype=model_dtype,
            device=device,
        )
        teacher_capture: list[Tensor] = []

        def capture_teacher_output(
            module: nn.Module,
            hook_inputs: tuple[Tensor, ...],
            output: Tensor,
        ) -> None:
            del module, hook_inputs
            if teacher_capture:
                raise RuntimeError("teacher output capture fired more than once")
            teacher_capture.append(output.detach())

        teacher_handle = down_proj.register_forward_hook(capture_teacher_output)
        print(
            f"[Sequential MLP C1] layer={layer_index}/{num_layers - 1} "
            "propagate_teacher_and_accumulate",
            flush=True,
        )
        try:
            for start in range(0, total_windows, args.batch_size):
                stop = start + args.batch_size
                teacher_capture.clear()
                teacher_result = layer(teacher_hidden[start:stop], **layer_kwargs)
                if not isinstance(teacher_result, Tensor) or len(teacher_capture) != 1:
                    raise RuntimeError("dense teacher layer capture failed")
                target = teacher_capture[0]
                if start < args.fit_windows:
                    x = student_regression_inputs[start:stop].reshape(
                        -1, regression_width
                    ).float()
                    y = target.reshape(-1, hidden_size).float()
                    c_yx.addmm_(y.T, x)
                    c_yy.addmm_(y.T, y)
                    del x, y
                else:
                    target_start = start - heldout_start
                    target_stop = stop - heldout_start
                    heldout_targets[target_start:target_stop].copy_(target)
                teacher_hidden[start:stop].copy_(teacher_result)
        finally:
            teacher_handle.remove()
        c_yx.div_(float(fit_rows))
        c_yy.div_(float(fit_rows))
        print(
            f"[Sequential MLP C1] layer={layer_index}/{num_layers - 1} "
            "solve_reduced_rank_regression absolute_damping="
            f"{tsqr_diagnostics['absolute_damping']:.8g}",
            flush=True,
        )
        if fixed_input_factor is None:
            factors = fit_mlp_down_c1_cross_moments_from_cholesky(
                input_cholesky,
                c_yx,
                c_yy,
                objective_input_cholesky=objective_input_cholesky,
                absolute_damping=float(tsqr_diagnostics["absolute_damping"]),
                relative_damping=args.relative_damping,
                rank=args.rank,
                factor_dtype=factor_dtype,
            )
            compressed_down: MLPDownC1Linear | None = MLPDownC1Linear(
                factors.input_factor.to(device=device),
                factors.output_basis.to(device=device),
            ).eval()
            fitted_decoder = None
        else:
            factors = fit_mlp_down_c1_fixed_encoder_decoder(
                fixed_input_factor,
                input_cholesky,
                c_yx,
                c_yy,
                factor_dtype=factor_dtype,
            )
            compressed_down = None
            fitted_decoder = factors.output_basis.to(device=device)
        heldout_output_error = 0.0
        heldout_output_energy = 0.0
        heldout_state_error = 0.0
        heldout_state_energy = 0.0
        for start in range(0, total_windows, args.batch_size):
            stop = start + args.batch_size
            if fitted_decoder is None:
                if compressed_down is None:
                    raise RuntimeError("full sequential decoder was not constructed")
                prediction = compressed_down(student_regression_inputs[start:stop])
            else:
                prediction = F.linear(
                    student_regression_inputs[start:stop], fitted_decoder
                )
            student_hidden[start:stop].add_(prediction)
            if start >= heldout_start:
                target_start = start - heldout_start
                target_stop = stop - heldout_start
                target = heldout_targets[target_start:target_stop]
                heldout_output_error += _tensor_squared_error(prediction, target)
                heldout_output_energy += _tensor_energy(target)
                heldout_state_error += _tensor_squared_error(
                    student_hidden[start:stop], teacher_hidden[start:stop]
                )
                heldout_state_energy += _tensor_energy(teacher_hidden[start:stop])

        factor_path = partial_dir / f"layer_{layer_index:03d}.safetensors"
        save_file(
            {
                "input_factor": factors.input_factor,
                "output_basis": factors.output_basis,
            },
            str(factor_path),
        )
        record = {
            "layer": layer_index,
            **dict(factors.metrics),
            "tsqr": tsqr_diagnostics,
            "fixed_encoder_source": source_encoder_record,
            "heldout_relative_output_mse": (
                heldout_output_error / heldout_output_energy
            ),
            "heldout_decoder_state_relative_mse": (
                heldout_state_error / heldout_state_energy
            ),
            "elapsed_seconds": time.perf_counter() - layer_started,
        }
        metrics.append(record)
        artifacts[str(layer_index)] = {
            "file": factor_path.name,
            "sha256": _sha256(factor_path),
            "input_factor_shape": list(factors.input_factor.shape),
            "output_basis_shape": list(factors.output_basis.shape),
            "factor_dtype": str(factors.input_factor.dtype),
            "metrics": record,
        }
        print(
            f"[Sequential MLP C1] layer={layer_index} fit_mse="
            f"{record['fit_relative_output_mse']:.8g} heldout_mlp_mse="
            f"{record['heldout_relative_output_mse']:.8g} heldout_state_mse="
            f"{record['heldout_decoder_state_relative_mse']:.8g} seconds="
            f"{record['elapsed_seconds']:.2f}",
            flush=True,
        )
        layer.to(device="cpu")
        del (
            student_regression_inputs,
            input_cholesky,
            objective_input_cholesky,
            c_yx,
            c_yy,
            heldout_targets,
            factors,
            compressed_down,
            fitted_decoder,
            fixed_input_factor,
        )
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    mean_fit = sum(row["fit_relative_output_mse"] for row in metrics) / len(metrics)
    mean_regularized = sum(
        row["fit_relative_regularized_objective"] for row in metrics
    ) / len(metrics)
    mean_heldout = sum(row["heldout_relative_output_mse"] for row in metrics) / len(
        metrics
    )
    manifest: Mapping[str, Any] = {
        "format": FACTOR_FORMAT,
        "schema_version": 1,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_started_utc": started_utc,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "model_type": "qwen3",
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_layers,
        },
        "calibration": {
            "windows": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest": windows_manifest,
            "fit_windows": args.fit_windows,
            "heldout_windows": args.heldout_windows,
            "sequence_length": sequence_length,
            "fit_rows": fit_rows,
            "heldout_rows": args.heldout_windows * sequence_length,
        },
        "fit_config": {
            "rank": args.rank,
            "tp_size": args.tp_size,
            "model_dtype": args.model_dtype,
            "factor_dtype": args.factor_dtype,
            "objective": (
                "dense_teacher_mlp_output_from_fixed_student_latent"
                if fixed_encoder_manifest is not None
                else "dense_teacher_mlp_output_from_sequential_compressed_student_input"
            ),
            "algorithm": (
                "sequential_fixed_encoder_decoder_least_squares"
                if fixed_encoder_manifest is not None
                else (
                    "sequential_exact_reduced_rank_regression"
                    if args.relative_damping == 0
                    else "sequential_ridge_reduced_rank_regression"
                )
            ),
            "solver": (
                "zero_damping_latent_streaming_tsqr_least_squares"
                if fixed_encoder_manifest is not None
                else (
                    "zero_damping_streaming_tsqr_reduced_rank_regression"
                    if args.relative_damping == 0
                    else "ridge_augmented_streaming_tsqr_reduced_rank_regression"
                )
            ),
            "relative_covariance_damping": args.relative_damping,
            "tsqr_windows_per_block": args.tsqr_windows_per_block,
            "layerwise_closure": True,
        },
        "fixed_encoder_source": (
            None
            if fixed_encoder_dir is None
            else {
                "directory": str(fixed_encoder_dir),
                "manifest_sha256": _sha256(fixed_encoder_dir / "manifest.json"),
                "algorithm": fixed_encoder_manifest["fit_config"]["algorithm"],
            }
        ),
        "communication": {
            "collective": "latent_allreduce",
            "dense_width": hidden_size,
            "latent_width": args.rank,
            "fraction_of_dense_allreduce": args.rank / hidden_size,
            "reduction_fraction": 1.0 - args.rank / hidden_size,
            "ring_elements_per_token_per_rank": (
                2.0 * (args.tp_size - 1) / args.tp_size * args.rank
            ),
        },
        "summary": {
            "mean_fit_relative_output_mse": mean_fit,
            "mean_fit_relative_regularized_objective": mean_regularized,
            "mean_heldout_relative_output_mse": mean_heldout,
            "final_heldout_decoder_state_relative_mse": metrics[-1][
                "heldout_decoder_state_relative_mse"
            ],
        },
        "layers": list(range(num_layers)),
        "artifacts": artifacts,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "safetensors": _installed_version("safetensors"),
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(partial_dir / "manifest.json", manifest)
    os.replace(partial_dir, output_dir)
    print(
        f"[Sequential MLP C1] complete output={output_dir} "
        f"mean_fit={mean_fit:.8g} mean_heldout={mean_heldout:.8g} "
        f"seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
