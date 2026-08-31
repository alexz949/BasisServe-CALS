#!/usr/bin/env python3
"""Run a two-layer Nemotron-H Mamba/attention C1-Wo quality pilot.

The model state and attention KV cache are untouched.  The command captures
post-gate Mamba ``out_proj`` and attention ``o_proj`` input covariances for the
first block of each kind, fits TP-source C1 factors, folds the factors into
quality-equivalent dense weights, and compares dense versus hybrid WikiText-2
perplexity.  Captured sufficient statistics are saved so rank/solver studies
can be repeated without rerunning the model.  Large checkpoints are sharded
across the visible GPUs while C1 fitting runs on a designated solver GPU.
"""

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
import sys
import time
from typing import Any, Mapping

from safetensors.torch import save_file
import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.nemotron_h_c1 import (  # noqa: E402
    NemotronC1Target,
    first_nemotron_h_c1_target_per_kind,
    nemotron_h_decoder_layers,
    projection_for_target,
)
from basisserve.core.tp_source_wo_c1 import (  # noqa: E402
    fold_factors_to_dense_weight,
)
from basisserve.core.tp_source_wo_fit import (  # noqa: E402
    TPSourceWOFitConfig,
    fit_tp_source_wo_c1,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)
from scripts.eval_svdllm_safetensors_ppl_accelerate import (  # noqa: E402
    _token_ids,
)


FORMAT = "basisserve.nemotron_h.c1_wo_two_layer_pilot.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--communication-reduction", type=float, default=0.75)
    parser.add_argument("--source-rank", type=int)
    parser.add_argument("--calibration-dataset", default="wikitext2")
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--fit-windows", type=int, default=16)
    parser.add_argument("--heldout-windows", type=int, default=8)
    parser.add_argument("--calibration-seqlen", type=int, default=512)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--eval-dataset", default="wikitext2")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--eval-seqlen", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-max-samples", type=int, default=8)
    parser.add_argument("--encoder-sweeps", type=int, default=1)
    parser.add_argument("--minimum-encoder-sweeps", type=int, default=0)
    parser.add_argument("--covariance-damping", type=float, default=1.0e-5)
    parser.add_argument("--encoder-relative-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--encoder-patience", type=int, default=1)
    parser.add_argument("--maximum-backtracks", type=int, default=8)
    parser.add_argument(
        "--work-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--factor-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument(
        "--model-device-map",
        choices=("single", "balanced", "balanced_low_0"),
        default="balanced_low_0",
        help="Accelerate placement strategy. Large checkpoints should use balanced_low_0.",
    )
    parser.add_argument(
        "--max-memory-per-gpu-gib",
        type=int,
        default=76,
        help="Hard model-loading budget for each visible GPU.",
    )
    parser.add_argument(
        "--mamba-implementation",
        choices=("fast", "torch"),
        default="fast",
        help="Require CUDA Mamba kernels or explicitly opt into the slow reference path.",
    )
    parser.add_argument("--solver-device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }[name]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_rank(args: argparse.Namespace, output_width: int) -> int:
    if args.source_rank is not None:
        if args.source_rank <= 0:
            raise ValueError("source rank must be positive")
        return int(args.source_rank)
    if not 0.0 < args.communication_reduction < 1.0:
        raise ValueError("communication reduction must lie in (0,1)")
    total_rank = 2.0 * output_width * (1.0 - args.communication_reduction)
    rounded = int(round(total_rank))
    if not math.isclose(total_rank, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("communication reduction does not produce an integer total rank")
    if rounded % args.tp:
        raise ValueError("total private rank is not divisible by TP")
    return rounded // args.tp


def _model_loading_configuration(
    *,
    strategy: str,
    solver_device: torch.device,
    cuda_device_count: int,
    max_memory_per_gpu_gib: int,
) -> tuple[str | dict[str, str], dict[int, str] | None]:
    if max_memory_per_gpu_gib <= 0:
        raise ValueError("model GPU memory budget must be positive")
    if strategy == "single":
        return {"": str(solver_device)}, None
    if strategy not in {"balanced", "balanced_low_0"}:
        raise ValueError(f"unknown model device-map strategy: {strategy}")
    if cuda_device_count < 2:
        raise ValueError(f"{strategy} model loading requires at least two CUDA devices")
    return strategy, {
        index: f"{max_memory_per_gpu_gib}GiB"
        for index in range(cuda_device_count)
    }


def _cuda_placement(model: nn.Module) -> dict[str, str]:
    raw = getattr(model, "hf_device_map", {})
    placement = {str(key): str(value) for key, value in raw.items()}
    if any(value in {"cpu", "disk"} for value in placement.values()):
        raise RuntimeError(f"Nemotron model parameters were offloaded: {placement}")
    return placement


def _require_fast_mamba(targets: tuple[NemotronC1Target, ...], model: nn.Module) -> None:
    import sys

    mamba = next(target for target in targets if target.layer_kind == "linear_attention")
    projection = projection_for_target(model, mamba)
    mixer_module = nemotron_h_decoder_layers(model)[mamba.layer_index].mixer
    implementation = sys.modules.get(type(mixer_module).__module__)
    if implementation is None or not bool(
        getattr(implementation, "is_fast_path_available", False)
    ):
        raise RuntimeError(
            "fast Nemotron Mamba path is unavailable; install the Hugging Face "
            "`kernels` package and its mamba-ssm/causal-conv1d kernels"
        )
    if projection.weight.device.type != "cuda":
        raise RuntimeError("Mamba output projection is not CUDA resident")


def _empty_cuda_caches() -> None:
    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()


class _CovarianceCapture:
    def __init__(
        self,
        model: nn.Module,
        targets: tuple[NemotronC1Target, ...],
    ) -> None:
        self.targets = {target.layer_index: target for target in targets}
        self.sums: dict[str, dict[int, Tensor]] = {"fit": {}, "heldout": {}}
        self.rows = {"fit": 0, "heldout": 0}
        self.active_split: str | None = None
        self.expected_rows = 0
        self.seen: set[int] = set()
        self.handles = [
            projection_for_target(model, target).register_forward_pre_hook(
                self._hook(target)
            )
            for target in targets
        ]

    def _hook(self, target: NemotronC1Target):
        def capture(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            del module
            if self.active_split is None or not inputs or not isinstance(inputs[0], Tensor):
                raise RuntimeError("Nemotron covariance hook fired outside an active batch")
            activation = inputs[0].detach()
            if int(activation.shape[-1]) != target.input_width:
                raise ValueError(
                    f"layer {target.layer_index} projection input has shape "
                    f"{tuple(activation.shape)}, expected width {target.input_width}"
                )
            flat = activation.reshape(-1, target.input_width).float()
            if len(flat) != self.expected_rows:
                raise ValueError("Nemotron covariance hook row count differs from batch")
            destination = self.sums[self.active_split].get(target.layer_index)
            if destination is None:
                destination = torch.zeros(
                    target.input_width,
                    target.input_width,
                    device=activation.device,
                    dtype=torch.float32,
                )
                self.sums[self.active_split][target.layer_index] = destination
            destination.addmm_(flat.transpose(0, 1), flat)
            self.seen.add(target.layer_index)

        return capture

    def begin(self, split: str, rows: int) -> None:
        if self.active_split is not None or split not in self.sums or rows <= 0:
            raise RuntimeError("invalid Nemotron covariance batch transition")
        self.active_split = split
        self.expected_rows = int(rows)
        self.seen.clear()

    def finish(self) -> None:
        expected = set(self.targets)
        if self.active_split is None or self.seen != expected:
            raise RuntimeError(
                f"Nemotron covariance capture missed layers {sorted(expected - self.seen)}"
            )
        self.rows[self.active_split] += self.expected_rows
        self.active_split = None
        self.expected_rows = 0
        self.seen.clear()

    def offload(self, split: str, expected_rows: int) -> None:
        if self.active_split is not None or self.rows[split] != expected_rows:
            raise RuntimeError(f"cannot offload incomplete {split} covariance")
        for layer_index in self.targets:
            matrix = self.sums[split][layer_index]
            self.sums[split][layer_index] = (
                matrix.div_(float(expected_rows)).cpu().contiguous()
            )
        _empty_cuda_caches()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def _capture_covariances(
    model: nn.Module,
    tokenizer: Any,
    targets: tuple[NemotronC1Target, ...],
    args: argparse.Namespace,
) -> _CovarianceCapture:
    total_windows = args.fit_windows + args.heldout_windows
    tokens = _token_ids(
        tokenizer,
        args.calibration_dataset,
        args.calibration_split,
        total_windows * args.calibration_seqlen,
    ).flatten()
    needed = total_windows * args.calibration_seqlen
    if int(tokens.numel()) < needed:
        raise ValueError("calibration split does not contain enough tokens")
    windows = tokens[:needed].reshape(total_windows, args.calibration_seqlen)
    input_device = model.get_input_embeddings().weight.device
    capture = _CovarianceCapture(model, targets)
    try:
        for split, first, stop in (
            ("fit", 0, args.fit_windows),
            ("heldout", args.fit_windows, total_windows),
        ):
            for start in range(first, stop, args.calibration_batch_size):
                end = min(stop, start + args.calibration_batch_size)
                batch = windows[start:end].to(input_device)
                capture.begin(split, int(batch.numel()))
                model(
                    input_ids=batch,
                    attention_mask=torch.ones_like(batch),
                    use_cache=False,
                )
                capture.finish()
                print(
                    f"[NemotronPilot] capture split={split} windows={end-first}/{stop-first}",
                    flush=True,
                )
            capture.offload(
                split,
                (stop - first) * args.calibration_seqlen,
            )
    except Exception:
        capture.close()
        raise
    capture.close()
    return capture


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if min(
        args.tp,
        args.fit_windows,
        args.heldout_windows,
        args.calibration_seqlen,
        args.calibration_batch_size,
        args.eval_seqlen,
        args.eval_batch_size,
        args.eval_max_samples,
    ) <= 0:
        raise ValueError("Nemotron pilot sizes must be positive")
    solver_device = torch.device(args.solver_device)
    if solver_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Nemotron-H C1 pilot requires CUDA")
    if solver_device.index is not None and solver_device.index >= torch.cuda.device_count():
        raise ValueError("solver device is not visible")
    torch.cuda.set_device(solver_device)
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)
    model_path = Path(args.model_path).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"local Nemotron-H checkpoint is absent: {model_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite Nemotron pilot: {output_dir}")
    partial_dir.mkdir(parents=True)

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    if str(config.model_type) != "nemotron_h":
        raise ValueError(f"expected Nemotron-H, found {config.model_type!r}")
    config.use_mamba_kernels = args.mamba_implementation == "fast"
    model_dtype = _dtype(args.model_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    model_device_map, max_memory = _model_loading_configuration(
        strategy=args.model_device_map,
        solver_device=solver_device,
        cuda_device_count=torch.cuda.device_count(),
        max_memory_per_gpu_gib=args.max_memory_per_gpu_gib,
    )
    print(
        f"[NemotronPilot] loading model={model_path} "
        f"device_map={model_device_map} max_memory={max_memory} "
        f"mamba_implementation={args.mamba_implementation}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        dtype=model_dtype,
        device_map=model_device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    ).eval()
    model.config.use_cache = False
    placement = _cuda_placement(model)
    print(f"[NemotronPilot] hf_device_map={placement}", flush=True)
    targets = first_nemotron_h_c1_target_per_kind(model)
    if args.mamba_implementation == "fast":
        _require_fast_mamba(targets, model)
    if len({target.output_width for target in targets}) != 1:
        raise ValueError("Mamba and attention C1 targets have different output widths")
    source_rank = _source_rank(args, targets[0].output_width)
    layouts = {
        target.layer_index: target.layout(
            tp_size=args.tp,
            source_rank=source_rank,
        )
        for target in targets
    }
    for target in targets:
        accounting = layouts[target.layer_index].accounting()
        print(
            f"[NemotronPilot] target layer={target.layer_index} "
            f"kind={target.layer_kind} input={target.input_width} "
            f"output={target.output_width} rank={source_rank} "
            f"comm_reduction={accounting['reduction_vs_dense_allreduce']:.6f}",
            flush=True,
        )

    dense_ppl = _eval_ppl_fp32_loss(
        model,
        tokenizer,
        dataset=args.eval_dataset,
        split=args.eval_split,
        seqlen=args.eval_seqlen,
        batch_size=args.eval_batch_size,
        max_samples=args.eval_max_samples,
        max_tokens=args.eval_max_samples * args.eval_seqlen,
    )
    capture = _capture_covariances(model, tokenizer, targets, args)

    fit_config = TPSourceWOFitConfig(
        encoder_sweeps=args.encoder_sweeps,
        minimum_encoder_sweeps=args.minimum_encoder_sweeps,
        covariance_damping=args.covariance_damping,
        encoder_relative_tolerance=args.encoder_relative_tolerance,
        encoder_patience=args.encoder_patience,
        maximum_backtracks=args.maximum_backtracks,
    )
    work_dtype = _dtype(args.work_dtype)
    factor_dtype = _dtype(args.factor_dtype)
    layer_records: list[dict[str, Any]] = []
    for target in targets:
        layer_started = time.perf_counter()
        projection = projection_for_target(model, target)
        weight = projection.weight.detach().cpu().contiguous()
        fit_covariance = capture.sums["fit"][target.layer_index]
        heldout_covariance = capture.sums["heldout"][target.layer_index]
        snapshot_name = f"layer_{target.layer_index:03d}_{target.layer_kind}_covariance.safetensors"
        snapshot_path = partial_dir / snapshot_name
        save_file(
            {
                "fit_covariance": fit_covariance,
                "heldout_covariance": heldout_covariance,
                "weight": weight,
            },
            str(snapshot_path),
        )
        print(
            f"[NemotronPilot] fit layer={target.layer_index} kind={target.layer_kind}",
            flush=True,
        )
        fitted = fit_tp_source_wo_c1(
            weight,
            fit_covariance,
            heldout_covariance,
            layouts[target.layer_index],
            config=fit_config,
            work_device=solver_device,
            work_dtype=work_dtype,
            factor_dtype=factor_dtype,
            objective_name=f"nemotron_h_layer_{target.layer_index:03d}_{target.layer_kind}",
        )
        factor_name = f"layer_{target.layer_index:03d}_{target.layer_kind}_c1.safetensors"
        factor_path = partial_dir / factor_name
        save_file(
            {
                "source_encoders": fitted.source_encoders,
                "source_decoders": fitted.source_decoders,
            },
            str(factor_path),
        )
        folded = fold_factors_to_dense_weight(
            fitted.source_encoders.to(device=solver_device, dtype=torch.float32),
            fitted.source_decoders.to(device=solver_device, dtype=torch.float32),
            layouts[target.layer_index],
        )
        projection.weight.copy_(
            folded.to(
                device=projection.weight.device,
                dtype=projection.weight.dtype,
            )
        )
        layer_records.append(
            {
                "target": asdict(target),
                "layout": layouts[target.layer_index].accounting(),
                "snapshot": {
                    "file": snapshot_name,
                    "sha256": _sha256(snapshot_path),
                },
                "factors": {
                    "file": factor_name,
                    "sha256": _sha256(factor_path),
                    "encoder_shape": list(fitted.source_encoders.shape),
                    "decoder_shape": list(fitted.source_decoders.shape),
                },
                "fit": {
                    "relative_mse": fitted.fit_relative_mse,
                    "heldout_relative_mse": fitted.heldout_relative_mse,
                    "quantized_relative_mse": fitted.quantized_fit_relative_mse,
                    "quantized_heldout_relative_mse": fitted.quantized_heldout_relative_mse,
                    "selected_boundary": fitted.selected_boundary,
                    "selected_sweep": fitted.selected_sweep,
                    "checkpoints": fitted.checkpoints,
                    "diagnostics": fitted.diagnostics,
                },
                "elapsed_seconds": time.perf_counter() - layer_started,
            }
        )
        del fitted, folded
        _empty_cuda_caches()

    hybrid_ppl = _eval_ppl_fp32_loss(
        model,
        tokenizer,
        dataset=args.eval_dataset,
        split=args.eval_split,
        seqlen=args.eval_seqlen,
        batch_size=args.eval_batch_size,
        max_samples=args.eval_max_samples,
        max_tokens=args.eval_max_samples * args.eval_seqlen,
    )
    elapsed = time.perf_counter() - started
    result = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "timestamp_started_utc": started_utc,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "model_type": str(config.model_type),
            "hidden_size": int(config.hidden_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "mamba_implementation": args.mamba_implementation,
            "device_map_strategy": args.model_device_map,
            "device_map": placement,
        },
        "scope": {
            "state_compression": "none",
            "kv_cache_compression": "none",
            "projection_compression": "first_mamba_and_first_attention_output_projection",
            "tp_size": args.tp,
            "source_rank": source_rank,
        },
        "calibration": {
            "dataset": args.calibration_dataset,
            "split": args.calibration_split,
            "fit_windows": args.fit_windows,
            "heldout_windows": args.heldout_windows,
            "sequence_length": args.calibration_seqlen,
            "fit_rows": args.fit_windows * args.calibration_seqlen,
            "heldout_rows": args.heldout_windows * args.calibration_seqlen,
        },
        "fit_config": asdict(fit_config),
        "layers": layer_records,
        "quality": {
            "dense": dense_ppl,
            "two_layer_c1": hybrid_ppl,
            "ppl_ratio": hybrid_ppl["ppl"] / dense_ppl["ppl"],
            "ppl_delta": hybrid_ppl["ppl"] - dense_ppl["ppl"],
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_devices": {
                str(device_index): torch.cuda.get_device_name(device_index)
                for device_index in range(torch.cuda.device_count())
            },
            "solver_device": str(solver_device),
            "peak_cuda_allocated_bytes_by_device": {
                str(device_index): int(torch.cuda.max_memory_allocated(device_index))
                for device_index in range(torch.cuda.device_count())
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(partial_dir / "results.json", result)
    os.replace(partial_dir, output_dir)
    print(
        f"[NemotronPilot] complete dense_ppl={dense_ppl['ppl']:.8f} "
        f"c1_ppl={hybrid_ppl['ppl']:.8f} output={output_dir} seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
