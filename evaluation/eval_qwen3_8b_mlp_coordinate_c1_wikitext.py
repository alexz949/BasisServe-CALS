#!/usr/bin/env python3
"""Fit and evaluate Qwen3-8B coordinate-selected MLP C1 decoders.

The fixed TP-source-local coordinates come from a completed Gram/SRRQR run.
On the same frozen C4 fit split, this evaluator accumulates the global selected
activation covariance and teacher-output cross moment, then solves

    D = argmin_D E[||A W_down.T - A[:, S] D.T||^2].

WikiText-2 evaluation uses a single-process compact gather/decode quality
equivalent.  It does not benchmark the sparse collective.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any, Mapping

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.mlp_coordinate_c1 import (  # noqa: E402
    CoordinateMLPDecoderRuntime,
    fit_mlp_coordinate_decoder,
)
from evaluation.analyze_qwen35_gated_attention_sparsity import (  # noqa: E402
    _git_commit,
    _installed_version,
)
from evaluation.eval_qwen35_postgate_topk_crossdomain import (  # noqa: E402
    _file_sha256,
    _serializable_ppl,
)
from evaluation.eval_qwen3_8b_mlp_gram_srrqr_wikitext import (  # noqa: E402
    _atomic_json,
    _atomic_text,
    _load_calibration_windows,
    _standard_wikitext_samples,
)
from scripts.eval_qwen35_projected_gdn_nll import _dtype, _evaluate  # noqa: E402


FORMAT = "basisserve.qwen3_8b.mlp_coordinate_c1_wikitext.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-windows", required=True)
    parser.add_argument("--selection-result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-offset", type=int, default=0)
    parser.add_argument("--calibration-count", type=int, default=256)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--relative-damping", type=float, default=0.0)
    parser.add_argument("--ppl-sequence-length", type=int, default=2048)
    parser.add_argument("--ppl-max-tokens", type=int)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_selection(
    result_dir: Path,
    *,
    expected_layers: int,
    expected_width: int,
    tp_size: int,
) -> tuple[dict[int, Tensor], dict[str, Any]]:
    results_path = result_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if payload.get("format") != "basisserve.qwen3_8b.mlp_gram_srrqr_wikitext.v2":
        raise ValueError("coordinate C1 requires a v2 Gram/SRRQR result")
    variants = payload.get("variants", [])
    if len(variants) != 1:
        raise ValueError("selection result must contain exactly one sparse variant")
    variant = variants[0]
    if int(payload["geometry"]["layers"]) != expected_layers:
        raise ValueError("selection result has the wrong layer count")
    if int(payload["geometry"]["intermediate_width"]) != expected_width:
        raise ValueError("selection result has the wrong MLP width")
    if int(payload["geometry"]["tp_size"]) != tp_size:
        raise ValueError("selection result has the wrong TP size")
    artifact = payload.get("selection_artifact", {})
    artifact_path = Path(str(artifact.get("path", ""))).expanduser().resolve()
    if artifact_path.parent != result_dir or not artifact_path.is_file():
        raise ValueError("selection artifact is outside the declared result directory")
    if _file_sha256(artifact_path) != artifact.get("sha256"):
        raise ValueError("selection artifact hash differs from results.json")
    tensors = load_file(str(artifact_path), device="cpu")
    name = str(variant["name"])
    local_width = expected_width // tp_size
    indices: dict[int, Tensor] = {}
    for layer_index in range(expected_layers):
        key = f"{name}__layer_{layer_index:02d}_mask"
        mask = tensors.get(key)
        if mask is None or tuple(mask.shape) != (expected_width,):
            raise ValueError(f"missing coordinate mask for layer {layer_index}")
        selected = torch.nonzero(mask.to(dtype=torch.bool), as_tuple=False).flatten()
        counts = torch.bincount(
            torch.div(selected, local_width, rounding_mode="floor"),
            minlength=tp_size,
        )
        if counts.numel() != tp_size or not torch.equal(
            counts, counts[:1].expand_as(counts)
        ):
            raise ValueError(f"layer {layer_index} selection is not TP balanced")
        if int(counts[0]) != int(variant["kept_per_source"]):
            raise ValueError(f"layer {layer_index} selection count differs from metadata")
        indices[layer_index] = selected.contiguous()
    return indices, {
        "result_dir": str(result_dir),
        "results_path": str(results_path),
        "results_sha256": _file_sha256(results_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact["sha256"],
        "variant": variant,
        "selection": payload["selection"],
        "calibration": payload["calibration"],
    }


class _CoordinateMomentCollector:
    def __init__(
        self,
        projections: list[nn.Linear],
        indices: Mapping[int, Tensor],
    ) -> None:
        self.rows = {layer: 0 for layer in range(len(projections))}
        self.second: dict[int, Tensor] = {}
        self.cross: dict[int, Tensor] = {}
        self.target_energy: dict[int, Tensor] = {}
        self.device_indices: dict[int, Tensor] = {}
        self.handles = []
        for layer_index, projection in enumerate(projections):
            selected = indices[layer_index].to(device=projection.weight.device)
            selected_width = int(selected.numel())
            self.device_indices[layer_index] = selected
            self.second[layer_index] = torch.zeros(
                selected_width,
                selected_width,
                dtype=torch.float32,
                device=projection.weight.device,
            )
            self.cross[layer_index] = torch.zeros(
                projection.out_features,
                selected_width,
                dtype=torch.float32,
                device=projection.weight.device,
            )
            self.target_energy[layer_index] = torch.zeros(
                (), dtype=torch.float32, device=projection.weight.device
            )
            self.handles.append(
                projection.register_forward_hook(self._hook(layer_index))
            )

    def _hook(self, layer_index: int):
        def accumulate(
            module: nn.Module,
            args: tuple[Any, ...],
            output: Tensor,
        ) -> None:
            del module
            if not args or not isinstance(args[0], Tensor) or not isinstance(output, Tensor):
                raise TypeError(f"layer {layer_index} coordinate capture is not tensor-valued")
            source = args[0].reshape(-1, args[0].shape[-1])
            selected = source.index_select(-1, self.device_indices[layer_index]).float()
            target = output.reshape(-1, output.shape[-1]).float()
            self.second[layer_index].addmm_(selected.T, selected)
            self.cross[layer_index].addmm_(target.T, selected)
            self.target_energy[layer_index].add_(target.square().sum())
            self.rows[layer_index] += int(selected.shape[0])

        return accumulate

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.device_indices.clear()


@torch.inference_mode()
def _capture_coordinate_moments(
    model: nn.Module,
    projections: list[nn.Linear],
    indices: Mapping[int, Tensor],
    windows: Tensor,
    *,
    batch_size: int,
    device: str,
) -> tuple[_CoordinateMomentCollector, int, float]:
    collector = _CoordinateMomentCollector(projections, indices)
    started = time.perf_counter()
    try:
        for start in range(0, len(windows), batch_size):
            stop = min(start + batch_size, len(windows))
            input_ids = windows[start:stop].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
            del outputs, input_ids
            print(f"[Coordinate capture] windows={stop}/{len(windows)}", flush=True)
    finally:
        collector.close()
    expected_rows = int(windows.numel())
    if set(collector.rows.values()) != {expected_rows}:
        raise RuntimeError(f"coordinate capture row counts differ: {collector.rows}")
    return collector, expected_rows, time.perf_counter() - started


def _fit_decoders(
    collector: _CoordinateMomentCollector,
    *,
    rows: int,
    input_width: int,
    relative_damping: float,
    factor_dtype: torch.dtype,
) -> tuple[dict[int, Tensor], list[dict[str, Any]], float]:
    decoders: dict[int, Tensor] = {}
    metrics: list[dict[str, Any]] = []
    started = time.perf_counter()
    for layer_index in sorted(collector.second):
        second = collector.second.pop(layer_index)
        cross = collector.cross.pop(layer_index)
        target_energy = collector.target_energy.pop(layer_index)
        second.div_(float(rows))
        cross.div_(float(rows))
        target_energy.div_(float(rows))
        fit = fit_mlp_coordinate_decoder(
            second,
            cross,
            target_energy,
            input_width=input_width,
            factor_dtype=factor_dtype,
            work_dtype=torch.float64,
            relative_damping=relative_damping,
        )
        decoders[layer_index] = fit.decoder_weight
        record = {"layer": layer_index, **dict(fit.metrics)}
        metrics.append(record)
        print(
            f"[Coordinate solve] layer={layer_index} "
            f"fit_mse={record['fit_relative_output_mse']:.8g} "
            f"stored_mse={record['stored_factor_relative_output_mse']:.8g} "
            f"chol_ratio={record['cholesky_diagonal_ratio']:.4g}",
            flush=True,
        )
        del second, cross, target_energy, fit
        torch.cuda.empty_cache()
    return decoders, metrics, time.perf_counter() - started


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    dense = payload["ppl"]["dense"]
    candidate = payload["ppl"]["candidate"]
    geometry = payload["geometry"]
    return "\n".join(
        [
            "# Qwen3-8B coordinate-selected MLP C1 WikiText-2",
            "",
            (
                "Static Gram/SRRQR coordinates use a free least-squares decoder. "
                "Runtime is a single-GPU quality equivalent; communication reduction "
                "is analytical."
            ),
            "",
            "| Variant | K/source | PPL | PPL delta | Delta NLL | Paired SE | Top-1 agreement | Mean calibration output MSE | Ideal communication reduction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| dense | {geometry['local_intermediate_width']} | {dense['perplexity']:.7g} | 0 | 0 | 0 | 1 | 0 | 0% |",
            (
                f"| coordinate C1 | {geometry['kept_per_source']} | "
                f"{candidate['perplexity']:.7g} | "
                f"{candidate['perplexity'] / dense['perplexity'] - 1:+.3%} | "
                f"{candidate['delta_mean_nll']:+.7g} | "
                f"{candidate['paired_window_standard_error']:.7g} | "
                f"{candidate['top1_agreement']:.6f} | "
                f"{payload['fit_summary']['mean_stored_factor_relative_output_mse']:.6f} | "
                f"{geometry['ideal_communication_reduction']:.3%} |"
            ),
            "",
        ]
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if min(
        args.calibration_count,
        args.calibration_batch_size,
        args.tp_size,
        args.ppl_sequence_length,
        args.ppl_batch_size,
        args.torch_num_threads,
    ) <= 0:
        raise ValueError("all counts, widths, and batch settings must be positive")
    if args.relative_damping < 0:
        raise ValueError("relative damping must be nonnegative")
    torch.set_num_threads(args.torch_num_threads)

    model_path = Path(args.model_path).expanduser().resolve()
    calibration_source = Path(args.calibration_windows).expanduser().resolve()
    selection_dir = Path(args.selection_result_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
    if not selection_dir.is_dir():
        raise FileNotFoundError(selection_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite coordinate C1 result: {output_dir}")
    plan = {
        "model": str(model_path),
        "calibration_windows": str(calibration_source),
        "calibration_offset": args.calibration_offset,
        "calibration_count": args.calibration_count,
        "selection_result_dir": str(selection_dir),
        "relative_damping": args.relative_damping,
        "output_dir": str(output_dir),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        return

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("coordinate C1 evaluation requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_float32_matmul_precision("high")
    calibration, calibration_metadata = _load_calibration_windows(
        calibration_source,
        offset=args.calibration_offset,
        count=args.calibration_count,
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=args.local_files_only
    )
    samples, wikitext_metadata = _standard_wikitext_samples(
        tokenizer,
        sequence_length=args.ppl_sequence_length,
        maximum_tokens=args.ppl_max_tokens,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=_dtype(args.dtype),
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    ).eval()
    model.config.use_cache = False
    if str(model.config.model_type) != "qwen3":
        raise ValueError("this evaluator requires a dense Qwen3 checkpoint")
    if calibration_metadata["model_config_sha256"] != _file_sha256(
        model_path / "config.json"
    ):
        raise ValueError("calibration windows belong to another model config")
    layers = model.model.layers
    projections = [layer.mlp.down_proj for layer in layers]
    if any(not isinstance(projection, nn.Linear) for projection in projections):
        raise TypeError("every layer must expose a linear MLP down projection")
    input_widths = {int(projection.in_features) for projection in projections}
    output_widths = {int(projection.out_features) for projection in projections}
    if len(input_widths) != 1 or len(output_widths) != 1:
        raise ValueError("MLP geometry is not uniform")
    input_width = input_widths.pop()
    output_width = output_widths.pop()
    if input_width % args.tp_size:
        raise ValueError("MLP input width is not TP divisible")

    indices, selection_metadata = _load_selection(
        selection_dir,
        expected_layers=len(layers),
        expected_width=input_width,
        tp_size=args.tp_size,
    )
    if (
        selection_metadata["calibration"]["artifact_sha256"]
        != calibration_metadata["artifact_sha256"]
        or int(selection_metadata["calibration"]["offset"])
        != args.calibration_offset
        or int(selection_metadata["calibration"]["count"])
        != args.calibration_count
    ):
        raise ValueError("coordinate fit and SRRQR selection must use the same C4 fit split")
    variant = selection_metadata["variant"]
    kept_per_source = int(variant["kept_per_source"])
    selected_width = int(next(iter(indices.values())).numel())

    collector, calibration_rows, capture_elapsed = _capture_coordinate_moments(
        model,
        projections,
        indices,
        calibration,
        batch_size=args.calibration_batch_size,
        device=args.device,
    )
    del calibration
    factor_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    decoders, fit_metrics, fit_elapsed = _fit_decoders(
        collector,
        rows=calibration_rows,
        input_width=input_width,
        relative_damping=args.relative_damping,
        factor_dtype=factor_dtype,
    )
    del collector
    torch.cuda.empty_cache()

    dense_raw = _evaluate(
        model,
        samples,
        batch_size=args.ppl_batch_size,
        device=args.device,
        label="dense",
    )
    dense = _serializable_ppl(dense_raw, baseline=None)
    with CoordinateMLPDecoderRuntime(
        model,
        indices,
        decoders,
        tp_size=args.tp_size,
    ) as runtime:
        candidate_raw = _evaluate(
            model,
            samples,
            batch_size=args.ppl_batch_size,
            device=args.device,
            label="mlp_coordinate_c1",
        )
        candidate = _serializable_ppl(candidate_raw, baseline=dense_raw)
        if runtime.kept_per_source != kept_per_source:
            raise RuntimeError("coordinate runtime geometry differs from selection")

    output_dir.mkdir(parents=True)
    factor_path = output_dir / "coordinate_c1.safetensors"
    temporary_factor = factor_path.with_suffix(".safetensors.tmp")
    factor_tensors: dict[str, Tensor] = {}
    for layer_index in range(len(layers)):
        factor_tensors[f"layer_{layer_index:02d}_indices"] = indices[
            layer_index
        ].contiguous()
        factor_tensors[f"layer_{layer_index:02d}_decoder_weight"] = decoders[
            layer_index
        ].contiguous()
    save_file(factor_tensors, str(temporary_factor))
    os.replace(temporary_factor, factor_path)

    exchange_fraction = selected_width / (2.0 * output_width)
    elapsed = time.perf_counter() - started
    fit_summary = {
        "mean_fit_relative_output_mse": statistics.fmean(
            row["fit_relative_output_mse"] for row in fit_metrics
        ),
        "mean_stored_factor_relative_output_mse": statistics.fmean(
            row["stored_factor_relative_output_mse"] for row in fit_metrics
        ),
        "minimum_cholesky_diagonal_ratio": min(
            row["cholesky_diagonal_ratio"] for row in fit_metrics
        ),
        "maximum_decoder_frobenius_norm": max(
            row["decoder_frobenius_norm"] for row in fit_metrics
        ),
    }
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "elapsed_seconds": elapsed,
        "model": {
            "path": str(model_path),
            "config_sha256": _file_sha256(model_path / "config.json"),
            "weights_updated": False,
        },
        "geometry": {
            "layers": len(layers),
            "hidden_width": output_width,
            "intermediate_width": input_width,
            "tp_size": args.tp_size,
            "local_intermediate_width": input_width // args.tp_size,
            "kept_per_source": kept_per_source,
            "selected_width": selected_width,
            "ideal_sparse_exchange_fraction_vs_standard_allreduce": exchange_fraction,
            "ideal_communication_reduction": 1.0 - exchange_fraction,
        },
        "calibration": {
            **calibration_metadata,
            "rows": calibration_rows,
            "capture_elapsed_seconds": capture_elapsed,
        },
        "selection_source": selection_metadata,
        "fit": {
            "objective": "min_D E[||A W_down.T - A[:,S] D.T||^2]",
            "global_cross_source_decoder": True,
            "relative_damping": args.relative_damping,
            "work_dtype": "float64",
            "factor_dtype": args.dtype,
            "elapsed_seconds": fit_elapsed,
            "layers": fit_metrics,
        },
        "fit_summary": fit_summary,
        "protocol": {
            "intervention_point": "post_swiglu_pre_down_projection",
            "static_coordinates": True,
            "compact_decoder": True,
            "single_process_quality_equivalent": True,
            "sparse_kernel_executed": False,
            "collective_modified": False,
            "actual_standard_tp_communication_reduction": 0.0,
            "target_tp_topology": "source_selected_activation_exchange_to_output_row_owners",
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        },
        "factor_artifact": {
            "path": str(factor_path),
            "sha256": _file_sha256(factor_path),
            "tensor_count": len(factor_tensors),
        },
        "ppl": {
            "dataset": "wikitext2",
            "source": wikitext_metadata,
            "dense": dense,
            "candidate": candidate,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "safetensors": _installed_version("safetensors"),
        },
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    _atomic_json(output_dir / "results.json", payload)
    _atomic_text(output_dir / "summary.md", _summary_markdown(payload))
    print(f"[Done] elapsed={elapsed:.2f}s output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
