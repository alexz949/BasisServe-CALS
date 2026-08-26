#!/usr/bin/env python3
"""Evaluate Qwen3-8B static TP-local MLP Gram/SRRQR on WikiText-2.

The selector is calibrated on frozen C4 windows.  For each layer and virtual
TP source shard it forms the full channel-contribution Gram

    E[a a^T] * (W_down^T W_down),

including off-diagonal activation and output-direction interactions.  A
square-root POD sketch followed by Gu--Eisenstat strong RRQR selects fixed real
channels.  An optional calibration solve reweights the selected contributions
to approximate the full output.  WikiText evaluation uses dense masked
execution, so measured wall time is not a sparse-kernel or collective result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from basisserve.core.mlp_gram_srrqr import (  # noqa: E402
    StaticMLPMaskRuntime,
    contribution_gram,
    gram_subset_reweighting,
    gram_srrqr_coordinates,
)
from basisserve.sketching.coordinate_selection import selected_index_sha256  # noqa: E402
from evaluation.analyze_qwen35_gated_attention_sparsity import (  # noqa: E402
    _git_commit,
    _installed_version,
)
from evaluation.eval_qwen35_postgate_topk_crossdomain import (  # noqa: E402
    _file_sha256,
    _serializable_ppl,
)
from scripts.eval_qwen35_projected_gdn_nll import _dtype, _evaluate  # noqa: E402
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids  # noqa: E402


FORMAT = "basisserve.qwen3_8b.mlp_gram_srrqr_wikitext.v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-windows", required=True)
    parser.add_argument("--calibration-offset", type=int, default=0)
    parser.add_argument("--calibration-count", type=int, default=16)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-ratios", default="0.25,0.375")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--pod-oversample", type=int, default=64)
    parser.add_argument("--srrqr-bound", type=float, default=4.0)
    parser.add_argument("--srrqr-max-swaps", type=int, default=32)
    parser.add_argument("--static-reweight", action="store_true")
    parser.add_argument("--ppl-sequence-length", type=int, default=512)
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


def _ratios(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(piece.strip()) for piece in raw.split(",") if piece.strip()}))
    if not values or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in values):
        raise ValueError("keep ratios must be finite values in (0,1)")
    return values


def _ratio_tag(ratio: float) -> str:
    return f"{ratio * 100:.6g}".replace(".", "p")


def _variant_name(ratio: float, *, reweighted: bool) -> str:
    suffix = "_reweighted" if reweighted else ""
    return f"mlp_static_gram_srrqr_{_ratio_tag(ratio)}_source_local{suffix}"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _load_calibration_windows(
    source: Path,
    *,
    offset: int,
    count: int,
) -> tuple[Tensor, dict[str, Any]]:
    windows_path = source / "windows.safetensors" if source.is_dir() else source
    manifest_path = windows_path.parent / "manifest.json"
    if not windows_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing calibration window artifact: {source}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "basisserve.calibration.c4_document_windows.v1":
        raise ValueError("unsupported calibration window manifest")
    artifact = manifest["artifact"]
    if artifact["file"] != windows_path.name or _file_sha256(windows_path) != artifact["sha256"]:
        raise ValueError("calibration window hash differs from manifest")
    payload = load_file(str(windows_path), device="cpu")
    input_ids = payload[artifact["tensor"]]
    if tuple(input_ids.shape) != tuple(artifact["shape"]) or input_ids.ndim != 2:
        raise ValueError("calibration window tensor shape differs from manifest")
    stop = offset + count
    fit = manifest.get("splits", {}).get("fit")
    if offset < 0 or count <= 0 or stop > int(input_ids.shape[0]):
        raise ValueError("requested calibration windows are out of range")
    if fit is not None and not (int(fit["offset"]) <= offset and stop <= int(fit["stop"])):
        raise ValueError("Gram/SRRQR selection must use only the declared C4 fit split")
    records = manifest.get("records", [])[offset:stop]
    return input_ids[offset:stop].to(torch.long).contiguous(), {
        "manifest": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "artifact": str(windows_path),
        "artifact_sha256": artifact["sha256"],
        "offset": offset,
        "count": count,
        "sequence_length": int(input_ids.shape[1]),
        "model_config_sha256": manifest["model"]["config_sha256"],
        "window_bank_command": manifest.get("command"),
        "document_ids_sha256": hashlib.sha256(
            "\n".join(str(record.get("document_id")) for record in records).encode("utf-8")
        ).hexdigest(),
    }


def _standard_wikitext_samples(
    tokenizer: Any,
    *,
    sequence_length: int,
    maximum_tokens: int | None,
) -> tuple[tuple[Tensor, ...], dict[str, Any]]:
    token_ids = _token_ids(tokenizer, "wikitext2", "test", maximum_tokens).squeeze(0)
    available = int(token_ids.numel())
    used = available - (available % sequence_length)
    if used < sequence_length:
        raise ValueError("WikiText-2 test has fewer than one complete evaluation chunk")
    samples = tuple(
        token_ids[start : start + sequence_length].contiguous()
        for start in range(0, used, sequence_length)
    )
    return samples, {
        "dataset": "wikitext2",
        "split": "test",
        "available_tokens_after_limit": available,
        "used_input_tokens": used,
        "sequence_length": sequence_length,
        "windows": len(samples),
        "evaluated_next_tokens": len(samples) * (sequence_length - 1),
        "tokenizer_add_special_tokens": True,
    }


class _SourceLocalMomentCollector:
    def __init__(self, projections: list[nn.Linear], tp_size: int) -> None:
        self.tp_size = int(tp_size)
        self.local_width = int(projections[0].in_features) // self.tp_size
        self.sums: dict[int, Tensor] = {}
        self.rows = {layer: 0 for layer in range(len(projections))}
        self.handles = []
        for layer_index, projection in enumerate(projections):
            self.sums[layer_index] = torch.zeros(
                self.tp_size,
                self.local_width,
                self.local_width,
                dtype=torch.float32,
                device=projection.weight.device,
            )
            self.handles.append(
                projection.register_forward_pre_hook(self._hook(layer_index))
            )

    def _hook(self, layer_index: int):
        def accumulate(module: nn.Module, args: tuple[Any, ...]) -> None:
            del module
            if not args or not isinstance(args[0], Tensor):
                raise TypeError(f"layer {layer_index} MLP input is not a tensor")
            source = args[0]
            flat = source.reshape(-1, self.tp_size, self.local_width)
            by_source = flat.permute(1, 0, 2).float()
            self.sums[layer_index].add_(
                torch.bmm(by_source.transpose(1, 2), by_source)
            )
            self.rows[layer_index] += int(flat.shape[0])

        return accumulate

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def _capture_source_local_moments(
    model: nn.Module,
    projections: list[nn.Linear],
    windows: Tensor,
    *,
    tp_size: int,
    batch_size: int,
    device: str,
) -> tuple[dict[int, Tensor], int, float]:
    collector = _SourceLocalMomentCollector(projections, tp_size)
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
            print(f"[Gram capture] windows={stop}/{len(windows)}", flush=True)
    finally:
        collector.close()
    expected_rows = int(windows.numel())
    if set(collector.rows.values()) != {expected_rows}:
        raise RuntimeError(f"captured MLP row counts differ: {collector.rows}")
    return collector.sums, expected_rows, time.perf_counter() - started


def _selection_sweep(
    moments: dict[int, Tensor],
    projections: list[nn.Linear],
    ratios: tuple[float, ...],
    *,
    rows: int,
    tp_size: int,
    pod_oversample: int,
    bound: float,
    max_swaps: int,
    reweight: bool,
) -> tuple[
    dict[float, dict[int, Tensor]],
    dict[float, dict[int, Tensor]] | None,
    list[dict[str, Any]],
    float,
]:
    width = int(projections[0].in_features)
    local_width = width // tp_size
    kept = {ratio: max(1, int(round(local_width * ratio))) for ratio in ratios}
    masks = {ratio: {} for ratio in ratios}
    scales = {ratio: {} for ratio in ratios} if reweight else None
    diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()
    for layer_index, projection in enumerate(projections):
        layer_moment = moments.pop(layer_index) / rows
        layer_masks = {
            ratio: torch.zeros(width, dtype=torch.bool) for ratio in ratios
        }
        layer_scales = (
            {ratio: torch.zeros(width, dtype=torch.float32) for ratio in ratios}
            if reweight
            else None
        )
        for source_rank in range(tp_size):
            start = source_rank * local_width
            stop = start + local_width
            gram = contribution_gram(
                layer_moment[source_rank],
                projection.weight[:, start:stop],
            )
            ones = torch.ones(local_width, dtype=gram.dtype, device=gram.device)
            full_output_energy = float((ones @ gram @ ones).detach().cpu())
            for ratio in ratios:
                result = gram_srrqr_coordinates(
                    gram,
                    kept[ratio],
                    pod_oversample=pod_oversample,
                    bound=bound,
                    max_swaps=max_swaps,
                )
                local_indices = result.indices
                layer_masks[ratio][start + local_indices] = True
                reweighting = None
                if layer_scales is not None:
                    reweighting = gram_subset_reweighting(gram, local_indices)
                    layer_scales[ratio][start + local_indices] = (
                        reweighting.coefficients
                    )
                omitted = torch.ones(local_width, dtype=torch.bool, device=gram.device)
                omitted[local_indices.to(gram.device)] = False
                omitted_gram = gram[omitted][:, omitted]
                omitted_ones = torch.ones(
                    int(omitted.sum()), dtype=gram.dtype, device=gram.device
                )
                zero_fill_error = float(
                    (omitted_ones @ omitted_gram @ omitted_ones).detach().cpu()
                )
                rrqr = result.diagnostics.to_dict()
                rrqr.pop("selected_columns")
                diagnostic = {
                        "layer": layer_index,
                        "source_rank": source_rank,
                        "ratio": ratio,
                        "kept": kept[ratio],
                        "selected_index_sha256": selected_index_sha256(local_indices),
                        "pod_dimension": result.pod_dimension,
                        "pod_energy_fraction": result.pod_energy_fraction,
                        "gram_trace": result.gram_trace,
                        "gram_min_eigenvalue": result.gram_min_eigenvalue,
                        "gram_negative_eigenvalue_fraction": result.gram_negative_eigenvalue_fraction,
                        "gram_positive_eigenvalue_count": result.gram_positive_eigenvalue_count,
                        "gram_effective_rank_fp32": result.gram_effective_rank_fp32,
                        "calibration_zero_fill_output_error_fraction": zero_fill_error
                        / max(abs(full_output_energy), 1e-30),
                        "rrqr": rrqr,
                    }
                if reweighting is not None:
                    diagnostic["reweighting"] = {
                        "solver": "fp64_cholesky",
                        "ridge": 0.0,
                        "relative_output_residual": reweighting.relative_residual,
                        "coefficient_minimum": reweighting.coefficient_minimum,
                        "coefficient_maximum": reweighting.coefficient_maximum,
                        "coefficient_mean": reweighting.coefficient_mean,
                        "coefficient_l2": reweighting.coefficient_l2,
                    }
                diagnostics.append(diagnostic)
                print(
                    f"[Gram SRRQR] layer={layer_index}/{len(projections) - 1} "
                    f"source={source_rank}/{tp_size - 1} "
                    f"ratio={ratio:g} swaps={rrqr['swaps']} "
                    f"rho={rrqr['final_max_rho']:.4g}",
                    flush=True,
                )
            del gram
        for ratio in ratios:
            masks[ratio][layer_index] = layer_masks[ratio]
            if scales is not None and layer_scales is not None:
                scales[ratio][layer_index] = layer_scales[ratio]
        del layer_moment
        torch.cuda.empty_cache()
    return masks, scales, diagnostics, time.perf_counter() - started


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    dense = payload["ppl"]["dense"]
    geometry = payload["geometry"]
    lines = [
        "# Qwen3-8B MLP static Gram/SRRQR WikiText-2",
        "",
        (
            "Fixed TP-source-local masks are calibrated on C4 with the full "
            "activation/output contribution Gram. Evaluation uses dense masked "
            "execution; reported sparse-exchange reductions are analytical."
        ),
        "",
        "| Variant | K/source | PPL | PPL delta | Delta NLL | Paired SE | Top-1 agreement | Retained input energy | Transformed input energy | Calibration output residual | Ideal sparse exchange reduction vs AllReduce |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| dense | {geometry['local_intermediate_width']} | {dense['perplexity']:.7g} | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0% |",
    ]
    metadata = {row["name"]: row for row in payload["variants"]}
    for name, result in payload["ppl"]["candidates"].items():
        row = metadata[name]
        calibration_residual = row["mean_calibration_output_residual"]
        calibration_residual_text = (
            "n/a" if calibration_residual is None else f"{calibration_residual:.6f}"
        )
        lines.append(
            f"| {name} | {row['kept_per_source']} | {result['perplexity']:.7g} | "
            f"{result['perplexity'] / dense['perplexity'] - 1:+.3%} | "
            f"{result['delta_mean_nll']:+.7g} | {result['paired_window_standard_error']:.7g} | "
            f"{result['top1_agreement']:.6f} | {row['mean_retained_input_energy']:.6f} | "
            f"{row['mean_transformed_input_energy']:.6f} | "
            f"{calibration_residual_text} | "
            f"{row['ideal_sparse_exchange_reduction_vs_standard_allreduce']:.3%} |"
        )
    lines.append("")
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    ratios = _ratios(args.keep_ratios)
    if min(
        args.tp_size,
        args.calibration_count,
        args.calibration_batch_size,
        args.ppl_batch_size,
        args.torch_num_threads,
    ) <= 0:
        raise ValueError("TP size, window counts, batch sizes, and thread count must be positive")
    if args.pod_oversample < 0 or args.srrqr_max_swaps < 0 or args.srrqr_bound < 1.0:
        raise ValueError("invalid POD/SRRQR controls")
    torch.set_num_threads(args.torch_num_threads)

    model_path = Path(args.model_path).expanduser().resolve()
    calibration_source = Path(args.calibration_windows).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Gram/SRRQR result: {output_dir}")
    plan = {
        "model": str(model_path),
        "calibration_windows": str(calibration_source),
        "calibration_offset": args.calibration_offset,
        "calibration_count": args.calibration_count,
        "static_reweight": args.static_reweight,
        "variants": [
            "dense",
            *[
                _variant_name(ratio, reweighted=args.static_reweight)
                for ratio in ratios
            ],
        ],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        return

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("Gram/SRRQR evaluation requires CUDA")
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    torch.set_float32_matmul_precision("high")

    calibration, calibration_metadata = _load_calibration_windows(
        calibration_source,
        offset=args.calibration_offset,
        count=args.calibration_count,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=args.local_files_only)
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
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if str(model.config.model_type) != "qwen3":
        raise ValueError("this evaluator requires a dense Qwen3 checkpoint")
    if calibration_metadata["model_config_sha256"] != _file_sha256(
        model_path / "config.json"
    ):
        raise ValueError("calibration window model config differs from Qwen3-8B")
    layers = model.model.layers
    projections = [getattr(getattr(layer, "mlp", None), "down_proj", None) for layer in layers]
    if not projections or any(not isinstance(projection, nn.Linear) for projection in projections):
        raise TypeError("every decoder layer must expose a linear MLP down projection")
    widths = {int(projection.in_features) for projection in projections}
    output_widths = {int(projection.out_features) for projection in projections}
    if len(widths) != 1 or len(output_widths) != 1:
        raise ValueError("MLP geometry is not uniform")
    intermediate_width = widths.pop()
    hidden_width = output_widths.pop()
    if intermediate_width % args.tp_size:
        raise ValueError("MLP intermediate width is not divisible by TP size")
    local_width = intermediate_width // args.tp_size

    moments, calibration_rows, capture_elapsed = _capture_source_local_moments(
        model,
        projections,
        calibration,
        tp_size=args.tp_size,
        batch_size=args.calibration_batch_size,
        device=args.device,
    )
    masks, scales, selection_diagnostics, selection_elapsed = _selection_sweep(
        moments,
        projections,
        ratios,
        rows=calibration_rows,
        tp_size=args.tp_size,
        pod_oversample=args.pod_oversample,
        bound=args.srrqr_bound,
        max_swaps=args.srrqr_max_swaps,
        reweight=args.static_reweight,
    )
    del moments, calibration
    torch.cuda.empty_cache()

    dense_raw = _evaluate(model, samples, batch_size=args.ppl_batch_size, device=args.device, label="dense")
    dense = _serializable_ppl(dense_raw, baseline=None)
    candidates: dict[str, Any] = {}
    variants: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    selection_tensors: dict[str, Tensor] = {}
    for ratio in ratios:
        name = _variant_name(ratio, reweighted=args.static_reweight)
        with StaticMLPMaskRuntime(
            model,
            masks[ratio],
            scales=None if scales is None else scales[ratio],
            tp_size=args.tp_size,
            profile=True,
        ) as runtime:
            candidate_raw = _evaluate(
                model,
                samples,
                batch_size=args.ppl_batch_size,
                device=args.device,
                label=name,
            )
            candidates[name] = _serializable_ppl(candidate_raw, baseline=dense_raw)
            summary = runtime.profile_snapshot()
            profiles.extend({"variant": name, **row} for row in summary)
            kept_per_source = int(runtime.kept_per_source or 0)
            kept_per_vector = kept_per_source * args.tp_size
            realized_ratio = kept_per_vector / intermediate_width
            exchange_fraction = kept_per_vector / (2.0 * hidden_width)
            reweight_diagnostics = [
                row["reweighting"]
                for row in selection_diagnostics
                if row["ratio"] == ratio and "reweighting" in row
            ]
            variants.append(
                {
                    "name": name,
                    "keep_ratio": ratio,
                    "realized_ratio": realized_ratio,
                    "kept_per_source": kept_per_source,
                    "kept_per_vector": kept_per_vector,
                    "mean_retained_input_energy": statistics.fmean(
                        row["retained_input_energy"] for row in summary
                    ),
                    "mean_transformed_input_energy": statistics.fmean(
                        row["transformed_input_energy"] for row in summary
                    ),
                    "mean_calibration_output_residual": (
                        statistics.fmean(
                            row["relative_output_residual"]
                            for row in reweight_diagnostics
                        )
                        if reweight_diagnostics
                        else None
                    ),
                    "current_dense_oracle_allreduce_reduction": 0.0,
                    "ideal_static_sparse_exchange_fraction_vs_standard_allreduce": exchange_fraction,
                    "ideal_sparse_exchange_reduction_vs_standard_allreduce": 1.0 - exchange_fraction,
                    "potential_down_projection_mac_reduction": 1.0 - realized_ratio,
                }
            )
            del candidate_raw
        for layer_index, mask in masks[ratio].items():
            selection_tensors[f"{name}__layer_{layer_index:02d}_mask"] = (
                mask.contiguous()
            )
            if scales is not None:
                selection_tensors[f"{name}__layer_{layer_index:02d}_scale"] = (
                    scales[ratio][layer_index].contiguous()
                )

    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True)
    selection_path = output_dir / "static_selection.safetensors"
    temporary_selection = selection_path.with_suffix(".safetensors.tmp")
    save_file(selection_tensors, str(temporary_selection))
    os.replace(temporary_selection, selection_path)
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
            "hidden_width": hidden_width,
            "intermediate_width": intermediate_width,
            "tp_size": args.tp_size,
            "local_intermediate_width": local_width,
            "standard_allreduce_elements_per_token": hidden_width,
            "dense_activation_allgather_fraction_vs_standard_allreduce": intermediate_width
            / (2.0 * hidden_width),
        },
        "calibration": {
            **calibration_metadata,
            "rows": calibration_rows,
            "capture_elapsed_seconds": capture_elapsed,
        },
        "selection": {
            "definition": "E[a a^T] hadamard (W_down^T W_down)",
            "scope": "tp_source_local",
            "pod_oversample": args.pod_oversample,
            "srrqr_bound": args.srrqr_bound,
            "srrqr_max_swaps": args.srrqr_max_swaps,
            "selected_channel_reweighting": args.static_reweight,
            "reweighting_objective": (
                "G[S,S] c = G[S,:] 1" if args.static_reweight else None
            ),
            "reweighting_solver": "fp64_cholesky" if args.static_reweight else None,
            "reweighting_ridge": 0.0 if args.static_reweight else None,
            "elapsed_seconds": selection_elapsed,
            "diagnostics": selection_diagnostics,
        },
        "protocol": {
            "intervention_point": "post_swiglu_pre_down_projection",
            "static_mask": True,
            "static_selected_channel_scale": args.static_reweight,
            "static_scale_foldable_into_down_projection": args.static_reweight,
            "runtime_index_payload_required": False,
            "runtime_scale_payload_required": False,
            "quality_oracle_dense_zero_fill": not args.static_reweight,
            "quality_oracle_dense_masked_execution": True,
            "sparse_kernel_executed": False,
            "collective_modified": False,
            "actual_standard_tp_communication_reduction": 0.0,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        },
        "variants": variants,
        "profiles": profiles,
        "selection_artifact": {
            "path": str(selection_path),
            "sha256": _file_sha256(selection_path),
            "tensor_count": len(selection_tensors),
        },
        "ppl": {
            "dataset": "wikitext2",
            "source": wikitext_metadata,
            "dense": dense,
            "candidates": candidates,
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
