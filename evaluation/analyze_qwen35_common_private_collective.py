#!/usr/bin/env python3
"""Run the Qwen3.5 common/private latent collective quality oracle.

The analysis is single-GPU and offline.  It simulates TP partitions, fits all
factors from document-isolated training rows, and only then loads held-out
validation rows.  No checkpoint or distributed runtime is modified.
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
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file
from torch import Tensor
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.analysis.common_private_collective import (  # noqa: E402
    CommonPrivateBudget,
    CommonPrivateFactors,
    collective_accounting,
    common_private_budgets,
    common_private_output_metrics,
    factors_from_bank,
    fit_private_pod_bank,
    shared_only_factors,
    uniform_private_ranks,
)
from basisserve.analysis.mlp_topk_chebyshev import polar_retract_columns  # noqa: E402
from basisserve.diagnostics.qwen35_gated_attention_sparsity import (  # noqa: E402
    reconstruct_bf16_runtime,
    reconstruct_gdn_silu_runtime,
)
from basisserve.sketching.coordinate_selection import (  # noqa: E402
    compute_uncentered_pod,
)
from evaluation.analyze_qwen35_activated_wo_topk_chebyshev import (  # noqa: E402
    DEFAULT_FULL_SNAPSHOTS,
    DEFAULT_GDN_SNAPSHOTS,
    _load_manifest,
    _validate_aligned_provenance,
    _validate_snapshot_bank,
)
from evaluation.analyze_qwen35_gated_attention_sparsity import (  # noqa: E402
    _git_commit,
    _installed_version,
    _load_o_proj as load_full_o_proj,
    _validate_manifest as validate_full_manifest,
    _validate_model as validate_full_model,
)
from evaluation.analyze_qwen35_gdn_silu_sparsity import (  # noqa: E402
    _load_o_proj as load_gdn_o_proj,
    _validate_manifest as validate_gdn_manifest,
    _validate_model as validate_gdn_model,
)
from evaluation.analyze_qwen35_mlp_topk_chebyshev import (  # noqa: E402
    _atomic_json,
    _atomic_text,
    _file_sha256,
    _load_down_weight,
    _load_manifest as load_mlp_manifest,
    _load_tensors,
    _model_metadata as mlp_model_metadata,
    _validate_snapshot_files as validate_mlp_snapshot_files,
)


FORMAT = "basisserve.qwen35.common_private_collective_oracle.v1"
ROW_FORMAT = "basisserve.qwen35.common_private_collective_oracle.row.v1"
DEFAULT_MLP_SNAPSHOTS = (
    REPO_ROOT / "results/qwen35_9b_mlp_topk_chebyshev/"
    "c4_t8192_d2048_v8192_s2048_p128_seed20260809_l0_16_31_snapshots"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-snapshots", default=str(DEFAULT_MLP_SNAPSHOTS))
    parser.add_argument("--full-snapshots", default=str(DEFAULT_FULL_SNAPSHOTS))
    parser.add_argument("--gdn-snapshots", default=str(DEFAULT_GDN_SNAPSHOTS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--mlp-layers", default="0,16,31")
    parser.add_argument("--full-layers", default="3,15,31")
    parser.add_argument("--gdn-layers", default="2,14,30")
    parser.add_argument("--baseline-ranks", default="1536")
    parser.add_argument("--shared-fractions", default="1,0.75,0.5,0.25,0")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--train-rows", type=int, default=8192)
    parser.add_argument("--validation-rows", type=int, default=8192)
    parser.add_argument("--pod-oversample", type=int, default=16)
    parser.add_argument("--pod-niter", type=int, default=4)
    parser.add_argument("--private-probe-multiplier", type=float, default=1.5)
    parser.add_argument("--private-probe-extra", type=int, default=16)
    parser.add_argument("--local-output-chunk-size", type=int, default=512)
    parser.add_argument("--metric-chunk-size", type=int, default=128)
    parser.add_argument("--active-tokens", type=int, default=1)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--substantial-improvement", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(
        sorted(set(int(piece.strip()) for piece in raw.split(",") if piece.strip()))
    )


def _csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(
        dict.fromkeys(float(piece.strip()) for piece in raw.split(",") if piece.strip())
    )


def _exact_split(
    *,
    block_type: str,
    split_record: Mapping[str, Any],
    rows: int,
    weight: Tensor,
    geometry: Any,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if block_type == "mlp_down":
        tensors = _load_tensors(split_record, ("C", "Y"))
        activation = tensors["C"][:rows].contiguous()
        teacher = tensors["Y"][:rows].contiguous()
    else:
        tensors = load_file(str(split_record["path"]), device="cpu")
        if block_type == "full_attention":
            runtime = reconstruct_bf16_runtime(
                tensors["h_pre_gate"][:rows].to(device),
                tensors["gate_logits"][:rows].to(device),
            )
        elif block_type == "gdn":
            runtime = reconstruct_gdn_silu_runtime(
                tensors["raw_core"][:rows].to(device),
                tensors["gate_preactivation"][:rows].to(device),
                tensors["norm_weight"].to(device),
                num_value_heads=int(geometry.num_value_heads),
                value_head_dim=int(geometry.value_head_dim),
                rms_norm_eps=float(geometry.rms_norm_eps),
            )
        else:
            raise ValueError(f"unsupported block type {block_type!r}")
        activation = runtime.c_post_gate.cpu().contiguous()
        teacher = F.linear(runtime.c_post_gate, weight.to(device=device), None)
        if teacher.dtype != torch.bfloat16:
            raise TypeError("checkpoint output projection did not preserve BF16")
        teacher = teacher.cpu().contiguous()
        del runtime
    if activation.dtype != torch.bfloat16 or teacher.dtype != torch.bfloat16:
        raise TypeError("quality oracle requires exact BF16 activation and teacher")
    return activation, teacher


@torch.no_grad()
def _local_outputs(
    activation: Tensor,
    weight: Tensor,
    *,
    tp_size: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[Tensor, ...]:
    rows, input_width = map(int, activation.shape)
    output_width = int(weight.shape[0])
    if tuple(weight.shape) != (output_width, input_width) or input_width % tp_size:
        raise ValueError("row-parallel activation/weight geometry is invalid")
    if chunk_size <= 0:
        raise ValueError("local-output chunk size must be positive")
    shard_width = input_width // tp_size
    result = []
    for shard in range(tp_size):
        start_channel = shard * shard_width
        stop_channel = start_channel + shard_width
        shard_weight = weight[:, start_channel:stop_channel].to(
            device=device,
            dtype=torch.float32,
        )
        output = torch.empty(rows, output_width, dtype=torch.float32, device="cpu")
        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            shard_activation = activation[start:stop, start_channel:stop_channel].to(
                device=device, dtype=torch.float32
            )
            output[start:stop] = F.linear(shard_activation, shard_weight, None).cpu()
        result.append(output)
        del shard_weight
    return tuple(result)


@torch.no_grad()
def _dense_local_consistency(
    teacher: Tensor,
    local_outputs: Sequence[Tensor],
    *,
    chunk_size: int,
) -> dict[str, float]:
    error = 0.0
    target = 0.0
    rows = int(teacher.shape[0])
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        exact = teacher[start:stop].double()
        local_sum = (
            torch.stack([value[start:stop] for value in local_outputs], dim=0)
            .sum(dim=0)
            .double()
        )
        error += float((exact - local_sum).square().sum())
        target += float(exact.square().sum())
    relative = error / max(target, 1.0e-300)
    if relative > 5.0e-5:
        raise RuntimeError(f"TP-local outputs disagree with dense teacher: {relative}")
    return {
        "relative_mse": relative,
        "error_energy": error,
        "teacher_energy": target,
    }


def _method_name(
    budget: CommonPrivateBudget,
    allocation: str,
) -> tuple[str, str]:
    if budget.total_private_rank == 0:
        return "pure_allreduce", "pure_allreduce"
    if budget.shared_rank == 0:
        return (
            f"pure_allgather_{allocation}",
            "pure_allgather",
        )
    return f"mixed_{allocation}", "mixed"


def _candidate(
    *,
    budget: CommonPrivateBudget,
    factors: CommonPrivateFactors,
    block_type: str,
    layer: int,
    allocation: str,
    input_shard_widths: Sequence[int],
    active_tokens: int,
    dtype_bytes: int,
    fit_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    method, collective_model = _method_name(budget, allocation)
    return {
        "factors": factors,
        "record": {
            "format": ROW_FORMAT,
            "block_type": block_type,
            "layer": int(layer),
            "baseline_rank": int(budget.baseline_rank),
            "method": method,
            "collective_model": collective_model,
            "allocation": allocation,
            "shared_fraction_requested": float(budget.shared_fraction),
            "shared_fraction_realized": factors.shared_rank / budget.baseline_rank,
            "shared_rank": factors.shared_rank,
            "private_ranks": list(factors.private_ranks),
            "total_private_rank": factors.total_private_rank,
            "fit": {
                "split": factors.fit_split,
                "validation_used": factors.validation_used_for_fit,
                **dict(fit_metadata),
            },
            "accounting": collective_accounting(
                factors,
                input_shard_widths=input_shard_widths,
                active_tokens=active_tokens,
                dtype_bytes=dtype_bytes,
            ),
            "splits": {},
        },
    }


def _summary_csv(records: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "block_type",
        "layer",
        "baseline_rank",
        "method",
        "allocation",
        "shared_rank",
        "total_private_rank",
        "private_ranks",
        "ideal_ring_units",
        "padded_ring_units",
        "decoder_parameters",
        "decoder_bytes_read",
        "total_local_encoder_parameters",
        "collective_launches",
        "validation_relative_mse",
        "validation_normalized_cross",
        "validation_prediction_teacher_energy",
        "validation_cosine_similarity",
        "validation_p95_token_error",
    )
    lines = [",".join(columns)]
    for record in records:
        accounting = record["accounting"]
        validation = record["splits"]["validation"]
        values = {
            **record,
            **accounting,
            "private_ranks": "|".join(map(str, record["private_ranks"])),
            "validation_relative_mse": validation["relative_mse"],
            "validation_normalized_cross": validation["normalized_cross"],
            "validation_prediction_teacher_energy": validation[
                "prediction_teacher_energy"
            ],
            "validation_cosine_similarity": validation["cosine_similarity"],
            "validation_p95_token_error": validation[
                "per_token_relative_squared_error"
            ]["p95"],
        }
        lines.append(
            ",".join(
                "" if values.get(column) is None else str(values.get(column, ""))
                for column in columns
            )
        )
    return "\n".join(lines) + "\n"


def _decisions(
    records: Sequence[Mapping[str, Any]],
    *,
    substantial_improvement: float,
) -> list[dict[str, Any]]:
    scopes = sorted(
        {
            (str(row["block_type"]), int(row["layer"]), int(row["baseline_rank"]))
            for row in records
        }
    )
    decisions = []
    for block_type, layer, baseline_rank in scopes:
        relevant = [
            row
            for row in records
            if row["block_type"] == block_type
            and int(row["layer"]) == layer
            and int(row["baseline_rank"]) == baseline_rank
        ]
        shared = next(row for row in relevant if row["method"] == "pure_allreduce")
        uniform = [row for row in relevant if row["allocation"] == "uniform"]
        spectrum = [row for row in relevant if row["allocation"] == "spectrum_aware"]
        best_uniform = min(
            uniform,
            key=lambda row: float(row["splits"]["validation"]["relative_mse"]),
        )
        best_spectrum = min(
            spectrum,
            key=lambda row: float(row["splits"]["validation"]["relative_mse"]),
        )
        shared_error = float(shared["splits"]["validation"]["relative_mse"])
        uniform_error = float(best_uniform["splits"]["validation"]["relative_mse"])
        spectrum_error = float(best_spectrum["splits"]["validation"]["relative_mse"])
        uniform_improvement = shared_error / max(uniform_error, 1.0e-300)
        spectrum_improvement = shared_error / max(spectrum_error, 1.0e-300)
        deployable_gain = uniform_improvement >= substantial_improvement
        ideal_only_gain = spectrum_improvement >= substantial_improvement and int(
            best_spectrum["accounting"]["padded_ring_units"]
        ) > int(shared["accounting"]["ideal_ring_units"])
        decisions.append(
            {
                "block_type": block_type,
                "layer": layer,
                "baseline_rank": baseline_rank,
                "pure_allreduce_validation_relative_mse": shared_error,
                "best_uniform_method": best_uniform["method"],
                "best_uniform_shared_rank": int(best_uniform["shared_rank"]),
                "best_uniform_validation_relative_mse": uniform_error,
                "best_uniform_improvement": uniform_improvement,
                "best_spectrum_method": best_spectrum["method"],
                "best_spectrum_shared_rank": int(best_spectrum["shared_rank"]),
                "best_spectrum_private_ranks": best_spectrum["private_ranks"],
                "best_spectrum_validation_relative_mse": spectrum_error,
                "best_spectrum_improvement": spectrum_improvement,
                "best_spectrum_padded_ring_units": int(
                    best_spectrum["accounting"]["padded_ring_units"]
                ),
                "substantial_deployable_uniform_gain": deployable_gain,
                "substantial_ideal_spectrum_gain_only": ideal_only_gain,
            }
        )
    return decisions


def _summary_markdown(decisions: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Qwen3.5 Common–Private Latent Collective Quality Oracle",
        "",
        (
            "All factors use training rows only. Validation rows are loaded after "
            "factor fitting. Spectrum-aware quality is reported with both ideal "
            "AllGatherv and standard padded-NCCL communication accounting."
        ),
        "",
        "| Block | Layer | r | Pure AR MSE | Best uniform | Gain | "
        "Best spectrum | Ideal gain | Spectrum padded units |",
        "|---|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for row in decisions:
        lines.append(
            f"| {row['block_type']} | {row['layer']} | {row['baseline_rank']} | "
            f"{float(row['pure_allreduce_validation_relative_mse']):.6g} | "
            f"{row['best_uniform_method']} rs={row['best_uniform_shared_rank']}, "
            f"MSE={float(row['best_uniform_validation_relative_mse']):.6g} | "
            f"{float(row['best_uniform_improvement']):.3f}x | "
            f"{row['best_spectrum_method']} rs={row['best_spectrum_shared_rank']}, "
            f"MSE={float(row['best_spectrum_validation_relative_mse']):.6g} | "
            f"{float(row['best_spectrum_improvement']):.3f}x | "
            f"{row['best_spectrum_padded_ring_units']} |"
        )
    lines.extend(
        [
            "",
            (
                "Uniform allocations have equal or near-equal private ranks and "
                "are the standard-AllGather comparison. Uneven spectrum-aware "
                "allocations may require padding; their ideal byte point is not a "
                "deployment claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _plots(records: Sequence[Mapping[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[CommonPrivate] plot skipped: {error}", flush=True)
        return []
    files = []
    scopes = sorted(
        {
            (str(row["block_type"]), int(row["layer"]), int(row["baseline_rank"]))
            for row in records
        }
    )
    for block_type, layer, baseline_rank in scopes:
        relevant = [
            row
            for row in records
            if row["block_type"] == block_type
            and int(row["layer"]) == layer
            and int(row["baseline_rank"]) == baseline_rank
        ]
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for allocation, label in (
            ("uniform", "uniform"),
            ("spectrum_aware", "spectrum-aware"),
        ):
            rows = sorted(
                (row for row in relevant if row["allocation"] == allocation),
                key=lambda row: float(row["shared_fraction_realized"]),
            )
            axes[0].plot(
                [float(row["shared_fraction_realized"]) for row in rows],
                [float(row["splits"]["validation"]["relative_mse"]) for row in rows],
                marker="o",
                label=label,
            )
            axes[1].plot(
                [
                    float(row["accounting"]["padded_ring_units"]) / (2 * baseline_rank)
                    for row in rows
                ],
                [float(row["splits"]["validation"]["relative_mse"]) for row in rows],
                marker="o",
                label=label,
            )
            axes[2].plot(
                [
                    float(row["accounting"]["decoder_bytes_read"]) / 2**20
                    for row in rows
                ],
                [float(row["splits"]["validation"]["relative_mse"]) for row in rows],
                marker="o",
                label=label,
            )
        axes[0].set_xlabel("Shared rank / baseline rank")
        axes[1].set_xlabel("Padded ring units / pure-AR units")
        axes[2].set_xlabel("Decoder bytes read (MiB)")
        for axis in axes:
            axis.set_ylabel("Validation relative MSE")
            axis.set_yscale("log")
            axis.grid(True, alpha=0.25)
            axis.legend()
        figure.suptitle(
            f"Qwen3.5 {block_type} layer {layer}, baseline rank {baseline_rank}"
        )
        figure.tight_layout()
        filename = f"{block_type}_layer_{layer:02d}_r{baseline_rank}_common_private.png"
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)
        files.append(filename)
    return files


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    mlp_layers = _csv_ints(args.mlp_layers)
    full_layers = _csv_ints(args.full_layers)
    gdn_layers = _csv_ints(args.gdn_layers)
    baseline_ranks = _csv_ints(args.baseline_ranks)
    shared_fractions = _csv_floats(args.shared_fractions)
    if (
        not mlp_layers
        or not full_layers
        or not gdn_layers
        or not baseline_ranks
        or min(baseline_ranks) <= 0
        or not shared_fractions
        or tuple(shared_fractions) != (1.0, 0.75, 0.5, 0.25, 0.0)
        or args.tp <= 1
        or min(args.train_rows, args.validation_rows) <= 0
        or min(args.local_output_chunk_size, args.metric_chunk_size) <= 0
        or args.pod_oversample < 0
        or args.pod_niter < 0
        or args.private_probe_multiplier < 1.0
        or args.private_probe_extra <= 0
        or args.active_tokens <= 0
        or args.dtype_bytes <= 0
        or args.substantial_improvement <= 1.0
    ):
        raise ValueError("invalid common/private oracle configuration")

    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    model_argument = args.model_path

    mlp_dir = Path(args.mlp_snapshots).expanduser().resolve()
    mlp_manifest, mlp_records = load_mlp_manifest(mlp_dir)
    mlp_model_path, _, mlp_weight_map, model_metadata = mlp_model_metadata(
        mlp_manifest,
        model_argument,
    )
    if sorted(set(mlp_layers) - set(mlp_records)):
        raise ValueError("a requested MLP layer is absent from snapshots")
    validated_mlp = validate_mlp_snapshot_files(mlp_dir, mlp_records, mlp_layers)
    for layer in mlp_layers:
        for split, required in (
            ("train", args.train_rows),
            ("validation", args.validation_rows),
        ):
            if int(validated_mlp[layer][split]["rows"]) < required:
                raise ValueError(f"MLP layer {layer} {split} has too few rows")

    full_dir = Path(args.full_snapshots).expanduser().resolve()
    gdn_dir = Path(args.gdn_snapshots).expanduser().resolve()
    full_manifest_path = full_dir / "manifest.json"
    gdn_manifest_path = gdn_dir / "manifest.json"
    full_manifest = _load_manifest(full_manifest_path)
    gdn_manifest = _load_manifest(gdn_manifest_path)
    full_geometry, full_records, full_provenance = validate_full_manifest(
        full_manifest,
        manifest_path=full_manifest_path,
    )
    gdn_geometry, gdn_records, gdn_provenance = validate_gdn_manifest(
        gdn_manifest,
        manifest_path=gdn_manifest_path,
    )
    if sorted(set(full_layers) - set(full_records)):
        raise ValueError("a requested full-attention layer is absent from snapshots")
    if sorted(set(gdn_layers) - set(gdn_records)):
        raise ValueError("a requested GDN layer is absent from snapshots")
    aligned_gated_provenance = _validate_aligned_provenance(
        full_manifest,
        gdn_manifest,
    )
    model_path = Path(model_argument or str(mlp_model_path)).expanduser().resolve()
    full_model, full_weight_map = validate_full_model(
        full_manifest,
        model_path,
        full_geometry,
    )
    gdn_model, gdn_weight_map = validate_gdn_model(
        gdn_manifest,
        model_path,
        gdn_geometry,
    )
    if model_path != mlp_model_path:
        raise ValueError("MLP and gated snapshots use different checkpoints")
    for key in ("config_sha256", "safetensors_index_sha256", "model_revision"):
        if full_model[key] != gdn_model[key] or full_model[key] != model_metadata[key]:
            raise ValueError(f"snapshot checkpoint metadata differs at {key}")
    row_targets = {
        "train": int(args.train_rows),
        "dev": 1,
        "validation": int(args.validation_rows),
    }
    validated_full = _validate_snapshot_bank(
        full_dir,
        full_manifest,
        layers=full_layers,
        family="full_attention",
        row_targets=row_targets,
    )
    validated_gdn = _validate_snapshot_bank(
        gdn_dir,
        gdn_manifest,
        layers=gdn_layers,
        family="gdn",
        row_targets=row_targets,
    )

    hidden = int(model_metadata["hidden_size"])
    if max(baseline_ranks) > hidden:
        raise ValueError("baseline rank exceeds model hidden size")
    layer_specs = (
        [(layer, "mlp_down", None, validated_mlp[layer]) for layer in mlp_layers]
        + [
            (layer, "full_attention", full_geometry, validated_full[layer])
            for layer in full_layers
        ]
        + [(layer, "gdn", gdn_geometry, validated_gdn[layer]) for layer in gdn_layers]
    )
    layer_specs.sort(key=lambda item: (item[0], item[1]))

    partial_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    cuda_index = (
        device.index if device.type == "cuda" and device.index is not None else 0
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device {device}")
        torch.cuda.set_device(cuda_index)
        torch.cuda.init()
    started = time.perf_counter()
    timestamp_started = datetime.now(timezone.utc).isoformat()
    all_records: list[dict[str, Any]] = []
    layer_metadata: dict[str, Any] = {}
    shard_hashes: dict[str, str] = {}
    peak_cuda = 0

    for layer, block_type, geometry, validated in layer_specs:
        layer_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(cuda_index)
        print(
            f"[CommonPrivate] layer={layer} block={block_type} phase=load_train",
            flush=True,
        )
        if block_type == "mlp_down":
            input_width = int(mlp_records[layer]["intermediate_size"])
            weight, weight_metadata = _load_down_weight(
                model_path,
                mlp_weight_map,
                layer,
                expected_shape=(hidden, input_width),
                shard_hashes=shard_hashes,
            )
        elif block_type == "full_attention":
            input_width = int(full_geometry.wire_width)
            weight, weight_metadata = load_full_o_proj(
                model_path,
                full_weight_map,
                layer,
                full_geometry,
            )
        else:
            input_width = int(gdn_geometry.wire_width)
            weight, weight_metadata = load_gdn_o_proj(
                model_path,
                gdn_weight_map,
                layer,
                gdn_geometry,
            )
        if input_width % args.tp:
            raise ValueError(f"layer {layer} input width is not divisible by TP")
        input_shard_widths = (input_width // args.tp,) * args.tp
        train_activation, train_teacher = _exact_split(
            block_type=block_type,
            split_record=validated["train"],
            rows=args.train_rows,
            weight=weight,
            geometry=geometry,
            device=device,
        )
        train_local = _local_outputs(
            train_activation,
            weight,
            tp_size=args.tp,
            chunk_size=args.local_output_chunk_size,
            device=device,
        )
        train_consistency = _dense_local_consistency(
            train_teacher,
            train_local,
            chunk_size=args.metric_chunk_size,
        )
        print(
            f"[CommonPrivate] layer={layer} block={block_type} phase=fit_shared",
            flush=True,
        )
        shared_pod = compute_uncentered_pod(
            train_teacher.to(device),
            max(baseline_ranks),
            oversample=args.pod_oversample,
            niter=args.pod_niter,
            seed=args.seed
            + 7919 * layer
            + {"mlp_down": 1, "full_attention": 2, "gdn": 3}[block_type],
        )
        maximum_shared, shared_retraction = polar_retract_columns(shared_pod.basis)
        maximum_shared = maximum_shared.cpu().contiguous()
        candidates: list[dict[str, Any]] = []
        pod_metadata = []
        for baseline_rank in baseline_ranks:
            for budget in common_private_budgets(baseline_rank, shared_fractions):
                shared = maximum_shared[:, : budget.shared_rank].contiguous()
                if budget.total_private_rank == 0:
                    factors = shared_only_factors(
                        shared,
                        baseline_rank=baseline_rank,
                        tp_size=args.tp,
                    )
                    candidates.append(
                        _candidate(
                            budget=budget,
                            factors=factors,
                            block_type=block_type,
                            layer=layer,
                            allocation="shared_only",
                            input_shard_widths=input_shard_widths,
                            active_tokens=args.active_tokens,
                            dtype_bytes=args.dtype_bytes,
                            fit_metadata={
                                "shared_basis": "train_teacher_output_pod_nested",
                                "private_basis": None,
                            },
                        )
                    )
                    continue
                print(
                    f"[CommonPrivate] layer={layer} block={block_type} "
                    f"phase=fit_private r={baseline_rank} rs={budget.shared_rank} "
                    f"rp_total={budget.total_private_rank}",
                    flush=True,
                )
                bank = fit_private_pod_bank(
                    train_local,
                    shared,
                    budget.total_private_rank,
                    maximum_private_ranks=input_shard_widths,
                    device=device,
                    oversample=args.pod_oversample,
                    niter=args.pod_niter,
                    seed=args.seed + 104729 * layer + 193 * budget.shared_rank,
                    probe_multiplier=args.private_probe_multiplier,
                    probe_extra=args.private_probe_extra,
                )
                uniform_ranks = uniform_private_ranks(
                    budget.total_private_rank,
                    args.tp,
                )
                for allocation, private_ranks in (
                    ("uniform", uniform_ranks),
                    ("spectrum_aware", bank.spectrum_ranks),
                ):
                    factors = factors_from_bank(
                        shared,
                        bank,
                        private_ranks,
                        baseline_rank=baseline_rank,
                        allocation=allocation,
                    )
                    candidates.append(
                        _candidate(
                            budget=budget,
                            factors=factors,
                            block_type=block_type,
                            layer=layer,
                            allocation=allocation,
                            input_shard_widths=input_shard_widths,
                            active_tokens=args.active_tokens,
                            dtype_bytes=args.dtype_bytes,
                            fit_metadata={
                                "shared_basis": "train_teacher_output_pod_nested",
                                "private_basis": "train_local_residual_output_pod",
                                "probe_ranks": list(bank.probe_ranks),
                                "spectrum_ranks": list(bank.spectrum_ranks),
                                "maximum_shared_private_orthogonality": bank.maximum_shared_private_orthogonality,
                            },
                        )
                    )
                pod_metadata.append(
                    {
                        "baseline_rank": baseline_rank,
                        "shared_rank": budget.shared_rank,
                        "total_private_rank": budget.total_private_rank,
                        "probe_ranks": list(bank.probe_ranks),
                        "spectrum_ranks": list(bank.spectrum_ranks),
                        "shards": list(bank.diagnostics),
                    }
                )
        print(
            f"[CommonPrivate] layer={layer} block={block_type} phase=metric_train",
            flush=True,
        )
        for candidate in candidates:
            candidate["record"]["splits"]["train"] = common_private_output_metrics(
                train_teacher,
                train_local,
                candidate["factors"],
                device=device,
                chunk_size=args.metric_chunk_size,
            )
        del train_activation, train_teacher, train_local
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"[CommonPrivate] layer={layer} block={block_type} phase=load_validation",
            flush=True,
        )
        validation_activation, validation_teacher = _exact_split(
            block_type=block_type,
            split_record=validated["validation"],
            rows=args.validation_rows,
            weight=weight,
            geometry=geometry,
            device=device,
        )
        validation_local = _local_outputs(
            validation_activation,
            weight,
            tp_size=args.tp,
            chunk_size=args.local_output_chunk_size,
            device=device,
        )
        validation_consistency = _dense_local_consistency(
            validation_teacher,
            validation_local,
            chunk_size=args.metric_chunk_size,
        )
        print(
            f"[CommonPrivate] layer={layer} block={block_type} phase=metric_validation",
            flush=True,
        )
        for candidate in candidates:
            candidate["record"]["splits"]["validation"] = common_private_output_metrics(
                validation_teacher,
                validation_local,
                candidate["factors"],
                device=device,
                chunk_size=args.metric_chunk_size,
            )
            all_records.append(candidate["record"])
        layer_peak = (
            int(torch.cuda.max_memory_allocated(cuda_index))
            if device.type == "cuda"
            else 0
        )
        peak_cuda = max(peak_cuda, layer_peak)
        key = f"{block_type}:layer_{layer:02d}"
        layer_metadata[key] = {
            "block_type": block_type,
            "layer": layer,
            "input_width": input_width,
            "input_shard_widths": list(input_shard_widths),
            "elapsed_seconds": time.perf_counter() - layer_started,
            "peak_cuda_allocated_bytes": layer_peak,
            "weight": weight_metadata,
            "train_local_dense_consistency": train_consistency,
            "validation_local_dense_consistency": validation_consistency,
            "shared_pod": {
                **shared_pod.diagnostics(),
                "polar_retraction": shared_retraction,
            },
            "private_pod": pod_metadata,
            "snapshot_files": {
                split: validated[split] for split in ("train", "validation")
            },
        }
        del candidates, validation_activation, validation_teacher, validation_local
        del weight, shared_pod, maximum_shared
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[CommonPrivate] layer={layer} block={block_type} done "
            f"seconds={layer_metadata[key]['elapsed_seconds']:.3f}",
            flush=True,
        )

    decisions = _decisions(
        all_records,
        substantial_improvement=args.substantial_improvement,
    )
    _atomic_text(
        partial_dir / "results.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_records),
    )
    _atomic_text(partial_dir / "summary.csv", _summary_csv(all_records))
    _atomic_text(partial_dir / "summary.md", _summary_markdown(decisions))
    plot_files = _plots(all_records, partial_dir)
    elapsed = time.perf_counter() - started
    manifest = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_started_utc": timestamp_started,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "model": full_model,
        "layers": {
            "mlp_down": list(mlp_layers),
            "full_attention": list(full_layers),
            "gdn": list(gdn_layers),
        },
        "baseline_ranks": list(baseline_ranks),
        "shared_fractions": list(shared_fractions),
        "tp_size": args.tp,
        "train_rows": args.train_rows,
        "validation_rows": args.validation_rows,
        "fit_split": "train",
        "validation_loaded_after_factor_fitting": True,
        "validation_used_for_fit": False,
        "snapshot_provenance": {
            "mlp": {
                "manifest": str(mlp_dir / "manifest.json"),
                "manifest_sha256": _file_sha256(mlp_dir / "manifest.json"),
                "combined_document_sha256": mlp_manifest["document_sampling"][
                    "combined_record_sha256"
                ],
            },
            "full_attention": full_provenance,
            "gdn": gdn_provenance,
            "gated_snapshots_aligned": aligned_gated_provenance,
        },
        "communication_model": {
            "ideal_ring_units": "2*shared_rank + sum(private_ranks)",
            "standard_padded_ring_units": "2*shared_rank + TP*max(private_ranks)",
            "spectrum_aware_allgather_is_idealized_when_uneven": True,
            "active_tokens": args.active_tokens,
            "dtype_bytes": args.dtype_bytes,
        },
        "private_fit": {
            "method": "independent_local_residual_output_pod",
            "global_joint_private_refit": False,
            "probe_multiplier": args.private_probe_multiplier,
            "probe_extra": args.private_probe_extra,
        },
        "decisions": decisions,
        "rows": len(all_records),
        "layer_metadata": layer_metadata,
        "plot_files": plot_files,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(cuda_index)
                if device.type == "cuda"
                else None
            ),
            "torch_num_threads": torch.get_num_threads(),
            "peak_cuda_allocated_bytes": peak_cuda,
        },
    }
    _atomic_json(partial_dir / "manifest.json", manifest)
    os.replace(partial_dir, output_dir)
    print(
        f"[CommonPrivate] complete rows={len(all_records)} output={output_dir} "
        f"seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
