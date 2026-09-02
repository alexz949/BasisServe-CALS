#!/usr/bin/env python3
"""Fit and export the Qwen3 S80-R32 factor bank."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file
import torch
import transformers


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.calibration.gqa_joint_routing_payload_s80_stats import (  # noqa: E402
    S80DirectResidualData,
    S80PayloadStatistics,
    S80RoutingShard,
    S80RoutingStatistics,
    routing_proxy_score_statistics,
)
from basisserve.checkpoint.gqa_joint_routing_payload_s80_qwen3 import (  # noqa: E402
    S80LayerExport,
    write_s80_factor_bank,
)
from basisserve.core.gqa_joint_routing_payload_s80 import (  # noqa: E402
    S80Factors,
    S80Layout,
    S80Objective,
    compose_routing_query_maps,
    evaluate_s80_objective,
    fit_s80_joint,
    fold_s80_factors,
    initialize_s80_from_c1_and_kq,
    s80_objective_from_statistics,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (  # noqa: E402
    S80CompactSoftmaxFisherRouting,
    compact_softmax_fisher_loss,
    prepare_compact_softmax_fisher_routing,
)


STATS_FORMAT = "basisserve.qwen3.gqa_joint_routing_payload_s80_stats.v2"
C1_FORMAT = "basisserve.qwen3_8b.gqa_c1_joint.v1"
KQ_FORMAT = "basisserve.qwen3_8b.post_rope_kqsvd.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--direct-capture-dir")
    parser.add_argument("--page-fisher-stats-dir")
    parser.add_argument("--validation-page-fisher-stats-dir")
    parser.add_argument("--c1-v80-init", required=True)
    parser.add_argument("--kq-r32-init", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--joint-rank", type=int, default=80)
    parser.add_argument("--routing-rank", type=int, default=32)
    parser.add_argument("--routing-weight", type=float, default=1.0)
    parser.add_argument(
        "--routing-metric",
        choices=("raw_qk", "page_fisher"),
        default="raw_qk",
    )
    parser.add_argument("--outer-sweeps", type=int, default=1)
    parser.add_argument(
        "--u-mode",
        choices=("frozen_u", "adapter_u", "full_u_final"),
        default="full_u_final",
    )
    parser.add_argument("--iterative-max-iterations", type=int, default=100)
    parser.add_argument("--iterative-tolerance", type=float, default=1e-5)
    parser.add_argument("--relative-damping", type=float, default=1e-5)
    parser.add_argument("--decoder-jitter", type=float, default=0.0)
    parser.add_argument(
        "--work-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument(
        "--factor-dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_layers(spec: str, total: int) -> tuple[int, ...]:
    if spec == "all":
        return tuple(range(total))
    selected: set[int] = set()
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(item))
    return tuple(sorted(selected))


class _SafetensorWeightReader:
    def __init__(self, model_root: Path) -> None:
        index_path = model_root / "model.safetensors.index.json"
        if index_path.is_file():
            payload = _load_json(index_path)
            self.weight_map = {
                str(key): model_root / str(value)
                for key, value in payload["weight_map"].items()
            }
        else:
            single = model_root / "model.safetensors"
            with safe_open(single, framework="pt", device="cpu") as handle:
                self.weight_map = {key: single for key in handle.keys()}

    def tensor(self, key: str) -> torch.Tensor | None:
        path = self.weight_map.get(key)
        if path is None:
            return None
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


def _artifact_path(root: Path, artifact: dict[str, Any]) -> Path:
    path = root / str(artifact["file"])
    return path


def _load_statistics(
    root: Path,
    manifest: dict[str, Any],
    *,
    split: str,
    layer_index: int,
) -> tuple[S80PayloadStatistics, S80RoutingStatistics]:
    entry = manifest["artifacts"][split][str(layer_index)]
    payload_tensors = load_file(
        str(_artifact_path(root, entry["payload"])),
        device="cpu",
    )
    payload = S80PayloadStatistics(
        covariance_blocks=payload_tensors["covariance_blocks"],
        row_count=int(payload_tensors["row_count"]),
        dense_output_energy=float(payload_tensors["dense_output_energy"]),
        value_dim=int(payload_tensors["value_dim"]),
        key_dim=int(payload_tensors["key_dim"]),
    )
    payload.validate()
    routing_tensors = load_file(
        str(_artifact_path(root, entry["routing"])),
        device="cpu",
    )
    shard_count = int(routing_tensors["query_grams"].shape[0])
    metadata = entry["routing"]["shard_metadata"]
    shards = tuple(
        S80RoutingShard(
            query_grams=routing_tensors["query_grams"][index],
            joint_grams=routing_tensors["joint_grams"][index],
            query_sums=routing_tensors["query_sums"][index],
            joint_sums=routing_tensors["joint_sums"][index],
            query_row_counts=routing_tensors["query_row_counts"][index],
            key_row_counts=routing_tensors["key_row_counts"][index],
            target_score_energy=float(routing_tensors["target_score_energy"][index]),
            metadata=metadata[index],
        )
        for index in range(shard_count)
    )
    routing = S80RoutingStatistics(
        shards=shards,
        head_to_kv_group=routing_tensors["head_to_kv_group"],
        value_dim=int(routing_tensors["value_dim"]),
        key_dim=int(routing_tensors["key_dim"]),
    )
    routing.validate()
    return payload, routing


def _load_c1_layer(
    root: Path,
    manifest: dict[str, Any],
    layer_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    artifact = manifest["artifacts"][str(layer_index)]
    path = root / artifact["file"]
    tensors = load_file(str(path), device="cpu")
    return tensors["value_coordinate_encoders"], tensors["head_output_decoders"]


def _load_direct_layer(
    root: Path,
    manifest: dict[str, Any],
    layer_index: int,
) -> S80DirectResidualData:
    tensors: dict[str, torch.Tensor] = {}
    for name, record in manifest["artifacts"][str(layer_index)].items():
        if name not in {"routing_queries", "routing_joint_rows"}:
            continue
        path = root / record["file"]
        shape = tuple(int(size) for size in record["shape"])
        values = 1
        for size in shape:
            values *= size
        tensors[name] = torch.from_file(
            str(path),
            shared=False,
            size=values,
            dtype=torch.bfloat16,
        ).reshape(shape)
    return S80DirectResidualData(**tensors)


def _load_compact_fisher_layer(
    root: Path,
    manifest: dict[str, Any],
    layer_index: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> S80CompactSoftmaxFisherRouting:
    artifact = manifest["artifacts"][str(layer_index)]
    tensors = load_file(str(root / artifact["file"]), device="cpu")
    return prepare_compact_softmax_fisher_routing(
        queries_by_head=tensors["queries_by_head"],
        fisher_grams_packed_by_head=tensors[
            "fisher_grams_packed_by_head"
        ],
        head_to_kv_group=tensors["head_to_kv_group"],
        value_dim=int(tensors["value_dim"]),
        key_dim=int(tensors["key_dim"]),
        scaling=float(tensors["scaling"]),
        teacher_fisher_energy=float(tensors["teacher_fisher_energy"]),
        device=device,
        dtype=dtype,
    )


def _compact_fisher_diagnostics(
    statistics: S80CompactSoftmaxFisherRouting,
    factors: S80Factors,
) -> dict[str, float | int]:
    value = compact_softmax_fisher_loss(
        statistics,
        routing_payload_encoders=factors.routing_payload_encoders,
        routing_query_factors=factors.routing_query_factors,
    )
    return {
        "documents": statistics.documents,
        "page_fisher_loss": value,
        "page_fisher_nmse": value / statistics.teacher_fisher_energy,
        "teacher_fisher_energy": statistics.teacher_fisher_energy,
    }


def _objective_on_device(
    objective: S80Objective,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> S80Objective:
    from basisserve.core.gqa_routed_ov_joint import RoutedOVQuadratic

    return S80Objective(
        payload=RoutedOVQuadratic(
            covariance=objective.payload.covariance.to(device=device, dtype=dtype),
            cross=objective.payload.cross.to(device=device, dtype=dtype),
            constant=objective.payload.constant.to(device=device, dtype=dtype),
            name=objective.payload.name,
        ),
        payload_target=objective.payload_target.to(device=device, dtype=dtype),
        routing=objective.routing,
        payload_normalizer=objective.payload_normalizer,
        routing_normalizer=objective.routing_normalizer,
        routing_weight=objective.routing_weight,
        normalization_epsilon=objective.normalization_epsilon,
    )


def _iteration_summary(steps: tuple[Any, ...]) -> dict[str, Any]:
    iterations = [int(step.lsqr.iterations) for step in steps]
    if not iterations:
        return {
            "systems": 0,
            "total_iterations": 0,
            "minimum_iterations": 0,
            "maximum_iterations": 0,
            "mean_iterations": 0.0,
            "converged_systems": 0,
        }
    return {
        "systems": len(iterations),
        "total_iterations": sum(iterations),
        "minimum_iterations": min(iterations),
        "maximum_iterations": max(iterations),
        "mean_iterations": sum(iterations) / len(iterations),
        "converged_systems": sum(bool(step.lsqr.converged) for step in steps),
    }


def _score_statistics_json(values: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "score_count": int(values["score_count"]),
        "raw_mean": float(values["raw_mean"]),
        "raw_variance": float(values["raw_variance"]),
        "scaled_logit_mean": float(values["scaled_mean"]),
        "scaled_logit_variance": float(values["scaled_variance"]),
        "score_counts_by_head": values["score_counts_by_head"].tolist(),
        "raw_mean_by_head": values["raw_mean_by_head"].tolist(),
        "raw_variance_by_head": values["raw_variance_by_head"].tolist(),
        "scaled_logit_mean_by_head": values["scaled_mean_by_head"].tolist(),
        "scaled_logit_variance_by_head": values["scaled_variance_by_head"].tolist(),
    }


def _routing_diagnostics(
    *,
    layout: S80Layout,
    factors: S80Factors,
    statistics: S80RoutingStatistics,
) -> dict[str, Any]:
    mapping = layout.head_to_kv_group(device=factors.joint_encoders.device)
    latent_maps = compose_routing_query_maps(
        layout=layout,
        routing_query_factors=factors.routing_query_factors,
    )
    encoders_by_head = factors.routing_payload_encoders.index_select(0, mapping)
    effective_maps = torch.bmm(latent_maps, encoders_by_head.mT)
    query_norms = torch.linalg.matrix_norm(
        factors.routing_query_factors,
        ord="fro",
        dim=(-2, -1),
    )
    effective_frobenius = torch.linalg.matrix_norm(
        effective_maps,
        ord="fro",
        dim=(-2, -1),
    )
    effective_spectral = torch.linalg.matrix_norm(
        effective_maps,
        ord=2,
        dim=(-2, -1),
    )
    score_statistics = routing_proxy_score_statistics(
        statistics,
        effective_maps,
        scaling=layout.key_dim**-0.5,
    )
    return {
        "routing_query_factor_frobenius_norm": float(
            torch.linalg.vector_norm(factors.routing_query_factors)
        ),
        "routing_query_factor_frobenius_norm_by_head": query_norms.tolist(),
        "effective_routing_map_frobenius_norm": float(
            torch.linalg.vector_norm(effective_maps)
        ),
        "effective_routing_map_frobenius_norm_by_head": effective_frobenius.tolist(),
        "effective_routing_map_spectral_norm_by_head": effective_spectral.tolist(),
        "proxy_scores": _score_statistics_json(score_statistics),
    }


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    started = time.monotonic()
    model_root = Path(args.model).expanduser().resolve()
    stats_root = Path(args.stats_dir).expanduser().resolve()
    direct_root = (
        None
        if args.direct_capture_dir is None
        else Path(args.direct_capture_dir).expanduser().resolve()
    )
    fisher_root = (
        None
        if args.page_fisher_stats_dir is None
        else Path(args.page_fisher_stats_dir).expanduser().resolve()
    )
    validation_fisher_root = (
        None
        if args.validation_page_fisher_stats_dir is None
        else Path(args.validation_page_fisher_stats_dir).expanduser().resolve()
    )
    c1_root = Path(args.c1_v80_init).expanduser().resolve()
    kq_root = Path(args.kq_r32_init).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    model_config_path = model_root / "config.json"
    model_config = _load_json(model_config_path)
    hidden = int(model_config["hidden_size"])
    query_heads = int(model_config["num_attention_heads"])
    kv_heads = int(model_config["num_key_value_heads"])
    head_dim = int(model_config.get("head_dim", hidden // query_heads))
    layers_total = int(model_config["num_hidden_layers"])
    layout = S80Layout(
        hidden_size=hidden,
        num_attention_heads=query_heads,
        num_key_value_heads=kv_heads,
        value_dim=head_dim,
        key_dim=head_dim,
        joint_rank=args.joint_rank,
        routing_rank=args.routing_rank,
    )
    selected_layers = _parse_layers(args.layers, layers_total)

    stats_manifest_path = stats_root / "manifest.json"
    stats_manifest = _load_json(stats_manifest_path)
    config_sha = _sha256(model_config_path)
    direct_manifest_path = (
        None if direct_root is None else direct_root / "manifest.json"
    )
    direct_manifest = (
        None if direct_manifest_path is None else _load_json(direct_manifest_path)
    )
    fisher_manifest_path = (
        None if fisher_root is None else fisher_root / "manifest.json"
    )
    fisher_manifest = (
        None if fisher_manifest_path is None else _load_json(fisher_manifest_path)
    )
    validation_fisher_manifest_path = (
        None
        if validation_fisher_root is None
        else validation_fisher_root / "manifest.json"
    )
    validation_fisher_manifest = (
        None
        if validation_fisher_manifest_path is None
        else _load_json(validation_fisher_manifest_path)
    )
    assert args.routing_metric != "raw_qk" or direct_root is not None
    assert args.routing_metric != "page_fisher" or fisher_root is not None
    assert (
        args.routing_metric != "page_fisher"
        or validation_fisher_root is not None
    )
    assert args.routing_metric == "raw_qk" or args.u_mode == "adapter_u"
    c1_manifest_path = c1_root / "results.json"
    c1_manifest = _load_json(c1_manifest_path)
    kq_manifest_path = kq_root / "result.json"
    kq_manifest = _load_json(kq_manifest_path)
    kq_path = kq_root / kq_manifest["artifacts"]["factors"]["file"]
    kq_tensors = load_file(str(kq_path), device="cpu")
    key_bank = kq_tensors["kq_svd_key_projector"]
    query_bank = kq_tensors["kq_svd_query_projector"]
    weights = _SafetensorWeightReader(model_root)
    work_dtype = torch.float64 if args.work_dtype == "float64" else torch.float32
    factor_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.factor_dtype]
    work_device = torch.device(args.work_device)

    exports: list[S80LayerExport] = []
    for ordinal, layer_index in enumerate(selected_layers, start=1):
        print(
            f"[S80 build] layer={layer_index} ({ordinal}/{len(selected_layers)}) loading",
            flush=True,
        )
        dense_v = weights.tensor(f"model.layers.{layer_index}.self_attn.v_proj.weight")
        dense_o = weights.tensor(f"model.layers.{layer_index}.self_attn.o_proj.weight")
        dense_v_bias = weights.tensor(
            f"model.layers.{layer_index}.self_attn.v_proj.bias",
        )
        dense_o_bias = weights.tensor(
            f"model.layers.{layer_index}.self_attn.o_proj.bias",
        )
        c1_a, c1_d = _load_c1_layer(c1_root, c1_manifest, layer_index)
        fit_payload, fit_routing = _load_statistics(
            stats_root,
            stats_manifest,
            split="fit",
            layer_index=layer_index,
        )
        fit_payload = S80PayloadStatistics(
            covariance_blocks=fit_payload.covariance_blocks.to(
                device=work_device, dtype=work_dtype
            ),
            row_count=fit_payload.row_count,
            dense_output_energy=fit_payload.dense_output_energy,
            value_dim=fit_payload.value_dim,
            key_dim=fit_payload.key_dim,
        )
        initial = initialize_s80_from_c1_and_kq(
            layout=layout,
            value_encoders=c1_a.to(device=work_device, dtype=work_dtype),
            value_decoders=c1_d.to(device=work_device, dtype=work_dtype),
            key_encoders=key_bank[layer_index, :, :, : args.routing_rank].to(
                device=work_device, dtype=work_dtype
            ),
            query_encoders=query_bank[layer_index, :, :, : args.routing_rank].to(
                device=work_device, dtype=work_dtype
            ),
            routing_statistics=fit_routing,
            cg_relative_tolerance=args.iterative_tolerance,
            cg_max_iterations=args.iterative_max_iterations,
            cg_relative_damping=args.relative_damping,
        )
        fit_objective = s80_objective_from_statistics(
            layout=layout,
            payload_statistics=fit_payload,
            routing_statistics=fit_routing,
            dense_o_proj_weight=dense_o.to(device=work_device, dtype=work_dtype),
            routing_weight=args.routing_weight,
        )
        direct_residuals = (
            None
            if direct_root is None or direct_manifest is None
            else _load_direct_layer(
                direct_root,
                direct_manifest,
                layer_index,
            )
        )
        fisher_statistics = (
            None
            if fisher_root is None or fisher_manifest is None
            else _load_compact_fisher_layer(
                fisher_root,
                fisher_manifest,
                layer_index,
                device=work_device,
                dtype=work_dtype,
            )
        )
        validation_fisher_statistics = (
            None
            if validation_fisher_root is None
            or validation_fisher_manifest is None
            else _load_compact_fisher_layer(
                validation_fisher_root,
                validation_fisher_manifest,
                layer_index,
                device=work_device,
                dtype=work_dtype,
            )
        )
        validation_payload, validation_routing = _load_statistics(
            stats_root,
            stats_manifest,
            split="validation",
            layer_index=layer_index,
        )
        validation_payload = S80PayloadStatistics(
            covariance_blocks=validation_payload.covariance_blocks.to(
                device=work_device,
                dtype=work_dtype,
            ),
            row_count=validation_payload.row_count,
            dense_output_energy=validation_payload.dense_output_energy,
            value_dim=validation_payload.value_dim,
            key_dim=validation_payload.key_dim,
        )
        validation_objective = s80_objective_from_statistics(
            layout=layout,
            payload_statistics=validation_payload,
            routing_statistics=validation_routing,
            dense_o_proj_weight=dense_o.to(device=work_device, dtype=work_dtype),
            routing_weight=args.routing_weight,
        )
        normalization_epsilon = fit_objective.normalization_epsilon

        def report_fit(item) -> None:
            relative_gain = (
                item.fit_before.total - item.fit_after_encoders.total
            ) / max(abs(item.fit_before.total), normalization_epsilon)
            validation_total = (
                float("nan")
                if item.validation_after_encoders is None
                else item.validation_after_encoders.total
            )
            print(
                f"[S80 build] layer={layer_index} "
                f"sweep={item.sweep}/{args.outer_sweeps} "
                f"fit={item.fit_after_encoders.total:.9g} "
                f"validation={validation_total:.9g} "
                f"relative_gain={relative_gain:.6g} "
                f"seconds={item.wall_time_seconds:.3f}",
                flush=True,
            )

        result = fit_s80_joint(
            layout=layout,
            objective=fit_objective,
            validation_objective=validation_objective,
            direct_residuals=direct_residuals,
            fisher_statistics=fisher_statistics,
            validation_fisher_statistics=validation_fisher_statistics,
            initial_factors=initial,
            routing_metric=args.routing_metric,
            outer_sweeps=args.outer_sweeps,
            u_mode=args.u_mode,
            lsqr_relative_tolerance=args.iterative_tolerance,
            lsqr_max_iterations=args.iterative_max_iterations,
            lsqr_relative_damping=args.relative_damping,
            decoder_relative_jitter=args.decoder_jitter,
            work_dtype=work_dtype,
            work_device=work_device,
            progress_callback=report_fit,
        )
        deployed = S80Factors(
            result.factors.routing_payload_encoders.to(
                device=work_device,
                dtype=work_dtype,
            ),
            result.factors.payload_only_encoders.to(
                device=work_device,
                dtype=work_dtype,
            ),
            result.factors.payload_decoders.to(device=work_device, dtype=work_dtype),
            result.factors.routing_query_factors.to(
                device=work_device,
                dtype=work_dtype,
            ),
        )
        heldout = evaluate_s80_objective(
            layout=layout,
            objective=_objective_on_device(
                validation_objective,
                device=work_device,
                dtype=work_dtype,
            ),
            factors=deployed,
        )
        fit_routing_diagnostics = _routing_diagnostics(
            layout=layout,
            factors=deployed,
            statistics=fit_routing,
        )
        validation_routing_diagnostics = _routing_diagnostics(
            layout=layout,
            factors=deployed,
            statistics=validation_routing,
        )
        fisher_fit_diagnostics = None
        fisher_validation_diagnostics = None
        if args.routing_metric == "page_fisher":
            fisher_fit_diagnostics = _compact_fisher_diagnostics(
                fisher_statistics,
                deployed,
            )
            fisher_validation_diagnostics = _compact_fisher_diagnostics(
                validation_fisher_statistics,
                deployed,
            )
        folded = fold_s80_factors(
            layout=layout,
            factors=result.factors,
            dense_v_proj_weight=dense_v,
            dense_v_proj_bias=dense_v_bias,
            dense_o_proj_bias=dense_o_bias,
            output_dtype=factor_dtype,
        )
        diagnostics = {
            "initial_fit_loss": asdict(result.initial_loss),
            "final_fit_loss": asdict(result.final_loss),
            "validation_metric_loss": asdict(
                result.diagnostics.validation_after_routing_queries
            ),
            "validation_raw_qk_loss": asdict(heldout),
            "routing_metric": result.routing_metric,
            "routing_normalizer": result.routing_normalizer,
            "fit": asdict(result.diagnostics),
            "final_routing_query_steps": [
                asdict(item) for item in result.final_routing_query_steps
            ],
            "iteration_summary": {
                "BF_by_sweep": [
                    {
                        "sweep": item.sweep,
                        **_iteration_summary(item.encoder_steps),
                    }
                    for item in result.diagnostics.sweeps
                ],
                "U_final": _iteration_summary(result.final_routing_query_steps),
            },
            "fit_routing_diagnostics": fit_routing_diagnostics,
            "validation_routing_diagnostics": validation_routing_diagnostics,
            "fit_page_fisher_diagnostics": fisher_fit_diagnostics,
            "validation_page_fisher_diagnostics": (
                fisher_validation_diagnostics
            ),
        }
        exports.append(
            S80LayerExport(
                layer_index=layer_index,
                factors=folded,
                routing_weight=args.routing_weight,
                payload_normalizer=fit_objective.payload_normalizer,
                routing_normalizer=result.routing_normalizer,
                fit_diagnostics=diagnostics,
            )
        )
        print(
            f"[S80 build] layer={layer_index} fit_total={result.final_loss.total:.9g} "
            "validation_total="
            f"{result.diagnostics.validation_after_routing_queries.total:.9g} "
            f"u_mode={args.u_mode} routing_metric={args.routing_metric}",
            flush=True,
        )
        del fit_payload, validation_payload, fit_objective, validation_objective
        del direct_residuals
        del fisher_statistics, validation_fisher_statistics
        del initial, result, deployed
        if work_device.type == "cuda":
            torch.cuda.empty_cache()

    solver_configuration = {
        "parameterization": "B32_route_payload,F48_payload_only,D,U",
        "update_order": "repeat(D_exact,BF_covariance_root,gauge),D_exact,U_final",
        "outer_sweeps": args.outer_sweeps,
        "u_mode": args.u_mode,
        "routing_metric": args.routing_metric,
        "routing_normalization": (
            "teacher_page_fisher_energy"
            if args.routing_metric == "page_fisher"
            else "teacher_raw_score_energy"
        ),
        "iterative_max_iterations": args.iterative_max_iterations,
        "iterative_tolerance": args.iterative_tolerance,
        "relative_damping": args.relative_damping,
        "initial_routing_solver": "jacobi_pcg",
        "bf_solver": "covariance_root_two_sided_preconditioned_lsqr",
        "encoder_lsqr_preconditioner": "partial_trace_kronecker",
        "routing_solver": "batched_gram_pcg",
        "decoder_jitter": args.decoder_jitter,
        "work_dtype": args.work_dtype,
        "factor_dtype": args.factor_dtype,
        "work_device": str(work_device),
        "elapsed_seconds_before_write": time.monotonic() - started,
    }
    manifest_path = write_s80_factor_bank(
        output_root,
        layout=layout,
        layers=exports,
        model_identifier=str(model_root),
        model_config_sha256=config_sha,
        initial_factor_sources={
            "c1_v80": {
                "manifest": str(c1_manifest_path),
                "sha256": _sha256(c1_manifest_path),
                "format": C1_FORMAT,
            },
            "kq_r32": {
                "manifest": str(kq_manifest_path),
                "sha256": _sha256(kq_manifest_path),
                "factor_file": str(kq_path),
                "factor_sha256": _sha256(kq_path),
                "format": KQ_FORMAT,
                "source_rank": int(kq_manifest["geometry"]["rank"]),
                "selected_rank": args.routing_rank,
            },
            "statistics": {
                "manifest": str(stats_manifest_path),
                "sha256": _sha256(stats_manifest_path),
                "format": STATS_FORMAT,
            },
            "direct_residuals": (
                None
                if direct_manifest_path is None or direct_manifest is None
                else {
                    "manifest": str(direct_manifest_path),
                    "sha256": _sha256(direct_manifest_path),
                    "format": str(direct_manifest["format"]),
                }
            ),
            "page_fisher_statistics": (
                None
                if fisher_manifest_path is None or fisher_manifest is None
                else {
                    "manifest": str(fisher_manifest_path),
                    "sha256": _sha256(fisher_manifest_path),
                    "format": str(fisher_manifest["format"]),
                }
            ),
            "validation_page_fisher_statistics": (
                None
                if validation_fisher_manifest_path is None
                or validation_fisher_manifest is None
                else {
                    "manifest": str(validation_fisher_manifest_path),
                    "sha256": _sha256(validation_fisher_manifest_path),
                    "format": str(validation_fisher_manifest["format"]),
                }
            ),
        },
        solver_configuration=solver_configuration,
        command=shlex.join(sys.argv),
        environment={
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_device": (
                torch.cuda.get_device_name(work_device)
                if work_device.type == "cuda"
                else None
            ),
        },
    )
    print(f"[S80 build] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
