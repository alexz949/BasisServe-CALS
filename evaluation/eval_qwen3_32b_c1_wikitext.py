#!/usr/bin/env python3
"""Evaluate folded Qwen3-32B GQA C1 factors on WikiText-2.

The reference path pads every low-rank physical Value writer back to head width
128 and pads each query-head decoder in ``o_proj``.  It therefore preserves the
compressed C1 function exactly without claiming compressed-cache runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.gqa_vo_svdllm import GQAVOLayout
from evaluation import allocate_qwen3_32b_c1_tp_source_global_kl as allocation_common
from evaluation.allocate_qwen3_32b_c1_tp_source_global_kl import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_QUERY_HEADS,
    _fold_ragged_to_padded_weights,
)
from evaluation.eval_attention_o_proj_collective_ppl import _eval_ppl_fp32_loss


FORMAT = "basisserve.qwen3_32b.gqa_c1_wikitext2_ppl.v1"
FACTOR_FORMAT = "basisserve.qwen3_32b.gqa_c1_v96_joint.v1"
RAGGED_FACTOR_FORMAT = "basisserve.qwen3_32b.gqa_c1.ragged_schedule_als.v1"
LAYER_ALLOCATION_FORMAT = (
    "basisserve.qwen3_32b.gqa_c1.layer_global_kl_allocation.v1"
)
EXTERNAL_SCHEDULE_FORMAT = "basisserve.c1.factorized_terminal_kl_schedule.v1"
MODEL_LABEL = "Qwen3-32B"


def activate_model_profile(name: str) -> None:
    global FORMAT, FACTOR_FORMAT, RAGGED_FACTOR_FORMAT, LAYER_ALLOCATION_FORMAT
    global MODEL_LABEL, HEAD_DIM, HIDDEN_SIZE, NUM_KV_HEADS, NUM_LAYERS
    global NUM_QUERY_HEADS
    allocation_common.activate_model_profile(name)
    if name == "qwen3_32b":
        slug = "qwen3_32b"
        MODEL_LABEL = "Qwen3-32B"
        FACTOR_FORMAT = "basisserve.qwen3_32b.gqa_c1_v96_joint.v1"
    elif name == "qwen3_8b":
        slug = "qwen3_8b"
        MODEL_LABEL = "Qwen3-8B-Base"
        FACTOR_FORMAT = "basisserve.qwen3_8b.gqa_c1_joint.v1"
    else:
        raise ValueError(f"unknown Qwen3 C1 model profile: {name}")
    HEAD_DIM = allocation_common.HEAD_DIM
    HIDDEN_SIZE = allocation_common.HIDDEN_SIZE
    NUM_KV_HEADS = allocation_common.NUM_KV_HEADS
    NUM_LAYERS = allocation_common.NUM_LAYERS
    NUM_QUERY_HEADS = allocation_common.NUM_QUERY_HEADS
    FORMAT = f"basisserve.{slug}.gqa_c1_wikitext2_ppl.v1"
    RAGGED_FACTOR_FORMAT = f"basisserve.{slug}.gqa_c1.ragged_schedule_als.v1"
    LAYER_ALLOCATION_FORMAT = (
        f"basisserve.{slug}.gqa_c1.layer_global_kl_allocation.v1"
    )


def _cuda_device_indices() -> tuple[int, ...]:
    indices = []
    for index in range(torch.cuda.device_count()):
        try:
            torch.cuda.get_device_properties(index)
        except RuntimeError:
            continue
        indices.append(index)
    return tuple(indices)


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


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("expected a Qwen-style decoder model")
    return layers


def _layout(config: Any, rank: int) -> GQAVOLayout:
    layout = GQAVOLayout(
        hidden_size=int(config.hidden_size),
        num_attention_heads=int(config.num_attention_heads),
        num_key_value_heads=int(config.num_key_value_heads),
        head_dim=int(getattr(config, "head_dim", 0)),
        rank=rank,
    )
    if (
        str(config.model_type) != "qwen3"
        or int(config.num_hidden_layers) != NUM_LAYERS
        or layout.hidden_size != HIDDEN_SIZE
        or layout.num_attention_heads != NUM_QUERY_HEADS
        or layout.num_key_value_heads != 8
        or layout.head_dim != 128
    ):
        raise ValueError(f"unexpected {MODEL_LABEL} geometry: {layout}")
    return layout


def _load_results(factor_dir: Path, model_path: Path) -> dict[str, Any]:
    path = factor_dir / "results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("format") != FACTOR_FORMAT or result.get("status") != "complete":
        raise ValueError("C1 factor result is incomplete or incompatible")
    config = result["fit_config"]
    if config.get("model_config_sha256") != _sha256(model_path / "config.json"):
        raise ValueError("C1 factors belong to another model config")
    if tuple(map(int, result.get("layers", ()))) != tuple(range(NUM_LAYERS)):
        raise ValueError(f"C1 result does not cover all {NUM_LAYERS} layers")
    return result


def _load_ragged_results(factor_dir: Path, model_path: Path) -> dict[str, Any]:
    path = factor_dir / "result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("format") != RAGGED_FACTOR_FORMAT
        or result.get("status") != "complete"
    ):
        raise ValueError("ragged C1 factor result is incomplete or incompatible")
    config = result.get("fit_config", {})
    if config.get("model_config_sha256") != _sha256(model_path / "config.json"):
        raise ValueError("ragged C1 factors belong to another model config")
    if tuple(map(int, result.get("layers", ()))) != tuple(range(NUM_LAYERS)):
        raise ValueError(
            f"ragged C1 result does not cover all {NUM_LAYERS} layers"
        )
    if set(map(int, result.get("artifacts", {}))) != set(range(NUM_LAYERS)):
        raise ValueError("ragged C1 result does not contain all layer artifacts")
    schedule = result.get("selection", {}).get("selected_schedule", ())
    if len(schedule) != NUM_LAYERS or any(
        len(layer) != NUM_KV_HEADS for layer in schedule
    ):
        raise ValueError("ragged C1 result has an incompatible rank schedule")
    flat = [int(rank) for layer in schedule for rank in layer]
    if any(not 0 < rank <= HEAD_DIM for rank in flat):
        raise ValueError("ragged C1 schedule contains an invalid source rank")
    if sum(flat) != int(result["selection"]["source_rank_sum"]):
        raise ValueError("ragged C1 schedule violates its recorded rank budget")
    if config.get("rank_schedule") != schedule:
        raise ValueError("ragged C1 fit config and selected schedule disagree")
    return result


def _load_layer_allocation_results(
    allocation_dir: Path,
    model_path: Path,
) -> dict[str, Any]:
    path = allocation_dir / "result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("format") != LAYER_ALLOCATION_FORMAT
        or result.get("status") != "complete"
    ):
        raise ValueError("per-layer Global-KL allocation is incomplete or incompatible")
    if result.get("model_config_sha256") != _sha256(model_path / "config.json"):
        raise ValueError("per-layer Global-KL allocation belongs to another model config")
    artifacts = result.get("selected_artifacts", {})
    if set(map(int, artifacts)) != set(range(NUM_LAYERS)):
        raise ValueError("per-layer allocation does not contain all selected artifacts")
    selection = result.get("selection", {})
    schedule = selection.get("selected_schedule", ())
    if len(schedule) != NUM_LAYERS or any(
        len(layer) != NUM_KV_HEADS for layer in schedule
    ):
        raise ValueError("per-layer allocation has an incompatible rank schedule")
    if any(len(set(map(int, layer))) != 1 for layer in schedule):
        raise ValueError("per-layer allocation contains a nonuniform layer rank")
    flat = [int(rank) for layer in schedule for rank in layer]
    if any(not 0 < rank <= HEAD_DIM for rank in flat):
        raise ValueError("per-layer allocation contains an invalid source rank")
    target = int(selection.get("target_source_rank_sum", -1))
    accounting = selection.get("selected_accounting", {})
    if sum(flat) != target or sum(flat) != int(
        accounting.get("source_rank_sum", -1)
    ):
        raise ValueError("per-layer allocation violates its recorded V-cache budget")
    schedules = result.get("schedules", {})
    if set(schedules) != {"uniform_anchor", "mean_dp", "ucb_dp"}:
        raise ValueError("per-layer allocation does not contain the three schedules")
    candidate_ranks = set(map(int, selection.get("candidate_ranks", ())))
    for name, row in schedules.items():
        candidate_schedule = row.get("schedule", ())
        if len(candidate_schedule) != NUM_LAYERS or any(
            len(layer) != NUM_KV_HEADS for layer in candidate_schedule
        ):
            raise ValueError(f"per-layer allocation schedule {name} is incompatible")
        candidate_flat = [
            int(rank) for layer in candidate_schedule for rank in layer
        ]
        if any(len(set(map(int, layer))) != 1 for layer in candidate_schedule):
            raise ValueError(
                f"per-layer allocation schedule {name} is not layer-uniform"
            )
        if any(rank not in candidate_ranks for rank in candidate_flat):
            raise ValueError(f"per-layer allocation schedule {name} uses an unknown rank")
        recorded_sum = int(row.get("accounting", {}).get("source_rank_sum", -1))
        if sum(candidate_flat) != target or recorded_sum != target:
            raise ValueError(f"per-layer allocation schedule {name} violates the budget")
    selected_name = str(selection.get("selected_candidate"))
    if (
        selected_name not in schedules
        or schedules[selected_name]["schedule"] != schedule
    ):
        raise ValueError("selected per-layer schedule and candidate name disagree")
    return result


def _load_external_layer_schedule(
    path: Path,
    *,
    allocation_result_path: Path,
    result: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Load and validate a schedule derived from an authenticated allocation."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("format") != EXTERNAL_SCHEDULE_FORMAT
        or payload.get("status") != "complete"
    ):
        raise ValueError("external layer schedule is incomplete or incompatible")
    source = payload.get("source", {})
    if source.get("allocation_result_sha256") != _sha256(allocation_result_path):
        raise ValueError("external layer schedule belongs to another allocation")
    name = str(payload.get("schedule_name", ""))
    if not name or name in result["schedules"]:
        raise ValueError("external layer schedule name is empty or collides")
    schedule = payload.get("schedule", ())
    if len(schedule) != NUM_LAYERS or any(
        len(layer) != NUM_KV_HEADS for layer in schedule
    ):
        raise ValueError("external layer schedule has incompatible geometry")
    if any(len(set(map(int, layer))) != 1 for layer in schedule):
        raise ValueError("external layer schedule is not layer-uniform")
    candidate_ranks = set(map(int, result["selection"]["candidate_ranks"]))
    flat = [int(rank) for layer in schedule for rank in layer]
    if any(rank not in candidate_ranks for rank in flat):
        raise ValueError("external layer schedule uses an unknown rank")
    target = int(result["selection"]["target_source_rank_sum"])
    if sum(flat) != target:
        raise ValueError("external layer schedule violates the allocation budget")
    accounting = dict(payload.get("accounting", {}))
    if int(accounting.get("source_rank_sum", -1)) != target:
        raise ValueError("external layer schedule accounting violates the budget")
    expected_histogram = {
        str(rank): sum(value == rank for value in flat)
        for rank in sorted(set(flat))
    }
    if accounting.get("source_rank_histogram") != expected_histogram:
        raise ValueError("external layer schedule rank histogram is inconsistent")
    row = {
        "schedule": [[int(rank) for rank in layer] for layer in schedule],
        "accounting": accounting,
    }
    return name, row, payload


@dataclass(frozen=True)
class LayerAllocationReconstructionState:
    snapshot_cache: Sequence[Any]
    factor_cache: Sequence[Mapping[int, Any]]
    anchor_rank: int
    covariance_damping: float
    decoder_relative_jitter: float


def _layer_allocation_profile_configuration(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    configurations = []
    for record in result.get("profile", {}).get("shards", ()):
        path = Path(record["path"]).expanduser().resolve()
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"per-layer profile shard hash mismatch: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"per-layer profile shard is incomplete: {path}")
        configurations.append(payload["configuration"])
    if not configurations or any(
        configuration != configurations[0] for configuration in configurations[1:]
    ):
        raise ValueError("per-layer profile shard configurations are missing or unequal")
    return dict(configurations[0])


def load_layer_allocation_reconstruction_state(
    result: Mapping[str, Any],
    *,
    model_path: Path,
) -> LayerAllocationReconstructionState:
    """Load and authenticate the exact ALS banks used by an allocation run."""

    configuration = _layer_allocation_profile_configuration(result)
    if Path(configuration["model"]).expanduser().resolve() != model_path:
        raise ValueError("per-layer profile configuration belongs to another model")
    if configuration["model_config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("per-layer profile model hash mismatch")
    snapshot_dir = Path(result["snapshot"]["path"]).expanduser().resolve()
    if Path(configuration["snapshot_dir"]).expanduser().resolve() != snapshot_dir:
        raise ValueError("per-layer profile and allocation covariance paths disagree")
    if (
        _sha256(snapshot_dir / "manifest.json")
        != result["snapshot"]["manifest_sha256"]
    ):
        raise ValueError("per-layer allocation covariance manifest hash mismatch")
    factor_dirs = {
        int(rank): Path(record["path"]).expanduser().resolve()
        for rank, record in result["factor_sources"].items()
    }
    candidate_ranks = set(map(int, result["selection"]["candidate_ranks"]))
    if set(factor_dirs) != candidate_ranks - {HEAD_DIM}:
        raise ValueError("per-layer allocation factor sources are incomplete")
    factor_results = allocation_common._load_factor_results(
        factor_dirs,
        model_config_sha256=_sha256(model_path / "config.json"),
        snapshot_dir=snapshot_dir,
    )
    for rank, directory in factor_dirs.items():
        digest = _sha256(directory / "results.json")
        if digest != result["factor_sources"][str(rank)]["results_sha256"]:
            raise ValueError(f"per-layer allocation factor result hash mismatch: {rank}")
        if digest != configuration["factor_results_sha256"][str(rank)]:
            raise ValueError(f"per-layer profile factor result hash mismatch: {rank}")
    snapshot_cache, _ = allocation_common._load_snapshot_cache(
        snapshot_dir,
        model_path=model_path,
    )
    factor_cache = allocation_common._load_factor_cache(factor_dirs, factor_results)
    return LayerAllocationReconstructionState(
        snapshot_cache=snapshot_cache,
        factor_cache=factor_cache,
        anchor_rank=int(configuration["anchor_rank"]),
        covariance_damping=float(configuration["covariance_damping"]),
        decoder_relative_jitter=float(configuration["decoder_relative_jitter"]),
    )


@torch.no_grad()
def fold_c1_to_padded_weights(
    *,
    dense_v_weight: Tensor,
    encoders: Tensor,
    decoders: Tensor,
    layout: GQAVOLayout,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Fold routed C1 factors and pad latent coordinates into dense HF slots."""

    if tuple(dense_v_weight.shape) != (layout.kv_width, layout.hidden_size):
        raise ValueError("dense v_proj weight has incompatible geometry")
    if tuple(encoders.shape) != (
        layout.num_key_value_heads,
        layout.head_dim,
        layout.rank,
    ):
        raise ValueError("C1 encoder tensor has incompatible geometry")
    if tuple(decoders.shape) != (
        layout.num_attention_heads,
        layout.rank,
        layout.hidden_size,
    ):
        raise ValueError("C1 decoder tensor has incompatible geometry")
    device = dense_v_weight.device
    work_dtype = torch.float32
    work_dense_v = dense_v_weight.to(dtype=work_dtype)
    work_A = encoders.to(device=device, dtype=work_dtype)
    work_D = decoders.to(device=device, dtype=work_dtype)
    padded_v = torch.zeros(
        layout.kv_width,
        layout.hidden_size,
        device=device,
        dtype=work_dtype,
    )
    padded_o = torch.zeros(
        layout.hidden_size,
        layout.query_width,
        device=device,
        dtype=work_dtype,
    )
    for group in range(layout.num_key_value_heads):
        target = slice(
            group * layout.head_dim,
            group * layout.head_dim + layout.rank,
        )
        dense_rows = slice(
            group * layout.head_dim,
            (group + 1) * layout.head_dim,
        )
        padded_v[target].copy_(work_A[group].T @ work_dense_v[dense_rows])
    for head in range(layout.num_attention_heads):
        target = slice(
            head * layout.head_dim,
            head * layout.head_dim + layout.rank,
        )
        padded_o[:, target].copy_(work_D[head].T)
    return padded_v, padded_o, {
        "maximum_dense_fold_error": 0.0,
        "maximum_qr_product_error": 0.0,
    }


@torch.no_grad()
def install_c1_factors(
    model: nn.Module,
    factor_dir: Path,
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rank = int(result["fit_config"]["cache_rank_per_head"])
    layout = _layout(model.config, rank)
    records: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(_decoder_layers(model)):
        started = time.perf_counter()
        artifact = result["artifacts"][str(layer_index)]
        path = factor_dir / artifact["file"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"C1 artifact hash mismatch at layer {layer_index}")
        payload = load_file(str(path), device="cpu")
        if set(payload) != {"value_coordinate_encoders", "head_output_decoders"}:
            raise ValueError(f"unexpected C1 tensors at layer {layer_index}")
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        dense_k = layer.self_attn.k_proj
        if (
            not isinstance(v_proj, nn.Linear)
            or not isinstance(o_proj, nn.Linear)
            or v_proj.bias is not None
            or o_proj.bias is not None
            or v_proj.weight.device.type != "cuda"
            or o_proj.weight.device.type != "cuda"
        ):
            raise TypeError(f"unsupported/offloaded Qwen projections at layer {layer_index}")
        padded_v, padded_o, diagnostics = fold_c1_to_padded_weights(
            dense_v_weight=v_proj.weight.detach(),
            encoders=payload["value_coordinate_encoders"],
            decoders=payload["head_output_decoders"],
            layout=layout,
        )
        v_proj.weight.copy_(padded_v.to(device=v_proj.weight.device, dtype=v_proj.weight.dtype))
        o_proj.weight.copy_(padded_o.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        if layer.self_attn.k_proj is not dense_k:
            raise RuntimeError(f"C1 installation changed K at layer {layer_index}")
        records.append(
            {
                "layer": layer_index,
                "factor_file": path.name,
                "factor_sha256": artifact["sha256"],
                "encoder_shape": list(payload["value_coordinate_encoders"].shape),
                "decoder_shape": list(payload["head_output_decoders"].shape),
                "cache_rank_per_physical_kv_head": rank,
                "runtime": (
                    f"rank-{rank} C1 coordinates padded into dense HF V/O slots"
                ),
                **diagnostics,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    return records


@torch.no_grad()
def _install_scheduled_c1_factors(
    model: nn.Module,
    factor_dir: Path,
    *,
    schedule: Sequence[Sequence[int]],
    artifacts: Mapping[str, Mapping[str, Any]],
    label: str,
    dense_v_weights: Sequence[Tensor] | None = None,
) -> list[dict[str, Any]]:
    _layout(model.config, max(int(rank) for layer in schedule for rank in layer))
    records: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(_decoder_layers(model)):
        started = time.perf_counter()
        artifact = artifacts[str(layer_index)]
        path = factor_dir / artifact["file"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"{label} artifact hash mismatch at layer {layer_index}")
        payload = load_file(str(path), device="cpu")
        if set(payload) != {
            "value_coordinate_encoders",
            "head_output_decoders",
            "source_ranks",
        }:
            raise ValueError(f"unexpected {label} tensors at layer {layer_index}")
        ranks = tuple(map(int, payload["source_ranks"].tolist()))
        if list(ranks) != list(map(int, schedule[layer_index])):
            raise ValueError(f"{label} rank mismatch at layer {layer_index}")
        maximum_rank = max(ranks)
        A = payload["value_coordinate_encoders"]
        D = payload["head_output_decoders"]
        if tuple(A.shape) != (NUM_KV_HEADS, HEAD_DIM, maximum_rank):
            raise ValueError(f"unexpected ragged encoder shape at layer {layer_index}")
        if tuple(D.shape) != (NUM_QUERY_HEADS, maximum_rank, HIDDEN_SIZE):
            raise ValueError(f"unexpected ragged decoder shape at layer {layer_index}")
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        dense_k = layer.self_attn.k_proj
        if (
            not isinstance(v_proj, nn.Linear)
            or not isinstance(o_proj, nn.Linear)
            or v_proj.bias is not None
            or o_proj.bias is not None
            or v_proj.weight.device.type != "cuda"
            or o_proj.weight.device.type != "cuda"
        ):
            raise TypeError(
                f"unsupported/offloaded Qwen projections at layer {layer_index}"
            )
        padded_v, padded_o = _fold_ragged_to_padded_weights(
            dense_v_weight=(
                v_proj.weight.detach()
                if dense_v_weights is None
                else dense_v_weights[layer_index].to(v_proj.weight.device)
            ),
            A=A,
            D=D,
            source_ranks=ranks,
        )
        v_proj.weight.copy_(padded_v.to(dtype=v_proj.weight.dtype))
        o_proj.weight.copy_(
            padded_o.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype)
        )
        if layer.self_attn.k_proj is not dense_k:
            raise RuntimeError(f"ragged C1 installation changed K at layer {layer_index}")
        records.append(
            {
                "layer": layer_index,
                "factor_file": str(path.relative_to(factor_dir)),
                "factor_sha256": artifact["sha256"],
                "source_ranks": list(ranks),
                "maximum_padded_rank": maximum_rank,
                "runtime": (
                    f"{label} coordinates zero-padded into dense HF V/O slots"
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        del payload, A, D, padded_v, padded_o
        torch.cuda.empty_cache()
    return records


@torch.no_grad()
def install_ragged_c1_factors(
    model: nn.Module,
    factor_dir: Path,
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return _install_scheduled_c1_factors(
        model,
        factor_dir,
        schedule=result["selection"]["selected_schedule"],
        artifacts=result["artifacts"],
        label="ragged C1",
    )


@torch.no_grad()
def install_layer_allocation_factors(
    model: nn.Module,
    allocation_dir: Path,
    result: Mapping[str, Any],
    *,
    dense_v_weights: Sequence[Tensor] | None = None,
) -> list[dict[str, Any]]:
    return _install_scheduled_c1_factors(
        model,
        allocation_dir,
        schedule=result["selection"]["selected_schedule"],
        artifacts=result["selected_artifacts"],
        label="per-layer Global-KL C1",
        dense_v_weights=dense_v_weights,
    )


@torch.no_grad()
def install_layer_allocation_schedule(
    model: nn.Module,
    allocation_dir: Path,
    result: Mapping[str, Any],
    *,
    schedule_name: str,
    dense_v_weights: Sequence[Tensor],
    reconstruction_state: LayerAllocationReconstructionState | None = None,
) -> list[dict[str, Any]]:
    if schedule_name not in result["schedules"]:
        raise ValueError(f"unknown per-layer allocation schedule: {schedule_name}")
    if schedule_name == result["selection"]["selected_candidate"]:
        return install_layer_allocation_factors(
            model,
            allocation_dir,
            result,
            dense_v_weights=dense_v_weights,
        )
    state = reconstruction_state or load_layer_allocation_reconstruction_state(
        result,
        model_path=Path(result["model"]).expanduser().resolve(),
    )
    _, diagnostics = allocation_common._install_schedule(
        model,
        schedule=result["schedules"][schedule_name]["schedule"],
        dense_v_weights=dense_v_weights,
        factor_cache=state.factor_cache,
        snapshot_cache=state.snapshot_cache,
        anchor_rank=state.anchor_rank,
        covariance_damping=state.covariance_damping,
        decoder_relative_jitter=state.decoder_relative_jitter,
        keep_factors=False,
    )
    return diagnostics


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C1 PPL evaluation requires CUDA")
    torch.cuda.set_device(device)
    cuda_indices = _cuda_device_indices()
    for index in cuda_indices:
        torch.cuda.reset_peak_memory_stats(index)
    model_path = Path(args.model).expanduser().resolve()
    allocation = args.allocation_dir is not None
    ragged = args.ragged_factor_dir is not None
    if args.allocation_schedule is not None and not allocation:
        raise ValueError("--allocation-schedule requires --allocation-dir")
    if args.allocation_schedule_file is not None and not allocation:
        raise ValueError("--allocation-schedule-file requires --allocation-dir")
    if (
        args.allocation_schedule is not None
        and args.allocation_schedule_file is not None
    ):
        raise ValueError(
            "--allocation-schedule and --allocation-schedule-file are mutually exclusive"
        )
    factor_dir = Path(
        args.allocation_dir
        if allocation
        else args.ragged_factor_dir
        if ragged
        else args.factor_dir
    ).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if allocation:
        result = _load_layer_allocation_results(factor_dir, model_path)
        allocation_result_path = factor_dir / "result.json"
        external_schedule_payload = None
        if args.allocation_schedule_file is not None:
            external_schedule_path = (
                Path(args.allocation_schedule_file).expanduser().resolve()
            )
            (
                external_schedule_name,
                external_schedule_row,
                external_schedule_payload,
            ) = _load_external_layer_schedule(
                external_schedule_path,
                allocation_result_path=allocation_result_path,
                result=result,
            )
            result["schedules"][external_schedule_name] = external_schedule_row
    elif ragged:
        result = _load_ragged_results(factor_dir, model_path)
    else:
        result = _load_results(factor_dir, model_path)
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in cuda_indices
        },
    ).eval()
    model.config.use_cache = False
    install_started = time.perf_counter()
    if allocation:
        schedule_name = (
            external_schedule_name
            if args.allocation_schedule_file is not None
            else args.allocation_schedule
            if args.allocation_schedule is not None
            else result["selection"]["selected_candidate"]
        )
        dense_v_weights = tuple(
            layer.self_attn.v_proj.weight.detach().cpu().clone()
            for layer in _decoder_layers(model)
        )
        installation = install_layer_allocation_schedule(
            model,
            factor_dir,
            result,
            schedule_name=schedule_name,
            dense_v_weights=dense_v_weights,
        )
        selection = result["selection"]
        accounting = result["schedules"][schedule_name]["accounting"]
        schedule = result["schedules"][schedule_name]["schedule"]
        flat_ranks = [int(rank) for layer in schedule for rank in layer]
        source_rank_sum = sum(flat_ranks)
        layout = _layout(model.config, max(flat_ranks))
        stage = result["profile"]["factor_stage"]
        sweeps = int(stage["encoder_sweeps"])
        arm = (
            f"c1_layer_global_kl_{schedule_name}_"
            f"postals_s{sweeps}"
        )
        result_path = factor_dir / "result.json"
        factor_result_record = {
            "path": str(result_path),
            "sha256": _sha256(result_path),
            "format": result["format"],
            "allocation_schedule": schedule_name,
            "original_selected_candidate": selection["selected_candidate"],
            "factor_stage": stage,
        }
        if args.allocation_schedule_file is not None:
            factor_result_record["external_schedule"] = {
                "path": str(external_schedule_path),
                "sha256": _sha256(external_schedule_path),
                "format": external_schedule_payload["format"],
                "method": external_schedule_payload["method"],
            }
        compression = {
            "target": "value_cache_only",
            "key_cache": "dense",
            "num_query_heads": layout.num_attention_heads,
            "num_physical_kv_heads": layout.num_key_value_heads,
            "head_dim": layout.head_dim,
            "source_rank_sum": source_rank_sum,
            "dense_source_rank_sum": NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM,
            "retained_v_ratio": (
                source_rank_sum / (NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM)
            ),
            "v_cache_compression_ratio": 1.0
            - source_rank_sum / (NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM),
            "layer_rank_histogram": accounting["layer_rank_histogram"],
        }
        runtime_description = (
            "per-layer-uniform C1 coordinates zero-padded into Hugging Face V/O "
            "slots; function-equivalent quality path, not a cache-performance benchmark"
        )
    elif ragged:
        installation = install_ragged_c1_factors(model, factor_dir, result)
        schedule = result["selection"]["selected_schedule"]
        flat_ranks = [int(rank) for layer in schedule for rank in layer]
        source_rank_sum = sum(flat_ranks)
        layout = _layout(model.config, max(flat_ranks))
        sweeps = int(result["fit_config"]["encoder_sweeps"])
        arm = f"c1_aasvd_global_kl_ragged_als_s{sweeps}"
        result_path = factor_dir / "result.json"
        factor_result_record = {
            "path": str(result_path),
            "sha256": _sha256(result_path),
            "format": result["format"],
            "aggregate": result["aggregate"],
        }
        compression = {
            "target": "value_cache_only",
            "key_cache": "dense",
            "num_query_heads": layout.num_attention_heads,
            "num_physical_kv_heads": layout.num_key_value_heads,
            "head_dim": layout.head_dim,
            "source_rank_sum": source_rank_sum,
            "dense_source_rank_sum": NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM,
            "retained_v_ratio": (
                source_rank_sum / (NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM)
            ),
            "v_cache_compression_ratio": 1.0
            - source_rank_sum / (NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM),
            "rank_histogram": result["fit_config"]["rank_histogram"],
        }
        runtime_description = (
            "ragged C1 coordinates zero-padded into Hugging Face V/O slots; "
            "function-equivalent quality path, not a cache-performance benchmark"
        )
    else:
        rank = int(result["fit_config"]["cache_rank_per_head"])
        layout = _layout(model.config, rank)
        installation = install_c1_factors(model, factor_dir, result)
        decoder_solver = result["fit_config"].get("decoder_solver")
        arm = (
            f"c1_joint_{decoder_solver}"
            if decoder_solver is not None
            else "c1_joint_als"
        )
        result_path = factor_dir / "results.json"
        factor_result_record = {
            "path": str(result_path),
            "sha256": _sha256(result_path),
            "format": result["format"],
            "aggregate": result["aggregate"],
        }
        compression = {
            "target": "value_cache_only",
            "key_cache": "dense",
            "num_query_heads": layout.num_attention_heads,
            "num_physical_kv_heads": layout.num_key_value_heads,
            "head_dim": layout.head_dim,
            "rank_per_physical_kv_head": rank,
            "rank_sum": layout.num_key_value_heads * rank,
            "retained_v_ratio": rank / layout.head_dim,
            "v_cache_compression_ratio": 1.0 - rank / layout.head_dim,
        }
        runtime_description = (
            "C1 latent coordinates padded into Hugging Face V/O slots; "
            "function-equivalent quality path, not a cache-performance benchmark"
        )
    install_seconds = time.perf_counter() - install_started
    ppl = _eval_ppl_fp32_loss(
        model,
        tokenizer,
        dataset=args.dataset,
        split=args.split,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
    )
    payload = {
        "format": FORMAT,
        "status": "complete",
        "arm": arm,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "factor_result": factor_result_record,
        "compression": compression,
        "quality_reference_runtime": {
            "description": runtime_description,
            "installation_seconds": install_seconds,
            "layers": installation,
        },
        "ppl": ppl,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in cuda_indices
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in cuda_indices
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_path, payload)
    print(f"[Result] arm={arm} ppl={ppl['ppl']:.9f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    factors = parser.add_mutually_exclusive_group(required=True)
    factors.add_argument("--allocation-dir")
    factors.add_argument("--factor-dir")
    factors.add_argument("--ragged-factor-dir")
    parser.add_argument(
        "--allocation-schedule",
        help="Evaluate this recorded schedule; defaults to the selected candidate.",
    )
    parser.add_argument(
        "--allocation-schedule-file",
        help="Evaluate an authenticated external schedule derived from the allocation.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
