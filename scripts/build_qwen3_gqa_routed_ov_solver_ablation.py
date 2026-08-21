#!/usr/bin/env python3
"""Build frozen-config routed GQA-OV solver-attribution endpoints.

Unlike the original candidate-grid builder, this script never instantiates the
full language model.  It reads only the dense ``v_proj``/``o_proj`` tensors
needed by the current layer from the checkpoint's safetensors shards.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.gqa_routed_ov_joint import (
    RoutedOVCheckpointDiagnostics,
    combine_quadratics,
    covariance_with_trace_damping,
    evaluate_quadratic,
    extract_value_coordinate_factors,
    fit_routed_ov_joint,
    fold_routed_ov_factors,
    function_prior_covariance,
    head_products,
    mask_routed_covariance,
    quadratic_from_target,
)
from basisserve.core.gqa_vo_svdllm import GQAVOLayout
from basisserve.core.joint_aa_gqa_o import resolve_head_to_kv_group


FORMAT = "basisserve.gqa_routed_ov.solver_ablation_grid.v1"
INDEX_FORMAT = "basisserve.gqa_routed_ov.candidate_index.v1"
LAYER_FORMAT = "basisserve.gqa_vo_svdllm.layer.v1"

PRIMARY_ENDPOINTS = (
    "anchor",
    "decoder_only_full_layer",
    "one_sweep_cg200_tol1e8",
    "two_sweeps_cg200_tol1e8",
    "five_sweeps_legacy_pre_red_cg200_tol1e8",
    "five_sweeps_final_red_cg200_tol1e8",
)
D_ONLY_ENDPOINTS = (
    "decoder_only_diagonal",
    "decoder_only_within_group",
)
ALL_ENDPOINTS = PRIMARY_ENDPOINTS + D_ONLY_ENDPOINTS


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _parse_layers(spec: str) -> set[int] | None:
    spec = spec.strip().lower()
    if spec in {"", "all", "*"}:
        return None
    selected: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid descending layer range: {item}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))
    if not selected:
        raise ValueError("--layers did not select any layers")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--a3-metrics", required=True)
    parser.add_argument("--anchor-rank-bank", required=True)
    parser.add_argument("--legacy-reference-rank-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--max-sweeps", type=int, default=5)
    parser.add_argument("--min-sweeps", type=int, default=1)
    parser.add_argument("--relative-objective-tolerance", type=float, default=1e-7)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--cg-tolerance", type=float, default=1e-8)
    parser.add_argument("--cg-max-iterations", type=int, default=200)
    parser.add_argument("--covariance-damping", type=float, default=1e-7)
    parser.add_argument("--decoder-jitter", type=float, default=0.0)
    parser.add_argument("--encoder-damping", type=float, default=1e-8)
    parser.add_argument("--maximum-backtracks", type=int, default=10)
    parser.add_argument(
        "--legacy-factor-tolerance",
        type=float,
        default=1e-4,
        help="maximum relative V or O factor difference from the selected bank",
    )
    parser.add_argument(
        "--work-dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument(
        "--factor-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--factor-device", default="cuda")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _layout_from_config(config: dict[str, Any], rank: int) -> GQAVOLayout:
    hidden_size = int(config["hidden_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(
        config.get("num_key_value_heads", num_attention_heads)
    )
    head_dim = int(
        config.get("head_dim", hidden_size // num_attention_heads)
    )
    return GQAVOLayout(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rank=rank,
    )


class _SafetensorWeightReader:
    def __init__(self, model_root: Path) -> None:
        self.model_root = model_root
        index_path = model_root / "model.safetensors.index.json"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = {
                str(key): model_root / str(value)
                for key, value in payload["weight_map"].items()
            }
        else:
            single = model_root / "model.safetensors"
            if not single.is_file():
                raise FileNotFoundError(
                    f"no safetensors checkpoint found under {model_root}"
                )
            with safe_open(single, framework="pt", device="cpu") as handle:
                self.weight_map = {key: single for key in handle.keys()}

    def tensor(self, key: str, *, required: bool = True) -> torch.Tensor | None:
        path = self.weight_map.get(key)
        if path is None:
            if required:
                raise KeyError(f"checkpoint is missing tensor {key!r}")
            return None
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


def _module_name(layer_index: int) -> str:
    return f"model.layers.{layer_index}.self_attn"


def _load_statistics(
    stats_root: Path,
    *,
    split: str,
    layer_index: int,
    layout: GQAVOLayout,
) -> tuple[torch.Tensor, int, float]:
    path = (
        stats_root
        / f"{split}_covariances"
        / f"layer_{layer_index:03d}.safetensors"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing routed statistics: {path}")
    payload = load_file(str(path), device="cpu")
    covariance = payload["covariance_blocks"]
    expected = (
        layout.num_attention_heads,
        layout.num_attention_heads,
        layout.head_dim,
        layout.head_dim,
    )
    if tuple(covariance.shape) != expected:
        raise ValueError(
            f"{path} covariance shape {tuple(covariance.shape)} != {expected}"
        )
    return (
        covariance,
        int(payload["row_count"].item()),
        float(payload["dense_output_energy"].item()),
    )


def _value_metrics(
    tensors: dict[str, torch.Tensor],
    *,
    layer_index: int,
    layout: GQAVOLayout,
) -> torch.Tensor:
    key = f"layers.{layer_index}.value_metric"
    if key not in tensors:
        raise KeyError(f"missing A3 value metric {key!r}")
    value = tensors[key]
    expected = (
        layout.num_key_value_heads,
        layout.head_dim,
        layout.head_dim,
    )
    if tuple(value.shape) != expected:
        raise ValueError(
            f"A3 value metric shape {tuple(value.shape)} != {expected}"
        )
    return value


def _load_source_profile(
    root: Path,
    *,
    model_name: str,
    layout: GQAVOLayout,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != "basisserve.a3_gqa_vo.rank_bank.v1":
        raise ValueError(f"unsupported source rank bank: {root}")
    profile = json.loads(
        (root / config["profile"]).read_text(encoding="utf-8")
    )
    source_model = str(profile.get("model", config.get("model")))
    source_cache_token = "models--" + source_model.replace("/", "--")
    same_local_path = False
    try:
        same_local_path = (
            Path(source_model).expanduser().resolve()
            == Path(model_name).expanduser().resolve()
        )
    except OSError:
        pass
    if (
        source_model != model_name
        and not same_local_path
        and source_cache_token not in model_name
    ):
        raise ValueError(f"source model mismatch: {source_model} vs {model_name}")
    model_config = profile["model_config"]
    actual = (
        int(model_config["hidden_size"]),
        int(model_config["num_query_heads"]),
        int(model_config["num_kv_heads"]),
        int(model_config["head_dim"]),
    )
    expected = (
        layout.hidden_size,
        layout.num_attention_heads,
        layout.num_key_value_heads,
        layout.head_dim,
    )
    if actual != expected:
        raise ValueError(f"source layout mismatch: {actual} vs {expected}")
    candidate_ranks = {int(value) for value in profile["candidate_ranks"]}
    if rank not in candidate_ranks or layout.head_dim not in candidate_ranks:
        raise ValueError(
            f"source bank must contain ranks {rank} and {layout.head_dim}"
        )
    return config, profile


def _source_payload(
    root: Path,
    profile: dict[str, Any],
    *,
    layer_index: int,
    rank: int,
) -> dict[str, Any]:
    entry = profile["layers"][str(layer_index)]["ranks"][str(rank)]
    payload = torch.load(
        root / entry["factor_path"],
        map_location="cpu",
        weights_only=True,
    )
    if payload.get("format") != LAYER_FORMAT:
        raise ValueError(f"invalid source payload for layer {layer_index}")
    if int(payload["rank_per_kv_head"]) != rank:
        raise ValueError(f"source rank mismatch for layer {layer_index}")
    return payload


def _save_layer(
    candidate_root: Path,
    *,
    layer_index: int,
    module_name: str,
    rank: int,
    factors: Any,
    o_bias: torch.Tensor | None,
    diagnostics: dict[str, Any],
    method: str = "routed_gqa_ov_solver_ablation",
) -> str:
    layer_dir = candidate_root / f"layer_{layer_index:04d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    factor_path = layer_dir / f"rank_{rank:04d}.pt"
    temporary = factor_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "format": LAYER_FORMAT,
            "method": method,
            "module_name": module_name,
            "layer_index": layer_index,
            "rank_per_kv_head": rank,
            "v_proj_compressed_weight": factors.v_proj_compressed_weight,
            "v_proj_compressed_bias": None,
            "o_decoder_weight": factors.o_decoder_weight,
            "o_decoder_bias": (
                None
                if o_bias is None
                else o_bias.detach().cpu().to(
                    factors.o_decoder_weight.dtype
                )
            ),
            "diagnostics": diagnostics,
        },
        temporary,
    )
    os.replace(temporary, factor_path)
    return str(factor_path.relative_to(candidate_root))


def _endpoint_mode(endpoint: str) -> str:
    if endpoint.endswith("_diagonal"):
        return "diagonal"
    if endpoint.endswith("_within_group"):
        return "within_group"
    return "full_layer"


def _relative_tensor_error(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().cpu()
    right = right.detach().float().cpu()
    denominator = torch.linalg.vector_norm(right).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return float(torch.linalg.vector_norm(left - right) / denominator)


def _checkpoint_attribution(
    checkpoints: dict[tuple[str, int], RoutedOVCheckpointDiagnostics],
    *,
    sweeps: int,
    include_final_redecoder: bool,
) -> dict[str, Any]:
    anchor_loss = checkpoints[("anchor", 0)].loss
    decoder_only_loss = checkpoints[("decoder_only", 0)].loss
    steps = []
    decoder_reduction = anchor_loss - decoder_only_loss
    encoder_reduction = 0.0
    endpoint_loss = decoder_only_loss
    for sweep in range(1, sweeps + 1):
        before_encoder = (
            decoder_only_loss
            if sweep == 1
            else checkpoints[("after_redecoder", sweep - 1)].loss
        )
        after_encoder = checkpoints[("after_encoder", sweep)].loss
        use_redecoder = sweep < sweeps or include_final_redecoder
        after_redecoder = (
            checkpoints[("after_redecoder", sweep)].loss
            if use_redecoder
            else None
        )
        encoder_delta = before_encoder - after_encoder
        redecoder_delta = (
            after_encoder - after_redecoder
            if after_redecoder is not None
            else 0.0
        )
        encoder_reduction += encoder_delta
        decoder_reduction += redecoder_delta
        endpoint_loss = (
            after_redecoder
            if after_redecoder is not None
            else after_encoder
        )
        steps.append(
            {
                "sweep": sweep,
                "loss_before_encoder": before_encoder,
                "loss_after_encoder": after_encoder,
                "loss_after_redecoder": after_redecoder,
                "encoder_reduction": encoder_delta,
                "redecoder_reduction": redecoder_delta,
            }
        )
    total = anchor_loss - endpoint_loss
    denominator = max(abs(total), 1e-300)
    return {
        "anchor_loss": anchor_loss,
        "decoder_only_loss": decoder_only_loss,
        "endpoint_loss": endpoint_loss,
        "initial_decoder_reduction": anchor_loss - decoder_only_loss,
        "steps": steps,
        "decoder_reduction": decoder_reduction,
        "encoder_reduction": encoder_reduction,
        "total_reduction": total,
        "decoder_fraction": decoder_reduction / denominator,
        "encoder_fraction": encoder_reduction / denominator,
        "identity_error": decoder_reduction + encoder_reduction - total,
    }


def _zero_sweep_attribution(
    *,
    anchor_loss: float,
    endpoint_loss: float,
    decoder_only: bool,
) -> dict[str, Any]:
    total = anchor_loss - endpoint_loss
    decoder = total if decoder_only else 0.0
    denominator = max(abs(total), 1e-300)
    return {
        "anchor_loss": anchor_loss,
        "decoder_only_loss": endpoint_loss if decoder_only else anchor_loss,
        "endpoint_loss": endpoint_loss,
        "initial_decoder_reduction": decoder,
        "steps": [],
        "decoder_reduction": decoder,
        "encoder_reduction": 0.0,
        "total_reduction": total,
        "decoder_fraction": decoder / denominator if total else 0.0,
        "encoder_fraction": 0.0,
        "identity_error": decoder - total,
    }


def _write_rank_bank(
    root: Path,
    *,
    args: argparse.Namespace,
    model_name: str,
    endpoint: str,
    layout: GQAVOLayout,
    num_layers: int,
    built_layers: list[dict[str, Any]],
) -> None:
    built = {int(item["layer_index"]): item for item in built_layers}
    profile = {
        "format": "basisserve.a3_gqa_vo.rank_profile.v1",
        "schema_version": 1,
        "method": "routed_gqa_ov_solver_ablation",
        "model": model_name,
        "candidate_ranks": [layout.rank, layout.head_dim],
        "model_config": {
            "num_layers": num_layers,
            "hidden_size": layout.hidden_size,
            "num_query_heads": layout.num_attention_heads,
            "num_kv_heads": layout.num_key_value_heads,
            "head_dim": layout.head_dim,
        },
        "layers": {
            str(layer_index): {
                "module_name": _module_name(layer_index),
                "ranks": (
                    {}
                    if layer_index not in built
                    else {
                        str(layout.rank): {
                            "factor_path": built[layer_index]["factor_path"]
                        }
                    }
                ),
            }
            for layer_index in range(num_layers)
        },
    }
    config = {
        "format": "basisserve.a3_gqa_vo.rank_bank.v1",
        "model": model_name,
        "model_type": "qwen3",
        "profile": "profile.json",
        "candidate_ranks": [layout.rank, layout.head_dim],
        "rank": layout.rank,
        "method": "routed_gqa_ov_solver_ablation",
        "endpoint": endpoint,
        "anchor": {
            "name": "bestkron",
            "rank_bank": str(Path(args.anchor_rank_bank).expanduser().resolve()),
            "target": "product",
        },
        "alpha": args.alpha,
        "head_coupling_mode": _endpoint_mode(endpoint),
        "solver": {
            "maximum_sweeps": args.max_sweeps,
            "minimum_sweeps": args.min_sweeps,
            "relative_objective_tolerance": args.relative_objective_tolerance,
            "patience": args.patience,
            "cg_tolerance": args.cg_tolerance,
            "cg_max_iterations": args.cg_max_iterations,
            "cg_fixed_iterations": False,
            "covariance_damping": args.covariance_damping,
            "decoder_jitter": args.decoder_jitter,
            "encoder_damping": args.encoder_damping,
            "maximum_backtracks": args.maximum_backtracks,
            "work_dtype": args.work_dtype,
            "factor_dtype": args.factor_dtype,
            "final_decoder_solve": (
                endpoint == "five_sweeps_final_red_cg200_tol1e8"
            ),
        },
    }
    schedule = {
        "format": "basisserve.a3_gqa_vo.group_rank_schedule.v1",
        "selected_ranks": [
            [
                layout.rank if layer_index in built else layout.head_dim
            ]
            * layout.num_key_value_heads
            for layer_index in range(num_layers)
        ],
        "rank_sum": sum(
            layout.num_key_value_heads
            * (layout.rank if layer_index in built else layout.head_dim)
            for layer_index in range(num_layers)
        ),
    }
    _write_json(root / "profile.json", profile)
    _write_json(root / "config.json", config)
    _write_json(root / "uniform_schedule.json", schedule)
    (root / "BUILD_COMPLETE").touch()


def _aggregate_attribution(layers: list[dict[str, Any]]) -> dict[str, Any]:
    attributes = [item["attribution"] for item in layers]
    sums = {
        key: sum(float(item[key]) for item in attributes)
        for key in (
            "anchor_loss",
            "decoder_only_loss",
            "endpoint_loss",
            "initial_decoder_reduction",
            "decoder_reduction",
            "encoder_reduction",
            "total_reduction",
            "identity_error",
        )
    }
    total = sums["total_reduction"]
    anchor = max(abs(sums["anchor_loss"]), 1e-300)
    total_denominator = max(abs(total), 1e-300)
    sweep_numbers = sorted(
        {
            int(step["sweep"])
            for item in attributes
            for step in item["steps"]
        }
    )
    steps = []
    for sweep in sweep_numbers:
        selected = [
            step
            for item in attributes
            for step in item["steps"]
            if int(step["sweep"]) == sweep
        ]
        steps.append(
            {
                "sweep": sweep,
                "encoder_reduction": sum(
                    float(item["encoder_reduction"]) for item in selected
                ),
                "redecoder_reduction": sum(
                    float(item["redecoder_reduction"]) for item in selected
                ),
            }
        )
    return sums | {
        "steps": steps,
        "decoder_fraction": sums["decoder_reduction"] / total_denominator,
        "encoder_fraction": sums["encoder_reduction"] / total_denominator,
        "total_reduction_normalized_by_anchor": total / anchor,
        "decoder_reduction_normalized_by_anchor": sums["decoder_reduction"]
        / anchor,
        "encoder_reduction_normalized_by_anchor": sums["encoder_reduction"]
        / anchor,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if args.max_sweeps < 5:
        raise ValueError("--max-sweeps must be at least 5 for required checkpoints")
    model_root = Path(args.model).expanduser().resolve()
    stats_root = Path(args.stats_dir).expanduser().resolve()
    metrics_path = Path(args.a3_metrics).expanduser().resolve()
    anchor_root = Path(args.anchor_rank_bank).expanduser().resolve()
    legacy_root = Path(args.legacy_reference_rank_bank).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    for path in (
        model_root / "config.json",
        stats_root / "config.json",
        metrics_path,
        anchor_root / "config.json",
        legacy_root / "config.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite solver ablation: {output}")

    model_config = json.loads(
        (model_root / "config.json").read_text(encoding="utf-8")
    )
    layout = _layout_from_config(model_config, args.rank)
    num_layers = int(model_config["num_hidden_layers"])
    requested_layers = _parse_layers(args.layers)
    selected_layers = [
        index
        for index in range(num_layers)
        if requested_layers is None or index in requested_layers
    ]
    if not selected_layers:
        raise ValueError("--layers selected no model layers")
    if requested_layers is not None and not requested_layers.issubset(
        set(range(num_layers))
    ):
        raise ValueError("--layers contains an out-of-range model layer")

    stats_config = json.loads(
        (stats_root / "config.json").read_text(encoding="utf-8")
    )
    if stats_config.get("status") != "complete":
        raise ValueError("routed statistics are not complete")
    missing_stats = set(selected_layers) - {
        int(item) for item in stats_config["layers"]
    }
    if missing_stats:
        raise ValueError(f"missing routed statistics for {sorted(missing_stats)}")

    work_dtype = _dtype(args.work_dtype)
    factor_dtype = _dtype(args.factor_dtype)
    factor_device = torch.device(args.factor_device)
    if factor_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA factor device requested but CUDA is unavailable")
    mapping = resolve_head_to_kv_group(layout).to(factor_device)
    reader = _SafetensorWeightReader(model_root)
    a3_tensors = load_file(str(metrics_path), device="cpu")
    _, anchor_profile = _load_source_profile(
        anchor_root,
        model_name=str(model_root),
        layout=layout,
        rank=layout.rank,
    )
    _, legacy_profile = _load_source_profile(
        legacy_root,
        model_name=str(model_root),
        layout=layout,
        rank=layout.rank,
    )

    output.mkdir(parents=True)
    endpoint_layers: dict[str, list[dict[str, Any]]] = {
        endpoint: [] for endpoint in ALL_ENDPOINTS
    }
    checkpoint_rows: list[dict[str, Any]] = []
    root_config = {
        "format": FORMAT,
        "status": "incomplete",
        "model": str(model_root),
        "stats_dir": str(stats_root),
        "a3_metrics": str(metrics_path),
        "anchors": {"bestkron": str(anchor_root)},
        "legacy_reference_rank_bank": str(legacy_root),
        "alphas": [args.alpha],
        "head_coupling_modes": [
            "diagonal",
            "within_group",
            "full_layer",
        ],
        "anchor_target": "product",
        "rank": layout.rank,
        "layers": selected_layers,
        "endpoints": list(ALL_ENDPOINTS),
        "command": shlex.join([sys.executable, *sys.argv]),
        "started_unix_time": time.time(),
    }
    _write_json(output / "config.json", root_config)

    legacy_maximum_v_error = 0.0
    legacy_maximum_o_error = 0.0
    legacy_all_exact = True
    for layer_index in selected_layers:
        started = time.monotonic()
        module_name = _module_name(layer_index)
        prefix = f"{module_name}."
        dense_v = reader.tensor(prefix + "v_proj.weight")
        dense_o_weight = reader.tensor(prefix + "o_proj.weight")
        o_bias = reader.tensor(prefix + "o_proj.bias", required=False)
        assert dense_v is not None and dense_o_weight is not None
        dense_v = dense_v.to(device=factor_device)
        dense_o = (
            dense_o_weight.to(device=factor_device, dtype=work_dtype)
            .transpose(0, 1)
            .reshape(
                layout.num_attention_heads,
                layout.head_dim,
                layout.hidden_size,
            )
        )
        fit_covariance, fit_rows, fit_energy = _load_statistics(
            stats_root,
            split="fit",
            layer_index=layer_index,
            layout=layout,
        )
        validation_covariance, validation_rows, validation_energy = (
            _load_statistics(
                stats_root,
                split="validation",
                layer_index=layer_index,
                layout=layout,
            )
        )
        fit_covariance, fit_damping = covariance_with_trace_damping(
            fit_covariance.to(device=factor_device, dtype=work_dtype),
            relative_damping=args.covariance_damping,
        )
        validation_covariance, validation_damping = (
            covariance_with_trace_damping(
                validation_covariance.to(
                    device=factor_device,
                    dtype=work_dtype,
                ),
                relative_damping=args.covariance_damping,
            )
        )
        value_metrics = _value_metrics(
            a3_tensors,
            layer_index=layer_index,
            layout=layout,
        ).to(device=factor_device, dtype=work_dtype)
        function_covariance = function_prior_covariance(
            value_metrics,
            head_to_kv_group=mapping,
        )
        source = _source_payload(
            anchor_root,
            anchor_profile,
            layer_index=layer_index,
            rank=layout.rank,
        )
        initial_A, initial_D, extraction = extract_value_coordinate_factors(
            layout=layout,
            dense_v_proj_weight=dense_v,
            compressed_v_proj_weight=source["v_proj_compressed_weight"],
            compressed_o_proj_weight=source["o_decoder_weight"],
            work_dtype=work_dtype,
            work_device=factor_device,
        )
        initial_A = initial_A.to(factor_device)
        initial_D = initial_D.to(factor_device)
        anchor_product = head_products(initial_A, initial_D, mapping)
        prior = quadratic_from_target(
            covariance=function_covariance,
            target=anchor_product,
            name="bestkron_product_prior",
            trace_normalize=True,
        )
        full_validation = quadratic_from_target(
            covariance=validation_covariance,
            target=dense_o,
            name="validation_full_routed",
            trace_normalize=False,
            precomputed_constant=(
                validation_energy if validation_damping == 0.0 else None
            ),
        )
        routed_fit: dict[str, Any] = {}
        routed_validation: dict[str, Any] = {}
        objectives: dict[str, Any] = {}
        for mode in ("diagonal", "within_group", "full_layer"):
            masked_fit = mask_routed_covariance(
                fit_covariance,
                head_to_kv_group=mapping,
                mode=mode,
            )
            masked_validation = mask_routed_covariance(
                validation_covariance,
                head_to_kv_group=mapping,
                mode=mode,
            )
            routed_fit[mode] = quadratic_from_target(
                covariance=masked_fit,
                target=dense_o,
                name=f"fit_routed_{mode}",
                trace_normalize=True,
                precomputed_constant=(
                    fit_energy
                    if mode == "full_layer" and fit_damping == 0.0
                    else None
                ),
            )
            routed_validation[mode] = quadratic_from_target(
                covariance=masked_validation,
                target=dense_o,
                name=f"validation_routed_{mode}",
                trace_normalize=False,
                precomputed_constant=(
                    validation_energy
                    if mode == "full_layer" and validation_damping == 0.0
                    else None
                ),
            )
            objectives[mode] = combine_quadratics(
                prior,
                routed_fit[mode],
                right_weight=args.alpha,
                name=f"bestkron_alpha_{args.alpha:g}_{mode}",
            )

        saved: dict[str, dict[str, Any]] = {}

        def save_endpoint(
            endpoint: str,
            checkpoint: RoutedOVCheckpointDiagnostics,
            checkpoint_A: torch.Tensor,
            checkpoint_D: torch.Tensor,
        ) -> None:
            mode = _endpoint_mode(endpoint)
            validation_mode_loss = evaluate_quadratic(
                routed_validation[mode],
                checkpoint_A,
                checkpoint_D,
                mapping,
            )
            validation_full_loss = evaluate_quadratic(
                full_validation,
                checkpoint_A,
                checkpoint_D,
                mapping,
            )
            factors = fold_routed_ov_factors(
                layout=layout,
                dense_v_proj_weight=dense_v,
                A_unique=checkpoint_A,
                D_heads=checkpoint_D,
                head_to_kv_group=mapping,
                thin_qr=True,
                output_dtype=factor_dtype,
            )
            diagnostics = {
                "endpoint": endpoint,
                "checkpoint": asdict(checkpoint),
                "fit_total_loss": checkpoint.loss,
                "fit_component_losses": dict(checkpoint.component_losses),
                "validation_mode_loss": validation_mode_loss,
                "validation_full_loss": validation_full_loss,
                "normalized_validation_mode_error": validation_mode_loss
                / max(float(routed_validation[mode].constant), 1e-30),
                "normalized_validation_full_error": validation_full_loss
                / max(float(full_validation.constant), 1e-30),
                "fit_rows": fit_rows,
                "validation_rows": validation_rows,
                "routed_absolute_damping": fit_damping,
                "validation_absolute_damping": validation_damping,
                "factor_extraction": extraction,
                "folding": {
                    "maximum_qr_product_error": (
                        factors.maximum_qr_product_error
                    ),
                    "maximum_dense_fold_error": (
                        factors.maximum_dense_fold_error
                    ),
                },
            }
            endpoint_root = output / "candidates" / endpoint
            relative_path = _save_layer(
                endpoint_root,
                layer_index=layer_index,
                module_name=module_name,
                rank=layout.rank,
                factors=factors,
                o_bias=o_bias,
                diagnostics=diagnostics,
            )
            saved[endpoint] = {
                "layer_index": layer_index,
                "module_name": module_name,
                "factor_path": relative_path,
                **diagnostics,
            }
            if endpoint == "five_sweeps_legacy_pre_red_cg200_tol1e8":
                reference = _source_payload(
                    legacy_root,
                    legacy_profile,
                    layer_index=layer_index,
                    rank=layout.rank,
                )
                v_error = _relative_tensor_error(
                    factors.v_proj_compressed_weight,
                    reference["v_proj_compressed_weight"],
                )
                o_error = _relative_tensor_error(
                    factors.o_decoder_weight,
                    reference["o_decoder_weight"],
                )
                exact = bool(
                    torch.equal(
                        factors.v_proj_compressed_weight.cpu(),
                        reference["v_proj_compressed_weight"].cpu(),
                    )
                    and torch.equal(
                        factors.o_decoder_weight.cpu(),
                        reference["o_decoder_weight"].cpu(),
                    )
                )
                saved[endpoint]["legacy_reference"] = {
                    "v_relative_error": v_error,
                    "o_relative_error": o_error,
                    "factor_tensors_exact": exact,
                }

        boundary_to_endpoint = {
            ("anchor", 0): "anchor",
            ("decoder_only", 0): "decoder_only_full_layer",
            ("after_redecoder", 1): "one_sweep_cg200_tol1e8",
            ("after_redecoder", 2): "two_sweeps_cg200_tol1e8",
            (
                "after_encoder",
                5,
            ): "five_sweeps_legacy_pre_red_cg200_tol1e8",
            (
                "after_redecoder",
                5,
            ): "five_sweeps_final_red_cg200_tol1e8",
        }

        def checkpoint_callback(
            checkpoint: RoutedOVCheckpointDiagnostics,
            checkpoint_A: torch.Tensor,
            checkpoint_D: torch.Tensor,
        ) -> None:
            checkpoint_rows.append(
                {
                    "layer_index": layer_index,
                    **asdict(checkpoint),
                }
            )
            endpoint = boundary_to_endpoint.get(
                (checkpoint.boundary, checkpoint.sweep)
            )
            if endpoint is not None:
                save_endpoint(
                    endpoint,
                    checkpoint,
                    checkpoint_A,
                    checkpoint_D,
                )

        primary = fit_routed_ov_joint(
            objective=objectives["full_layer"],
            initial_A=initial_A,
            initial_D=initial_D,
            head_to_kv_group=mapping,
            coupling_mode="full_layer",
            maximum_sweeps=args.max_sweeps,
            minimum_sweeps=args.min_sweeps,
            relative_objective_tolerance=args.relative_objective_tolerance,
            patience=args.patience,
            decoder_relative_jitter=args.decoder_jitter,
            encoder_relative_damping=args.encoder_damping,
            cg_relative_tolerance=args.cg_tolerance,
            cg_max_iterations=args.cg_max_iterations,
            maximum_backtracks=args.maximum_backtracks,
            component_objectives={
                "function_prior": prior,
                "routed": routed_fit["full_layer"],
            },
            checkpoint_callback=checkpoint_callback,
            final_decoder_solve=True,
            work_dtype=work_dtype,
            work_device=factor_device,
        )
        if set(PRIMARY_ENDPOINTS) - set(saved):
            raise RuntimeError(
                "primary solve did not produce all required checkpoints: "
                f"{sorted(set(PRIMARY_ENDPOINTS) - set(saved))}"
            )
        checkpoint_map = {
            (item.boundary, item.sweep): item for item in primary.checkpoints
        }
        primary_specs = {
            "anchor": (0, False),
            "decoder_only_full_layer": (0, False),
            "one_sweep_cg200_tol1e8": (1, True),
            "two_sweeps_cg200_tol1e8": (2, True),
            "five_sweeps_legacy_pre_red_cg200_tol1e8": (5, False),
            "five_sweeps_final_red_cg200_tol1e8": (5, True),
        }
        for endpoint, (sweep_count, include_final_red) in primary_specs.items():
            if endpoint == "anchor":
                attribution = _zero_sweep_attribution(
                    anchor_loss=primary.initial_loss,
                    endpoint_loss=primary.initial_loss,
                    decoder_only=False,
                )
            elif endpoint == "decoder_only_full_layer":
                attribution = _zero_sweep_attribution(
                    anchor_loss=primary.initial_loss,
                    endpoint_loss=primary.decoder_only_loss,
                    decoder_only=True,
                )
            else:
                attribution = _checkpoint_attribution(
                    checkpoint_map,
                    sweeps=sweep_count,
                    include_final_redecoder=include_final_red,
                )
            saved[endpoint]["attribution"] = attribution
            saved[endpoint]["sweeps"] = [
                asdict(item)
                for item in primary.sweeps[:sweep_count]
            ]

        for mode, endpoint in (
            ("diagonal", "decoder_only_diagonal"),
            ("within_group", "decoder_only_within_group"),
        ):
            result = fit_routed_ov_joint(
                objective=objectives[mode],
                initial_A=initial_A,
                initial_D=initial_D,
                head_to_kv_group=mapping,
                coupling_mode=mode,
                maximum_sweeps=0,
                minimum_sweeps=0,
                decoder_relative_jitter=args.decoder_jitter,
                component_objectives={
                    "function_prior": prior,
                    "routed": routed_fit[mode],
                },
                work_dtype=work_dtype,
                work_device=factor_device,
            )
            checkpoint = result.checkpoints[-1]
            save_endpoint(
                endpoint,
                checkpoint,
                result.A_unique.to(factor_device),
                result.D_heads.to(factor_device),
            )
            saved[endpoint]["attribution"] = _zero_sweep_attribution(
                anchor_loss=result.initial_loss,
                endpoint_loss=result.final_loss,
                decoder_only=True,
            )
            saved[endpoint]["sweeps"] = []

        if set(ALL_ENDPOINTS) != set(saved):
            raise RuntimeError(
                f"layer {layer_index} endpoint mismatch: {sorted(saved)}"
            )
        comparison = saved["five_sweeps_legacy_pre_red_cg200_tol1e8"][
            "legacy_reference"
        ]
        legacy_maximum_v_error = max(
            legacy_maximum_v_error,
            float(comparison["v_relative_error"]),
        )
        legacy_maximum_o_error = max(
            legacy_maximum_o_error,
            float(comparison["o_relative_error"]),
        )
        legacy_all_exact = (
            legacy_all_exact and comparison["factor_tensors_exact"]
        )
        if max(
            float(comparison["v_relative_error"]),
            float(comparison["o_relative_error"]),
        ) > args.legacy_factor_tolerance:
            raise RuntimeError(
                "legacy five-sweep factors do not reproduce the selected bank "
                f"at layer {layer_index}: {comparison}"
            )
        for endpoint in ALL_ENDPOINTS:
            endpoint_layers[endpoint].append(saved[endpoint])
        print(
            f"[Layer] index={layer_index} "
            f"legacy_v={comparison['v_relative_error']:.3e} "
            f"legacy_o={comparison['o_relative_error']:.3e} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
        del (
            dense_v,
            dense_o,
            fit_covariance,
            validation_covariance,
            initial_A,
            initial_D,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    candidates = []
    for endpoint in ALL_ENDPOINTS:
        layers = sorted(
            endpoint_layers[endpoint],
            key=lambda item: int(item["layer_index"]),
        )
        endpoint_root = output / "candidates" / endpoint
        _write_rank_bank(
            endpoint_root,
            args=args,
            model_name=str(model_root),
            endpoint=endpoint,
            layout=layout,
            num_layers=num_layers,
            built_layers=layers,
        )
        item = {
            "name": endpoint,
            "anchor": "bestkron",
            "alpha": args.alpha,
            "head_coupling_mode": _endpoint_mode(endpoint),
            "rank_bank": str(endpoint_root),
            "layers": layers,
            "sum_normalized_validation_mode_error": sum(
                float(layer["normalized_validation_mode_error"])
                for layer in layers
            ),
            "sum_normalized_validation_full_error": sum(
                float(layer["normalized_validation_full_error"])
                for layer in layers
            ),
            "attribution": _aggregate_attribution(layers),
        }
        item["mean_normalized_validation_full_error"] = (
            item["sum_normalized_validation_full_error"] / len(layers)
        )
        _write_json(endpoint_root / "diagnostics.json", item)
        candidates.append(item)
    candidates.sort(
        key=lambda item: (
            item["sum_normalized_validation_full_error"],
            ALL_ENDPOINTS.index(item["name"]),
        )
    )
    _write_json(
        output / "candidate_index.json",
        {
            "format": INDEX_FORMAT,
            "candidates": candidates,
        },
    )

    aggregated_checkpoints = []
    checkpoint_keys = sorted(
        {
            (str(item["boundary"]), int(item["sweep"]))
            for item in checkpoint_rows
        },
        key=lambda item: (
            item[1],
            (
                "anchor",
                "decoder_only",
                "after_encoder",
                "after_redecoder",
            ).index(item[0]),
        ),
    )
    for boundary, sweep in checkpoint_keys:
        selected = [
            item
            for item in checkpoint_rows
            if item["boundary"] == boundary and int(item["sweep"]) == sweep
        ]
        component_names = {
            name
            for item in selected
            for name, _ in item["component_losses"]
        }
        aggregated_checkpoints.append(
            {
                "boundary": boundary,
                "sweep": sweep,
                "loss": sum(float(item["loss"]) for item in selected),
                "component_losses": {
                    name: sum(
                        float(dict(item["component_losses"])[name])
                        for item in selected
                    )
                    for name in sorted(component_names)
                },
                "maximum_decoder_relative_stationarity": max(
                    float(item["decoder_relative_stationarity"])
                    for item in selected
                ),
            }
        )
    _write_json(
        output / "attribution.json",
        {
            "format": "basisserve.gqa_routed_ov.solver_attribution.v1",
            "primary_endpoint": "five_sweeps_final_red_cg200_tol1e8",
            "legacy_endpoint": (
                "five_sweeps_legacy_pre_red_cg200_tol1e8"
            ),
            "endpoint_attribution": {
                item["name"]: item["attribution"] for item in candidates
            },
            "checkpoints": aggregated_checkpoints,
            "legacy_reproduction": {
                "reference_rank_bank": str(legacy_root),
                "maximum_v_relative_error": legacy_maximum_v_error,
                "maximum_o_relative_error": legacy_maximum_o_error,
                "all_factor_tensors_exact": legacy_all_exact,
                "tolerance": args.legacy_factor_tolerance,
            },
        },
    )
    root_config["status"] = "complete"
    root_config["completed_unix_time"] = time.time()
    root_config["legacy_reproduction"] = {
        "maximum_v_relative_error": legacy_maximum_v_error,
        "maximum_o_relative_error": legacy_maximum_o_error,
        "all_factor_tensors_exact": legacy_all_exact,
    }
    _write_json(output / "config.json", root_config)
    (output / "BUILD_COMPLETE").touch()
    print(f"[Done] solver ablation={output}", flush=True)


if __name__ == "__main__":
    main()
