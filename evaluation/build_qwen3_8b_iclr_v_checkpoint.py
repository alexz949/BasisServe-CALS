#!/usr/bin/env python3
"""Build 8B-model V-side checkpoints for the ICLR experiment matrix.

The entry point covers the dense reference, per-physical-head Weight-SVD, and
uniform or Fisher-allocated activation-aware PaLU M/G2/G4 checkpoints.  It
intentionally does not contain a C1 path.
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
from typing import Any, Mapping, Sequence, cast

import torch
from torch import Tensor, nn
from transformers import AutoConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import build_llama31_8b_palu_m_checkpoint as palu_builder  # noqa: E402
from evaluation.reproduce_palu_paper_llama2_distributed import (  # noqa: E402
    official_fisher_uniform_rank_map,
)
from palu.model.modules.svd_linear import HeadwiseLowRankModule  # noqa: E402


EXPERIMENT_PROFILES = {
    "qwen3_8b": {
        "format": "basisserve.qwen3_8b.iclr_v_factors.v1",
        "fisher_format": "basisserve.gqa.palu_projection_fisher_stats.v1",
        "run_id_prefix": "Q3-8B",
    },
    "llama31_8b": {
        "format": "basisserve.llama31_8b.iclr_v_factors.v1",
        "fisher_format": "basisserve.gqa.palu_projection_fisher_stats.v1",
        "run_id_prefix": "L31-8B",
    },
    "llama2_7b": {
        "format": "basisserve.llama2_7b.iclr_v_factors.v1",
        "fisher_format": "basisserve.gqa.palu_projection_fisher_stats.v1",
        "run_id_prefix": "L2-7B",
    },
    "qwen3_32b": {
        "format": "basisserve.qwen3_32b.iclr_v_factors.v1",
        "fisher_format": "basisserve.gqa.palu_projection_fisher_stats.v1",
        "run_id_prefix": "Q3-32B",
    },
    "llama31_70b": {
        "format": "basisserve.llama31_70b.iclr_v_factors.v1",
        "fisher_format": "basisserve.gqa.palu_projection_fisher_stats.v1",
        "run_id_prefix": "L31-70B",
    },
}
FORMAT = str(EXPERIMENT_PROFILES["qwen3_8b"]["format"])
PINNED_FISHER_FORMAT = str(EXPERIMENT_PROFILES["qwen3_8b"]["fisher_format"])
PROFILE = "qwen3_8b"
RUN_ID_PREFIX = str(EXPERIMENT_PROFILES["qwen3_8b"]["run_id_prefix"])
FISHER_RANK_BLOCK_SIZE = 32
SUPPORTED_EQUIVALENT_RANKS = (64, 80, 96)
SUPPORTED_HEAD_GROUP_SIZES = (1, 2, 4)


def activate_experiment_profile(name: str) -> None:
    experiment = EXPERIMENT_PROFILES[name]
    global FORMAT, PINNED_FISHER_FORMAT, PROFILE, RUN_ID_PREFIX
    FORMAT = str(experiment["format"])
    PINNED_FISHER_FORMAT = str(experiment["fisher_format"])
    PROFILE = name
    RUN_ID_PREFIX = str(experiment["run_id_prefix"])
    palu_builder.activate_model_profile(name)


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _immutable_model_matches(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    keys = (
        "huggingface_repo",
        "revision",
        "config_sha256",
        "safetensors_index_sha256",
    )
    return all(observed.get(key) == expected.get(key) for key in keys)


def run_id(method: str, equivalent_rank: int | None, head_group_size: int) -> str:
    if method == "dense":
        return f"{RUN_ID_PREFIX}-Dense"
    if method == "weight-svd":
        return f"{RUN_ID_PREFIX}-SVD-R{equivalent_rank}"
    geometry = "PALUM" if head_group_size == 1 else f"PALUG{head_group_size}"
    if method == "palu-uniform":
        return f"{RUN_ID_PREFIX}-{geometry}-U-R{equivalent_rank}"
    return f"{RUN_ID_PREFIX}-{geometry}-R{equivalent_rank}"


def _factor_diagnostics(
    weight: Tensor,
    writer: Tensor,
    decoder: Tensor,
    ranks: Sequence[int],
    cholesky: Tensor | None = None,
) -> dict[str, float]:
    group_width = int(weight.shape[0]) // len(ranks)
    weight64 = weight.to(torch.float64)
    error_sq = 0.0
    reference_sq = 0.0
    weighted_error_sq = 0.0
    weighted_reference_sq = 0.0
    scale = cholesky.to(torch.float64) if cholesky is not None else None
    offset = 0
    for group_index, rank in enumerate(ranks):
        dense = weight64[
            group_index * group_width : (group_index + 1) * group_width
        ]
        reconstructed = (
            decoder[group_index].to(torch.float64)
            @ writer[offset : offset + rank].to(torch.float64)
        )
        residual = dense - reconstructed
        error_sq += float(residual.square().sum())
        reference_sq += float(dense.square().sum())
        if scale is not None:
            weighted_error_sq += float((residual @ scale).square().sum())
            weighted_reference_sq += float((dense @ scale).square().sum())
        offset += rank
    diagnostics = {
        "relative_frobenius_error": (error_sq / reference_sq) ** 0.5
    }
    if scale is not None:
        diagnostics["relative_activation_weighted_error"] = (
            weighted_error_sq / weighted_reference_sq
        ) ** 0.5
    return diagnostics


@torch.no_grad()
def factorize_weight_svd_projection(
    weight: Tensor,
    *,
    ranks: Sequence[int],
    output_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Factor each physical V head with ordinary truncated weight SVD."""

    linear = nn.Linear(
        int(weight.shape[1]),
        int(weight.shape[0]),
        bias=False,
        device="cpu",
        dtype=weight.dtype,
    )
    linear.weight.copy_(weight)
    replacement = HeadwiseLowRankModule.from_linear(
        linear,
        list(ranks),
        shared_basis_rank=0,
        latent_recovery_ranks=[0] * len(ranks),
        preserve_budget=False,
    )
    writer = replacement.VT.weight.detach().to(output_dtype).cpu().contiguous()
    decoder = (
        torch.stack([module.weight.detach() for module in replacement.U])
        .to(output_dtype)
        .cpu()
        .contiguous()
    )
    return writer, decoder, _factor_diagnostics(weight, writer, decoder, ranks)


@torch.no_grad()
def factorize_palu_projection(
    weight: Tensor,
    cholesky: Tensor,
    *,
    ranks: Sequence[int],
    output_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Apply PaLU's activation-weighted SVD without forming a dense inverse."""

    group_width = int(weight.shape[0]) // len(ranks)
    scale = cholesky.to(device="cpu", dtype=torch.float64)
    weight_groups = weight.to(torch.float64).reshape(
        len(ranks), group_width, int(weight.shape[1])
    )
    left_factors: list[Tensor] = []
    right_factors: list[Tensor] = []
    for group_index, rank in enumerate(ranks):
        weighted = weight_groups[group_index] @ scale
        left_singular, singular_values, right_singular = torch.linalg.svd(
            weighted,
            full_matrices=False,
        )
        right_unwhitened = torch.linalg.solve_triangular(
            scale.transpose(0, 1),
            right_singular[:rank].transpose(0, 1),
            upper=True,
        ).transpose(0, 1)
        sigma_root = singular_values[:rank].sqrt()
        left_factors.append(
            (left_singular[:, :rank] * sigma_root.unsqueeze(0)).to(output_dtype)
        )
        right_factors.append(
            (sigma_root.unsqueeze(1) * right_unwhitened).to(output_dtype)
        )
    writer = torch.cat(right_factors, dim=0).cpu().contiguous()
    decoder = torch.stack(left_factors).cpu().contiguous()
    diagnostics = _factor_diagnostics(
        weight,
        writer,
        decoder,
        ranks,
        cholesky,
    )
    return writer, decoder, diagnostics


def fisher_layer_ranks(
    fisher_scalars: Mapping[str, float],
    *,
    equivalent_rank: int,
    head_group_size: int,
) -> tuple[list[list[int]], int, int]:
    rank_map, rank_sum, total_rank = official_fisher_uniform_rank_map(
        dict(fisher_scalars),
        output_width=palu_builder.NUM_KV_HEADS * palu_builder.HEAD_DIM,
        num_heads=palu_builder.NUM_KV_HEADS,
        head_group_size=head_group_size,
        retained_ratio=equivalent_rank / palu_builder.HEAD_DIM,
        block_size=FISHER_RANK_BLOCK_SIZE,
    )
    layer_ranks = [
        rank_map[f"model.layers.{layer_index}.self_attn.v_proj"]
        for layer_index in range(palu_builder.NUM_LAYERS)
    ]
    return layer_ranks, rank_sum, total_rank


def _load_palu_inputs(
    fisher_path: Path,
    whitening_dir: Path,
    model_metadata: Mapping[str, Any],
    *,
    equivalent_rank: int,
    head_group_size: int,
) -> tuple[
    dict[str, Any],
    list[Tensor],
    dict[str, Any],
    list[list[int]],
    int,
    int,
] | None:
    fisher = json.loads(fisher_path.read_text(encoding="utf-8"))
    valid = all(
        (
            _check(
                fisher.get("format") == PINNED_FISHER_FORMAT,
                "unexpected Fisher format",
            ),
            _check(fisher.get("status") == "complete", "Fisher result is incomplete"),
            _check(
                _immutable_model_matches(fisher.get("model", {}), model_metadata),
                "Fisher result belongs to another model snapshot",
            ),
            _check(
                fisher.get("fisher", {}).get("target") == "v_proj_only",
                "Fisher result is not V-only",
            ),
        )
    )
    if not valid:
        return None
    whitening, whitening_manifest = palu_builder._load_whitening(
        whitening_dir, model_metadata
    )
    same_shape = (
        int(whitening_manifest["samples"])
        == int(fisher["fisher"]["samples"])
        and int(whitening_manifest["sequence_length"])
        == int(fisher["fisher"]["sequence_length"])
    )
    if not _check(same_shape, "Fisher and whitening calibration shapes differ"):
        return None
    same_windows = (
        whitening_manifest["windows"]["sha256"]
        == fisher["calibration_windows"]["sha256"]
    )
    if not _check(same_windows, "Fisher and whitening use different C4 windows"):
        return None
    scalars = fisher["fisher"]["scalars"]
    if not _check(
        len(scalars) == palu_builder.NUM_LAYERS,
        "Fisher scalars do not cover every decoder layer",
    ):
        return None
    layer_ranks, rank_sum, total_rank = fisher_layer_ranks(
        scalars,
        equivalent_rank=equivalent_rank,
        head_group_size=head_group_size,
    )
    return (
        fisher,
        whitening,
        whitening_manifest,
        layer_ranks,
        rank_sum,
        total_rank,
    )


def _dense_manifest(model_metadata: Mapping[str, Any]) -> dict[str, Any]:
    dense_rank_sum = (
        palu_builder.NUM_LAYERS
        * palu_builder.NUM_KV_HEADS
        * palu_builder.HEAD_DIM
    )
    return {
        "format": FORMAT,
        "status": "complete",
        "run_id": run_id("dense", None, 1),
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_metadata,
        "compression": {
            "target": "dense_kv_cache",
            "key_cache": "dense",
            "value_cache": "dense",
            "projection": None,
            "method": "dense",
            "method_label": "Dense",
            "num_query_heads": palu_builder.NUM_QUERY_HEADS,
            "num_physical_kv_heads": palu_builder.NUM_KV_HEADS,
            "head_dim": palu_builder.HEAD_DIM,
            "rank_sum_across_layers": dense_rank_sum,
            "dense_rank_sum_across_layers": dense_rank_sum,
            "realized_retained_v_ratio": 1.0,
            "realized_v_cache_compression_ratio": 0.0,
        },
        "artifact": None,
        "layers": [],
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
        },
    }


@torch.no_grad()
def build(args: argparse.Namespace) -> int:
    palu_builder.activate_model_profile(PROFILE)
    torch.set_num_threads(args.torch_num_threads)
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not _check(not output_dir.exists(), f"output directory already exists: {output_dir}"):
        return 2
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    palu_builder._validate_config(config)
    model_metadata = palu_builder._model_metadata(model_path)

    if args.method == "dense":
        output_dir.mkdir(parents=True)
        palu_builder._atomic_json(
            output_dir / "manifest.json", _dense_manifest(model_metadata)
        )
        print(f"[Result] dense reference={output_dir}", flush=True)
        return 0

    equivalent_rank = int(args.equivalent_rank or 0)
    valid = all(
        (
            _check(
                equivalent_rank in SUPPORTED_EQUIVALENT_RANKS,
                f"equivalent rank must be one of {SUPPORTED_EQUIVALENT_RANKS}",
            ),
            _check(
                args.head_group_size in SUPPORTED_HEAD_GROUP_SIZES,
                f"head group size must be one of {SUPPORTED_HEAD_GROUP_SIZES}",
            ),
            _check(
                args.method != "weight-svd" or args.head_group_size == 1,
                "Weight-SVD is defined per physical V head",
            ),
        )
    )
    if not valid:
        return 2

    fisher: dict[str, Any] | None = None
    whitening: list[Tensor] | None = None
    whitening_manifest: dict[str, Any] | None = None
    fisher_path: Path | None = None
    whitening_dir: Path | None = None
    if args.method == "weight-svd":
        layer_ranks = [
            [equivalent_rank] * palu_builder.NUM_KV_HEADS
            for _ in range(palu_builder.NUM_LAYERS)
        ]
        rank_sum = sum(sum(ranks) for ranks in layer_ranks)
        total_rank = (
            palu_builder.NUM_LAYERS
            * palu_builder.NUM_KV_HEADS
            * palu_builder.HEAD_DIM
        )
    elif args.method == "palu-uniform":
        if not _check(
            args.whitening_dir is not None,
            "uniform PaLU requires --whitening-dir",
        ):
            return 2
        whitening_dir = Path(args.whitening_dir).expanduser().resolve()
        whitening, whitening_manifest = palu_builder._load_whitening(
            whitening_dir,
            model_metadata,
        )
        group_count = palu_builder.NUM_KV_HEADS // args.head_group_size
        group_rank = args.head_group_size * equivalent_rank
        layer_ranks = [
            [group_rank] * group_count for _ in range(palu_builder.NUM_LAYERS)
        ]
        rank_sum = sum(sum(ranks) for ranks in layer_ranks)
        total_rank = (
            palu_builder.NUM_LAYERS
            * palu_builder.NUM_KV_HEADS
            * palu_builder.HEAD_DIM
        )
    else:
        if not _check(
            args.fisher_result is not None and args.whitening_dir is not None,
            "PaLU requires --fisher-result and --whitening-dir",
        ):
            return 2
        fisher_path = Path(args.fisher_result).expanduser().resolve()
        whitening_dir = Path(args.whitening_dir).expanduser().resolve()
        loaded = _load_palu_inputs(
            fisher_path,
            whitening_dir,
            model_metadata,
            equivalent_rank=equivalent_rank,
            head_group_size=args.head_group_size,
        )
        if loaded is None:
            return 2
        (
            fisher,
            whitening,
            whitening_manifest,
            layer_ranks,
            rank_sum,
            total_rank,
        ) = loaded

    factor_payload: dict[str, Tensor] = {}
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for layer_index, ranks in enumerate(layer_ranks):
        layer_started = time.perf_counter()
        tensor_name = f"model.layers.{layer_index}.self_attn.v_proj.weight"
        dense = palu_builder._load_indexed_tensor(model_path, tensor_name)
        expected_shape = (
            palu_builder.NUM_KV_HEADS * palu_builder.HEAD_DIM,
            palu_builder.HIDDEN_SIZE,
        )
        if not _check(
            tuple(dense.shape) == expected_shape,
            f"unexpected v_proj shape at layer {layer_index}: {tuple(dense.shape)}",
        ):
            return 2
        if args.method == "weight-svd":
            writer, decoder, diagnostics = factorize_weight_svd_projection(
                dense,
                ranks=ranks,
                output_dtype=torch.bfloat16,
            )
        else:
            whitening = cast(list[Tensor], whitening)
            writer, decoder, diagnostics = factorize_palu_projection(
                dense,
                whitening[layer_index],
                ranks=ranks,
                output_dtype=torch.bfloat16,
            )
        factor_payload[f"layers.{layer_index}.v_writer.weight"] = writer
        factor_payload[f"layers.{layer_index}.v_decoder.weight"] = decoder
        records.append(
            {
                "layer": layer_index,
                "ranks": list(ranks),
                "source_tensor": tensor_name,
                "source_dtype": str(dense.dtype),
                "writer_shape": list(writer.shape),
                "decoder_shape": list(decoder.shape),
                "factor_dtype": str(writer.dtype),
                **diagnostics,
                "elapsed_seconds": time.perf_counter() - layer_started,
            }
        )
        metric = (
            diagnostics["relative_frobenius_error"]
            if args.method == "weight-svd"
            else diagnostics["relative_activation_weighted_error"]
        )
        print(
            f"[{args.method}] layer={layer_index}/{palu_builder.NUM_LAYERS - 1} "
            f"group_rank={ranks[0]} error={metric:.6f}",
            flush=True,
        )

    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "v_factors.safetensors"
    palu_builder._atomic_safetensors(artifact_path, factor_payload)
    group_count = palu_builder.NUM_KV_HEADS // args.head_group_size
    method_label = (
        "Weight-SVD"
        if args.method == "weight-svd"
        else (
            "PaLU M-LRD"
            if args.head_group_size == 1
            else f"PaLU G-LRD{args.head_group_size}"
        )
    )
    realized_retained_ratio = rank_sum / total_rank
    compression = {
        "target": "value_cache_only",
        "key_cache": "dense",
        "value_cache": "low_rank",
        "projection": "v",
        "method": args.method,
        "method_label": method_label,
        "allocation": (
            "uniform_per_physical_head"
            if args.method == "weight-svd"
            else (
                "uniform_per_group"
                if args.method == "palu-uniform"
                else "official_palu_fisher_uniform"
            )
        ),
        "equivalent_rank_target": equivalent_rank,
        "nominal_retained_v_ratio": equivalent_rank / palu_builder.HEAD_DIM,
        "nominal_v_cache_compression_ratio": (
            1.0 - equivalent_rank / palu_builder.HEAD_DIM
        ),
        "rank_block_size": (
            FISHER_RANK_BLOCK_SIZE if args.method == "palu-fisher" else None
        ),
        "num_query_heads": palu_builder.NUM_QUERY_HEADS,
        "num_physical_kv_heads": palu_builder.NUM_KV_HEADS,
        "head_dim": palu_builder.HEAD_DIM,
        "head_group_size": args.head_group_size,
        "groups": group_count,
        "group_width": args.head_group_size * palu_builder.HEAD_DIM,
        "nominal_group_rank": args.head_group_size * equivalent_rank,
        "layer_ranks": layer_ranks,
        "rank_sum_across_layers": rank_sum,
        "dense_rank_sum_across_layers": total_rank,
        "realized_retained_v_ratio": realized_retained_ratio,
        "realized_v_cache_compression_ratio": 1.0 - realized_retained_ratio,
        "factorization_work_device": "cpu",
        "factorization_work_dtype": (
            "torch.float32" if args.method == "weight-svd" else "torch.float64"
        ),
        "stored_factor_dtype": "torch.bfloat16",
    }
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "status": "complete",
        "run_id": run_id(args.method, equivalent_rank, args.head_group_size),
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_metadata,
        "compression": compression,
        "artifact": {
            "file": artifact_path.name,
            "sha256": palu_builder._sha256(artifact_path),
            "tensor_count": len(factor_payload),
            "bytes": artifact_path.stat().st_size,
        },
        "layers": records,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    if args.method in ("palu-uniform", "palu-fisher"):
        whitening_dir = cast(Path, whitening_dir)
        whitening_manifest = cast(dict[str, Any], whitening_manifest)
        manifest["calibration"] = {
            "dataset": "allenai/c4",
            "samples": int(whitening_manifest["samples"]),
            "sequence_length": int(whitening_manifest["sequence_length"]),
            "whitening_manifest": str(whitening_dir / "manifest.json"),
            "whitening_manifest_sha256": palu_builder._sha256(
                whitening_dir / "manifest.json"
            ),
            "whitening_artifact_sha256": whitening_manifest["artifact"]["sha256"],
            "windows": whitening_manifest["windows"],
        }
    if args.method == "palu-fisher":
        fisher = cast(dict[str, Any], fisher)
        fisher_path = cast(Path, fisher_path)
        manifest["fisher"] = {
            "result": str(fisher_path),
            "result_sha256": palu_builder._sha256(fisher_path),
            "format": fisher["format"],
            "official_palu_commit": fisher["official_palu_commit"],
            "calibration": fisher["fisher"],
            "source_allocation_ignored": True,
        }
    palu_builder._atomic_json(output_dir / "manifest.json", manifest)
    print(
        f"[Result] run_id={manifest['run_id']} "
        f"realized_retained_v_ratio={realized_retained_ratio:.9f} "
        f"artifact={artifact_path}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("dense", "weight-svd", "palu-uniform", "palu-fisher"),
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--equivalent-rank", type=int)
    parser.add_argument("--head-group-size", type=int, default=1)
    parser.add_argument("--fisher-result")
    parser.add_argument("--whitening-dir")
    parser.add_argument("--torch-num-threads", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(build(parse_args()))
