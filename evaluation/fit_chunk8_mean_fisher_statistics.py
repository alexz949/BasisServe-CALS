"""Fit Mean-128 Chunk8 landmarks from exact compact Fisher statistics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file

from basisserve.core.chunk_fisher_landmark import (
    CompactChunkFisherDataset,
    adapt_token_residual_initialization,
    compact_chunk_fisher_metrics,
    fit_compact_chunk_fisher_landmarks,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    unpack_symmetric_fisher_grams,
)
from evaluation.fit_k_routing_streaming import verified
from evaluation.v96kl_common import configure, read_json, save_tensors, sha256, write_json


FORMAT = "basisserve.chunk8_mean_fisher.fit.v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--statistics", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--layer", type=int, required=True)
    result.add_argument("--sweeps", type=int, default=8)
    result.add_argument("--relative-damping", type=float, default=1e-5)
    result.add_argument("--cg-tolerance", type=float, default=1e-5)
    result.add_argument("--cg-iterations", type=int, default=50)
    result.add_argument("--smoke", action="store_true")
    return result


def load_split(
    statistics_root: Path,
    record: dict,
    split: str,
) -> CompactChunkFisherDataset:
    queries = []
    grams = []
    crosses = []
    energy = 0.0
    artifacts = [item for item in record["windows"] if item["split"] == split]
    assert artifacts
    for item in artifacts:
        path = statistics_root / item["file"]
        assert sha256(path) == item["sha256"]
        payload = load_file(str(path))
        query = payload["queries_by_head"].float()
        gram = unpack_symmetric_fisher_grams(
            payload["fisher_grams_packed_by_head"],
            dimension=128,
        )
        cross = payload["target_cross_by_head"].float()
        assert query.shape[:2] == gram.shape[:2] == cross.shape[:2]
        queries.append(query)
        grams.append(gram)
        crosses.append(cross)
        energy += float(payload["target_fisher_energy"])
    mapping = torch.arange(32, dtype=torch.long) // 4
    result = CompactChunkFisherDataset(
        queries_by_head=torch.cat(queries, dim=1),
        fisher_grams_by_head=torch.cat(grams, dim=1),
        target_cross_by_head=torch.cat(crosses, dim=1),
        head_to_group=mapping,
        scaling=128**-0.5,
        target_fisher_energy=energy,
    )
    result.validate()
    return result


def dataset_to(
    dataset: CompactChunkFisherDataset,
    device: torch.device,
) -> CompactChunkFisherDataset:
    result = CompactChunkFisherDataset(
        queries_by_head=dataset.queries_by_head.to(device=device, dtype=torch.float32),
        fisher_grams_by_head=dataset.fisher_grams_by_head.to(
            device=device,
            dtype=torch.float32,
        ),
        target_cross_by_head=dataset.target_cross_by_head.to(
            device=device,
            dtype=torch.float32,
        ),
        head_to_group=dataset.head_to_group.to(device),
        scaling=dataset.scaling,
        target_fisher_energy=dataset.target_fisher_energy,
    )
    result.validate()
    return result


def markdown(result: dict) -> str:
    initial = result["metrics"]["initial"]
    final = result["metrics"]["final"]
    lines = [
        f"# Layer {result['protocol']['layer']} Mean-128 Chunk8 Fisher Fit",
        "",
        "| Split | Fisher NMSE init | Fisher NMSE final |",
        "|---|---:|---:|",
        f"| fit | {initial['fit']['fisher_nmse']:.6g} | {final['fit']['fisher_nmse']:.6g} |",
        f"| heldout | {initial['heldout']['fisher_nmse']:.6g} | {final['heldout']['fisher_nmse']:.6g} |",
        "",
        "| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for sweep in result["sweeps"]:
        query = sweep["query_cg"]
        encoder = sweep["encoder_cg"]
        lines.append(
            f"| {sweep['sweep']} | {sweep['train_fisher_nmse']:.6g} | "
            f"{sweep['heldout_fisher_nmse']:.6g} | "
            f"{query['mean_iterations']:.1f}/{query['maximum_iterations']}/"
            f"{query['hit_iteration_limit_count']} | "
            f"{encoder['mean_iterations']:.1f}/{encoder['maximum_iterations']}/"
            f"{encoder['hit_iteration_limit_count']} | {sweep['wall_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parser().parse_args()
    configure()
    assert 0 <= args.layer < 32
    assert args.sweeps == 8
    assert args.cg_iterations > 0
    record_path = args.statistics / f"layer_{args.layer:03d}.json"
    capture = read_json(record_path)
    protocol = capture["protocol"]
    assert capture["status"] == "complete" and capture["layer"] == args.layer
    assert protocol["format"] == "basisserve.chunk8_mean_fisher.compact.v1"
    assert protocol["scope"] == ("smoke" if args.smoke else "formal")
    assert protocol["feature"] == "mean" and protocol["feature_dim"] == 128
    assert protocol["root"] == str(args.root)

    factor_path = args.root / "ours_b16r16" / f"layer_{args.layer:03d}.safetensors"
    factors, factor_record = verified(factor_path)
    assert factor_record["v_rank"] == 128
    assert capture["factor_source_sha256"] == sha256(factor_path)

    load_start = time.perf_counter()
    train = load_split(args.statistics, capture, "fit")
    heldout = load_split(args.statistics, capture, "heldout")
    load_wall_seconds = time.perf_counter() - load_start
    device = torch.device("cuda")
    train = dataset_to(train, device)
    heldout = dataset_to(heldout, device)
    initial_encoder, initial_query = adapt_token_residual_initialization(
        factors["residual_encoder_b16_r16"],
        factors["residual_query_b16_r16"],
        chunk_size=8,
        feature="mean",
    )
    initial_encoder = initial_encoder.to(device=device, dtype=torch.float32)
    initial_query = initial_query.to(device=device, dtype=torch.float32)
    initial = {
        "fit": compact_chunk_fisher_metrics(train, initial_encoder, initial_query),
        "heldout": compact_chunk_fisher_metrics(
            heldout,
            initial_encoder,
            initial_query,
        ),
    }
    torch.cuda.synchronize()
    fit_start = time.perf_counter()
    fitted = fit_compact_chunk_fisher_landmarks(
        train,
        heldout,
        initial_encoders=initial_encoder,
        initial_query_factors=initial_query,
        sweeps=args.sweeps,
        relative_damping=args.relative_damping,
        relative_tolerance=args.cg_tolerance,
        max_iterations=args.cg_iterations,
    )
    torch.cuda.synchronize()
    fit_wall_seconds = time.perf_counter() - fit_start
    final = {
        "fit": compact_chunk_fisher_metrics(train, fitted.encoders, fitted.query_factors),
        "heldout": compact_chunk_fisher_metrics(
            heldout,
            fitted.encoders,
            fitted.query_factors,
        ),
    }
    tensors = {
        "chunk_encoder_r16": fitted.encoders.float().cpu().contiguous(),
        "chunk_query_r16": fitted.query_factors.float().cpu().contiguous(),
    }
    tensor_path = args.output / f"layer_{args.layer:03d}_mean.safetensors"
    save_tensors(tensor_path, tensors)

    half_steps = [asdict(item) for item in fitted.half_steps]
    sweeps = []
    for sweep in range(1, args.sweeps + 1):
        query_step = next(
            item
            for item in half_steps
            if item["sweep"] == sweep and item["boundary"] == "query"
        )
        encoder_step = next(
            item
            for item in half_steps
            if item["sweep"] == sweep and item["boundary"] == "encoder"
        )
        cg_keys = (
            "solver_count",
            "converged_count",
            "hit_iteration_limit_count",
            "total_iterations",
            "mean_iterations",
            "maximum_iterations",
        )
        sweeps.append(
            {
                "sweep": sweep,
                "train_fisher_nmse": encoder_step["train"]["fisher_nmse"],
                "heldout_fisher_nmse": encoder_step["heldout"]["fisher_nmse"],
                "query_cg": {key: query_step[key] for key in cg_keys},
                "encoder_cg": {key: encoder_step[key] for key in cg_keys},
                "wall_seconds": query_step["total_wall_seconds"]
                + encoder_step["total_wall_seconds"],
            }
        )

    result = {
        "status": "complete",
        "protocol": {
            "format": FORMAT,
            "scope": "smoke" if args.smoke else "formal all-layer Mean-only fit",
            "model": protocol["model"],
            "model_variant": protocol["model_variant"],
            "layer": args.layer,
            "feature": "mean",
            "feature_dim": 128,
            "residual_rank": 16,
            "als_sweeps": args.sweeps,
            "final_query_closure": True,
            "relative_damping": args.relative_damping,
            "cg_tolerance": args.cg_tolerance,
            "cg_iterations": args.cg_iterations,
            "objective": "exact fixed-teacher direct Chunk8 softmax-Fisher over routed historical candidates",
            "candidate_mask": "exclude fixed sink32 and exact recent64",
            "statistics_source": str(record_path),
            "statistics_source_sha256": sha256(record_path),
            "factor_source": str(factor_path),
            "factor_source_sha256": sha256(factor_path),
            "source_sha256": {
                name: sha256(Path(name))
                for name in (
                    "basisserve/core/chunk_fisher_landmark.py",
                    "evaluation/fit_chunk8_mean_fisher_statistics.py",
                )
            },
        },
        "audits": {
            "capture_is_mean_only": protocol["feature"] == "mean",
            "statistics_are_exact": protocol["statistics"].startswith("exact"),
            "flat_not_captured_or_fitted": True,
            "query_positions_reuse_audited_query_gram_selection": True,
            "sink_chunks_excluded_from_fisher": True,
            "recent64_excluded_from_fisher": True,
            "base_is_frozen": True,
            "final_query_closure": True,
        },
        "metrics": {"initial": initial, "final": final},
        "half_steps": half_steps,
        "sweeps": sweeps,
        "timing": {
            "statistics_load_wall_seconds": load_wall_seconds,
            "fit_wall_seconds_including_diagnostics_and_final_closure": fit_wall_seconds,
        },
        "artifact": {
            "file": tensor_path.name,
            "sha256": sha256(tensor_path),
            "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
        },
        "command": shlex.join(sys.argv),
        "python": sys.executable,
    }
    record_output = args.output / f"layer_{args.layer:03d}.json"
    write_json(record_output, result)
    text = markdown(result)
    markdown_path = args.output / f"layer_{args.layer:03d}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
