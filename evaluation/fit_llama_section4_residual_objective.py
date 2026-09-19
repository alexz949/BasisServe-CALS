"""Fit the Section 4 Residual-MSE and Score-MSE B16R16 controls."""

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch

from basisserve.core.gqa_joint_routing_payload_s80_ablation import fit_page_fisher_router
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    S80CompactSoftmaxFisherRouting,
    compact_softmax_fisher_loss,
)
from evaluation.fit_k_routing_streaming import layer_file, save_record, verified
from evaluation.streaming_k_statistics import load_fisher_windows
from evaluation.v96kl_common import configure, read_json, sha256, write_json


OBJECTIVES = ("residual_mse", "score_mse")


def score_statistics(stats_root, layer, split, *, smoke):
    indices = [0] if split == "fit" and smoke else [64] if smoke else (
        range(64) if split == "fit" else range(64, 80)
    )
    payloads = []
    records = []
    for index in indices:
        path = stats_root / ("smoke" if smoke else "") / "score_mse" / f"l{layer:03d}" / f"w{index:03d}.safetensors"
        payload, record = verified(path)
        assert record["layer"] == layer and record["split"] == split and record["window_id"] == index
        payloads.append(payload)
        records.append(record)
    result = load_fisher_windows(payloads, 32, 8, 128)
    return result, records


def residual_statistics(stats_root, layer, split, *, smoke):
    path = stats_root / ("smoke" if smoke else "") / "residual_mse" / f"layer_{layer:03d}.safetensors"
    payload, record = verified(path)
    covariance = payload[f"{split}_covariance"].float()
    mapping = torch.arange(32) // 4
    by_head = covariance.index_select(0, mapping)
    queries = torch.eye(128).expand(32, -1, -1).contiguous()
    grams = by_head[:, None].expand(-1, 128, -1, -1).contiguous()
    scale = 128 ** -0.5
    energy = 0.5 * scale**2 * float(by_head.diagonal(dim1=-2, dim2=-1).sum())
    assert energy > 0
    return S80CompactSoftmaxFisherRouting(
        queries_by_head=queries,
        fisher_grams_by_head=grams,
        head_to_kv_group=mapping,
        value_dim=0,
        key_dim=128,
        scaling=scale,
        teacher_fisher_energy=energy,
    ), [record]


def to_device(statistics, device):
    return S80CompactSoftmaxFisherRouting(
        queries_by_head=statistics.queries_by_head.to(device),
        fisher_grams_by_head=statistics.fisher_grams_by_head.to(device),
        head_to_kv_group=statistics.head_to_kv_group.to(device),
        value_dim=0,
        key_dim=statistics.key_dim,
        scaling=statistics.scaling,
        teacher_fisher_energy=statistics.teacher_fisher_energy,
    )


def initialize(statistics, rank):
    mapping = statistics.head_to_kv_group
    encoders = []
    for group in range(8):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        gram = statistics.fisher_grams_by_head.index_select(0, heads).sum(dim=(0, 1))
        _, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.mT))
        encoders.append(eigenvectors[:, -rank:])
    encoder = torch.stack(encoders)
    return encoder, encoder.index_select(0, mapping)


def page_fisher_initialization(root, layer, *, smoke):
    indices = [0] if smoke else range(64)
    payloads = []
    hashes = []
    for index in indices:
        path = root / "fisher" / f"l{layer:03d}" / f"w{index:03d}.safetensors"
        payload, record = verified(path)
        assert record["layer"] == layer and record["split"] == "fit" and record["window_id"] == index
        payloads.append(payload)
        hashes.append(record["sha256"])
    statistics = to_device(load_fisher_windows(payloads, 32, 8, 128), torch.device("cuda"))
    encoder, query = initialize(statistics, 16)
    return encoder, query, hashes


def fit(args, *, smoke):
    layer = args.layer
    root = args.root
    stats_root = args.statistics
    base_path = layer_file(root, "base", layer)
    base, base_meta = verified(base_path)
    assert base_meta["protocol"]["base_rank"] == 16 and base_meta["identity_sha256"] == sha256(root / "manifests/v128.json")
    loader = residual_statistics if args.objective == "residual_mse" else score_statistics
    fit_cpu, fit_records = loader(stats_root, layer, "fit", smoke=smoke)
    heldout_cpu, heldout_records = loader(stats_root, layer, "heldout", smoke=smoke)
    device = torch.device("cuda")
    fit_stats = to_device(fit_cpu, device)
    heldout_stats = to_device(heldout_cpu, device)
    initial_encoder, initial_query, initialization_hashes = page_fisher_initialization(
        root, layer, smoke=smoke
    )
    initial_fit = compact_softmax_fisher_loss(
        fit_stats,
        routing_payload_encoders=initial_encoder,
        routing_query_factors=initial_query,
    )
    sweeps = 2 if smoke else 40
    started = time.monotonic()
    result = fit_page_fisher_router(
        fit_stats,
        initial_routing_encoders=initial_encoder,
        initial_query_factors=initial_query,
        active_joint_rows=torch.arange(128, device=device),
        sweeps=sweeps,
        relative_damping=1e-5,
        relative_tolerance=1e-5,
        max_iterations=100,
    )
    wall_time = time.monotonic() - started
    fit_loss = compact_softmax_fisher_loss(
        fit_stats,
        routing_payload_encoders=result.routing_encoders,
        routing_query_factors=result.routing_query_factors,
    )
    heldout_loss = compact_softmax_fisher_loss(
        heldout_stats,
        routing_payload_encoders=result.routing_encoders,
        routing_query_factors=result.routing_query_factors,
    )
    assert fit_loss <= initial_fit * 1.00001
    tensors = {
        "base_left_b16": base["left"].float(),
        "base_right_b16": base["right"].float(),
        "base_bias_b16": base["bias"].float(),
        "residual_encoder_b16_r16": result.routing_encoders.float().cpu(),
        "residual_query_b16_r16": result.routing_query_factors.float().cpu(),
    }
    protocol = {
        **base_meta["protocol"],
        "residual_rank": 16,
        "objective": args.objective,
        "objective_definition": (
            "ordinary Euclidean post-RoPE residual reconstruction, repeated across associated query heads"
            if args.objective == "residual_mse"
            else "unweighted causal residual QK score squared error on the frozen Page-Fisher query sample"
        ),
        "initialization": "same top-16 eigenspace of summed fit Page-Fisher Grams and U0=E0 used by Page-Fisher",
        "initialization_statistics_sha256": initialization_hashes,
        "statistics_sha256": sorted({record["sha256"] for record in fit_records + heldout_records}),
        "smoke": smoke,
        "source_sha256": {
            "evaluation/fit_llama_section4_residual_objective.py": sha256(Path(__file__)),
            "basisserve/core/gqa_joint_routing_payload_s80_ablation.py": sha256(
                Path("basisserve/core/gqa_joint_routing_payload_s80_ablation.py")
            ),
        },
    }
    output = args.output / "smoke" if smoke else args.output
    path = output / "objectives" / args.objective / "ours_b16r16" / f"layer_{layer:03d}.safetensors"
    save_record(
        path,
        tensors,
        {
            "protocol": protocol,
            "layer": layer,
            "v_rank": 128,
            "identity_sha256": sha256(root / "manifests/v128.json"),
            "base_sha256": base_meta["sha256"],
            "sweeps": sweeps,
            "pcg_iterations": 100,
            "fit_wall_time_seconds": wall_time,
            "losses": {
                "initial_fit": initial_fit,
                "fit": fit_loss,
                "heldout": heldout_loss,
                "fit_normalized": fit_loss / fit_stats.teacher_fisher_energy,
                "heldout_normalized": heldout_loss / heldout_stats.teacher_fisher_energy,
                "sweeps": [asdict(item) for item in result.sweeps],
            },
        },
    )
    print(
        {
            "objective": args.objective,
            "layer": layer,
            "initial": initial_fit,
            "fit": fit_loss,
            "heldout": heldout_loss,
            "seconds": wall_time,
        },
        flush=True,
    )


def audit(args):
    hashes = {}
    wall = 0.0
    for layer in range(32):
        path = args.output / "objectives" / args.objective / "ours_b16r16" / f"layer_{layer:03d}.safetensors"
        _, meta = verified(path)
        assert meta["layer"] == layer and meta["protocol"]["objective"] == args.objective
        assert meta["sweeps"] == 40 and meta["pcg_iterations"] == 100
        assert meta["losses"]["fit"] <= meta["losses"]["initial_fit"] * 1.00001
        hashes[str(layer)] = meta["sha256"]
        wall += meta["fit_wall_time_seconds"]
    write_json(
        args.output / "objectives" / args.objective / "fit_audit.json",
        {"status": "complete", "objective": args.objective, "bank_sha256": hashes, "fit_wall_time_seconds": wall},
    )
    print({"objective": args.objective, "layers": 32, "fit_wall_time_seconds": wall}, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("smoke", "fit", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, required=True)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    configure()
    assert 0 <= args.layer < 32
    if args.stage == "audit":
        audit(args)
    else:
        fit(args, smoke=args.stage == "smoke")


if __name__ == "__main__":
    main()
