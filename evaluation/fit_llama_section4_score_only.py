"""Fit the clean Score-only Section 4 B16R16 checkpoint."""

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
from evaluation.v96kl_common import configure, sha256, write_json


def score_statistics(stats_root, layer, split, *, smoke):
    indices = [0] if split == "fit" and smoke else [64] if smoke else (
        range(64) if split == "fit" else range(64, 80)
    )
    payloads = []
    records = []
    for index in indices:
        path = stats_root / ("smoke" if smoke else "") / "score" / f"l{layer:03d}" / f"w{index:03d}.safetensors"
        payload, record = verified(path)
        assert record["layer"] == layer and record["split"] == split and record["window_id"] == index
        assert record["protocol"]["fisher_artifacts_read"] == []
        payloads.append(payload)
        records.append(record)
    return load_fisher_windows(payloads, 32, 8, 128), records


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


def fit(args, *, smoke):
    layer = args.layer
    base, base_meta = verified(layer_file(args.root, "base", layer))
    identity_sha256 = sha256(args.root / "manifests/v128.json")
    assert base_meta["protocol"]["base_rank"] == 16 and base_meta["identity_sha256"] == identity_sha256
    fit_cpu, fit_records = score_statistics(args.statistics, layer, "fit", smoke=smoke)
    heldout_cpu, heldout_records = score_statistics(args.statistics, layer, "heldout", smoke=smoke)
    fit_stats = to_device(fit_cpu, torch.device("cuda"))
    heldout_stats = to_device(heldout_cpu, torch.device("cuda"))
    initial_encoder, initial_query = initialize(fit_stats, 16)
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
        active_joint_rows=torch.arange(128, device="cuda"),
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
    capture_protocol = fit_records[0]["protocol"]
    assert all(record["protocol"] == capture_protocol for record in fit_records + heldout_records)
    protocol = {
        "format": "basisserve.section4.score_only_b16r16.v1",
        "identity_sha256": identity_sha256,
        "model_config_sha256": capture_protocol["model_config_sha256"],
        "windows_sha256": capture_protocol["windows_sha256"],
        "sequence_length": capture_protocol["sequence_length"],
        "fit_ids": capture_protocol["fit_ids"],
        "heldout_ids": capture_protocol["heldout_ids"],
        "fit_queries": capture_protocol["fit_queries"],
        "heldout_queries": capture_protocol["heldout_queries"],
        "fit_positions": capture_protocol["fit_positions"],
        "heldout_positions": capture_protocol["heldout_positions"],
        "position_sampling": capture_protocol["position_sampling"],
        "value_mode": "dense original V and Wo",
        "base_rank": 16,
        "residual_rank": 16,
        "objective": "score_only",
        "objective_definition": "unweighted causal residual QK score squared error",
        "initialization": "top-16 eigenspace of the summed fit Score-MSE residual Grams; U0=E0",
        "fisher_artifacts_read": [],
        "base_sha256": base_meta["sha256"],
        "initialization_statistics_sha256": sorted(record["sha256"] for record in fit_records),
        "statistics_sha256": sorted(record["sha256"] for record in fit_records + heldout_records),
        "solver": {
            "sweeps": sweeps,
            "relative_damping": 1e-5,
            "relative_tolerance": 1e-5,
            "pcg_max_iterations": 100,
        },
        "smoke": smoke,
        "source_sha256": {
            "evaluation/fit_llama_section4_score_only.py": sha256(Path(__file__)),
            "evaluation/capture_llama_section4_score_only_statistics.py": sha256(
                Path("evaluation/capture_llama_section4_score_only_statistics.py")
            ),
            "basisserve/core/gqa_joint_routing_payload_s80_ablation.py": sha256(
                Path("basisserve/core/gqa_joint_routing_payload_s80_ablation.py")
            ),
        },
    }
    output = args.output / "smoke" if smoke else args.output
    save_record(
        output / "ours_b16r16" / f"layer_{layer:03d}.safetensors",
        {
            "base_left_b16": base["left"].float(),
            "base_right_b16": base["right"].float(),
            "base_bias_b16": base["bias"].float(),
            "residual_encoder_b16_r16": result.routing_encoders.float().cpu(),
            "residual_query_b16_r16": result.routing_query_factors.float().cpu(),
        },
        {
            "protocol": protocol,
            "layer": layer,
            "v_rank": 128,
            "identity_sha256": identity_sha256,
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
        {"objective": "score_only", "layer": layer, "initial": initial_fit, "fit": fit_loss,
         "heldout": heldout_loss, "seconds": wall_time},
        flush=True,
    )


def audit(args):
    hashes = {}
    wall = 0.0
    for layer in range(32):
        path = args.output / "ours_b16r16" / f"layer_{layer:03d}.safetensors"
        _, meta = verified(path)
        assert meta["layer"] == layer and meta["protocol"]["objective"] == "score_only"
        assert meta["protocol"]["fisher_artifacts_read"] == []
        assert meta["sweeps"] == 40 and meta["pcg_iterations"] == 100
        assert meta["losses"]["fit"] <= meta["losses"]["initial_fit"] * 1.00001
        hashes[str(layer)] = meta["sha256"]
        wall += meta["fit_wall_time_seconds"]
    write_json(
        args.output / "fit_audit.json",
        {"status": "complete", "objective": "score_only", "bank_sha256": hashes,
         "fit_wall_time_seconds": wall},
    )
    print({"objective": "score_only", "layers": 32, "fit_wall_time_seconds": wall}, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("smoke", "fit", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
