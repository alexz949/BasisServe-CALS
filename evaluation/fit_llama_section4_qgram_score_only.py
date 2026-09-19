"""Fit Fisher-free Score-MSE using the existing Query-Gram position coreset."""

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch

from basisserve.core.gqa_joint_routing_payload_s80_ablation import fit_page_fisher_router
from basisserve.core.gqa_joint_routing_payload_s80_fisher import compact_softmax_fisher_loss
from evaluation.fit_k_routing_streaming import layer_file, save_record, verified
from evaluation.fit_llama_section4_residual_objective import score_statistics
from evaluation.fit_llama_section4_score_only import initialize, to_device
from evaluation.v96kl_common import configure, sha256, write_json


def fit(args, *, smoke):
    layer = args.layer
    identity_sha256 = sha256(args.root / "manifests/v128.json")
    base, base_meta = verified(layer_file(args.root, "base", layer))
    assert base_meta["protocol"]["base_rank"] == 16
    assert base_meta["identity_sha256"] == identity_sha256

    fit_cpu, fit_records = score_statistics(args.statistics, layer, "fit", smoke=smoke)
    heldout_cpu, heldout_records = score_statistics(args.statistics, layer, "heldout", smoke=smoke)
    selection_payload, selection_meta = verified(layer_file(args.root, "moments", layer))
    assert selection_payload
    selections = selection_meta["selections"]
    assert selection_meta["selection_fit_only"]
    assert len(selections["fit"]["selected_positions"]) == 64
    assert len(selections["heldout"]["selected_positions"]) == 32
    assert all(len(item["bins"]) == 4 for item in selections.values())
    for record in fit_records + heldout_records:
        protocol = record["protocol"]
        assert protocol["score_mse"] == (
            "unweighted residual QK score squared error; no softmax or Page-Fisher weights"
        )

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
    statistics_protocol = fit_records[0]["protocol"]
    assert all(record["protocol"] == statistics_protocol for record in fit_records + heldout_records)
    protocol = {
        "format": "basisserve.section4.qgram_score_only_b16r16.v1",
        "identity_sha256": identity_sha256,
        "model_config_sha256": statistics_protocol["model_config_sha256"],
        "windows_sha256": statistics_protocol["windows_sha256"],
        "sequence_length": statistics_protocol["sequence_length"],
        "fit_ids": statistics_protocol["fit_ids"],
        "heldout_ids": statistics_protocol["heldout_ids"],
        "fit_queries": statistics_protocol["fit_queries"],
        "heldout_queries": statistics_protocol["heldout_queries"],
        "position_selection": {
            "method": "deterministic stratified pivots of uncentered head-whitened Query Grams",
            "candidate_stride": 64,
            "excluded_query_prefix": 32,
            "length_bins": 4,
            "fit_queries_per_bin": 16,
            "heldout_queries_per_bin": 8,
            "selection_fit_only": True,
            "fit_positions": selections["fit"]["selected_positions"],
            "heldout_positions": selections["heldout"]["selected_positions"],
            "moments_sha256": selection_meta["sha256"],
        },
        "value_mode": "dense original V and Wo",
        "base_rank": 16,
        "residual_rank": 16,
        "objective": "qgram_score_only",
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
            "evaluation/fit_llama_section4_qgram_score_only.py": sha256(Path(__file__)),
            "evaluation/fit_llama_section4_residual_objective.py": sha256(
                Path("evaluation/fit_llama_section4_residual_objective.py")
            ),
            "basisserve/core/query_position_sampling.py": sha256(
                Path("basisserve/core/query_position_sampling.py")
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
        {
            "objective": "qgram_score_only",
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
        path = args.output / "ours_b16r16" / f"layer_{layer:03d}.safetensors"
        _, meta = verified(path)
        assert meta["layer"] == layer
        assert meta["protocol"]["objective"] == "qgram_score_only"
        assert meta["protocol"]["fisher_artifacts_read"] == []
        assert meta["sweeps"] == 40 and meta["pcg_iterations"] == 100
        assert meta["losses"]["fit"] <= meta["losses"]["initial_fit"] * 1.00001
        hashes[str(layer)] = meta["sha256"]
        wall += meta["fit_wall_time_seconds"]
    write_json(
        args.output / "fit_audit.json",
        {
            "status": "complete",
            "objective": "qgram_score_only",
            "bank_sha256": hashes,
            "fit_wall_time_seconds": wall,
        },
    )
    print({"objective": "qgram_score_only", "layers": 32, "fit_wall_time_seconds": wall}, flush=True)


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
