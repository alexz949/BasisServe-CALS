"""Build the Dense-V Base-only banks for the Section 4 rank sweep."""

import argparse
import csv
from pathlib import Path

import torch

from evaluation.fit_k_routing_streaming import encoder_for, layer_file, save_record, verified
from evaluation.streaming_k_statistics import base_from_moments, base_mse
from evaluation.v96kl_common import configure, read_json, sha256, write_json


RANKS = (4, 8, 16, 24, 32, 48, 64, 80, 96)


def centered_energy(moments):
    count = int(moments["count"])
    centered = moments["kk"] - torch.einsum(
        "gd,ge->gde", moments["sum_k"], moments["sum_k"]
    ) / count
    value = float(centered.diagonal(dim1=-2, dim2=-1).sum())
    assert value > 0
    return value


def split_moments(payload):
    return {
        split: {
            key.removeprefix(split + "_"): value
            for key, value in payload.items()
            if key.startswith(split + "_")
        }
        for split in ("fit", "heldout")
    }


def protocol(root, output, identity, moments_meta, *, smoke):
    return {
        "format": "basisserve.section4.base_only.v1",
        "model_config_sha256": identity["model_config_sha256"],
        "identity_sha256": sha256(root / "manifests/v128.json"),
        "moments_protocol": moments_meta["protocol"],
        "moments_root": str(root / "moments"),
        "value_mode": "dense original V and Wo",
        "ranks": list(RANKS),
        "residual_rank": 0,
        "smoke": smoke,
        "source_sha256": {
            "evaluation/build_llama_section4_base_sweep.py": sha256(Path(__file__)),
            "evaluation/streaming_k_statistics.py": sha256(Path("evaluation/streaming_k_statistics.py")),
            "basisserve/core/c1_v_conditional_k_router.py": sha256(
                Path("basisserve/core/c1_v_conditional_k_router.py")
            ),
        },
        "output": str(output),
    }


def build_layer(args, layer, *, smoke):
    root = args.root
    output = args.output / "smoke" if smoke else args.output
    identity = read_json(root / "manifests/v128.json")
    assert identity["status"] == "complete" and identity["layer_ranks"] == [128] * 32
    moments_path = layer_file(root, "moments", layer)
    payload, moments_meta = verified(moments_path)
    splits = split_moments(payload)
    encoder = encoder_for(identity, layer)
    spec = protocol(root, output, identity, moments_meta, smoke=smoke)
    full = base_from_moments(splits["fit"], encoder, rank=128)[128]
    full_metrics = {split: base_mse(moment, encoder, full) for split, moment in splits.items()}
    centered = {split: centered_energy(moment) for split, moment in splits.items()}
    full_captured = {
        split: centered[split] - full_metrics[split]["squared_error"]
        for split in splits
    }
    assert all(value > 0 for value in full_captured.values())
    selected = RANKS
    for rank in selected:
        bases = base_from_moments(splits["fit"], encoder, rank=rank)[rank]
        metrics = {}
        for split, moment in splits.items():
            raw = base_mse(moment, encoder, bases)
            error = raw["squared_error"]
            captured = centered[split] - error
            metrics[split] = {
                **raw,
                "centered_key_energy": centered[split],
                "centered_relative_mse": error / centered[split],
                "centered_explained_fraction": captured / centered[split],
                "full_rank_predictable_energy": full_captured[split],
                "predictable_energy_captured_fraction": captured / full_captured[split],
            }
        tensors = {
            f"base_{name}_b{rank}": torch.stack([getattr(base, name) for base in bases]).float()
            for name in ("left", "right", "bias")
        }
        tensors[f"residual_encoder_b{rank}_r0"] = torch.empty(8, 128, 0)
        tensors[f"residual_query_b{rank}_r0"] = torch.empty(32, 128, 0)
        path = output / f"b{rank}r0" / f"layer_{layer:03d}.safetensors"
        save_record(
            path,
            tensors,
            {
                "protocol": {
                    **moments_meta["protocol"],
                    "base_rank": rank,
                    "residual_rank": 0,
                    "objective": "affine pre-RoPE K reduced-rank regression",
                    "section4_protocol": spec,
                    "smoke": smoke,
                },
                "layer": layer,
                "v_rank": 128,
                "identity_sha256": sha256(root / "manifests/v128.json"),
                "moments_sha256": moments_meta["sha256"],
                "metrics": metrics,
                "sweeps": 0,
                "pcg_iterations": 0,
            },
        )
        print({"layer": layer, "rank": rank, "heldout": metrics["heldout"]}, flush=True)


def summarize(args):
    records = []
    for rank in RANKS:
        for layer in range(32):
            path = args.output / f"b{rank}r0" / f"layer_{layer:03d}.safetensors"
            _, meta = verified(path)
            assert meta["layer"] == layer and meta["protocol"]["base_rank"] == rank
            records.append(
                {
                    "base_rank": rank,
                    "layer": layer,
                    **meta["metrics"]["heldout"],
                    "factor_checkpoint": str(path),
                }
            )
    summary = {}
    for rank in RANKS:
        rows = [row for row in records if row["base_rank"] == rank]
        summary[str(rank)] = {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in (
                "centered_relative_mse",
                "centered_explained_fraction",
                "predictable_energy_captured_fraction",
            )
        }
    write_json(args.output / "fit_summary.json", {"status": "complete", "ranks": summary})
    csv_path = args.repo_output / "base_rank_sweep_fit_layers.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("smoke", "build", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-output", type=Path, default=Path("results/section4_ablation"))
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    configure()
    if args.stage == "summarize":
        summarize(args)
    else:
        assert 0 <= args.layer < 32
        build_layer(args, args.layer, smoke=args.stage == "smoke")


if __name__ == "__main__":
    main()
