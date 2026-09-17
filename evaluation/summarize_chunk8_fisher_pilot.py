"""Summarize the early/middle/late direct Chunk8 Fisher convergence pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.v96kl_common import read_json, sha256, write_json


LAYERS = (0, 15, 31)
FEATURES = ("mean", "flat")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fit-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def markdown(result: dict) -> str:
    lines = [
        "# Chunk8 Fisher ALS Convergence Pilot",
        "",
        "Llama-3.1-8B-Instruct layers 0, 15, and 31; Dense V128; frozen Base16; direct Chunk8 R16; 64×64K fit and 16×64K held-out windows; PCG capped at 50 iterations.",
        "",
        "| Layer | Feature | Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in LAYERS:
        record = result["records"][str(layer)]
        for feature in FEATURES:
            for sweep in record["features"][feature]["sweeps"]:
                query = sweep["query_cg"]
                encoder = sweep["encoder_cg"]
                lines.append(
                    f"| {layer} | {feature} | {sweep['sweep']} | "
                    f"{sweep['train_fisher_nmse']:.6g} | "
                    f"{sweep['heldout_fisher_nmse']:.6g} | "
                    f"{query['mean_iterations']:.1f}/{query['maximum_iterations']}/"
                    f"{query['hit_iteration_limit_count']} | "
                    f"{encoder['mean_iterations']:.1f}/{encoder['maximum_iterations']}/"
                    f"{encoder['hit_iteration_limit_count']} | "
                    f"{sweep['wall_seconds']:.2f} |"
                )
    lines.extend(
        [
            "",
            "`hit` is the number of independent block solves that reached the 50-iteration cap; it is diagnostic, not a failure condition. The reported sweep endpoint is after the encoder half-step. A final query closure is fitted and saved separately from the four sweep endpoints.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parser().parse_args()
    records = {}
    artifacts = {}
    for layer in LAYERS:
        path = args.fit_root / f"layer_{layer:03d}.json"
        record = read_json(path)
        assert record["status"] == "complete"
        assert record["protocol"]["scope"] == "three-layer convergence pilot"
        assert record["protocol"]["layer"] == layer
        assert record["protocol"]["cg_iterations"] == 50
        assert record["protocol"]["als_sweeps"] == 4
        assert all(record["audits"].values())
        records[str(layer)] = record
        artifacts[str(layer)] = {"file": path.name, "sha256": sha256(path)}
    result = {
        "status": "complete",
        "layers": list(LAYERS),
        "features": list(FEATURES),
        "records": records,
        "artifacts": artifacts,
    }
    write_json(args.output / "pilot_summary.json", result)
    text = markdown(result)
    output = args.output / "summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
