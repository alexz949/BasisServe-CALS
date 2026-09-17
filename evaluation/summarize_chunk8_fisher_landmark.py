"""Summarize all 32 formal direct Chunk8 Fisher layer fits."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics

from evaluation.v96kl_common import read_json, sha256, write_json


FEATURES = ("mean", "flat")
SPLITS = ("fit", "heldout")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fit-root", type=Path, required=True)
    result.add_argument("--geometry-result", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def aggregate(records: list[dict]) -> dict:
    summary = {}
    for feature in FEATURES:
        summary[feature] = {}
        for split in SPLITS:
            initial_loss = sum(
                record["features"][feature]["initial"][split]["fisher_loss"]
                for record in records
            )
            final_loss = sum(
                record["features"][feature]["final"][split]["fisher_loss"]
                for record in records
            )
            energy = sum(
                record["features"][feature]["final"][split]["base_fisher_energy"]
                for record in records
            )
            summary[feature][split] = {
                "initial_fisher_nmse_pooled": initial_loss / energy,
                "final_fisher_nmse_pooled": final_loss / energy,
                "final_chunk_logit_rel_mse_layer_mean": statistics.fmean(
                    record["features"][feature]["final"][split][
                        "chunk_logit_rel_mse"
                    ]
                    for record in records
                ),
                "final_exact_chunk_support_recall_layer_mean": statistics.fmean(
                    record["features"][feature]["final"][split][
                        "exact_chunk_support_recall"
                    ]
                    for record in records
                ),
                "final_routed_candidate_attention_mass_layer_mean": statistics.fmean(
                    record["features"][feature]["final"][split][
                        "routed_candidate_attention_mass"
                    ]
                    for record in records
                ),
            }
    return summary


def markdown(result: dict) -> str:
    lines = [
        "# Llama-3.1-8B-Instruct Direct Chunk8 Fisher Fit",
        "",
        "All 32 layers use frozen Base16, direct Chunk8 Fisher R16, 64×64K fit windows, 16×64K held-out windows, four ALS sweeps, and a final query closure solve.",
        "",
        "| Feature | Split | Fisher NMSE init | Fisher NMSE final | Layer-mean chunk rel-MSE | Exact support recall | Routed-candidate mass |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for feature in FEATURES:
        for split in SPLITS:
            metric = result["summary"][feature][split]
            lines.append(
                f"| {feature} | {split} | "
                f"{metric['initial_fisher_nmse_pooled']:.6g} | "
                f"{metric['final_fisher_nmse_pooled']:.6g} | "
                f"{metric['final_chunk_logit_rel_mse_layer_mean']:.6g} | "
                f"{metric['final_exact_chunk_support_recall_layer_mean']:.4f} | "
                f"{metric['final_routed_candidate_attention_mass_layer_mean']:.4f} |"
            )
    lines.extend(
        [
            "",
            "The fitting objective excludes the four fixed sink chunks and exact recent64. Both variants store one Base128 landmark plus one learned R16 residual code per Chunk8; token-level R16 is not part of deployment.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parser().parse_args()
    records = []
    artifacts = {}
    common = None
    for layer in range(32):
        path = args.fit_root / f"layer_{layer:03d}.json"
        record = read_json(path)
        assert record["status"] == "complete"
        assert record["protocol"]["scope"] == "formal 64-fit/16-heldout fit"
        assert record["protocol"]["layer"] == layer
        assert all(record["audits"].values())
        shared = {
            key: value
            for key, value in record["protocol"].items()
            if key
            not in (
                "layer",
                "factor_source",
                "factor_source_sha256",
                "moments_source",
                "moments_source_sha256",
            )
        }
        common = shared if common is None else common
        assert shared == common
        records.append(record)
        artifacts[str(layer)] = {
            "record": path.name,
            "record_sha256": sha256(path),
            "factors": record["artifacts"],
        }
    geometry = read_json(args.geometry_result)
    assert geometry["status"] == "complete"
    assert geometry["audits"]["all_runtime_audits_passed"]
    result = {
        "status": "complete",
        "protocol": common,
        "layers": 32,
        "summary": aggregate(records),
        "artifacts": artifacts,
        "geometry_reference": {
            "file": str(args.geometry_result),
            "sha256": sha256(args.geometry_result),
            "budget_accounting": geometry["protocol"]["budget_accounting"],
        },
    }
    write_json(args.output / "fit_summary.json", result)
    text = markdown(result)
    path = args.output / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
