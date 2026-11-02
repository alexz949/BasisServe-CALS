"""Verify and summarize a completed three-arm Experiment B workload."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    reference_prompts = None
    reference_sources = None
    reference_assets = None
    commands = {}
    for mode in ("dense", "als_full", "joint"):
        cohort_summary = json.loads((args.input / mode / "cohorts.json").read_text())
        assert cohort_summary["status"] == "complete" and cohort_summary["cohorts"] == 3
        assert cohort_summary["separate_engine_warmup"]
        trials = []
        prompts = []
        mode_job_id = None
        for cohort in ("warmup", "cohort_0", "cohort_1", "cohort_2"):
            records = [json.loads((args.input / mode / cohort / f"rank_{rank}.json").read_text())
                       for rank in range(8)]
            first = records[0]
            if mode_job_id is None:
                mode_job_id = first["slurm_job_id"]
            for rank, row in enumerate(records):
                assert row["status"] == "complete" and row["rank"] == rank and row["world_size"] == 8
                assert row["slurm_job_id"] == mode_job_id
                assert row["all_logits_finite"] and row["measured_decode_steps"] == 128
                assert row["discard_first_decode_steps"] == 16 and row["steps"] == 145
                assert row["measurement_cache_length_before_first_forward"] == row["length"] + 16
                assert row["measurement_cache_length_after_last_forward"] == row["length"] + 144
                assert row["is_engine_warmup"] == (cohort == "warmup")
                for name in ("generated_token_ids", "source_sha256", "asset_manifest_sha256",
                             "prompt_sha256", "dispatch", "decode_max_rank_ms"):
                    assert row[name] == first[name]
                assert len(row["generated_token_ids"]) == row["batch"]
                assert all(len(tokens) == 145 for tokens in row["generated_token_ids"])
            assert all(max(row["decode_local_rank_ms"][step] for row in records)
                       == first["decode_max_rank_ms"][step] for step in range(144))
            if reference_sources is None:
                reference_sources = first["source_sha256"]
                reference_assets = first["asset_manifest_sha256"]
            assert first["source_sha256"] == reference_sources
            assert first["asset_manifest_sha256"] == reference_assets
            commands[mode] = first["argv"]
            prompts.append(first["prompt_sha256"])
            if cohort == "warmup":
                continue
            trials.append({
                "cohort": cohort, "mean_ms": first["decode_mean_ms"],
                "p50_ms": first["decode_p50_ms"], "p95_ms": first["decode_p95_ms"],
                "tokens_per_second": first["steady_tokens_per_second"],
                "prefill_peak_allocated_gib": max(row["prefill_peak_allocated_bytes"] for row in records) / 2**30,
                "decode_ready_allocated_gib": max(row["decode_ready_allocated_bytes"] for row in records) / 2**30,
                "peak_reserved_gib": max(max(row["prefill_peak_reserved_bytes"], row["decode_peak_reserved_bytes"])
                                         for row in records) / 2**30,
                "host_exact_k_capacity_gib_replica": sum(row["host_exact_k_capacity_bytes"] for row in records) / 2**30,
            })
        assert len(set(prompts)) == 4
        if reference_prompts is None:
            reference_prompts = prompts
        assert prompts == reference_prompts
        summaries.append({
            "mode": mode, "slurm_job_id": mode_job_id, "trials": trials,
            "median_mean_ms": statistics.median(row["mean_ms"] for row in trials),
            "median_tokens_per_second": statistics.median(row["tokens_per_second"] for row in trials),
            "max_reserved_gib": max(row["peak_reserved_gib"] for row in trials),
        })
    for name, expected in reference_sources.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    for summary in summaries:
        summary["speedup_vs_dense"] = summaries[0]["median_mean_ms"] / summary["median_mean_ms"]
    result = {
        "status": "verified", "length": cohort_summary["length"], "batch": cohort_summary["batch"],
        "measured_trials": 9, "warmup_trials": 3, "rank_records": 96,
        "results": summaries, "commands": commands, "conda_environment": "basis",
        "asset_manifest_sha256": reference_assets, "source_sha256": reference_sources,
        "router_training_manifest_matches_current": cohort_summary["router_training_manifest_matches_current"],
        "limitations": ["Rebuilt A5000 eager runner, not an identical L40S software/checkpoint reproduction.",
                        "Decode-only Experiment B; prefill timings are diagnostic, not a request benchmark.",
                        "Only final-step slot counts are retained; no measured interval slot-hit average.",
                        "GPU memory is PyTorch allocator memory; no NVML peak or host RSS sampling."],
    }
    (args.input / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
