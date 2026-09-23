"""Validate and summarize the corrected TP8 decode pilot without file hashing."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_trials(root):
    rows, profiles, generations = [], [], []
    for phase in ("smoke", "timing", "profile"):
        manifest_path = root / phase / "decode_grid_trials.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        for trial in manifest["trials"]:
            command = trial["command"]
            tag = command[command.index("--tag") + 1]
            arm, length, batch, cohort = (trial[key] for key in ("arm", "prompt_tokens", "batch", "cohort"))
            folder = root / phase / f"{tag}_{arm}_p{length}_b{batch}_r{cohort}"
            row = dict(phase=phase, arm=arm, prompt_tokens=length, batch=batch, cohort=cohort,
                       status="failed", rank_files=0, mean_ms=None, p50_ms=None, p95_ms=None,
                       tokens_s=None, decode_gpu_maxrank_gib=None, host_exact_key_total_gib=None, ranks_identical=False,
                       launcher_log=trial["log"])
            if trial["returncode"] == 0:
                paths = [folder / f"rank{rank}.json" for rank in range(8)]
                assert all(path.is_file() for path in paths), str(folder)
                ranks = [json.loads(path.read_text()) for path in paths]
                first = ranks[0]
                measured = manifest["measure_steps"]
                generated = measured + manifest["conditioning_steps"] + 1
                assert all(value["status"] == "complete" for value in ranks)
                assert all(len(value["decode_step_ms"]) == measured for value in ranks)
                assert all(value["generated_token_ids"] == first["generated_token_ids"] for value in ranks)
                assert all(value["decode_step_ms"] == first["decode_step_ms"] for value in ranks)
                assert len(first["generated_token_ids"]) == batch
                assert all(len(tokens) == generated for tokens in first["generated_token_ids"])
                row.update(status="complete", rank_files=8, mean_ms=first["decode_step_mean_ms"],
                           p50_ms=first["decode_step_p50_ms"], p95_ms=first["decode_step_p95_ms"],
                           tokens_s=first["decode_tokens_per_second"], ranks_identical=True,
                           decode_gpu_maxrank_gib=max(value["decode_resident_allocated_bytes"] for value in ranks) / 2**30,
                           host_exact_key_total_gib=sum(value["host_persistent_exact_key_bytes"] for value in ranks) / 2**30)
                if phase == "timing":
                    generations.append(dict(arm=arm, prompt_tokens=length, batch=batch, cohort=cohort,
                                            token_ids=first["generated_token_ids"][0], model=first["model"]))
                if phase == "profile":
                    for component in sorted(first["component_profile"]):
                        values = [value["component_profile"][component]["mean_ms_per_step"] for value in ranks]
                        profiles.append(dict(arm=arm, prompt_tokens=length, batch=batch, cohort=cohort,
                                             component=component, rank0_ms=values[0],
                                             median_rank_ms=statistics.median(values), min_rank_ms=min(values),
                                             max_rank_ms=max(values)))
            rows.append(row)
    return rows, profiles, generations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/system_benchmarks/tp8_path_opt"))
    args = parser.parse_args()
    rows, profiles, generations = read_trials(args.root)
    write_csv(args.root / "decode_summary.csv", rows)
    write_csv(args.root / "component_profile.csv", profiles)
    comparisons = []
    for small in generations:
        if small["batch"] != 1:
            continue
        large = next((value for value in generations if value["batch"] == 8 and all(
            value[key] == small[key] for key in ("arm", "prompt_tokens", "cohort"))), None)
        if large is None:
            continue
        pairs = list(zip(small["token_ids"], large["token_ids"], strict=True))
        prefix = next((index for index, (a, b) in enumerate(pairs) if a != b), len(pairs))
        comparisons.append(dict(arm=small["arm"], prompt_tokens=small["prompt_tokens"],
                                token_agreement=sum(a == b for a, b in pairs) / len(pairs),
                                exact_prefix_tokens=prefix, compared_tokens=len(pairs)))
    write_csv(args.root / "batch_consistency.csv", comparisons)
    if generations:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(generations[0]["model"], local_files_only=True)
        for sample in generations:
            sample["text"] = tokenizer.decode(sample["token_ids"], skip_special_tokens=True)
        (args.root / "generated_samples.json").write_text(json.dumps(generations, indent=2) + "\n")
    counts = {phase: {status: sum(row["phase"] == phase and row["status"] == status for row in rows)
                      for status in ("complete", "failed")} for phase in ("smoke", "timing", "profile")}
    lines = ["# Corrected TP8 Decode Pilot", "", "Conda environment: `basis`. Jobs ran directly on 8 x L40S.", "",
             "This is a single-cohort pilot, not the full three-cohort grid or a task-quality evaluation.", "",
             "## Completion", "", "```json", json.dumps(counts, indent=2), "```", "",
             "Every successful trial has eight rank JSON files, complete step arrays, and identical tokens/timings across ranks.", "",
             "## Uninstrumented Timing", "", "| Prompt | B | Arm | Mean ms | P50 ms | P95 ms | tokens/s | vs Dense | vs ALS-full | GPU GiB/rank | Host K GiB total |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    timing = [row for row in rows if row["phase"] == "timing" and row["status"] == "complete"]
    for row in timing:
        dense = next((value for value in timing if value["arm"] == "dense" and all(
            value[key] == row[key] for key in ("prompt_tokens", "batch", "cohort"))), None)
        als = next((value for value in timing if value["arm"] == "als_full" and all(
            value[key] == row[key] for key in ("prompt_tokens", "batch", "cohort"))), None)
        speedup = f"{dense['mean_ms'] / row['mean_ms']:.3f}x" if dense else "-"
        als_speedup = f"{als['mean_ms'] / row['mean_ms']:.3f}x" if als else "-"
        lines.append(f"| {row['prompt_tokens']} | {row['batch']} | {row['arm']} | {row['mean_ms']:.2f} | "
                     f"{row['p50_ms']:.2f} | {row['p95_ms']:.2f} | {row['tokens_s']:.2f} | {speedup} | "
                     f"{als_speedup} | {row['decode_gpu_maxrank_gib']:.2f} | {row['host_exact_key_total_gib']:.2f} |")
    lines += ["", "GPU GiB is the maximum per-rank allocated decode-resident memory, not prefill peak or total host memory.",
              "Latency is the mean of per-step rank maxima; tokens/s uses maximum-rank wall time. These are not exact reciprocals.",
              "The timed loop includes greedy token selection, finite-logit checks, and generated-token copies for all arms.",
              "", "## Component Profiles", "",
              "Profiles contain CUDA-event instrumentation. They are not comparable to uninstrumented latency.",
              "`attention_total` contains attention subcomponents. Do not add it to those components; do not sum per-component rank maxima.",
              "Collective intervals can include rank arrival skew and CPU launch gaps, not only transfer time.", "",
              "| Prompt | B | Component | Rank 0 ms/step | Median rank ms/step | Max rank ms/step |",
              "|---:|---:|---|---:|---:|---:|"]
    for row in profiles:
        lines.append(f"| {row['prompt_tokens']} | {row['batch']} | {row['component']} | {row['rank0_ms']:.3f} | "
                     f"{row['median_rank_ms']:.3f} | {row['max_rank_ms']:.3f} |")
    collective_paths = sorted(args.root.glob("collective_b*.json"))
    if collective_paths:
        lines += ["", "## Collective Smoke Measurements", "",
                  "TP8 BF16, local width 384, 20 warmup calls and 100 measured calls per backend. Each completed backend passes exact output comparison.",
                  "The timing helper now executes graph replay and timing events on the same CUDA stream; the old side-stream measurement was not reliable.",
                  "These short standalone measurements do not reproduce model rank-arrival skew or its CPU affinity; they do not establish end-to-end backend speedups.",
                  "", "| B | Backend | Median us | Mean us | Status |", "|---:|---|---:|---:|---|"]
        for path in collective_paths:
            payload = json.loads(path.read_text())
            for record in payload["records"]:
                if record["status"] == "complete":
                    lines.append(f"| {payload['tokens']} | {record['backend']} | {record['p50_us']:.2f} | "
                                 f"{record['mean_us']:.2f} | complete |")
                else:
                    lines.append(f"| {payload['tokens']} | {record['backend']} | - | - | {record['status']} |")
    lines += ["", "## Validation Limits", "",
              "- Finite logits are checked by the runner. Generated first-sequence samples are in `generated_samples.json`.",
              "- `batch_consistency.csv` compares the same first prompt at B=1 and B=8; floating-point changes can alter greedy trajectories.",
              f"- Observed B=1/B=8 trajectory divergence in {sum(row['exact_prefix_tokens'] < row['compared_tokens'] for row in comparisons)}/{len(comparisons)} arm/context pairs. Dense also diverges; teacher-forced comparisons are needed to diagnose this, not free-generation token agreement alone.",
              "- Cross-rank agreement and finite logits do not establish task accuracy or numerical equivalence to Dense.",
              "- Historical B>1 Basis-joint runs had an output-stride error and are not a correctness baseline.",
              "- CPU affinity is set, but strict pinned-host NUMA placement is unavailable (`set_mempolicy: Operation not permitted`).",
              "- No SHA256 validation was performed.", "", "## Commands and Logs", "",
              "Exact per-trial torchrun commands and return codes are in each phase's `decode_grid_trials.json`.",
              "Top-level launcher logs are `smoke.log`, `timing.log`, and `profile.log`; each trial also has launcher and rank logs.",
              "Timing uses 16 conditioning + 128 measured steps; profile uses 16 conditioning + 16 measured steps.",
              "Source changes relative to the archived baseline are recorded in `source.patch`.", ""]
    (args.root / "SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps({"counts": counts, "summary": str(args.root / "SUMMARY.md")}), flush=True)


if __name__ == "__main__":
    main()
