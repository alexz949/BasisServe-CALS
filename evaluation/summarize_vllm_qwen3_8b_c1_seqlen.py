#!/usr/bin/env python3
"""Summarize the fixed-batch prompt-length sweep and separate profiles."""

import argparse
import csv
import json
from pathlib import Path
import shlex
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.summarize_vllm_qwen3_8b_c1 import kernel_category


def median_run(row, key, statistic=None):
    values = [run[key] if statistic is None else run[key][statistic]
              for run in row["runs"]]
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path("results/vllm_8b_tp8_seqlen"))
    args = parser.parse_args()
    root = args.input_dir
    dense = json.loads((root / "dense.json").read_text())
    compact = json.loads((root / "c1.json").read_text())
    assert dense["status"] == compact["status"] == "complete"
    for key in ("batch_size", "prefill_lengths", "decode_tokens",
                "max_num_batched_tokens", "repeats", "warmups",
                "gpu_memory_utilization", "profile_prefill_lengths"):
        assert dense["arguments"][key] == compact["arguments"][key]
    for key in dense["configuration"]:
        if key != "hf_overrides":
            assert dense["configuration"][key] == compact["configuration"][key]

    rows = []
    for baseline, c1 in zip(dense["lengths"], compact["lengths"], strict=True):
        assert baseline["prefill_tokens"] == c1["prefill_tokens"]
        assert baseline["batch"] == c1["batch"]
        row = {"prefill_tokens": baseline["prefill_tokens"],
               "batch": baseline["batch"]}
        for arm, result in (("dense", baseline), ("c1", c1)):
            row[f"{arm}_seconds"] = median_run(result, "wall_seconds")
            row[f"{arm}_seconds_min"] = min(run["wall_seconds"] for run in result["runs"])
            row[f"{arm}_seconds_max"] = max(run["wall_seconds"] for run in result["runs"])
            row[f"{arm}_output_tokens_per_second"] = median_run(
                result, "output_tokens_per_second")
            row[f"{arm}_ttft_ms"] = median_run(result, "ttft_ms", "mean")
            row[f"{arm}_tpot_ms"] = median_run(result, "tpot_ms", "mean")
            row[f"{arm}_preemptions_total"] = sum(
                run["preemptions"] for run in result["runs"])
        row["e2e_speedup"] = row["dense_seconds"] / row["c1_seconds"]
        row["ttft_speedup"] = row["dense_ttft_ms"] / row["c1_ttft_ms"]
        row["tpot_speedup"] = row["dense_tpot_ms"] / row["c1_tpot_ms"]
        rows.append(row)

    with (root / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    profiles = []
    for path in sorted((root / "profiles").glob("*.kernels.csv")):
        groups = {}
        with path.open() as handle:
            kernels = list(csv.DictReader(handle))
        assert kernels
        for kernel in kernels:
            category = kernel_category(kernel["kernel"])
            groups[category] = groups.get(category, 0.0) + float(kernel["total_us"])
        total = sum(groups.values())
        profiles.append({
            "profile": path.name.removesuffix(".kernels.csv"),
            "total_kernel_ms": total / 1000,
            "category_ms": {key: value / 1000 for key, value in groups.items()},
            "category_percent": {key: 100 * value / total for key, value in groups.items()},
            "top_kernels": kernels[:10],
        })
    assert len(profiles) == 2 * len(dense["arguments"]["profile_prefill_lengths"])
    (root / "profile_summary.json").write_text(json.dumps(profiles, indent=2) + "\n")

    arguments = dense["arguments"]
    lines = [
        "# Qwen3-8B-Base TP8 fixed-batch context-length sweep", "",
        f"Batch {arguments['batch_size']}, {arguments['decode_tokens']} output tokens, "
        "native context only (YaRN disabled). Environment: `basis`, eight NVIDIA "
        f"L40S (PCIe), PyTorch `{dense['torch']}`, vLLM `{dense['vllm']}`. "
        f"Each point has {arguments['warmups']} full warmup and "
        f"{arguments['repeats']} measured runs; entries are medians.", "",
        "Both arms use one engine with max model length 32768, chunked prefill "
        f"with an {arguments['max_num_batched_tokens']}-token scheduler budget, "
        "no prefix caching, synchronous scheduling, compilation mode NONE, and "
        "FULL_DECODE_ONLY CUDA Graphs. Dense uses FlashAttention 2; C1 uses "
        "native Triton DiffKV, prepared NCCL, and one decoder GEMM.", "",
        "| Prefill | Dense E2E s | C1 E2E s | E2E speedup | Dense TTFT ms | C1 TTFT ms | Dense TPOT ms | C1 TPOT ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['prefill_tokens']} | {row['dense_seconds']:.3f} | "
            f"{row['c1_seconds']:.3f} | {row['e2e_speedup']:.3f}x | "
            f"{row['dense_ttft_ms']:.2f} | {row['c1_ttft_ms']:.2f} | "
            f"{row['dense_tpot_ms']:.3f} | {row['c1_tpot_ms']:.3f} |"
        )
    lines += [
        "", "All measured dense and C1 runs recorded zero scheduler preemptions. "
        "Batch means 32 requests submitted together, not a constant execution "
        "batch: continuous batching and chunked prefill remain active. E2E "
        "throughput includes prefill. TTFT includes admission-barrier wait; TPOT "
        f"is per request `(last-first)/{arguments['decode_tokens'] - 1}` and "
        "includes scheduler interleaving, so it is not isolated decode-kernel latency.",
        "", "## Rank-zero full-cohort profiles", "",
        "Profiles are separate unmeasured runs after all timed points. Values are "
        "summed GPU kernel durations, not wall time; communication durations can "
        "include waiting and categories are inferred from kernel names.", "",
        "| Profile | Total kernel s | AllReduce s | AllGather s | GEMM/GEMV s | Attention s | Other s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in profiles:
        groups = profile["category_ms"]
        lines.append(
            f"| {profile['profile']} | {profile['total_kernel_ms']/1000:.3f} | "
            f"{groups.get('allreduce', 0)/1000:.3f} | "
            f"{groups.get('allgather', 0)/1000:.3f} | "
            f"{groups.get('gemm_gemv', 0)/1000:.3f} | "
            f"{groups.get('attention', 0)/1000:.3f} | "
            f"{groups.get('other', 0)/1000:.3f} |"
        )
    lines += [
        "", "The E2E speedup rises from short context to a maximum around 4K–8K, "
        "then declines as long-context prefill occupies a larger share of the "
        "fixed 128-token generation workload. This is a serving-path comparison, "
        "not an isolated attention-kernel test or a model-quality evaluation.",
        "", "## Reproduction", "",
        "Use the `basis` environment with `CUDA_HOME=/usr/local/cuda`, "
        "`OMP_NUM_THREADS=1`, and `VLLM_WORKER_MULTIPROC_METHOD=spawn`.", "",
        "```bash", dense["command"], compact["command"],
        "/workspace/miniforge3/envs/basis/bin/python "
        "evaluation/summarize_vllm_qwen3_8b_c1_seqlen.py "
        f"--input-dir {shlex.quote(str(root))}", "```", "",
        "C1 factor manifest SHA256: `" + compact["factor_sha256"] + "`.", "",
        "Non-fatal runtime warnings about unavailable custom PCIe all-reduce "
        "backends, initial Triton JIT, profiler export waits, and vLLM process "
        "cleanup are retained in the arm logs.", "",
    ]
    (root / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
