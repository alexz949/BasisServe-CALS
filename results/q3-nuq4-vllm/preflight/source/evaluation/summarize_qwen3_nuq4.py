"""Summarize matched Qwen NUQ4 TP8 request benchmarks, without mixing phases."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median


def summarize(root):
    phase = root.name
    assert phase in ("preflight", "formal")
    inputs = {name: root / name / f"graph_{kernel}_4096.json"
              for name, kernel in (("dense", "flash"), ("r64", "splitk"), ("r96", "splitk"))}
    payloads = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    reference = payloads["dense"]["configuration"]
    rows = []
    for name, payload in payloads.items():
        assert payload["status"] == "complete" and payload["phase"] == phase
        assert payload["metric_notes"]["output_length"] == 128
        for key in ("model", "tensor_parallel_size", "dtype", "max_model_len", "max_num_seqs",
                    "max_num_batched_tokens", "gpu_memory_utilization", "block_size",
                    "enable_prefix_caching", "enable_chunked_prefill", "async_scheduling",
                    "compilation_config"):
            assert payload["configuration"][key] == reference[key], key
        expected = [1, 256] if phase == "preflight" else [1, 2, 4, 8, 16, 32, 64, 128, 256]
        assert [entry["batch"] for entry in payload["batches"]] == expected
        for entry in payload["batches"]:
            runs, workers = entry["runs"], entry["workers"]
            assert len(runs) == (1 if phase == "preflight" else 3) and len(workers) == 8
            if name != "dense":
                assert all(not worker["overflow_layers"] and worker["loaded_value_layers"] == 36
                           and worker["capture_calls"] > 0 for worker in workers)
            for run in runs:
                assert len(run["requests"]) == entry["batch"]
                assert all(len(request["output_token_ids"]) == 128 for request in run["requests"])
            rows.append(dict(arm=name, batch=entry["batch"], repeats=len(runs),
                wall_s=median(run["wall_seconds"] for run in runs),
                throughput_tokens_s=median(run["output_tokens_per_second"] for run in runs),
                ttft_ms=median(run["ttft_ms"]["median"] for run in runs),
                tpot_ms=median(run["tpot_ms"]["median"] for run in runs),
                preemptions_max=max(run["preemptions"] for run in runs),
                peak_allocated_gib=max(worker["peak_cuda_allocated_bytes"] for worker in workers)/2**30))
    dense = {row["batch"]: row for row in rows if row["arm"] == "dense"}
    for row in rows:
        row["request_speedup"] = dense[row["batch"]]["wall_s"] / row["wall_s"]
    return rows, inputs, payloads


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows, inputs, payloads = summarize(args.root)
    with (args.root / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# Qwen3-8B-Base TP8 NUQ4: {args.root.name}", "",
        "Environment: `basis`; 8 x L40S; 4096 input + 128 output tokens (127 decode forwards).",
        "Scheduler budget 8192; GPU memory budget 80%; one warmup per batch; CUDA Graph decode.", "",
        ("Single-repeat endpoint checks, not formal speed conclusions." if args.root.name == "preflight"
         else "Three measured repeats per batch; table reports medians."), "",
        "| Arm | Batch | Request s | Output tok/s | TTFT ms | TPOT ms | Speedup | Preemptions max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda row: (row["batch"], row["arm"])):
        lines.append(f"| {row['arm']} | {row['batch']} | {row['wall_s']:.3f} | "
                     f"{row['throughput_tokens_s']:.2f} | {row['ttft_ms']:.2f} | "
                     f"{row['tpot_ms']:.3f} | {row['request_speedup']:.3f}x | {row['preemptions_max']} |")
    lines += ["", "## Interpretation", "",
        "Speedup is Dense request wall time divided by quantized request wall time at the same batch.",
        "Throughput includes prefill and all output tokens; it is not isolated steady decode throughput.",
        "TTFT includes paused admission time. TPOT spans first to last token and can include scheduling interleaving.",
        "Per-request medians are computed within each repeat, then across repeats.",
        "CUDA peak allocation in CSV is cumulative since engine initialization, not per-trial resident cache memory.",
        "No outlier pool overflow was reported in accepted NUQ4 trials. No output-equivalence or PPL claim is made here.",
        "Packed cache includes BF16 outliers and metadata; the end-to-end storage reduction is not 4x.",
        "The quantized path includes global BF16 V-statistics AllGather and uint8 latent AllGather plus W8A8 GEMM.",
        "Full-context attention; no sparse routing or MLP changes; no SHA256.", "", "## Commands and Raw Data", "",
        "All benchmark commands use the `basis` environment and `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.",
        "Additional environment: `CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.", ""]
    for name, payload in payloads.items():
        lines += [f"- [{name} raw JSON]({inputs[name].relative_to(args.root)}):",
                  f"  `/workspace/miniforge3/bin/conda run --no-capture-output -n basis python {payload['command']}`"]
    lines += ["", "Summary command:", "", "```bash",
              f"/workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/summarize_qwen3_nuq4.py --root {args.root}",
              "```", ""]
    (args.root / "SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
