#!/usr/bin/env python3
"""Summarize the paired fixed-cohort benchmark without mixing profile timings."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import shlex
import statistics


def median_runs(batch, key, statistic=None):
    values = [run[key] if statistic is None else run[key][statistic] for run in batch["runs"]]
    return statistics.median(values)


def kernel_category(name):
    lowered = name.lower()
    if "nccl" in lowered:
        return "allgather" if "allgather" in lowered else "allreduce" if "allreduce" in lowered else "other_collective"
    if "gemm" in lowered or "gemv" in lowered:
        return "gemm_gemv"
    if any(part in lowered for part in (
        "flash_fwd", "attn", "attention", "reduce_segments",
        "diffkv_prefill", "diffkv_decode",
    )):
        return "attention"
    return "other"


def decode_boundary_kernels(events, compact, layers):
    """Attribute principal GEMM kernels using the verified per-layer graph order."""
    launches = {}
    for event in events:
        if event.get("cat") == "kernel" and event.get("args", {}).get("graph id", 0):
            launches.setdefault(event["args"]["correlation"], []).append(event)
    gemm_us, collective_us, pack_us = [], [], []
    matched = 0
    for kernels in launches.values():
        kernels.sort(key=lambda event: event["ts"])
        collective = "allgather" if compact else "allreduce"
        indices = [i for i, event in enumerate(kernels) if kernel_category(event["name"]) == collective]
        if len(indices) != (layers if compact else 2 * layers + 1):
            continue
        # Dense: embedding AllReduce, then attention/MLP pairs for every layer.
        selected = indices if compact else indices[1::2]
        matched += 1
        for index in selected:
            candidates = kernels[index+1:] if compact else reversed(kernels[:index])
            gemm = next(event for event in candidates if kernel_category(event["name"]) == "gemm_gemv")
            gemm_us.append(gemm["dur"])
            collective_us.append(kernels[index]["dur"])
            if compact:
                pack_us.append(kernels[index-1]["dur"])
    assert matched > 0 and len(gemm_us) == layers*matched
    result = dict(matched_graph_launches=matched, total_graph_launches=len(launches),
                  principal_decoder_gemm_us_mean=statistics.fmean(gemm_us),
                  boundary_collective_us_mean=statistics.fmean(collective_us),
                  note="Profiled kernels, not operator or wall time. For batches >1, graph replay batch sizes can vary as requests finish.")
    if compact:
        result["pre_allgather_kernel_us_mean"] = statistics.fmean(pack_us)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/vllm_tp8"))
    args = parser.parse_args()
    root = args.input_dir
    dense = json.loads((root / "dense.json").read_text())
    compact = json.loads((root / "c1.json").read_text())
    model_config = json.loads((Path(dense["configuration"]["model"]) / "config.json").read_text())
    layers = model_config["num_hidden_layers"]
    model_label = "Qwen3-8B-Base" if layers == 36 else "Qwen3-32B"
    decode_paths = sorted({worker["decode_specialization"]
                           for batch in compact["batches"] for worker in batch["workers"]})
    decode_label = "the recorded decode path(s): " + ", ".join(f"`{path}`" for path in decode_paths)
    validation_label = (
        "Factor validation: configuration, manifest structure and tensor shapes only; "
        "no SHA256 checks were performed. Factor contents were not authenticated."
        if compact["factor_validation"] == "structure"
        else f"C1 manifest SHA256: `{compact['factor_sha256']}`."
    )
    arguments = dense["arguments"]
    assert dense["status"] == compact["status"] == "complete"
    for key in ("batch_sizes", "prefill_tokens", "decode_tokens", "max_num_batched_tokens", "repeats", "warmups", "gpu_memory_utilization"):
        assert dense["arguments"][key] == compact["arguments"][key]
    for key in dense["configuration"]:
        if key != "hf_overrides":
            assert dense["configuration"][key] == compact["configuration"][key]
    rows = []
    for baseline, c1 in zip(dense["batches"], compact["batches"], strict=True):
        assert baseline["batch"] == c1["batch"]
        row = {"batch": baseline["batch"]}
        for arm, batch in (("dense", baseline), ("c1", c1)):
            row[f"{arm}_seconds"] = median_runs(batch, "wall_seconds")
            row[f"{arm}_output_tokens_per_second"] = median_runs(batch, "output_tokens_per_second")
            row[f"{arm}_ttft_ms"] = median_runs(batch, "ttft_ms", "mean")
            row[f"{arm}_tpot_ms"] = median_runs(batch, "tpot_ms", "mean")
            row[f"{arm}_seconds_min"] = min(run["wall_seconds"] for run in batch["runs"])
            row[f"{arm}_seconds_max"] = max(run["wall_seconds"] for run in batch["runs"])
            row[f"{arm}_preemptions_total"] = sum(run["preemptions"] for run in batch["runs"])
        row["e2e_speedup"] = row["dense_seconds"] / row["c1_seconds"]
        row["tpot_speedup"] = row["dense_tpot_ms"] / row["c1_tpot_ms"]
        rows.append(row)
    with (root / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    profile_rows = []
    for path in sorted((root / "profiles").glob("*.kernels.csv")):
        with path.open() as handle:
            kernels = list(csv.DictReader(handle))
        assert kernels, f"No GPU kernels recorded: {path}"
        groups = {}
        for kernel in kernels:
            group = kernel_category(kernel["kernel"])
            groups[group] = groups.get(group, 0.) + float(kernel["total_us"])
        total = sum(groups.values())
        profile_rows.append(dict(profile=path.stem, total_kernel_ms=total/1000,
                                 category_ms={key: value/1000 for key, value in groups.items()},
                                 category_percent={key: 100*value/total for key, value in groups.items()},
                                 top_kernels=kernels[:10]))
        trace_path = path.with_name(path.name.replace(".kernels.csv", ".trace.json.gz"))
        with gzip.open(trace_path) as handle:
            trace = json.load(handle)
        graph_groups = {}
        graph_launches = set()
        for event in trace["traceEvents"]:
            if event.get("cat") == "kernel" and event.get("args", {}).get("graph id", 0):
                category = kernel_category(event["name"])
                graph_groups[category] = graph_groups.get(category, 0.) + event["dur"]
                graph_launches.add(event["args"]["correlation"])
        graph_total = sum(graph_groups.values())
        assert graph_total > 0
        profile_rows[-1]["decode_graph"] = dict(
            launches=len(graph_launches), total_kernel_ms=graph_total/1000,
            category_ms={key: value/1000 for key, value in graph_groups.items()},
            category_percent={key: 100*value/graph_total for key, value in graph_groups.items()},
        )
        profile_rows[-1]["decode_boundary"] = decode_boundary_kernels(
            trace["traceEvents"], path.name.startswith("c1_"), layers)
        del trace
    (root / "profile_summary.json").write_text(json.dumps(profile_rows, indent=2)+"\n")
    lines = [f"# {model_label} TP8 dense / C1 benchmark", "",
             "Environment: `basis`, eight NVIDIA L40S (PCIe), "
             f"PyTorch `{dense['torch']}`, vLLM `{dense['vllm']}`.", "",
             f"Model: {model_label} BF16; C1 uses R64-S6 factors. Fixed cohorts of "
             f"{arguments['prefill_tokens']} prompt tokens and exactly {arguments['decode_tokens']} output tokens per request. "
             f"{arguments['warmups']} full warmup(s) and {arguments['repeats']} measured runs per batch; table entries are medians over runs. "
             f"Both arms use chunked prefill ({arguments['max_num_batched_tokens']}-token budget), no prefix caching, "
             "synchronous scheduling, FULL_DECODE_ONLY CUDA Graphs, and compilation mode NONE. "
             "This is a matched serving configuration, not a claim of globally optimal "
             "production vLLM tuning. Dense uses FlashAttention 2; C1 uses the SM89-specialized "
             f"QK128/V64 Triton DiffKV prefill kernel and {decode_label}.", "",
             "| Batch | Dense E2E s | C1 E2E s | E2E speedup | Dense output tok/s | C1 output tok/s | Dense TPOT ms | C1 TPOT ms |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['batch']} | {row['dense_seconds']:.3f} | {row['c1_seconds']:.3f} | "
                     f"{row['e2e_speedup']:.3f}x | {row['dense_output_tokens_per_second']:.1f} | "
                     f"{row['c1_output_tokens_per_second']:.1f} | {row['dense_tpot_ms']:.3f} | {row['c1_tpot_ms']:.3f} |")
    lines += ["", "## Scheduler preemptions", "",
              "Counts are summed over measured runs only, excluding warmup and profiling. "
              "Nonzero values mark capacity-affected results that must not be attributed "
              "solely to communication/kernel changes. Zero preemptions do not by themselves "
              "rule out cache-capacity-limited admission; also inspect KV utilization logs.", "",
              "| Batch | Dense preemptions | C1 preemptions |",
              "| ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['batch']} | {row['dense_preemptions_total']} | {row['c1_preemptions_total']} |")
    lines += ["", "E2E throughput includes prefill. TPOT is computed separately for each request "
              f"as `(last_token_ts - first_token_ts) / {arguments['decode_tokens'] - 1}`, then averaged over requests. "
              "It includes scheduler interleaving and is not isolated decode-kernel latency. "
              "Batch denotes the number of requests submitted together, not a constant "
              "GPU execution batch: continuous batching and chunked prefill remain active. "
              "TTFT uses engine queue-to-first-token timestamps and includes admission-barrier wait. "
              "Raw request timestamps, output token IDs, TTFT, run ranges, and worker statistics "
              "are retained in the JSON/CSV files. GPU peak allocation is cumulative and includes "
              "vLLM's preallocated KV pool; it is not live per-request KV usage.", "",
              "## Rank-zero GPU profiles", "",
              "Profiles are separate full-cohort runs after all timed measurements. "
              "They include prefill and decode. Kernel durations are summed, not wall time; "
              "profiling overhead can affect communication wait. Categories are inferred from "
              "kernel names, so consult raw traces for attribution.", "",
              "| Profile | AllReduce % | AllGather % | GEMM/GEMV % | Attention % | Other % |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for profile in profile_rows:
        groups = profile["category_percent"]
        lines.append(f"| {profile['profile']} | {groups.get('allreduce', 0):.1f} | "
                     f"{groups.get('allgather', 0):.1f} | {groups.get('gemm_gemv', 0):.1f} | "
                     f"{groups.get('attention', 0):.1f} | {groups.get('other', 0):.1f} |")
    lines += ["", "### Batch-one decode graph attribution", "",
              "The following values use post-prefill graph replays only. "
              "Decoder attribution follows the per-layer graph order: after AllGather "
              f"for C1, before the attention-output AllReduce for dense. All {layers} layer "
              "boundaries are checked per replay. Values are mean GPU kernel microseconds "
              "per layer, not isolated operator benchmarks or unprofiled wall latency.", "",
              "The last kernel before AllGather can be a segment reduction rather than a layout copy; consult trace names. "
              "A reduction classified as attention is already included in attention time and must not be added again.", "",
              "| Arm | Principal O/decoder kernel us | Output collective us | Attention kernels us | Pre-AllGather kernel us |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for profile in profile_rows:
        if profile["profile"] not in ("dense_b1.kernels", "c1_b1.kernels"):
            continue
        boundary = profile["decode_boundary"]
        attention = profile["decode_graph"]["category_ms"]["attention"]*1000/(layers*boundary["matched_graph_launches"])
        lines.append(f"| {profile['profile'].split('_')[0]} | {boundary['principal_decoder_gemm_us_mean']:.2f} | "
                     f"{boundary['boundary_collective_us_mean']:.2f} | {attention:.2f} | "
                     f"{boundary.get('pre_allgather_kernel_us_mean', 0):.2f} |")
    lines += ["", "The path is functional and offline-tuned for the measured SM89 prefill "
              "shapes, but is not established as globally optimal. Retain one decoder GEMM. "
              "The remaining priorities are realistic full-model/cold-cache small-batch "
              "decoder layout and GEMM/GEMV selection, followed by broader-shape prefill tuning. "
              f"The C1 replicated decoder is [{model_config['num_attention_heads']*64},{model_config['hidden_size']}] per rank versus a dense local "
              f"O projection of [{model_config['num_attention_heads']//8*128},{model_config['hidden_size']}]: four times the BF16 weight bytes and matrix-product "
              "work per rank. Repeated resident-weight microbenchmarks do not reproduce the "
              "full model's cache working set. Cold-weight traffic is a hypothesis to test, "
              "not a measured DRAM-bandwidth diagnosis. Compare the measured output "
              "preparation cost with decoder and attention costs before prioritizing fusion.", "",
              "Warnings: native custom AllReduce variants are unsupported for eight PCIe-only "
              "GPUs, so vLLM uses PyNccl. Consult each arm's log for startup and JIT messages. "
              "vLLM process teardown can emit forced EngineCore cleanup and "
              "Python shared-memory/semaphore resource-tracker warnings; retain logs. "
              "Completed runs validate exact output lengths, non-corrupted request status, "
              f"and all {layers} C1 V layers plus graph-capture counters on every rank."]
    lines += ["", "## Reproduction", "", "Use the `basis` environment with "
              "`CUDA_HOME=/usr/local/cuda`, `OMP_NUM_THREADS=1`, and "
              "`VLLM_WORKER_MULTIPROC_METHOD=spawn`. No Slurm is available; the runs "
              "execute sequentially on all eight local GPUs.", "", "```bash",
              dense["command"], compact["command"],
              "/workspace/miniforge3/envs/basis/bin/python evaluation/summarize_vllm_qwen3_8b_c1.py "
              f"--input-dir {shlex.quote(str(root))}", "```", "",
              validation_label, "",
              "Model startup, JIT compilation, profiling, and trace export are excluded from "
              "timed measurements. This is a synthetic fixed-length throughput test, not a "
              "quality evaluation or an online arrival-rate benchmark.", ""]
    (root / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
