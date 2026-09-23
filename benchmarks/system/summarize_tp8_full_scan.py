"""Validate the frozen full-scan TP8 grid and summarize all three cohorts."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import tarfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/system_benchmarks/llama31_8b_tp8_full_scan"
ROUTING = "full_scan_b16r16_persistent_slots"
ARMS = ("dense", "als_full", "basis_joint")
MATRIX = {4096: (1, 8, 32, 128), 16384: (1, 4, 8, 16),
          65536: (1, 4, 8, 16), 130048: (1, 4, 8, 16)}
METRICS = ("mean_ms", "p50_ms", "p95_ms", "tokens_s", "prefill_gpu_gib",
           "resident_gpu_gib", "decode_peak_gpu_gib", "host_k_gib", "slot_hit_rate")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classify_failure(log):
    match = re.search(r"(?:torch\.(?:cuda\.)?OutOfMemoryError|CUDA out of memory)[^\n]*", log)
    if match:
        stage = "decode" if '"status": "prefill_complete"' in log else (
            "prefill" if '"status": "model_ready"' in log else "setup")
        return f"gpu_oom_{stage}", match.group(0)
    match = re.search(r"(?:Cannot allocate memory|MemoryError)[^\n]*", log)
    if match:
        return "host_oom", match.group(0)
    return "process_failure", "Inspect launcher log; no recognized OOM message."


def read_trial(root, trial):
    arm, length, batch, cohort = (trial[key] for key in ("arm", "prompt_tokens", "batch", "cohort"))
    command = trial["command"]
    tag = command[command.index("--tag") + 1]
    folder = root / f"{tag}_{arm}_p{length}_b{batch}_r{cohort}"
    row = dict(arm=arm, prompt_tokens=length, batch=batch, cohort=cohort,
               status="complete", failure_detail="", rank_files=0,
               routing_mode=ROUTING if arm == "basis_joint" else "full",
               model=None, pytorch=None, cuda=None, nccl=None,
               ranks_identical=False, **dict.fromkeys(METRICS),
               result_directory=str(folder.relative_to(root)), launcher_log=trial["log"])
    if trial["returncode"] != 0:
        row["status"], row["failure_detail"] = classify_failure(Path(trial["log"]).read_text(errors="replace"))
        row["rank_files"] = len(list(folder.glob("rank[0-7].json")))
        return row
    ranks = [json.loads((folder / f"rank{rank}.json").read_text()) for rank in range(8)]
    first = ranks[0]
    for rank, value in enumerate(ranks):
        assert value["status"] == "complete" and value["rank"] == rank
        assert (value["arm"], value["prompt_tokens"], value["batch"], value["repeat"]) == (arm, length, batch, cohort)
        assert (value["tp"], value["dp"], value["pp"]) == (8, 1, 1)
        assert value["conditioning_steps"] == 16 and value["measured_steps"] == 128
        assert value["dtype"] == "bfloat16" and not value["trace_enabled"]
        assert value["component_profile"] is None and not value["metadata"]["torch_matmul_tf32"]
        assert value["routing_mode"] == row["routing_mode"]
        assert value["decode_step_ms"] == first["decode_step_ms"]
        assert len(value["decode_step_ms"]) == 128
        assert all(math.isfinite(t) and t > 0 for t in value["decode_step_ms"])
        assert value["generated_token_ids"] == first["generated_token_ids"]
        assert value["prompt_cohort"] == first["prompt_cohort"]
        assert value["model"] == first["model"]
        assert all(value["metadata"][key] == first["metadata"][key] for key in ("pytorch", "cuda", "nccl"))
        assert len(value["generated_token_ids"]) == batch
        assert all(len(tokens) == 145 for tokens in value["generated_token_ids"])
        assert math.isclose(value["decode_step_mean_ms"], statistics.fmean(value["decode_step_ms"]), rel_tol=1e-9)
        assert math.isfinite(value["decode_tokens_per_second"]) and value["decode_tokens_per_second"] > 0
        assert math.isclose(value["decode_tokens_per_second"], batch * 128 / value["decode_wall_seconds"], rel_tol=1e-9)
        if arm == "basis_joint":
            assert (value["value_rank"], value["base_rank"], value["residual_rank"]) == (96, 16, 16)
            assert (value["page_size"], value["routed_pages"], value["recent_tokens"], value["physical_support"]) == (32, 62, 64, 2048)
    row.update(rank_files=8, ranks_identical=True, model=first["model"],
               pytorch=first["metadata"]["pytorch"], cuda=first["metadata"]["cuda"],
               nccl=json.dumps(first["metadata"]["nccl"]),
               mean_ms=first["decode_step_mean_ms"], p50_ms=first["decode_step_p50_ms"],
               p95_ms=first["decode_step_p95_ms"], tokens_s=first["decode_tokens_per_second"],
               prefill_gpu_gib=max(v["prefill_peak_allocated_bytes"] for v in ranks) / 2**30,
               resident_gpu_gib=max(v["decode_resident_allocated_bytes"] for v in ranks) / 2**30,
               decode_peak_gpu_gib=max(v["decode_peak_allocated_bytes"] for v in ranks) / 2**30,
               host_k_gib=sum(v["host_persistent_exact_key_bytes"] for v in ranks) / 2**30)
    if arm == "basis_joint":
        hits = sum(v["slot_counts"]["last_step_hits"] for v in ranks)
        valid = sum(v["slot_counts"]["last_step_valid"] for v in ranks)
        row["slot_hit_rate"] = hits / valid
    return row


def aggregate(rows):
    groups = []
    for length, batches in MATRIX.items():
        for batch in batches:
            for arm in ARMS:
                trials = [r for r in rows if (r["prompt_tokens"], r["batch"], r["arm"]) == (length, batch, arm)]
                assert len(trials) == 3 and {r["cohort"] for r in trials} == {0, 1, 2}
                complete = all(r["status"] == "complete" for r in trials)
                row = dict(prompt_tokens=length, batch=batch, arm=arm,
                           successful_trials=sum(r["status"] == "complete" for r in trials),
                           status="complete" if complete else ";".join(sorted({r["status"] for r in trials})),
                           routing_mode=trials[0]["routing_mode"], **dict.fromkeys(METRICS),
                           speedup_vs_dense=None, speedup_vs_als_full=None)
                if complete:
                    for metric in METRICS:
                        values = [r[metric] for r in trials]
                        if all(v is not None for v in values):
                            row[metric] = statistics.median(values)
                groups.append(row)
    indexed = {(r["prompt_tokens"], r["batch"], r["arm"]): r for r in groups}
    for row in groups:
        if row["status"] != "complete":
            continue
        for arm in ("dense", "als_full"):
            reference = indexed[row["prompt_tokens"], row["batch"], arm]
            if reference["status"] == "complete":
                row[f"speedup_vs_{arm}"] = reference["mean_ms"] / row["mean_ms"]
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "decode_grid_trials.json").read_text())
    assert manifest["status"] in ("complete", "complete_with_failures")
    assert manifest["routing_mode"] == ROUTING and manifest["tag"] == "full_scan_decode"
    assert manifest["conditioning_steps"] == 16 and manifest["measure_steps"] == 128
    assert not manifest["profile_components"]
    expected = {(arm, length, batch, cohort) for length, batches in MATRIX.items()
                for batch in batches for cohort in range(3) for arm in ARMS}
    keys = [(t["arm"], t["prompt_tokens"], t["batch"], t["cohort"]) for t in manifest["trials"]]
    assert len(keys) == 144 and set(keys) == expected
    frozen = json.loads((root / "freeze/manifest.json").read_text())
    unchanged = 0
    with tarfile.open(root / frozen["source_archive"], "r:gz") as archive:
        for name in frozen["source_files"]:
            assert (ROOT / name).read_bytes() == archive.extractfile(name).read(), name
            unchanged += 1
    trials = [read_trial(root, trial) for trial in manifest["trials"]]
    successful = [row for row in trials if row["status"] == "complete"]
    software = {(row["pytorch"], row["cuda"], row["nccl"]) for row in successful}
    assert len(software) == 1
    assert {row["model"] for row in successful} == {frozen["model"]}
    groups = aggregate(trials)
    write_csv(root / "decode_trial_summary.csv", trials)
    write_csv(root / "summary.csv", groups)
    validation = dict(trials=144, successful_trials=sum(r["status"] == "complete" for r in trials),
                      failed_trials=sum(r["status"] != "complete" for r in trials),
                      validated_rank_results=sum(r["rank_files"] for r in trials if r["status"] == "complete"),
                      frozen_source_files_unchanged=unchanged, source_verification="direct byte comparison; no SHA256",
                      routing_mode=ROUTING, conda_environment="basis",
                      model=frozen["model"],
                      software=dict(zip(("pytorch", "cuda", "nccl"), next(iter(software)))))
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    fmt = lambda value: "-" if value is None else f"{value:.3f}"
    lines = ["# Frozen TP8 Full-Scan Decode Grid", "", "Conda environment: `basis`; direct execution on 8 x NVIDIA L40S.",
             "", "## Validation", "", "```json", json.dumps(validation, indent=2), "```", "",
             "Every successful trial has 8 complete rank files, identical generated tokens and synchronized step arrays across ranks,",
             "128 measured steps after 16 conditioning steps, and 145 generated tokens per sequence including the prefill prediction.",
             "All frozen source files were compared directly against the pre-run archive. No SHA256 was computed.", "",
             "## Three-Cohort Medians", "",
             "A row receives median metrics only when all three cohorts completed. Speed ratios compare median mean-step latencies",
             "and are omitted if either arm lacks a complete three-cohort result. P95 is the median of per-cohort P95 values.", "",
             "| Prompt | Batch | Arm | Status | Mean ms | P50 ms | P95 ms | tokens/s | vs Dense | vs ALS | Prefill GPU GiB | Resident GPU GiB | Host K GiB | Slot hit |",
             "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    columns = ("prompt_tokens", "batch", "arm", "status", "mean_ms", "p50_ms", "p95_ms", "tokens_s",
               "speedup_vs_dense", "speedup_vs_als_full", "prefill_gpu_gib", "resident_gpu_gib", "host_k_gib", "slot_hit_rate")
    for row in groups:
        lines.append("| " + " | ".join(str(row[k]) if k in columns[:4] else fmt(row[k]) for k in columns) + " |")
    failures = [r for r in trials if r["status"] != "complete"]
    lines += ["", "## Failures", ""]
    for row in failures:
        lines.append(f"- P={row['prompt_tokens']}, B={row['batch']}, {row['arm']}, cohort {row['cohort']}: {row['status']}; `{Path(row['launcher_log']).name}`.")
    if not failures:
        lines.append("None.")
    lines += ["", "## Method and Limits", "",
              "- TP=8, DP=1, PP=1; one rank per L40S, no NVLink. Captured topology: `freeze/nvidia_topology.txt`.",
              "- BF16, TF32 disabled, eager execution. MLP and Dense were not optimized for this rerun.",
              "- Dense uses this repository's DenseTP8Attention / gpu_paged_attention backend with HF TP8. Ratios are against this matched control, not tuned vLLM, SGLang, or TensorRT-LLM deployments.",
              "- Basis-joint: V96, all-history B16R16 Page32 routing, 62 selected pages plus recent64, 2048 support slots. No coarse screening or 512-candidate limit.",
              "- Dense and ALS-full cache on GPU; Basis keeps historical exact K in pinned host memory with persistent GPU K slots.",
              "- GPU memory is max-rank PyTorch allocated memory; host K is summed allocation capacity, not measured host peak RSS.",
              "- Latency covers the full-model decode step, including MLP and collectives. It is not divided by batch size. Throughput uses maximum-rank wall time.",
              "- Slot hit rate describes the final measured step only. Profiling instrumentation was disabled.",
              "- Container restrictions prevent strict NUMA host-memory binding (`set_mempolicy: Operation not permitted`); CPU affinity is applied.",
              "- NCCL emits current-device inference warnings for barriers. Any failures remain in the manifest and logs.",
              "- Prefill OOM is not a decode-cache-only capacity limit. This is not a capacity search, TTFT/request benchmark, or task-quality evaluation.",
              "- Cross-rank agreement and finite logits do not establish task accuracy or equivalence to Dense.",
              "- Historical two-stage measurements were preserved and are not combined with these full-scan results.", "",
              "## Frozen Source and Command", "", f"Base commit: `{frozen['base_commit']}` plus the working-tree source in `freeze/source.tar.gz`.",
              "`freeze/manifest.json` lists archived files, inputs, environment and command; `freeze/source.patch` records tracked changes.",
              "The archive, rather than the base commit alone, identifies the tested implementation.", "", "```bash",
              "CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \\",
              "/workspace/miniforge3/bin/conda run --no-capture-output -n basis \\",
              " ".join(frozen["command"]), "```", "",
              "Summary command: `conda run --no-capture-output -n basis python benchmarks/system/summarize_tp8_full_scan.py`.",
              "Exact per-trial torchrun commands, timestamps and return codes are in `decode_grid_trials.json`.",
              "Top-level progress is in `run.log`; per-trial launcher and rank logs remain alongside raw JSON results.", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
