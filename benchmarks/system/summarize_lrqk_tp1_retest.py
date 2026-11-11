"""Summary of the LRQK TP1 retest (run_lrqk_tp1_retest.py): runs.csv, summary.csv, manifest.json and SUMMARY.md.

  python benchmarks/system/summarize_lrqk_tp1_retest.py --root results/rebuttal/v2q8_l40s_sparse/tp1_lrqk_retest
"""

import argparse
import csv
import json
from pathlib import Path
import statistics

LABELS = {"lrqk_upstream_koff_v128": "K offload, V on GPU", "lrqk_upstream_kvoff_v128": "K and V offload",
          "lrqk_upstream_koff_noempty_v128": "K offload, no decode empty_cache (trial)",
          "lrqk_upstream_kvoff_noempty_v128": "K and V offload, no decode empty_cache (trial)"}
GIB = 2 ** 30
# (key, title, unit, digits); request and steady metrics stay separate.
METRICS = (
    ("decode_ready_s", "Decode-ready (prefill until all rows placed)", "s", 2),
    ("request_e2e_s", "Request end to end, 128 tokens", "s", 2),
    ("request_decode_mean_ms", "Request: mean decode step (127 steps)", "ms", 1),
    ("steady_mean_ms", "Steady: mean step", "ms", 1),
    ("steady_p50_ms", "Steady: p50 step", "ms", 1),
    ("steady_p95_ms", "Steady: p95 step", "ms", 1),
    ("steady_tokens_per_s", "Steady: tokens per second", "tok/s", 2),
    ("gpu_resident_gib", "GPU resident, allocated", "GiB", 2),
    ("prefill_peak_allocated_gib", "Prefill peak allocated", "GiB", 2),
    ("prefill_peak_reserved_gib", "Prefill peak reserved", "GiB", 2),
    ("decode_peak_allocated_gib", "Decode peak allocated", "GiB", 2),
    ("decode_peak_reserved_gib", "Decode peak reserved", "GiB", 2),
    ("host_offload_pinned_gib", "Pinned host memory holding offloaded state", "GiB", 2),
    ("process_rss_gib", "Process RSS", "GiB", 2),
)


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def row(folder):
    record, result = load(folder / "run.json"), load(folder / "result.json")
    arm, mode, cohort = folder.name.split("_request_")[0].split("_steady_")[0], \
        "request" if "_request_" in folder.name else "steady", int(folder.name.rsplit("_c", 1)[1])
    values = dict(arm=arm, mode=mode, cohort=cohort, status=record.get("status", "missing"),
                  exit_code=record.get("exit_code"), started_utc=record.get("started_utc"),
                  finished_utc=record.get("finished_utc"), wall_seconds=record.get("seconds"),
                  warning_lines=record.get("warning_lines"), failure=record.get("failure", ""),
                  **{f"load_{key}": value for key, value in record.get("load", {}).items()
                     if key != "other_gpu_processes_seen"},
                  other_gpu_processes=len(record.get("load", {}).get("other_gpu_processes_seen", [])))
    if result.get("status") != "complete":
        return values
    rank = result["ranks"][0]
    measured = rank["measurement"]
    environment = result["environment"]
    values.update(threads=rank["cpu_placement"]["torch_threads"],
                  torch_compile_disabled=environment["effective"]["torch_compile_disabled"],
                  float32_matmul_precision=environment["effective"]["float32_matmul_precision"],
                  cuda_matmul_allow_tf32=environment["effective"]["cuda_matmul_allow_tf32"],
                  ordered_prefill="OrderedOffloadPrefill" in " ".join(result["configuration"].get("prefill_class_mro", [])),
                  physical_gpu=environment["gpus"].split(",")[3].strip(), cpus=len(rank["cpu_placement"]["cpus"]),
                  allocator=environment["pytorch_cuda_alloc_conf"] or "PyTorch default")
    if mode == "request":
        after_prefill, after_decode = measured["memory_after_prefill"], measured["memory_after_decode"]
        values.update(ttft_s=measured["ttft_seconds"][0], placement_s=measured["placement_seconds"],
                      decode_ready_s=measured["batch_ready_seconds"], request_e2e_s=measured["request_seconds"],
                      request_decode_mean_ms=measured["decode_mean_ms_per_step"],
                      output_tokens=len(measured["generated_tokens"][0]),
                      # Request mode reads memory twice: after prefill and placement, and after the last decode step.
                      request_gpu_resident_gib=after_decode["allocated"] / GIB,
                      request_prefill_peak_allocated_gib=after_prefill["peak_allocated"] / GIB,
                      request_prefill_peak_reserved_gib=after_prefill["peak_reserved"] / GIB,
                      request_decode_peak_allocated_gib=after_decode["peak_allocated"] / GIB,
                      request_decode_peak_reserved_gib=after_decode["peak_reserved"] / GIB,
                      request_process_rss_gib=after_decode["process_rss"] / GIB)
    else:
        wall = measured["wall_ms"]
        ordered = sorted(wall)
        phases, inventory = measured["memory"], rank["inventory"]
        values.update(
            steady_steps=len(wall), steady_mean_ms=statistics.fmean(wall), steady_p50_ms=statistics.median(wall),
            steady_p95_ms=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            steady_min_ms=ordered[0], steady_max_ms=ordered[-1], steady_tokens_per_s=1000 / statistics.fmean(wall),
            steady_cuda_p50_ms=statistics.median(measured["cuda_ms"]),
            steady_prefill_s=measured["prefill_seconds"],
            gpu_resident_gib=phases["decode_ready"]["allocated"] / GIB,
            prefill_peak_allocated_gib=phases["after_prefill"]["peak_allocated"] / GIB,
            prefill_peak_reserved_gib=phases["after_prefill"]["peak_reserved"] / GIB,
            decode_peak_allocated_gib=phases["decode_end"]["peak_allocated"] / GIB,
            decode_peak_reserved_gib=phases["decode_end"]["peak_reserved"] / GIB,
            host_offload_pinned_gib=sum(item["backing_bytes"] for item in inventory["rows"]
                                        if item["device"] == "host" and item["pinned"]) / GIB,
            host_inventory_gib=inventory["host_backing_bytes"] / GIB,
            process_rss_gib=max(phase["process_rss"] for phase in phases.values()) / GIB)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = sorted((row(folder) for folder in sorted((args.root / "formal").glob("*_c[0-9]"))),
                  key=lambda item: (item["mode"], item["cohort"], item["arm"]))
    ARMS = {arm: LABELS.get(arm, arm) for arm in sorted({item["arm"] for item in rows})}
    columns = list(dict.fromkeys(key for item in rows for key in item))
    with (args.root / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for arm in ARMS:
        for key, title, unit, digits in METRICS:
            by_cohort = {item["cohort"]: item[key] for item in rows if item["arm"] == arm and key in item}
            values = list(by_cohort.values())
            summary.append(dict(arm=arm, metric=key, unit=unit, runs=len(values),
                                **{f"cohort_{cohort}": by_cohort.get(cohort) for cohort in (0, 1, 2)},
                                median=statistics.median(values) if values else None,
                                minimum=min(values, default=None), maximum=max(values, default=None)))
    with (args.root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    manifests = [load(path) for path in sorted(args.root.glob("manifest_*.json"))]
    (args.root / "manifest.json").write_text(json.dumps(dict(
        invocations=manifests, formal_runs=[dict(name=f"{item['arm']}_{item['mode']}_b1_t65536_c{item['cohort']}",
                                                 status=item["status"]) for item in rows],
        deliverables=["correctness-summary.json", "manifest.json", "runs.csv", "summary.csv", "SUMMARY.md"]),
        indent=2) + "\n")

    correctness = load(args.root / "correctness-summary.json")
    latest = manifests[-1] if manifests else {}
    cell = lambda value, digits: "n/a" if value is None else f"{value:.{digits}f}"
    lines = ["# LRQK TP1 retest after the prefill offload stream-order fix", ""]
    complete = sum(item["status"] == "complete" for item in rows)
    lines += [f"Runs complete: {complete} of {len(rows)} present (arms: {', '.join(ARMS)}).",
              "The three cohorts are three different prompts (row 0 of cohort 0, 1, 2 of the TP4 64K inputs), not "
              "repetitions of one prompt.", ""]
    if latest:
        gpu, cpu, software = latest["gpu"], latest["cpu"], latest["software"]
        lines += ["## Setup", "",
                  f"- Llama-3.1-8B-Instruct, BF16, original V128/Wo, revision `{latest['model']['revision']}`; TP1, B1, "
                  "65536 prompt tokens.",
                  f"- LRQK: rank {latest['lrqk']['rank']}, top-k {latest['lrqk']['topk']}, {latest['lrqk']['lite']} lite "
                  f"tokens, iterations {tuple(latest['lrqk']['iterations'])}, tolerance {tuple(latest['lrqk']['tolerance'])}; "
                  f"upstream `tenghuilee/LRQK@{latest['repository']['lrqk_head'][:8]}`.",
                  f"- Repository `{latest['repository']['head'][:7]}` (contains the fix `be7c8a2`: "
                  f"{latest['repository']['contains_fix_be7c8a2']}), with uncommitted entry-point changes "
                  f"(diff sha256 `{latest['repository']['uncommitted_diff_sha256'][:12]}`).",
                  f"- GPU: {gpu['name']}, driver {gpu['driver_version']}, host `{latest['host']}`, one GPU.",
                  f"- CPU: {latest['cpu_model']}; {cpu['threads']} threads on {cpu['physical_cores']} physical cores "
                  f"(both hyperthreads) of NUMA node {cpu['gpu_numa_node']}, `OMP_NUM_THREADS={cpu['threads']}`.",
                  f"- Software: conda env `{software['conda_environment']}`, Python {software['python']}, PyTorch "
                  f"{software['torch']}, CUDA {software['cuda_runtime']}, Transformers {software['transformers']}, "
                  f"NCCL {software['nccl']}; {software['allocator']}.",
                  "- Each run: fresh process, same-shape warmup request, fresh state, then the measurement. Request = "
                  "128 tokens; steady = 16 conditioning + 128 measured steps. No component timers.", ""]
    lines += ["## Correctness", ""]
    order = correctness.get("offload_order", {})
    if order:
        wrong = {name: order["ordered"][name]["wrong"] for name in ("K", "V", "A_K")} if order.get("ordered") else {}
        before = {name: order["upstream_unordered"][name]["wrong"] for name in ("K", "V", "A_K")}
        lines += [f"- Stream-order check (`check_lrqk_offload_order.py`, read from its `result.json`): ordered copy "
                  f"wrong elements {wrong}; passed: {order['passed']}. The unordered upstream copy in the same probe: "
                  f"{before}. {order['scope']}"]
    for key, title in (("smoke_8k", "8192-token smoke"), ("probe_64k", "65536-token probe")):
        for item in correctness.get(key, []):
            lines += [f"- {title}, `{item['name']}`: status {item['status']}, {item['output_tokens']} tokens, "
                      f"log-probabilities finite: {item['logprobs_finite']}, imported files match the reviewed "
                      f"hashes: {all(item['loaded']['files_match'].values())}, ordered prefill class in use: "
                      f"{item['loaded']['ordered_prefill_in_mro']}; passed: {item['passed']}."]
    consistency = correctness.get("placement_consistency", {})
    if consistency:
        lines += ["", f"Placement consistency (8192 tokens). {consistency['method']}", "",
                  "| Cohort | Steps | K+V argmax = K-only token | Max abs log-prob difference | Mean abs difference |",
                  "|---|---|---|---|---|"]
        for item in consistency["cohorts"]:
            lines += [f"| {item['cohort']} | {item['steps']} | {item['argmax_matches']} / {item['steps']} | "
                      f"{item['logprob_max_abs_difference']} | {item['logprob_mean_abs_difference']} |"]
        lines += ["", f"Exact agreement (top-1 100 %, maximum difference 0): {consistency['exact']}. "
                  f"{consistency['scope']}", ""]
    lines += ["## Latency and memory", "",
              "Median of the three cohorts, with the range in brackets. Request and steady step times are different "
              "metrics and are not merged.", "",
              "| | " + " | ".join(ARMS.values()) + " |", "|---|" + "---|" * len(ARMS)]
    for key, title, unit, digits in METRICS:
        cells = []
        for arm in ARMS:
            item = next(entry for entry in summary if entry["arm"] == arm and entry["metric"] == key)
            cells.append("n/a" if not item["runs"] else
                         f"{cell(item['median'], digits)} {unit} ({cell(item['minimum'], digits)} to "
                         f"{cell(item['maximum'], digits)}; n={item['runs']})")
        lines += [f"| {title} | " + " | ".join(cells) + " |"]
    lines += ["", "Per cohort:", "", "| Arm | Mode | Cohort | Status | Decode-ready s | E2E s | Request step ms | "
              "Steady mean / p50 / p95 ms | Others on pinned CPUs, max |", "|---|---|---|---|---|---|---|---|---|"]
    for item in rows:
        steady = (f"{cell(item.get('steady_mean_ms'), 1)} / {cell(item.get('steady_p50_ms'), 1)} / "
                  f"{cell(item.get('steady_p95_ms'), 1)}") if item["mode"] == "steady" else ""
        lines += [f"| `{item['arm']}` | {item['mode']} | {item['cohort']} | {item['status']} | "
                  f"{cell(item.get('decode_ready_s'), 2) if item['mode'] == 'request' else ''} | "
                  f"{cell(item.get('request_e2e_s'), 2) if item['mode'] == 'request' else ''} | "
                  f"{cell(item.get('request_decode_mean_ms'), 1) if item['mode'] == 'request' else ''} | {steady} | "
                  f"{item.get('load_pinned_busy_by_others_max')} |"]
    failed = [item for item in rows if item["status"] != "complete"]
    lines += ["", "## Warnings, failures and interference", ""]
    lines += [f"- `{item['arm']}` {item['mode']} cohort {item['cohort']}: {item['status']}; {item['failure']}"
              for item in failed] or ["- No formal run failed."]
    history = [json.loads(line) for line in (args.root / "runs.jsonl").read_text().splitlines()]
    lines += [f"- Earlier attempt, kept in `runs.jsonl`: `{item['name']}` started {item['started_utc']}: {item['status']} "
              f"(exit {item['exit_code']}); {item['failure']}" for item in history
              if item["status"] not in ("complete", "exit0")]
    effective = sorted({(item.get("torch_compile_disabled"), item.get("float32_matmul_precision"),
                         item.get("cuda_matmul_allow_tf32")) for item in rows if item["status"] == "complete"})
    lines += [f"- Effective settings (torch.compile disabled, float32 matmul precision, cuda.matmul.allow_tf32), "
              f"read after each measurement: {effective}.",
              "- Warning lines per run (torch.compile recompiles, autotune, deprecations) are counted in `runs.csv` "
              "and listed in each `run.json`.",
              "- `load_pinned_busy_by_others_max`: the most logical CPUs of the 24 pinned ones that other processes "
              "kept busy in any 5 s sample; `load.jsonl` in each run directory has the samples.", "",
              "## Limits", "",
              "- This is the absolute TP1 latency of the fixed LRQK on this host. The earlier TP1/64K numbers (K+V "
              "about 369 ms/step, K-only about 430 ms/step) were measured before the fix, on another host (Xeon "
              "Platinum 8558, driver 570, Transformers 5.22.2) and under CPU contention; differences from them are "
              "not attributable to the stream fix.",
              "- BasisKV and ShadowKV were not rerun here; any table that puts these numbers next to theirs compares "
              "different hosts and sessions, not a matched pair.",
              "- The correctness checks cover implementation and placement consistency, not quality against Dense.", ""]
    (args.root / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
