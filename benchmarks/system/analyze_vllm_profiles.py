"""Summarize per-rank GPU kernels; kernel sums are not request wall latency."""

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re


STAGES = {
    "_append": "append_v_r_and_host_k", "_request_epoch": "request_epoch",
    "_bounds": "exact_k_bounds", "_coarse": "coarse_scores", "_select_coarse": "candidates512",
    "_query_code": "query_r16", "_fine": "fine_page_scores", "_select_fine": "select62",
    "_pack": "pack_support", "_reset": "slot_request_reset", "_lookup": "slot_lookup",
    "_plan": "slot_plan", "_fetch": "host_k_fetch", "_attention": "selected_qk_pv",
    "_merge": "attention_merge", "_gather": "prefill_kv_staging",
}


def stage(name):
    lower = name.lower()
    if "nccl" in lower:
        return "nccl_allgather" if "allgather" in lower else "nccl_allreduce" if "allreduce" in lower else "nccl_other"
    for kernel, label in STAGES.items():
        if re.search(r"(?:^|[ :])" + re.escape(kernel) + r"(?:$|[ <])", name):
            return label
    if "flash" in lower or "attention" in lower:
        return "native_attention"
    if "gemm" in lower or "gemv" in lower or "cutlass" in lower:
        return "linear_unattributed"
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.rglob("*.pt.trace.json*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            trace = json.load(handle)
        groups = defaultdict(lambda: dict(count=0, gpu_duration_sum_us=0., maximum_kernel_us=0., launches=[]))
        for event in trace.get("traceEvents", []):
            if event.get("cat") != "kernel" or "dur" not in event:
                continue
            name = event["name"]
            data = groups[(stage(name), name)]
            data["count"] += 1
            data["gpu_duration_sum_us"] += event["dur"]
            data["maximum_kernel_us"] = max(data["maximum_kernel_us"], event["dur"])
            launch = {key: value for key, value in event.get("args", {}).items()
                      if key in ("grid", "block", "registers per thread", "shared memory", "blocks per SM", "warps per SM")}
            if launch not in data["launches"]:
                data["launches"].append(launch)
        for (label, name), data in groups.items():
            rows.append(dict(trace=str(path), stage=label, kernel=name, **data))
    assert rows
    output = args.root / "kernel_profile_summary.json"
    output.write_text(json.dumps(dict(
        scope="per-trace sums of GPU kernel durations; not additive end-to-end wall times",
        launch_metadata_scope="static launch attributes; not measured SM utilization or PCIe bytes",
        kernels=rows), indent=2) + "\n")
    totals = defaultdict(lambda: dict(kernel_calls=0, gpu_duration_sum_us=0.))
    for row in rows:
        total = totals[(row["trace"], row["stage"])]
        total["kernel_calls"] += row["count"]
        total["gpu_duration_sum_us"] += row["gpu_duration_sum_us"]
    with (args.root / "kernel_stages.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trace", "stage", "kernel_calls", "gpu_duration_sum_us"])
        writer.writeheader()
        for (trace, label), total in sorted(totals.items()):
            writer.writerow(dict(trace=trace, stage=label, **total))
    print(f"Saved {len(rows)} kernel records from {len(set(r['trace'] for r in rows))} traces.")


if __name__ == "__main__":
    main()
