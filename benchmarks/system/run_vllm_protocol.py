"""Run isolated vLLM workloads from the supplied TP4/TP8 experiment protocol."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time


MATRIX = {4096: (1, 8, 32, 128), 16384: (1, 4, 8, 16),
          65536: (1, 4, 8, 16), 130048: (1, 4, 8, 16)}
ARMS = ("dense", "als_full", "basis_joint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", choices=("anchor", "matrix", "decode", "profiles"), required=True)
    args = parser.parse_args()
    assert Path(sys.prefix).name == "basis"
    assert os.environ.get("SLURM_JOB_ID")
    if args.plan == "anchor":
        workloads = [(16384, 1, experiment) for experiment in ("request", "decode", "profile")]
    elif args.plan == "profiles":
        workloads = [(16384, 1, "profile"), (130048, 8, "profile")]
    else:
        workloads = [(p, b, experiment) for p, batches in MATRIX.items()
                     for b in batches for experiment in (("decode",) if args.plan == "decode" else ("request", "decode"))]
    args.output.mkdir(parents=True, exist_ok=True)
    for prompt, batch, experiment in workloads:
        for arm in ARMS:
            folder = args.output / f"p{prompt}_b{batch}" / arm / experiment
            folder.mkdir(parents=True, exist_ok=True)
            existing = folder / "config.json"
            if existing.exists() and json.loads(existing.read_text()).get("status") == "complete":
                print(f"PROTOCOL retained completed result {folder}", flush=True)
                continue
            command = ["bash", "benchmarks/system/launch_vllm_bench.sh", str(args.tp), str(folder),
                       arm, str(prompt), str(batch), experiment]
            logfile = folder / f"run-{os.environ['SLURM_JOB_ID']}.log"
            print("PROTOCOL " + shlex.join(command), flush=True)
            started = time.time()
            with logfile.open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            log_text = logfile.read_text()
            configuration = json.loads(existing.read_text()) if existing.exists() else {}
            status = "complete" if result.returncode == 0 and configuration.get("status") == "complete" else "failed"
            failure = None
            if status != "complete":
                if configuration.get("failure_reason"):
                    failure = configuration["failure_reason"]
                elif "OutOfMemoryError" in log_text or "CUDA out of memory" in log_text:
                    failure = "gpu_oom"
                elif "host_bytes <= layer.joint_host_budget" in log_text:
                    failure = "host_budget_exceeded"
                elif "No available memory for the cache blocks" in log_text:
                    failure = "gpu_cache_capacity"
                elif "Free memory on device" in log_text and "is less than desired GPU memory utilization" in log_text:
                    failure = "gpu_occupied"
                else:
                    failure = "runtime_failure"
            record = dict(prompt_tokens=prompt, batch=batch, arm=arm, experiment=experiment,
                          tp=args.tp, status=status, failure_reason=failure, exit_code=result.returncode,
                          started_unix=started, ended_unix=time.time(), command=command, log=str(logfile))
            with (args.output / "runs.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print("PROTOCOL " + json.dumps(record), flush=True)
            # Resource failures are recorded per workload; an implementation error
            # stops the sweep so it does not consume GPUs repeating the same fault.
            if failure == "runtime_failure":
                return 1
    rows = []
    for path in sorted(args.output.glob("p*_b*/*/*/summary.json")):
        result = json.loads(path.read_text())
        trials = result["trials"]
        if len(trials) != 3 or any(r["status"] != "complete" for r in trials):
            continue
        row = {key: trials[0][key] for key in ("experiment", "arm", "tp", "prompt_tokens", "batch")}
        row.update(result["medians"])
        row["trial_count"] = len(trials)
        row["prompt_cohort_hashes"] = json.dumps([t["prompt_cohort_sha256"] for t in trials])
        row["source"] = str(path)
        rows.append(row)
    for row in rows:
        dense = next((d for d in rows if d["arm"] == "dense" and all(
            row[k] == d[k] for k in ("experiment", "tp", "prompt_tokens", "batch", "prompt_cohort_hashes")
        )), None)
        metric = "cohort_e2e_s" if row["experiment"] == "request" else "decode_step_mean_ms"
        if dense is not None:
            dense_trials = json.loads(Path(dense["source"]).read_text())["trials"]
            method_trials = json.loads(Path(row["source"]).read_text())["trials"]
            dense_by_cohort = {t["prompt_cohort_sha256"]: t for t in dense_trials}
            row["speedup_vs_dense"] = statistics.median(
                dense_by_cohort[t["prompt_cohort_sha256"]][metric] / t[metric] for t in method_trials)
        else:
            row["speedup_vs_dense"] = None
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (args.output / "summary.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
