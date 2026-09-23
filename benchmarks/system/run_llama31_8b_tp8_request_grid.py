"""Run fresh TP8 request and steady-decode trials with every attempt retained."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks/system/bench_llama31_8b_tp8_request.py"
PROMPTS = Path("/workspace/runs/l31-cal128/tp8-benchmark-prompts")
DEFAULT_OUTPUT = ROOT / "results/system_benchmarks/tp8_external_baselines/shadowkv"
FULL_GRID = {
    "arms": ["dense", "basis_joint", "shadowkv"],
    "modes": ["request", "steady"],
    "contexts": [65536, 130048],
    "batches": [1, 4, 8],
    "cohorts": [0, 1, 2],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def failed_rank_records(folder):
    return [rank for rank in range(8) if not (folder / f"rank{rank}.json").is_file()]


def failure_details(folder, log_path):
    text = log_path.read_text()
    oom = "out of memory" in text.lower() or "outofmemoryerror" in text.lower()
    allocation = re.search(r"Tried to allocate ([0-9.]+ [MG]iB)", text)
    stages = {}
    truncated = []
    for rank in range(8):
        path = folder / f"rank{rank}.log"
        if path.is_file():
            lines = path.read_text().splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                truncated.append(rank)
            entries = [json.loads(line) for line in lines if line.endswith("\n")]
            stages[str(rank)] = entries[-1]["status"] if entries else "not_started"
        else:
            stages[str(rank)] = "not_started"
    return {"oom": oom, "allocation_request": allocation.group(1) if allocation else None,
            "missing_rank_results": failed_rank_records(folder), "last_rank_stages": stages,
            "truncated_rank_logs": truncated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=("dense", "basis_joint", "shadowkv"),
                        default=("dense", "basis_joint", "shadowkv"))
    parser.add_argument("--modes", nargs="+", choices=("request", "steady"),
                        default=("request", "steady"))
    parser.add_argument("--contexts", nargs="+", type=int, choices=(65536, 130048),
                        default=(65536, 130048))
    parser.add_argument("--batches", nargs="+", type=int, choices=(1, 4, 8),
                        default=(1, 4, 8))
    parser.add_argument("--cohorts", nargs="+", type=int, choices=(0, 1, 2),
                        default=(0, 1, 2))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    assert args.arms and args.modes and args.contexts and args.batches and args.cohorts
    assert BENCHMARK.is_file() and PROMPTS.is_dir()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "grid_trials.json"
    selection = {"arms": args.arms, "modes": args.modes, "contexts": args.contexts,
                 "batches": args.batches, "cohorts": args.cohorts}
    assert all(set(values) <= set(FULL_GRID[name]) for name, values in selection.items())
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["grid"] == FULL_GRID
    else:
        manifest = {"schema": "basisserve.llama31_8b.tp8_request_grid.v1",
                    "status": "running", "created_at_utc": now(),
                    "grid": FULL_GRID, "attempts": []}

    for length in args.contexts:
        for batch in args.batches:
            for cohort in args.cohorts:
                tokens = PROMPTS / f"p{length}_c{cohort}.safetensors"
                prompt_manifest = PROMPTS / f"p{length}_c{cohort}.json"
                assert tokens.is_file() and prompt_manifest.is_file()
                for arm in args.arms:
                    for mode in args.modes:
                        key = {"arm": arm, "mode": mode, "prompt_tokens": length,
                               "batch": batch, "cohort": cohort}
                        previous = [row for row in manifest["attempts"]
                                    if all(row[field] == value for field, value in key.items())]
                        if any(row["status"] == "complete" and
                               Path(row["output_dir"], "replica.json").is_file()
                               for row in previous):
                            continue
                        folder = (args.output_root / "raw" /
                                  f"{arm}_{mode}_p{length}_b{batch}_c{cohort}" /
                                  f"attempt{len(previous)}")
                        folder.mkdir(parents=True, exist_ok=True)
                        command = ["torchrun", "--standalone", "--nproc-per-node=8", str(BENCHMARK),
                                   "--arm", arm, "--mode", mode,
                                   "--length", str(length), "--batch", str(batch),
                                   "--cohort", str(cohort), "--tokens", str(tokens),
                                   "--prompt-manifest", str(prompt_manifest),
                                   "--output-dir", str(folder)]
                        log_path = folder / "launcher.log"
                        row = {**key, "attempt": len(previous), "output_dir": str(folder),
                               "launcher_log": str(log_path), "command": command,
                               "environment": {name: os.environ.get(name) for name in (
                                   "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "MAX_JOBS",
                                   "TORCH_CUDA_ARCH_LIST", "CONDA_DEFAULT_ENV")},
                               "started_at_utc": now(), "status": "running"}
                        manifest["attempts"].append(row)
                        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                        print("RUN", " ".join(command), flush=True)
                        with log_path.open("a", encoding="utf-8") as log:
                            completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                                       stdout=log, stderr=subprocess.STDOUT,
                                                       text=True)
                        replica = folder / "replica.json"
                        row["returncode"] = completed.returncode
                        row["finished_at_utc"] = now()
                        row["status"] = (
                            "complete" if completed.returncode == 0 and replica.is_file()
                            and not failed_rank_records(folder) else "failed")
                        if row["status"] == "failed":
                            row["failure"] = failure_details(folder, log_path)
                        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                        print(json.dumps({"trial": key, "status": row["status"],
                                          "returncode": row["returncode"]}), flush=True)

    latest = {}
    for row in manifest["attempts"]:
        key = tuple(row[field] for field in ("arm", "mode", "prompt_tokens", "batch", "cohort"))
        latest[key] = row
    failures = sum(row["status"] != "complete" for row in latest.values())
    expected = 1
    for values in FULL_GRID.values():
        expected *= len(values)
    manifest["status"] = (
        "partial" if len(latest) < expected
        else "complete" if failures == 0 else "complete_with_failures"
    )
    manifest["finished_at_utc"] = now()
    manifest["successful_trials"] = len(latest) - failures
    manifest["failed_trials"] = failures
    manifest["expected_trials"] = expected
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "successful_trials": len(latest) - failures,
                      "failed_trials": failures}), flush=True)


if __name__ == "__main__":
    main()
