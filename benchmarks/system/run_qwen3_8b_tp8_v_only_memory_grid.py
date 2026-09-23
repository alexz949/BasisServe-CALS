"""Run the fixed Qwen3 TP8 V-only capacity grid and preserve every attempt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks/system/bench_qwen3_8b_tp8_v_only_memory.py"
PROMPTS = ROOT / "results/system_benchmarks/tp8_external_baselines/starkv_v_only/prompts"
DEFAULT_OUTPUT = ROOT / "results/system_benchmarks/tp8_external_baselines/starkv_v_only"
FULL_GRID = {
    "arms": ["basis_v64", "star_v_adaptive"],
    "contexts": [4096, 16384, 32768, 65536, 98304, 130048],
    "batches": [1, 4, 8],
    "cohorts": [0],
    "chunk_size": 4096,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def missing_rank_results(folder: Path) -> list[int]:
    return [rank for rank in range(8) if not (folder / f"rank{rank}.json").is_file()]


def failure_details(folder: Path, launcher: Path) -> dict:
    text = launcher.read_text()
    oom = "out of memory" in text.lower() or "outofmemoryerror" in text.lower()
    allocation = re.search(r"Tried to allocate ([0-9.]+ [MG]iB)", text)
    oom_ranks = sorted({int(rank) for rank in re.findall(
        r"\[rank(\d+)\]:[^\n]*(?:out of memory|OutOfMemoryError)", text, re.IGNORECASE
    )})
    stages = {}
    truncated = []
    for rank in range(8):
        path = folder / f"rank{rank}.log"
        lines = path.read_text().splitlines(keepends=True) if path.is_file() else []
        if lines and not lines[-1].endswith("\n"):
            truncated.append(rank)
        valid = [json.loads(line) for line in lines if line.startswith("{") and line.rstrip().endswith("}")]
        stages[str(rank)] = valid[-1]["status"] if valid else "not_started"
    phase_by_stage = {
        "model_load_start": "model_load",
        "cache_allocate_layer": "cache_state_allocation",
        "prefill_chunk_start": "transient_prefill_workspace",
        "decode_ready": "decode_workspace",
        "decode_start": "decode_workspace",
        "decode_step_complete": "decode_workspace",
    }
    relevant = [stages[str(rank)] for rank in oom_ranks] if oom_ranks else list(stages.values())
    phases = [phase_by_stage.get(stage) for stage in relevant]
    phase = phases[0] if phases and None not in phases and len(set(phases)) == 1 else "unclassified"
    return {
        "oom": oom,
        "oom_ranks": oom_ranks,
        "phase": phase,
        "oom_rank_stages": {str(rank): stages[str(rank)] for rank in oom_ranks},
        "allocation_request": allocation.group(1) if allocation else None,
        "missing_rank_results": missing_rank_results(folder),
        "last_rank_stages": stages,
        "truncated_rank_logs": truncated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=FULL_GRID["arms"], default=FULL_GRID["arms"])
    parser.add_argument("--contexts", nargs="+", type=int, choices=FULL_GRID["contexts"],
                        default=FULL_GRID["contexts"])
    parser.add_argument("--batches", nargs="+", type=int, choices=FULL_GRID["batches"],
                        default=FULL_GRID["batches"])
    parser.add_argument("--cohorts", nargs="+", type=int, choices=FULL_GRID["cohorts"],
                        default=FULL_GRID["cohorts"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    assert args.arms and args.contexts and args.batches and args.cohorts
    assert all(set(getattr(args, name)) <= set(FULL_GRID[name])
               for name in ("arms", "contexts", "batches", "cohorts"))
    assert BENCHMARK.is_file() and PROMPTS.is_dir()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "grid_trials.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["grid"] == FULL_GRID
        for previous in manifest["attempts"]:
            if previous["status"] == "running":
                previous["status"] = "failed"
                previous["finished_at_utc"] = now()
                previous["returncode"] = None
                folder = Path(previous["output_dir"])
                launcher = Path(previous["launcher_log"])
                previous["failure"] = (
                    failure_details(folder, launcher) if launcher.is_file()
                    else {"oom": False, "phase": "startup_or_other",
                          "oom_ranks": [], "oom_rank_stages": {}, "allocation_request": None,
                          "missing_rank_results": missing_rank_results(folder),
                          "last_rank_stages": {}, "truncated_rank_logs": []}
                )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        manifest = {
            "schema": "basisserve.qwen3_8b.tp8_v_only_memory_grid.v1",
            "status": "running",
            "created_at_utc": now(),
            "grid": FULL_GRID,
            "attempts": [],
        }

    for length in args.contexts:
        for batch in args.batches:
            for cohort in args.cohorts:
                tokens = PROMPTS / f"p{length}_c{cohort}.safetensors"
                prompt_manifest = PROMPTS / f"p{length}_c{cohort}.json"
                assert tokens.is_file() and prompt_manifest.is_file()
                for arm in args.arms:
                    key = {"arm": arm, "prompt_tokens": length, "batch": batch, "cohort": cohort}
                    previous = [row for row in manifest["attempts"]
                                if all(row[name] == value for name, value in key.items())]
                    if any(row["status"] == "complete" and
                           not missing_rank_results(Path(row["output_dir"])) for row in previous):
                        continue
                    if any(row["status"] == "failed" and row["failure"]["oom"] for row in previous):
                        continue
                    folder = (args.output_root / "raw" / f"{arm}_p{length}_b{batch}_c{cohort}" /
                              f"attempt{len(previous)}")
                    folder.mkdir(parents=True, exist_ok=True)
                    cmd = ["torchrun", "--standalone", "--nproc-per-node=8", str(BENCHMARK),
                           "--arm", arm, "--length", str(length), "--batch", str(batch),
                           "--cohort", str(cohort), "--chunk-size", str(FULL_GRID["chunk_size"]),
                           "--tokens", str(tokens), "--prompt-manifest", str(prompt_manifest),
                           "--output-dir", str(folder)]
                    launcher = folder / "launcher.log"
                    row = {
                        **key,
                        "attempt": len(previous),
                        "chunk_size": FULL_GRID["chunk_size"],
                        "output_dir": str(folder),
                        "launcher_log": str(launcher),
                        "command": cmd,
                        "environment": {name: os.environ.get(name) for name in (
                            "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CONDA_DEFAULT_ENV")},
                        "started_at_utc": now(),
                        "status": "running",
                    }
                    manifest["attempts"].append(row)
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    print("RUN", " ".join(cmd), flush=True)
                    with launcher.open("a", encoding="utf-8") as log:
                        completed = subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(),
                                                   stdout=log, stderr=subprocess.STDOUT, text=True)
                    row["returncode"] = completed.returncode
                    row["finished_at_utc"] = now()
                    row["status"] = (
                        "complete" if completed.returncode == 0 and not missing_rank_results(folder)
                        else "failed"
                    )
                    if row["status"] == "failed":
                        row["failure"] = failure_details(folder, launcher)
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    print(json.dumps({"trial": key, "status": row["status"],
                                      "returncode": row["returncode"]}), flush=True)

    latest = {}
    for row in manifest["attempts"]:
        key = tuple(row[field] for field in ("arm", "prompt_tokens", "batch", "cohort"))
        latest[key] = row
    failed = sum(row["status"] != "complete" for row in latest.values())
    expected = (len(FULL_GRID["arms"]) * len(FULL_GRID["contexts"]) *
                len(FULL_GRID["batches"]) * len(FULL_GRID["cohorts"]))
    manifest["status"] = (
        "partial" if len(latest) < expected
        else "complete" if failed == 0 else "complete_with_failures"
    )
    manifest["finished_at_utc"] = now()
    manifest["successful_trials"] = len(latest) - failed
    manifest["failed_trials"] = failed
    manifest["expected_trials"] = expected
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"],
                      "successful_trials": len(latest) - failed,
                      "failed_trials": failed}), flush=True)


if __name__ == "__main__":
    main()
