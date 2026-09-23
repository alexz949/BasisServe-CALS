"""Run the fixed 16K prefill, 128-output-token TP8 V-only batch sweep."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.system.run_qwen3_8b_tp8_v_only_memory_grid import (
    BENCHMARK,
    PROMPTS,
    failure_details,
    missing_rank_results,
    now,
)


DEFAULT_OUTPUT = ROOT / "results/system_benchmarks/tp8_external_baselines/starkv_v_only/batch_sweep_16k"
GRID = {
    "arms": ["basis_v64", "star_v_adaptive"],
    "prompt_tokens": 16384,
    "batches": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "cohort": 0,
    "chunk_size": 256,
    "output_tokens": 128,
    "reserve_decode_tokens": 128,
    "repeat_prompts": True,
}


def trial_command(arm: str, batch: int, output_dir: Path) -> list[str]:
    tokens = PROMPTS / f"p{GRID['prompt_tokens']}_c{GRID['cohort']}.safetensors"
    prompt_manifest = PROMPTS / f"p{GRID['prompt_tokens']}_c{GRID['cohort']}.json"
    return [
        "torchrun", "--standalone", "--nproc-per-node=8", str(BENCHMARK),
        "--arm", arm, "--length", str(GRID["prompt_tokens"]), "--batch", str(batch),
        "--cohort", str(GRID["cohort"]), "--chunk-size", str(GRID["chunk_size"]),
        "--output-tokens", str(GRID["output_tokens"]),
        "--reserve-decode-tokens", str(GRID["reserve_decode_tokens"]),
        "--repeat-prompts", "--tokens", str(tokens),
        "--prompt-manifest", str(prompt_manifest), "--output-dir", str(output_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.output_root.resolve()
    assert BENCHMARK.is_file()
    assert (PROMPTS / "p16384_c0.safetensors").is_file()
    assert (PROMPTS / "p16384_c0.json").is_file()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "grid_trials.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["grid"] == GRID
        for previous in manifest["attempts"]:
            if previous["status"] == "running":
                previous["status"] = "failed"
                previous["finished_at_utc"] = now()
                previous["returncode"] = None
                folder = Path(previous["output_dir"])
                launcher = Path(previous["launcher_log"])
                previous["failure"] = (
                    failure_details(folder, launcher) if launcher.is_file()
                    else {"oom": False, "phase": "startup_or_other", "oom_ranks": [],
                          "oom_rank_stages": {}, "allocation_request": None,
                          "missing_rank_results": missing_rank_results(folder),
                          "last_rank_stages": {}, "truncated_rank_logs": []}
                )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        manifest = {
            "schema": "basisserve.qwen3_8b.tp8_v_only_batch_sweep.v1",
            "status": "running",
            "created_at_utc": now(),
            "grid": GRID,
            "attempts": [],
        }

    for batch in GRID["batches"]:
        for arm in GRID["arms"]:
            key = {"arm": arm, "prompt_tokens": GRID["prompt_tokens"],
                   "batch": batch, "cohort": GRID["cohort"]}
            previous = [row for row in manifest["attempts"]
                        if all(row[name] == value for name, value in key.items())]
            if any(row["status"] == "complete" and
                   not missing_rank_results(Path(row["output_dir"])) for row in previous):
                continue
            if any(row["status"] == "failed" and row["failure"]["oom"] for row in previous):
                continue
            folder = root / "raw" / f"{arm}_p16384_b{batch}_c0" / f"attempt{len(previous)}"
            folder.mkdir(parents=True, exist_ok=True)
            cmd = trial_command(arm, batch, folder)
            launcher = folder / "launcher.log"
            row = {
                **key,
                "attempt": len(previous),
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
        latest[(row["arm"], row["batch"])] = row
    failed = sum(row["status"] != "complete" for row in latest.values())
    expected = len(GRID["arms"]) * len(GRID["batches"])
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
