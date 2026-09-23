"""Run the frozen TP8 pure-decode grid in fresh processes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = REPOSITORY_ROOT / "benchmarks/system/bench_llama31_8b_tp8_combined.py"
PROMPT_ROOT = Path("/workspace/runs/l31-cal128/tp8-benchmark-prompts")
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/system_benchmarks/llama31_8b_tp8_combined"
MATRIX = {
    4096: (1, 8, 32, 128),
    16384: (1, 4, 8, 16),
    65536: (1, 4, 8, 16),
    130048: (1, 4, 8, 16),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms", nargs="+", choices=("dense", "als_full", "basis_joint"),
        default=("dense", "als_full", "basis_joint"),
    )
    parser.add_argument(
        "--contexts", nargs="+", type=int, choices=tuple(MATRIX), default=tuple(MATRIX)
    )
    parser.add_argument("--cohorts", nargs="+", type=int, choices=(0, 1, 2), default=(0, 1, 2))
    parser.add_argument("--conditioning-steps", type=int, default=16)
    parser.add_argument("--measure-steps", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    assert args.conditioning_steps >= 0 and args.measure_steps > 0
    assert PROMPT_ROOT.is_dir() and BENCHMARK.is_file()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "decode_grid_trials.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] in ("running", "complete_with_failures", "complete")
    else:
        manifest = {
            "status": "running",
            "format": "basisserve.llama31_8b.tp8_combined_decode_grid.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "matrix": {str(key): list(value) for key, value in MATRIX.items()},
            "arms": list(args.arms),
            "contexts": list(args.contexts),
            "cohorts": list(args.cohorts),
            "conditioning_steps": args.conditioning_steps,
            "measure_steps": args.measure_steps,
            "trials": [],
        }
    indexed = {
        (row["arm"], row["prompt_tokens"], row["batch"], row["cohort"]): row
        for row in manifest["trials"]
    }

    for prompt_tokens in args.contexts:
        tokens = PROMPT_ROOT / f"p{prompt_tokens}_c0.safetensors"
        assert tokens.is_file()
        for batch in MATRIX[prompt_tokens]:
            for cohort in args.cohorts:
                tokens = PROMPT_ROOT / f"p{prompt_tokens}_c{cohort}.safetensors"
                prompt_manifest = PROMPT_ROOT / f"p{prompt_tokens}_c{cohort}.json"
                assert tokens.is_file() and prompt_manifest.is_file()
                for arm in args.arms:
                    key = (arm, prompt_tokens, batch, cohort)
                    previous = indexed.get(key)
                    if previous is not None and previous["returncode"] == 0:
                        continue
                    command = [
                        "torchrun",
                        "--standalone",
                        "--nproc-per-node=8",
                        str(BENCHMARK),
                        "--arm", arm,
                        "--length", str(prompt_tokens),
                        "--batch", str(batch),
                        "--conditioning-steps", str(args.conditioning_steps),
                        "--measure-steps", str(args.measure_steps),
                        "--repeat", str(cohort),
                        "--tag", f"formal_decode_c{cohort}",
                        "--tokens", str(tokens),
                        "--prompt-manifest", str(prompt_manifest),
                        "--output-root", str(args.output_root),
                    ]
                    trial_log = args.output_root / (
                        f"launcher_{arm}_p{prompt_tokens}_b{batch}_c{cohort}.log"
                    )
                    started = datetime.now(timezone.utc).isoformat()
                    print("RUN", " ".join(command), flush=True)
                    with trial_log.open("a", encoding="utf-8") as log:
                        completed = subprocess.run(
                            command,
                            cwd=REPOSITORY_ROOT,
                            env=os.environ.copy(),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                    row = {
                        "arm": arm,
                        "prompt_tokens": prompt_tokens,
                        "batch": batch,
                        "cohort": cohort,
                        "command": command,
                        "started_at": started,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "returncode": completed.returncode,
                        "log": str(trial_log),
                    }
                    if previous is None:
                        manifest["trials"].append(row)
                    else:
                        manifest["trials"][manifest["trials"].index(previous)] = row
                    indexed[key] = row
                    manifest["status"] = "running"
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                "event": "trial_complete",
                                "arm": arm,
                                "prompt_tokens": prompt_tokens,
                                "batch": batch,
                                "cohort": cohort,
                                "returncode": completed.returncode,
                            }
                        ),
                        flush=True,
                    )
    failures = sum(row["returncode"] != 0 for row in manifest["trials"])
    manifest["status"] = "complete" if failures == 0 else "complete_with_failures"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "failures": failures}), flush=True)


if __name__ == "__main__":
    main()
