"""Serial TP1 offline-representation grid with resumable, explicit outcomes."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("dense", "basis", "shadowkv", "lrqk")
CONTEXTS = (16384, 32768, 65536, 130048)


def grid(phase):
    if phase == "smoke":
        return [("request", m, 8192, 0, 4) for m in METHODS] + [
            ("steady", m, 8192, 0, 128) for m in METHODS[:2]]
    if phase == "capacity":
        return [("request", "shadowkv", 8192, 0, 512)] + [
            ("request", m, 130048, 0, 4) for m in METHODS]
    trials = []
    for cohort in range(3):
        for context in CONTEXTS:
            trials += [("steady", m, context, cohort, 128) for m in METHODS[:2]]
        for context in CONTEXTS[1:]:
            trials += [("request", m, context, cohort, 128) for m in METHODS]
        for count in (32, 512):
            trials += [("request", m, 130048, cohort, count) for m in METHODS[1:]]
    return trials


def prepare(root):
    from safetensors.torch import load_file, save_file
    directory = root / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    prompts = directory / "prompts.safetensors"
    if not prompts.exists():
        source = Path("/workspace/runs/l31-cal128/calibration/windows.safetensors")
        windows = load_file(str(source))["input_ids"]
        save_file({f"cohort_{i}": windows[i, :130048].clone().contiguous() for i in range(3)}, str(prompts))
        (directory / "manifest.json").write_text(json.dumps(dict(source=str(source),
            rows=[0, 1, 2], length=130048, protocol="Same frozen prefixes for every method; greedy fixed-length generation"), indent=2) + "\n")
    return prompts


def summarize(root, phase, outcomes):
    rows = []
    for item in outcomes:
        row = {k: item[k] for k in ("mode", "method", "context", "cohort", "output_tokens", "status", "phase")}
        result_path = Path(item["directory"]) / "result.json"
        if result_path.exists():
            data = json.loads(result_path.read_text())
            measured = data["measurement"]
            row["peak_allocated_gib"] = data["peak_allocated_gib"]
            for key in ("request_seconds", "prefill_inclusive_seconds", "post_prefill_phase_seconds",
                        "decode_phase_seconds", "post_prefill_ready_seconds", "ready_overlaps_first_decode",
                        "steady_tail_wall_mean_ms", "cuda_median_ms", "cuda_mean_ms", "wall_median_ms"):
                row[key] = measured.get(key)
            profile = data.get("build_profile", {})
            row["profile_construction_seconds"] = profile.get("construction_including_preparation_seconds", profile.get("prompt_specific_fit_seconds"))
            attention = data.get("attention_profile", {}).get("components", [])
            row["profile_attention_block_mean_ms"] = sum(r["attention_block"] for r in attention)/len(attention) if attention else None
        rows.append(row)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (root / f"{phase}_summary.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (root / f"{phase}_outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "capacity", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/system_benchmarks/tp1_offline")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    inputs = prepare(root)
    trials = grid(args.phase)
    if args.phase == "formal":
        checks = root / "smoke_outcomes.json"
        assert checks.exists()
        assert len(json.loads(checks.read_text())) == 6
        assert all(r["status"] == "complete" for r in json.loads(checks.read_text()))
        capacity = json.loads((root / "capacity_outcomes.json").read_text())
        assert len(capacity) == 5 and capacity[0]["status"] == "complete"
        assert all(r["status"] in ("complete", "gpu_oom") for r in capacity)
    # Each phase has an immutable runner snapshot. Resume only unchanged code.
    source = root / args.phase / "source"
    source.mkdir(parents=True, exist_ok=True)
    for filename in ("audit_tp1_request.py", "bench_tp1_offline_request.py", "run_tp1_offline_grid.py"):
        original = Path(__file__).with_name(filename)
        destination = source / filename
        if destination.exists():
            assert destination.read_bytes() == original.read_bytes()
        else:
            shutil.copy2(original, destination)
    outcomes = []
    for mode, method, context, cohort, count in trials:
        directory = root / args.phase / f"{mode}_{method}_t{context}_c{cohort}_n{count}"
        directory.mkdir(parents=True, exist_ok=True)
        outcome_path = directory / "outcome.json"
        if outcome_path.exists():
            outcomes.append(json.loads(outcome_path.read_text()))
            summarize(root, args.phase, outcomes)
            continue
        command = [sys.executable, str(Path(__file__).with_name("bench_tp1_offline_request.py")),
            "--method", method, "--mode", mode, "--length", str(context), "--cohort", str(cohort),
            "--output-tokens", str(count), "--inputs", str(inputs), "--output", str(directory)]
        if args.phase == "smoke":
            command.append("--smoke")
        print(json.dumps(dict(trial=len(outcomes)+1, total=len(trials), command=command)), flush=True)
        with (directory / "run.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                stdout=log, stderr=subprocess.STDOUT)
        result_path = directory / "result.json"
        log_text = (directory / "run.log").read_text()
        state_path = directory / "progress.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        status = (json.loads(result_path.read_text())["status"] if process.returncode == 0 and result_path.exists()
                  else "gpu_oom" if "CUDA out of memory" in log_text or "torch.OutOfMemoryError" in log_text else "error")
        item = dict(mode=mode, method=method, context=context, cohort=cohort, output_tokens=count,
            status=status, phase=state.get("phase", "launch"), directory=str(directory), returncode=process.returncode)
        outcome_path.write_text(json.dumps(item, indent=2) + "\n")
        outcomes.append(item)
        summarize(root, args.phase, outcomes)
        print(json.dumps(item), flush=True)
        if status not in ("complete", "gpu_oom"):
            return
    print(json.dumps(dict(phase=args.phase, trials=len(outcomes),
        complete=sum(r["status"] == "complete" for r in outcomes),
        gpu_oom=sum(r["status"] == "gpu_oom" for r in outcomes))), flush=True)


if __name__ == "__main__":
    main()
