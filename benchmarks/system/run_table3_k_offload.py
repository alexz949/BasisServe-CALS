"""Launch the Table 3 Key-offload arms: one fresh process per arm and prompt cohort."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from compare_table3_placement import compare
from run_tp1_paper_scaling import ROOT, now, prepare_runtime, write_json

DRIVER = Path(__file__).with_name("bench_table3_k_offload.py")
ARMS = ["dense_resident", "basis_resident", "basis_resident_nobase", "basis_k_offload_nobase"]
NO_BASE = ARMS[2:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "check", "formal"), required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=[130048])
    parser.add_argument("--cohorts", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--page-sizes", nargs="+", type=int, default=[],
                        help="Page-size study: only the no-Base arms, on the page-size-general router")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/rebuttal/table3_k_offload_128k")
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime, prompts = prepare_runtime(output)
    phase = output / ("pages" if args.page_sizes else "arms") / args.phase
    phase.mkdir(parents=True, exist_ok=True)
    contexts = [24576] if args.phase == "smoke" else args.contexts
    cohorts = [0] if args.phase != "formal" else args.cohorts
    sizes = args.page_sizes or [None]
    arms = NO_BASE if args.page_sizes else ARMS
    trials = [(context, cohort, "steady", arm, size) for cohort in cohorts for context in contexts
              for size in sizes for arm in arms] if args.phase != "check" else []
    if args.phase != "formal":
        trials += [(context, 0, "check", arm, size) for context in contexts for size in sizes
                   for arm in arms if arm != "dense_resident"]
    pairs = [(first + tag, second + tag) for size in sizes for tag in [f"_p{size}" if size else ""]
             for first, second in [NO_BASE] + ([] if size else [("basis_resident", NO_BASE[0])])]
    for name in (DRIVER.name, "table3_page_router.py", "table3_page_select.cu"):
        shutil.copy2(DRIVER.with_name(name), phase / name)
    write_json(phase / "manifest.json", dict(
        phase=args.phase, contexts=contexts, cohorts=cohorts, created_utc=now(),
        trials=[dict(context=c, cohort=k, mode=m, arm=a, page_size=s) for c, k, m, a, s in trials],
        command=shlex.join([sys.executable, *sys.argv]), environment=Path(sys.prefix).name,
        launch_git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        runtime_manifest="../../runtime_manifest.json", driver=DRIVER.name,
        protocol="Archived Table 3 steady decode: same-shape warmup request, fresh state, full "
                 "prefill, 16 conditioning and 128 measured decode steps (2 and 4 in smoke); "
                 "token selection is outside the timed step"))
    env = os.environ.copy()
    env.update(CUDA_HOME="/usr/local/cuda", MAX_JOBS="2", TORCH_CUDA_ARCH_LIST="8.9",
               PYTHONPATH=str(runtime), PYTHONNOUSERSITE="1")
    outcomes = []
    for context, cohort, mode, arm, size in trials:
        prefix = "check_" if mode == "check" else ""
        tag = f"_p{size}" if size else ""
        directory = phase / f"{prefix}{arm}{tag}_t{context}_c{cohort}"
        directory.mkdir(exist_ok=True)
        outcome_path = directory / "outcome.json"
        if outcome_path.exists():
            outcome = json.loads(outcome_path.read_text())
        else:
            command = [sys.executable, str(DRIVER), "--runtime", str(runtime), "--arm", arm,
                       "--mode", mode, "--length", str(context), "--cohort", str(cohort),
                       "--inputs", str(prompts), "--output", str(directory)]
            if args.phase == "smoke":
                command.append("--smoke")
            if size:
                command += ["--page-size", str(size)]
            started = now()
            print(shlex.join(command), flush=True)
            with (directory / "run.log").open("w") as log:
                process = subprocess.run(command, cwd=runtime, env=env,
                                         stdout=log, stderr=subprocess.STDOUT)
            result = directory / "result.json"
            log_text = (directory / "run.log").read_text()
            progress = directory / "progress.json"
            state = json.loads(progress.read_text()) if progress.exists() else {}
            status = (json.loads(result.read_text())["status"]
                      if process.returncode == 0 and result.exists() else
                      "gpu_oom" if "CUDA out of memory" in log_text or
                      "torch.OutOfMemoryError" in log_text else "error")
            outcome = dict(arm=arm, mode=mode, context=context, cohort=cohort, page_size=size,
                status=status,
                phase=state.get("phase", "startup"), command=shlex.join(command),
                directory=str(directory), started_utc=started, finished_utc=now(),
                returncode=process.returncode)
            write_json(outcome_path, outcome)
        outcomes.append(outcome)
        write_json(phase / "outcomes.json", outcomes)
        print(json.dumps({k: v for k, v in outcome.items() if k != "command"}), flush=True)
        if outcome["status"] != "complete":
            print("Stopping: inspect the failed trial before resuming.", flush=True)
            break
    complete = len(outcomes) == len(trials) and outcomes[-1]["status"] == "complete"
    if complete and args.phase != "formal":
        write_json(phase / "placement_check.json", [
            compare(phase / f"check_{first}_t{context}_c0", phase / f"check_{second}_t{context}_c0")
            for context in contexts for first, second in pairs])
    print(json.dumps(dict(phase=args.phase, trials=len(trials), completed=len(outcomes),
                          all_complete=complete)), flush=True)


if __name__ == "__main__":
    main()
