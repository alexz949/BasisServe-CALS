"""Seven-point TP1 scaling sweep using the archived formal full-scan runtime."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "results/system_benchmarks/tp1_offline"
CONTEXTS = [16384, 24576, 32768, 49152, 65536, 98304, 130048]
SNAPSHOT = "results/system_benchmarks/tp1_sparse_local/frozen_source/"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare_runtime(output):
    runtime = output / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    members = []

    def preserve(name, data):
        destination = runtime / name
        assert destination.resolve().is_relative_to(runtime.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert destination.read_bytes() == data, str(destination)
        else:
            destination.write_bytes(data)
        members.append(name)

    with tarfile.open(ARCHIVE / "freeze/dependencies.tar.gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and (member.name.startswith(SNAPSHOT) or
                    member.name == "benchmarks/system/bench_tp1_router_config.py"):
                preserve(member.name, archive.extractfile(member).read())
    for filename in ("audit_tp1_request.py", "bench_tp1_offline_request.py"):
        preserve("benchmarks/system/" + filename,
                 (ARCHIVE / "formal/source" / filename).read_bytes())
    # The archived formal driver imports the complete frozen TP1 dependency tree.
    assert runtime.joinpath(SNAPSHOT, "basisserve/kernels/csrc/conditional_router_page32.cu").exists()
    frozen = json.loads((ARCHIVE / "freeze/provenance.json").read_text())
    write_json(output / "runtime_manifest.json", dict(
        source_archive=str((ARCHIVE / "freeze/dependencies.tar.gz").relative_to(ROOT)),
        formal_driver_archive=str((ARCHIVE / "formal/source").relative_to(ROOT)),
        archived_provenance=frozen, runtime_source_files=members,
        source_policy="Byte-exact copies from archived formal full-scan runtime, not current dirty source",
        validation="Direct bytes; no SHA256 checks"))
    prompt_source = ARCHIVE / "inputs/prompts.safetensors"
    prompts = output / "prompts.safetensors"
    if prompts.exists():
        assert prompts.read_bytes() == prompt_source.read_bytes()
    else:
        shutil.copy2(prompt_source, prompts)
    return runtime, prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS)
    parser.add_argument("--cohorts", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/system_benchmarks/tp1_paper_scaling")
    args = parser.parse_args()
    assert len(set(args.contexts)) == len(args.contexts)
    assert all(0 < n <= 130048 and n % 32 == 0 for n in args.contexts)
    assert args.cohorts == [0, 1, 2]
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime, prompts = prepare_runtime(output)
    phase = output / args.phase
    phase.mkdir(exist_ok=True)
    contexts = [24576] if args.phase == "smoke" else args.contexts
    cohorts = [0] if args.phase == "smoke" else args.cohorts
    settings = dict(phase=args.phase, contexts=contexts, cohorts=cohorts,
                    methods=["dense", "basis"], warmup=2 if args.phase == "smoke" else 16,
                    measured_steps=4 if args.phase == "smoke" else 128,
                    separate_attention_steps=4 if args.phase == "smoke" else 32)
    manifest_path = phase / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["settings"] == settings
    else:
        write_json(manifest_path, dict(settings=settings, created_utc=now(),
            command=shlex.join([sys.executable, *sys.argv]), environment="basis",
            launch_git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            runtime_manifest="../runtime_manifest.json", cuda_graph=False,
            input_source=str(ARCHIVE / "inputs/prompts.safetensors"),
            hardware=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version",
                                               "--format=csv"], text=True)))
    runner_copy = phase / Path(__file__).name
    if runner_copy.exists():
        assert runner_copy.read_bytes() == Path(__file__).read_bytes()
    else:
        shutil.copy2(__file__, runner_copy)
    if args.phase == "formal":
        smoke = json.loads((output / "smoke/outcomes.json").read_text())
        assert len(smoke) == 2 and all(row["status"] == "complete" for row in smoke)
    env = os.environ.copy()
    env.update(CUDA_HOME="/usr/local/cuda", MAX_JOBS="2", TORCH_CUDA_ARCH_LIST="8.9",
               PYTHONPATH=str(runtime), PYTHONNOUSERSITE="1")
    outcomes = []
    for cohort in cohorts:
        for context in contexts:
            for method in ("dense", "basis"):
                directory = phase / f"steady_{method}_t{context}_c{cohort}_n128"
                directory.mkdir(exist_ok=True)
                outcome_path = directory / "outcome.json"
                if outcome_path.exists():
                    outcome = json.loads(outcome_path.read_text())
                    assert outcome["status"] == "complete", "Inspect failed trial before resuming"
                else:
                    command = [sys.executable, str(runtime / "benchmarks/system/bench_tp1_offline_request.py"),
                        "--method", method, "--mode", "steady", "--length", str(context),
                        "--cohort", str(cohort), "--output-tokens", "128", "--inputs", str(prompts),
                        "--output", str(directory)]
                    if args.phase == "smoke":
                        command.append("--smoke")
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
                    outcome = dict(method=method, context=context, cohort=cohort,
                        status=status, phase=state.get("phase", "startup"),
                        command=shlex.join(command), directory=str(directory),
                        started_utc=started, finished_utc=now(), returncode=process.returncode)
                    write_json(outcome_path, outcome)
                outcomes.append(outcome)
                write_json(phase / "outcomes.json", outcomes)
                print(json.dumps({k: v for k, v in outcome.items() if k != "command"}), flush=True)
                assert outcome["status"] == "complete", outcome
    prepare_runtime(output)
    write_json(phase / "source_check.json", dict(status="passed", compared="direct bytes against archive"))
    print(json.dumps(dict(phase=args.phase, completed=len(outcomes))), flush=True)


if __name__ == "__main__":
    main()
