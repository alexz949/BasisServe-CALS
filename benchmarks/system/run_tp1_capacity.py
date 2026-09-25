"""Run isolated TP1 capacity trials, retaining OOM phase and all logs."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", nargs="+", type=int, default=[65536, 131072])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--methods", nargs="+", default=["dense_local", "dense_k_offload", "basis_k_offload"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["settings"] == settings
    else:
        with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
            for folder in ("basisserve", "benchmarks/system", "evaluation"):
                for path in sorted((ROOT / folder).rglob("*")):
                    if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".cpp", ".h"):
                        archive.add(path, arcname=str(path.relative_to(ROOT)))
        manifest = dict(settings=settings, command=shlex.join([sys.executable, *sys.argv]),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            worktree_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
            gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv"], text=True),
            source_archive="source.tar.gz", validation="No SHA256 checks")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=env.get("CUDA_VISIBLE_DEVICES", "0"), CUDA_HOME="/usr/local/cuda",
               MAX_JOBS="2", TORCH_CUDA_ARCH_LIST="8.9")
    records = []
    for context in args.contexts:
        for method in args.methods:
            stopped = False
            for batch in args.batches:
                if stopped:
                    break
                for repeat in range(args.repeats):
                    directory = args.output / f"{method}_t{context}_b{batch}_r{repeat}"
                    directory.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, str(ROOT / "benchmarks/system/bench_tp1_capacity.py"),
                        "--method", method, "--context", str(context), "--batch", str(batch),
                        "--warmup", str(args.warmup), "--steps", str(args.steps), "--output", str(directory.resolve())]
                    if args.validate:
                        command.append("--validate")
                    record = dict(method=method, context=context, batch=batch, repeat=repeat,
                                  command=shlex.join(command), directory=str(directory))
                    (directory / "command.json").write_text(json.dumps(record, indent=2) + "\n")
                    if not (directory / "result.json").exists():
                        print(record["command"], flush=True)
                        with (directory / "run.log").open("w") as log:
                            run = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                        record["returncode"] = run.returncode
                    if (directory / "result.json").exists():
                        result = json.loads((directory / "result.json").read_text())
                        assert (result["batch"], result["decode_steps"], result["warmup_steps"], result["validation"]) == (
                            batch, args.steps, args.warmup, args.validate)
                        record.update(status="success", result=result)
                    else:
                        log = (directory / "run.log").read_text()
                        gpu_oom = "CUDA out of memory" in log or "torch.OutOfMemoryError" in log
                        record["status"] = "gpu_oom" if gpu_oom else "failed"
                        progress = directory / "progress.json"
                        record["phase"] = json.loads(progress.read_text())["phase"] if progress.exists() else "startup"
                        stopped = True
                    records.append(record)
                    (args.output / "outcomes.json").write_text(json.dumps(records, indent=2) + "\n")
                    print(json.dumps({k: v for k, v in record.items() if k != "result"}), flush=True)
                    if record["status"] == "failed":
                        return
                    if stopped:
                        break
    with tarfile.open(args.output / "source.tar.gz", "r:gz") as archive:
        changed = [member.name for member in archive.getmembers()
                   if archive.extractfile(member).read() != (ROOT / member.name).read_bytes()]
    (args.output / "source_check.json").write_text(json.dumps(dict(changed_files=changed), indent=2) + "\n")
    assert not changed


if __name__ == "__main__":
    main()
