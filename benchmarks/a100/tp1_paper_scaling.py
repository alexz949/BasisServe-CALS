"""Paper Table 3 (TP1/B1 sparse vs dense decoding, Llama-3.1-8B-Instruct) on one A100-SXM4-80GB.

Runs the archived L40S formal runtime (HF `system_benchmarks/tp1_offline`, the source of
`results/system_benchmarks/tp1_paper_scaling`) with the same contexts, cohorts, prompts, warmup and measured steps.
The archives are read, never modified; the runtime copy gets five asserted exact replacements, each recorded with
before/after sha256 in `runtime_manifest.json`:

1-2. `conditional_router_page32.cu`: `cub::Max()` -> `::cuda::maximum<>{}` (+ `<cuda/functional>`). CUDA 13 CCCL
     removed `cub::Max`; both are the `a < b ? b : a` functor, so routing is unchanged.
3.   router geometry `w16u1` -> `w8u1`. On A100 the 16-warp block chosen on L40S is 2.5-9% slower than the 8-warp
     block at 16K-128K (`bench_tp1_router_config.py --phase micro`); scores and pages are bitwise identical.
4-5. model snapshot and router-bank paths of the L40S host -> local paths (same model revision, same archived bank).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
REPO, REVISION = "alexz949/BasisServe-CALS", "63a35906f354cc239339465db97fab136ccdb1de"
ARCHIVE = "system_benchmarks/tp1_offline"
LFS = {
    "freeze/dependencies.tar.gz": "6e42b00f3180929a065834f1ba0c35574ef86087a88e68119f28c6b83629c71b",
    "freeze/factors.tar.gz": "57b8b9e48a6fb4f6a478fec56381993d124942a254850542d041bdce61caeb14",
    "inputs/prompts.safetensors": "7c6485069e8d58022f587313f28ddd22e27e2eb53cdddb590b9c610f52cdd68b",
}
DRIVERS = ("formal/source/audit_tp1_request.py", "formal/source/bench_tp1_offline_request.py")
MODEL_REPO, MODEL_REVISION = "meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"
CONTEXTS = [16384, 24576, 32768, 49152, 65536, 98304, 130048]
SNAPSHOT = "results/system_benchmarks/tp1_sparse_local/frozen_source/"
CUDA_HOME = Path.home() / "cuda-13.0"  # benchmarks/a100/setup_cuda_home.sh


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def patches(model, bank):
    router = SNAPSHOT + "basisserve/kernels/csrc/conditional_router_page32.cu"
    audit = "benchmarks/system/audit_tp1_request.py"
    sparse = SNAPSHOT + "benchmarks/system/bench_tp1_sparse_full_v6.py"
    old_model = ('MODEL = Path("/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/'
                 'snapshots/0e9e39f249a16976918f6564b8830bc894c89659")')
    return [
        dict(file=router, old="cub::Max()", new="::cuda::maximum<>{}", count=1,
             reason="CUDA 13 CCCL removed cub::Max; cuda::maximum is the same a < b ? b : a functor"),
        dict(file=router, old="#include <cub/block/block_reduce.cuh>\n",
             new="#include <cub/block/block_reduce.cuh>\n#include <cuda/functional>\n", count=1,
             reason="declares cuda::maximum"),
        dict(file=audit, old='config.build("w16u1")', new='config.build("w8u1")', count=1,
             reason="A100 router geometry: 8 warps per block; w16u1 (L40S choice) is 2.5-9% slower at 16K-128K"),
        dict(file=audit, old=old_model, new=f'MODEL = Path("{model}")', count=1,
             reason="local snapshot of the same model revision"),
        dict(file=sparse, old='FACTOR_ROOT = Path("/workspace/runs/l31-router-source/v128-router/ours_b16r16")',
             new=f'FACTOR_ROOT = Path("{bank}")', count=1, reason="router bank extracted from freeze/factors.tar.gz"),
    ]


def fetch():
    files = {}
    for name, digest in LFS.items():
        files[name] = Path(hf_hub_download(REPO, f"{ARCHIVE}/{name}", revision=REVISION))
        assert sha256(files[name].read_bytes()) == digest, name
    for name in DRIVERS:
        files[name] = Path(hf_hub_download(REPO, f"{ARCHIVE}/{name}", revision=REVISION))
    model = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=["*.json", "*.safetensors"]))
    return files, model


def preserve(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, str(path)
    else:
        path.write_bytes(data)


def prepare_runtime(output, files, model):
    runtime, bank = output / "runtime", output / "factors" / "ours_b16r16"
    sources = {}
    with tarfile.open(files["freeze/dependencies.tar.gz"]) as archive:
        for member in archive.getmembers():
            if member.isfile() and (member.name.startswith(SNAPSHOT) or
                                    member.name == "benchmarks/system/bench_tp1_router_config.py"):
                sources[member.name] = archive.extractfile(member).read()
    for name in DRIVERS:
        sources["benchmarks/system/" + Path(name).name] = files[name].read_bytes()
    records = []
    for patch in patches(model, bank):
        text = sources[patch["file"]].decode()
        assert text.count(patch["old"]) == patch["count"], (patch["file"], patch["old"])
        before = sha256(sources[patch["file"]])
        sources[patch["file"]] = text.replace(patch["old"], patch["new"]).encode()
        records.append(dict(patch, sha256_before=before, sha256_after=sha256(sources[patch["file"]])))
    for name, data in sources.items():
        destination = runtime / name
        assert destination.resolve().is_relative_to(runtime.resolve())
        preserve(destination, data)
    with tarfile.open(files["freeze/factors.tar.gz"]) as archive:
        for member in archive.getmembers():
            if member.isfile():
                assert member.name.startswith("ours_b16r16/") and ".." not in member.name
                preserve(output / "factors" / member.name, archive.extractfile(member).read())
    assert runtime.joinpath(SNAPSHOT, "basisserve/kernels/csrc/conditional_router_page32.cu").exists()
    prompts = output / "prompts.safetensors"
    preserve(prompts, files["inputs/prompts.safetensors"].read_bytes())
    nvcc = subprocess.check_output([str(CUDA_HOME / "bin/nvcc"), "--version"], text=True).strip().splitlines()[-1]
    manifest = dict(hf_repo=REPO, hf_revision=REVISION, archive=ARCHIVE, archive_sha256=LFS,
                    drivers={name: sha256(files[name].read_bytes()) for name in DRIVERS},
                    model=dict(repo=MODEL_REPO, revision=MODEL_REVISION, path=str(model)),
                    patches=records, runtime_files=sorted(sources), cuda_home=str(CUDA_HOME), nvcc=nvcc,
                    host_compiler="/usr/bin/g++-11", torch_cuda_arch_list="8.0",
                    source_policy="archive bytes plus the recorded asserted replacements; archives unmodified")
    path = output / "runtime_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(manifest))
    else:
        write_json(path, manifest)
    return runtime, prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS)
    parser.add_argument("--cohorts", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/system_benchmarks/a100/tp1_paper_scaling")
    args = parser.parse_args()
    assert len(set(args.contexts)) == len(args.contexts)
    assert all(0 < n <= 130048 and n % 32 == 0 for n in args.contexts)
    assert args.cohorts == [0, 1, 2]
    assert (CUDA_HOME / "bin/nvcc").is_file() and Path("/usr/bin/g++-11").is_file()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files, model = fetch()
    runtime, prompts = prepare_runtime(output, files, model)
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
            launch_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            runtime_manifest="../runtime_manifest.json", cuda_graph=False,
            input_source=f"hf://{REPO}@{REVISION}/{ARCHIVE}/inputs/prompts.safetensors",
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
    env.update(CUDA_HOME=str(CUDA_HOME), CC="/usr/bin/gcc-11", CXX="/usr/bin/g++-11", MAX_JOBS="16",
               TORCH_CUDA_ARCH_LIST="8.0", PYTHONPATH=str(runtime), PYTHONNOUSERSITE="1",
               CUDA_VISIBLE_DEVICES=str(args.gpu), PATH=f"{Path(sys.executable).parent}:{env['PATH']}")
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
                        process = subprocess.run(command, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
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
    prepare_runtime(output, files, model)
    write_json(phase / "source_check.json", dict(status="passed", compared="direct bytes against archive + patches"))
    print(json.dumps(dict(phase=args.phase, completed=len(outcomes))), flush=True)


if __name__ == "__main__":
    main()
