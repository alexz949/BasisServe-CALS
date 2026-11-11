"""V2Q8 TP1 smoke on this node's L40S GPUs: builds, short-context smoke of every arm and mode, selection
checks against the quality references, and 130048-token capacity probes. Each job is a fresh process with its
own log; jobs are spread over the GPUs, one job per GPU at a time. Smoke timings are not results.

Usage: python benchmarks/system/run_v2q8_tp1_smoke.py [--gpus 0,1,2,3] [--stages build,short,check,capacity]
           [--arms dense,basis_v96,...] [--output DIR]
Other hosts: BASIS_ARTIFACTS (asset root), V2Q8_PYTHON (benchmark interpreter), V2Q8_CUDA_HOME (toolkit for the
JIT builds), V2Q8_BANK_V128 (V128 router bank) and BENCH_MODEL (model snapshot, read by audit_tp1_request).
"""

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("BASIS_ARTIFACTS", "/workspace/basis_artifacts"))
PYTHON = Path(os.environ.get("V2Q8_PYTHON", ARTIFACTS / "envs/lowrank/bin/python"))
CUDA = Path(os.environ.get("V2Q8_CUDA_HOME", ARTIFACTS / "cuda/12.9"))
HF = ARTIFACTS / "sparse/hf"
OUT = ROOT / "results/rebuttal/v2q8_l40s_sparse/tp1_128k/smoke"  # --output replaces it
COMMON = [
    "--inputs", str(HF / "system_benchmarks/tp1_offline/inputs/prompts.safetensors"),
    "--value-checkpoint", str(ARTIFACTS / "checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix"),
    "--bank-v96", str(HF / "checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix/router_page4_b16r16"),
    "--bank-v128", os.environ.get("V2Q8_BANK_V128", "/workspace/runs/l31-router-source/v128-router/ours_b16r16"),
    "--build-root", str(ARTIFACTS / "sparse/build"),
]
ARMS = ("dense", "basis_v128", "basis_v96", "basis_v96_reuse", "basis_v96_resident",
        "fullk_v96", "lrqk_v96", "shadowkv_v96", "lrqk_native", "shadowkv_native")


def environment(gpu):
    cuda = CUDA
    env = dict(os.environ, CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="16",
               CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT), NVIDIA_TF32_OVERRIDE="0", OMP_NUM_THREADS="24",
               PATH=f"{cuda}/bin:{PYTHON.parent}:{os.environ['PATH']}",
               LD_LIBRARY_PATH=f"{PYTHON.parent.parent}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}")
    return env


def run(name, command, gpu, cwd=ROOT):
    log = OUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    begin = time.time()
    with log.open("w") as handle:
        handle.write(json.dumps(dict(command=command, gpu=gpu, cwd=str(cwd))) + "\n")
        handle.flush()
        code = subprocess.run(command, cwd=cwd, env=environment(gpu), stdout=handle, stderr=subprocess.STDOUT).returncode
    tail = log.read_text().strip().splitlines()[-1:] if log.exists() else []
    record = dict(name=name, gpu=gpu, exit=code, seconds=round(time.time() - begin, 1), last_line=tail[0][:400] if tail else "")
    print(json.dumps(record), flush=True)
    return record


def bench(arm, mode, length, smoke=True, label="short"):
    output = OUT / label / f"{arm}_{mode}_t{length}"
    return f"{label}_{arm}_{mode}_t{length}", [str(PYTHON), "benchmarks/system/bench_v2q8_tp1.py", "--arm", arm,
        "--mode", mode, "--length", str(length), "--cohort", "0", "--output", str(output), *COMMON] + (
        ["--smoke"] if smoke else [])


RULER = HF / "evaluation/llama31-8b-instruct-ruler128k-1100-seed42/prompts.safetensors"  # paper Table 2 prompts


def check(arm, length, ruler_index=None):
    """Reference and runtime in separate processes (different kernel trees), then the comparison.
    With ruler_index: that Table 2 RULER-128K sample, reference greedy and runtime forced with its tokens."""
    label = f"ruler{ruler_index:04d}" if ruler_index is not None else f"t{length}"
    output = OUT / "check" / f"{arm}_{label}"
    base = [str(PYTHON), "benchmarks/system/check_v2q8_tp1_selection.py", "--arm", arm, "--length", str(length),
            "--steps", "16", "--output", str(output), *COMMON] + (
        ["--ruler-prompts", str(RULER), "--ruler-index", str(ruler_index)] if ruler_index is not None else [])
    script = " && ".join(" ".join(base + ["--side", side]) for side in ("reference", "runtime", "compare"))
    return f"check_{arm}_{label}", ["bash", "-c", script]


def parallel(jobs, gpus):
    pending, records = queue.Queue(), []
    for job in jobs:
        pending.put(job)

    def worker(gpu):
        while True:
            try:
                name, command = pending.get_nowait()
            except queue.Empty:
                return
            records.append(run(name, command, gpu))

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return records


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--stages", default="build,short,check,capacity")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    gpus = [int(g) for g in args.gpus.split(",")]
    stages = args.stages.split(",")
    arms = args.arms.split(",")
    records = []
    if "build" in stages:
        if "shadowkv_native" in arms:
            records.append(run("build_shadowkv", [str(PYTHON), "setup.py", "build_ext", "--inplace"], gpus[0],
                               cwd=ROOT / "external/ShadowKV"))
        if "lrqk_native" in arms:
            records.append(run("build_lrqk_gather", [str(PYTHON), "setup.py", "build_ext", "--inplace"], gpus[0],
                               cwd=ROOT / "external/LRQK/cpp_kernel"))
        # Compile each BasisKV router geometry once, serially, before any parallel job imports it.
        for arm in ("basis_v96", "basis_v96_reuse", "basis_v128"):
            if arm in arms:
                records.append(run(*bench(arm, "request", 8192, label="build"), gpus[0]))
    if "short" in stages:
        records += parallel([bench(arm, mode, 8192) for arm in arms for mode in ("request", "steady", "profile")], gpus)
    if "check" in stages:
        jobs = [check(arm, length) for arm in ("basis_v96", "basis_v96_reuse", "basis_v96_resident")
                for length in (8192, 32768) if arm in arms]
        jobs += [check(arm, 8192) for arm in ("fullk_v96", "lrqk_v96", "shadowkv_v96") if arm in arms]
        # Ordinal-0 sample of each of the 11 RULER tasks (100 samples per task, task-major indices).
        jobs += [check(arm, 0, ruler_index=100 * task) for arm in ("basis_v96", "lrqk_v96", "shadowkv_v96")
                 for task in range(11) if arm in arms]
        # Runtime-paired RULER quality smoke: two Table 2 samples per arm through the timing runtime.
        for arm in ("basis_v96", "lrqk_v96", "shadowkv_v96", "fullk_v96"):
            if arm in arms:
                jobs.append((f"ruler_{arm}", [str(PYTHON), "benchmarks/system/bench_v2q8_tp1_ruler.py", "--arm", arm,
                    "--prompts", str(RULER.parent), "--max-ordinal", "1", "--shards", "6",
                    "--output", str(OUT / "ruler" / arm), *COMMON[2:]]))
        records += parallel(jobs, gpus)
    if "capacity" in stages:
        records += parallel([bench(arm, "request", 130048, label="capacity") for arm in arms], gpus)
    summary = OUT / f"runner_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    summary.write_text(json.dumps(records, indent=2) + "\n")
    print(f"summary {summary}", flush=True)


if __name__ == "__main__":
    main()
