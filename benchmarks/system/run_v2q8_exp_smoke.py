"""Smoke for the requester's V2Q8 experiments (V128 first): BasisKV (K offload), ShadowKV (V offload), LRQK (K
offload) through one runner (bench_v2q8_tp.py) at TP1 and TP4. Each job is one fresh torchrun replica with its own
log. Smoke timings are not results.

Round 2 (default, smoke_v3/): the baselines are their upstream runtimes, unmodified (shadowkv_upstream_v128,
lrqk_upstream_koff_v128; TP4 through the minimal ports in basisserve/core), run with PyTorch's default CUDA
allocator as upstream does; BasisKV keeps expandable segments. smoke_v2/ is the same round before the fixes from
the round-2 review (COMPATIBILITY.md section 12); round 1 (smoke/) used the earlier adapters.

Stages:
  build     ShadowKV CUDA extension (CUTLASS v3.5.1) and the LRQK CPU gather extension
  checks    ShadowKV TP adapter vs upstream (factors, cache, short generation) at TP4; dense HF TP4 vs one-GPU
            logits (the TP noise floor); LRQK factory vs upstream (hit sets, GPU hit cache, B factors per step) at
            W=1 and TP4; TP1 vs TP4 generation on real text for dense, ShadowKV and LRQK
  exp1      TP1: request / steady / breakdown at 8192 tokens, then a 130048-token capacity probe, per arm
            (W=1 jobs spread over the GPUs, one per GPU at a time); LRQK also with upstream torch.compile kept
  exp2      TP4: request / steady / breakdown at 8192 tokens B4, then a 65536-token B16 capacity probe, per arm

Usage: python benchmarks/system/run_v2q8_exp_smoke.py [--gpus 0,1,2,3] [--stages build,checks,exp1,exp2]
"""

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/workspace/basis_artifacts")
PYTHON = ARTIFACTS / "envs/lowrank/bin/python"
HF = ARTIFACTS / "sparse/hf"
OUT = ROOT / "results/rebuttal/v2q8_l40s_sparse/smoke_v3"
MODEL = Path("/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
             "0e9e39f249a16976918f6564b8830bc894c89659")
ARMS = ("basis_koff_v128", "basis_koff_v128_reuse", "shadowkv_upstream_v128", "lrqk_upstream_koff_v128")
# Upstream baselines run in their own environment: PyTorch's default CUDA allocator.
UPSTREAM_ENV = {"PYTORCH_CUDA_ALLOC_CONF": ""}


def arm_env(arm):
    return UPSTREAM_ENV if "upstream" in arm else None
INPUTS = {1: HF / "system_benchmarks/tp1_offline/inputs/prompts.safetensors",
          4: ROOT / "results/rebuttal/v2q8_l40s_sparse/tp4_64k/inputs/prompts.safetensors"}
COMMON = [
    "--value-checkpoint", str(ARTIFACTS / "checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix"),
    "--bank-v96", str(HF / "checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix/router_page4_b16r16"),
    "--bank-v128", "/workspace/runs/l31-router-source/v128-router/ours_b16r16",
    "--build-root", str(ARTIFACTS / "sparse/build"),
]
PORT = [29800]
LOCK = threading.Lock()


def environment(gpus, extra=None):
    cuda = ARTIFACTS / "cuda/12.9"
    # The redistributable CUDA 12.9 subset has no cuSPARSE/cuBLAS/cuSOLVER headers, which torch's
    # ATen/cuda/CUDAContextLight.h includes; the matching headers ship in the env's nvidia-* wheels.
    wheels = PYTHON.parent.parent / "lib/python3.11/site-packages/nvidia"
    headers = ":".join(str(wheels / name / "include") for name in ("cusparse", "cublas", "cusolver", "cufft", "curand"))
    env = dict(os.environ, CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="16", CPATH=headers,
               CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), PYTHONPATH=str(ROOT),
               OMP_NUM_THREADS="24",
               # Same allocator setting for every arm: avoids OOM from fragmentation in long prefills.
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PATH=f"{cuda}/bin:{PYTHON.parent}:{os.environ['PATH']}",
               LD_LIBRARY_PATH=f"{PYTHON.parent.parent}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}")
    env.update(extra or {})
    return env


def run(name, command, gpus, output=None, cwd=ROOT, extra=None):
    log = OUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if output is not None:  # a relaunch must not inherit an earlier attempt's result files
        for stale in [output / "result.json", *output.glob("rank_*.json")] if output.exists() else []:
            stale.unlink()
    begin = time.time()
    with log.open("w") as handle:
        handle.write(json.dumps(dict(command=command, gpus=gpus, extra_env=extra)) + "\n")
        handle.flush()
        code = subprocess.run(command, cwd=cwd, env=environment(gpus, extra), stdout=handle,
                              stderr=subprocess.STDOUT).returncode
    result = output / "result.json" if output else None
    status = (json.loads(result.read_text())["status"] if result and result.exists() and code == 0
              else "exit0" if code == 0 else "failed")
    text = log.read_text(errors="replace").strip().splitlines()
    error = next((line for line in reversed(text) if "Error" in line), "")
    record = dict(name=name, gpus=gpus, exit=code, status=status, seconds=round(time.time() - begin, 1),
                  error=error[:300])
    print(json.dumps(record), flush=True)
    return record


def torchrun(script, gpus):
    with LOCK:
        PORT[0] += 1
        port = PORT[0]
    return [str(PYTHON), "-m", "torch.distributed.run", f"--nproc-per-node={len(gpus)}", f"--master-port={port}",
            *script]


def bench(stage, arm, mode, length, batch, gpus, smoke=True, extra=None, tag=""):
    extra = arm_env(arm) if extra is None else extra
    name = f"{stage}_{arm}{tag}_{mode}_b{batch}_t{length}"
    output = OUT / stage / f"{arm}{tag}_{mode}_b{batch}_t{length}"
    script = ["benchmarks/system/bench_v2q8_tp.py", "--arm", arm, "--mode", mode, "--length", str(length),
              "--batch", str(batch), "--cohort", "0", "--inputs", str(INPUTS[len(gpus)]), "--output", str(output),
              *COMMON] + (["--smoke"] if smoke else [])
    return name, script, output, extra


def single_gpu(jobs, gpus):
    """W=1 jobs, one per GPU at a time."""
    pending, records = queue.Queue(), []
    for job in jobs:
        pending.put(job)

    def worker(gpu):
        while True:
            try:
                name, script, output, extra = pending.get_nowait()
            except queue.Empty:
                return
            records.append(run(name, torchrun(script, [gpu]), [gpu], output, extra=extra))

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--stages", default="build,checks,exp1,exp2")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    stages, records = args.stages.split(","), []
    if "build" in stages:
        records.append(run("build_shadowkv", [str(PYTHON), "setup.py", "build_ext", "--inplace"], gpus[:1],
                           cwd=ROOT / "external/ShadowKV"))
        records.append(run("build_lrqk_gather", [str(PYTHON), "setup.py", "build_ext", "--inplace"], gpus[:1],
                           cwd=ROOT / "external/LRQK/cpp_kernel"))
    if "checks" in stages:
        # Every arm's TP4 path against its own TP1 run, teacher-forced on real text, dense as the noise reference.
        import check_v2q8_tp_consistency
        records += check_v2q8_tp_consistency.main()
    if "exp1" in stages:
        # Compile each BasisKV router geometry once before parallel jobs import it.
        name, script, output, extra = bench("exp1", "basis_koff_v128", "request", 8192, 1, gpus[:1], tag="_build")
        records.append(run(name, torchrun(script, gpus[:1]), gpus[:1], output, extra=extra))
        jobs = [bench("exp1", arm, mode, 8192, 1, [0]) for arm in ARMS for mode in ("request", "steady", "breakdown")]
        records += single_gpu(jobs, gpus)
        jobs = [bench("exp1", arm, mode, 130048, 1, [0]) for arm in ARMS for mode in ("request", "breakdown")]
        # Informational: LRQK's 130K capacity under the allocator setting BasisKV uses.
        jobs.append(bench("exp1", "lrqk_upstream_koff_v128", "request", 130048, 1, [0],
                          extra={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, tag="_expandable"))
        records += single_gpu(jobs, gpus)
    if "exp2" in stages:
        for arm in ARMS:
            for mode in ("request", "steady", "breakdown"):
                name, script, output, extra = bench("exp2", arm, mode, 8192, 4, gpus)
                records.append(run(name, torchrun(script, gpus), gpus, output, extra=extra))
        for arm in ARMS:
            name, script, output, extra = bench("exp2", arm, "request", 65536, 16, gpus)
            records.append(run(name, torchrun(script, gpus), gpus, output, extra=extra))
    summary = OUT / f"runner_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(records, indent=2) + "\n")
    print(f"summary {summary}", flush=True)


if __name__ == "__main__":
    main()
