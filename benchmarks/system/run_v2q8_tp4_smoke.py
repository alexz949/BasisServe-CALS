"""V2Q8 TP4 smoke: the three distributed ShadowKV adapter checks at TP4 (factors vs upstream get_svd, cache vs
the full-head official cache, short real-text generation vs the single-GPU official adapter), short-context
correctness smoke of every arm (B1 and B4), W=1 vs W=4 agreement of the same modules, and 65536-token capacity
probes at B4/B8/B16. Each job is one fresh torchrun replica on all listed GPUs (one after another). Smoke timings
are not results.

Usage: python benchmarks/system/run_v2q8_tp4_smoke.py [--gpus 0,1,2,3] [--stages shadowkv,lrqk,short,agreement,capacity]
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/workspace/basis_artifacts")
PYTHON = ARTIFACTS / "envs/lowrank/bin/python"
OUT = ROOT / "results/rebuttal/v2q8_l40s_sparse/tp4_64k/smoke"
ARMS = ("dense", "basis_v96", "basis_v96_reuse", "fullk_v96", "lrqk_v96", "shadowkv_v96", "shadowkv_native",
        "lrqk_native")
COMMON = [
    "--inputs", str(ROOT / "results/rebuttal/v2q8_l40s_sparse/tp4_64k/inputs/prompts.safetensors"),
    "--value-checkpoint", str(ARTIFACTS / "checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix"),
    "--bank-v96", str(ARTIFACTS / "sparse/hf/checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix/router_page4_b16r16"),
    "--build-root", str(ARTIFACTS / "sparse/build"),
]
PORT = [29700]


def bench(name, arm, mode, length, batch, gpus, smoke=True, extra_env=None):
    output = OUT / name
    return run(name, ["benchmarks/system/bench_v2q8_tp.py", "--arm", arm, "--mode", mode, "--length", str(length),
                      "--batch", str(batch), "--cohort", "0", "--output", str(output), *COMMON]
               + (["--smoke"] if smoke else []), gpus, output, extra_env)


def run(name, script, gpus, output, extra_env=None):
    log = OUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    PORT[0] += 1
    command = [str(PYTHON), "-m", "torch.distributed.run", f"--nproc-per-node={len(gpus)}",
               f"--master-port={PORT[0]}", *script]
    cuda = ARTIFACTS / "cuda/12.9"
    env = dict(os.environ, CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="16",
               CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), PYTHONPATH=str(ROOT), NVIDIA_TF32_OVERRIDE="0",
               OMP_NUM_THREADS="24", PATH=f"{cuda}/bin:{PYTHON.parent}:{os.environ['PATH']}",
               LD_LIBRARY_PATH=f"{PYTHON.parent.parent}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
               **(extra_env or {}))
    begin = time.time()
    with log.open("w") as handle:
        handle.write(json.dumps(dict(command=command, gpus=gpus)) + "\n")
        handle.flush()
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    results = [output / "result.json", output / "model_reference.json"]
    result = next((r for r in results if r.exists()), None)
    status = json.loads(result.read_text())["status"] if result else ("exit0" if code == 0 else "no_result")
    tail = log.read_text().strip().splitlines()[-1][:300] if log.exists() else ""
    record = dict(name=name, exit=code, status=status, seconds=round(time.time() - begin, 1), last_line=tail)
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--stages", default="shadowkv,lrqk,short,agreement,capacity")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    stages, records = args.stages.split(","), []
    if "shadowkv" in stages:
        from safetensors.torch import load_file, save_file
        tokens = OUT / "shadowkv_checks/p4096_c0.safetensors"
        tokens.parent.mkdir(parents=True, exist_ok=True)
        prompts = load_file(COMMON[1])["cohort_0"]
        save_file({"input_ids": prompts[:1, :4096].contiguous()}, str(tokens))
        upstream = ["--upstream", str(ROOT / "external/ShadowKV")]
        for check in ("factors", "cache", "model"):
            output = OUT / "shadowkv_checks" / check
            extra = ["--tokens", str(tokens), "--model", str(Path(os.environ.get("BENCH_MODEL_PATH",
                "/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
                "0e9e39f249a16976918f6564b8830bc894c89659")))] if check == "model" else []
            records.append(run(f"shadowkv_{check}_tp{len(gpus)}",
                               [f"tests/distributed_shadowkv_tp8_{check}_smoke.py", *upstream, "--output",
                                str(output), *extra], gpus, output))
    if "lrqk" in stages:
        # TP LRQK factory vs the unchanged upstream factory: hit sets, GPU hit cache and B factors per step.
        for world in (1, len(gpus)):
            output = OUT / "lrqk_checks" / f"w{world}"
            records.append(run(f"lrqk_factory_w{world}", ["tests/distributed_lrqk_tp_smoke.py", "--output", str(output)],
                               gpus[:world], output))
    if "short" in stages:
        for arm in ARMS:
            for batch in (1, 4):
                for mode in ("request", "steady"):
                    records.append(bench(f"short_{arm}_{mode}_b{batch}_t8192", arm, mode, 8192, batch, gpus))
        # Upstream LRQK with its torch.compile(max_autotune) kept, to pick the faster LRQK configuration.
        for batch in (1, 4):
            records.append(bench(f"short_lrqk_native_compile_steady_b{batch}_t8192", "lrqk_native", "steady", 8192,
                                 batch, gpus, extra_env={"LRQK_TORCH_COMPILE": "1"}))
    if "agreement" in stages:
        # Same modules at W=1 and W=4 on one request; compare generated tokens and physical support.
        for arm in ARMS:
            for world in (1, len(gpus)):
                records.append(bench(f"agreement_{arm}_w{world}_t8192", arm, "steady", 8192, 1, gpus[:world]))
    if "capacity" in stages:
        for batch in (4, 8, 16):
            for arm in ARMS:
                records.append(bench(f"capacity_{arm}_b{batch}_t65536", arm, "request", 65536, batch, gpus))
    summary = OUT / f"runner_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    summary.write_text(json.dumps(records, indent=2) + "\n")
    print(f"summary {summary}", flush=True)


if __name__ == "__main__":
    main()
