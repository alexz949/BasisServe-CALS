"""Checks and short smoke of the ShadowKV TP SVD modes (basisserve/core/shadowkv_upstream_tp.py), TP4 on this node.
The three arms differ only in where the port runs upstream's joint per-request SVD:
  shadowkv_upstream_v128          centralized (the layer's owner decomposes the whole prefill group)
  shadowkv_upstream_reqpar_v128   request_parallel (request j of the group on rank j % W)
  shadowkv_upstream_repl_v128     replicated (every rank projects all Keys and decomposes every request)
Prefill groups are upstream's 4 in every arm. Every job is a fresh torchrun replica with its own log under
svd_modes/logs. Smoke timings are not results; formal runs are not started here.

Stages:
  layout   check_shadowkv_svd_modes.py --part layout at TP4 (NCCL, GPU tensors) and with 8 ranks (gloo, CPU tensors:
           one KV head per rank, ranks 4-7 decompose nothing in request_parallel)
  model    check_shadowkv_svd_modes.py --part model at TP4 (real model, 8192 tokens, one prefill group)
  forced   teacher-forced check at batch 8 (two prefill groups), cohort 0: upstream ShadowKV at W=1 generates, each
           arm at TP4 is fed those tokens; dense W=1 vs TP4 is the reference for TP rounding noise
  smoke    request / steady / breakdown at 8192 tokens B4, and request at 8192 tokens B16 (four prefill groups)
  probe    one 65536-token B16 request per arm with the smoke settings (8 output tokens)

Usage: python benchmarks/system/run_shadowkv_svd_modes_smoke.py [--stages layout,model,forced,smoke,probe]
"""

import argparse
import json
import time

import run_v2q8_exp_smoke as smoke

OUT = smoke.ROOT / "results/rebuttal/v2q8_l40s_sparse/svd_modes"
ARMS = ("shadowkv_upstream_v128", "shadowkv_upstream_reqpar_v128", "shadowkv_upstream_repl_v128")
GPUS = [0, 1, 2, 3]


def check(part, ranks, backend):
    name = f"check_{part}_w{ranks}_{backend}"
    output = OUT / "checks" / f"{part}_w{ranks}_{backend}"
    command = [str(smoke.PYTHON), "-m", "torch.distributed.run", f"--nproc-per-node={ranks}", "--master-port=29790",
               "benchmarks/system/check_shadowkv_svd_modes.py", "--part", part, "--backend", backend,
               "--output", str(output)]
    return smoke.run(name, command, GPUS, output, extra=smoke.UPSTREAM_ENV)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="layout,model,forced,smoke,probe")
    args = parser.parse_args()
    stages, records = args.stages.split(","), []
    smoke.OUT = OUT
    if "layout" in stages:
        records += [check("layout", 4, "nccl"), check("layout", 8, "gloo")]
    if "model" in stages:
        records.append(check("model", 4, "nccl"))
    if "forced" in stages:
        import check_v2q8_tp_consistency
        records += check_v2q8_tp_consistency.main(("dense_v128", *ARMS), cohorts=(0,), batch=8,
                                                  reference={arm: ARMS[0] for arm in ARMS[1:]})
    if "smoke" in stages:
        for arm in ARMS:
            for mode, batch in (("request", 4), ("steady", 4), ("breakdown", 4), ("request", 16)):
                name, script, output, extra = smoke.bench("exp2", arm, mode, 8192, batch, GPUS)
                records.append(smoke.run(name, smoke.torchrun(script, GPUS), GPUS, output, extra=extra))
    if "probe" in stages:
        for arm in ARMS:
            name, script, output, extra = smoke.bench("exp2", arm, "request", 65536, 16, GPUS)
            records.append(smoke.run(name, smoke.torchrun(script, GPUS), GPUS, output, extra=extra))
    summary = OUT / f"runner_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(records, indent=2) + "\n")
    print(f"summary {summary}", flush=True)


if __name__ == "__main__":
    main()
