"""Smoke probe for Exp 3 (TP1, 65536 tokens, B1, row 0 of the Exp 2 cohort 0): one short request run per arm (8
output tokens), one arm per GPU. Not a result: it only shows whether each arm fits and runs."""
import sys

sys.path.insert(0, "/workspace/BasisServe-CALS/benchmarks/system")
import run_v2q8_exp_smoke as smoke

ARMS = ("basis_koff_v128_reuse_opt", "shadowkv_upstream_v128", "lrqk_upstream_kvoff_v128", "lrqk_upstream_koff_v128")
jobs = []
for arm in ARMS:
    output = smoke.OUT / "exp3" / f"{arm}_request_b1_t65536"
    script = ["benchmarks/system/bench_v2q8_tp.py", "--arm", arm, "--mode", "request", "--length", "65536",
              "--batch", "1", "--cohort", "0", "--inputs", str(smoke.INPUTS[4]), "--output", str(output),
              *smoke.COMMON, "--smoke"]
    jobs.append((f"exp3_{arm}_request_b1_t65536", script, output, smoke.arm_env(arm)))
smoke.single_gpu(jobs, [0, 1, 2, 3])
