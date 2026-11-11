"""Trial (requester): does upstream LRQK's decode depend on how many CPU cores it gets? Arm
lrqk_upstream_kvoff_t9_v128 (K+V offload, the paper's Table 9 settings), Exp 3 geometry (TP1, 65536 tokens, B1,
GPU 0, cohort 0), steady mode (128 measured decode steps), one run per CPU set, serial. The formal policy is 24
logical CPUs = 12 physical cores and their hyperthreads on the GPU's NUMA node (already measured: steady median
335.6 ms). Written under smoke_v3/exp3: trials, not formal results."""
import sys

sys.path.insert(0, "/workspace/BasisServe-CALS/benchmarks/system")
import run_v2q8_exp_smoke as smoke

ARM = "lrqk_upstream_kvoff_t9_v128"
# label: (cpu range, threads). CPUs 0-23 are the 24 physical cores of NUMA node 0, 0-47 the 48 of socket 0.
SETS = {"c4_t4": ("0-3", 4), "c12_t12": ("0-11", 12), "c24_t24": ("0-23", 24), "c48_t48": ("0-47", 48)}
for label, (cpus, threads) in SETS.items():
    output = smoke.OUT / "exp3" / f"{ARM}_threads_{label}_steady_b1_t65536"
    script = ["benchmarks/system/bench_v2q8_tp.py", "--arm", ARM, "--mode", "steady", "--length", "65536",
              "--batch", "1", "--cohort", "0", "--inputs", str(smoke.INPUTS[4]), "--output", str(output),
              *smoke.COMMON]
    extra = dict(smoke.arm_env(ARM), V2Q8_TRIAL_CPUS=cpus, V2Q8_TRIAL_THREADS=str(threads),
                 OMP_NUM_THREADS=str(threads))
    smoke.run(f"exp3_{ARM}_threads_{label}_steady_b1_t65536", smoke.torchrun(script, [0]), [0], output, extra=extra)
