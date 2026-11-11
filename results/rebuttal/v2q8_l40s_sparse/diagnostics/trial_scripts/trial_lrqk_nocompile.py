"""Trial (requester): upstream LRQK with torch.compile turned off by PyTorch's own switch (TORCH_COMPILE_DISABLE=1:
the decorated upstream functions then run as plain Python; upstream's source is not edited). Arm
lrqk_upstream_kvoff_t9_v128 (K+V offload, the paper's Table 9 settings), Exp 3 geometry (TP1, 65536 tokens, B1,
GPU 0, cohort 0): steady (128 measured steps) and breakdown, serial. With compile on, the same arm measured a steady
median of 335.6 ms. Written under smoke_v3/exp3: trials, not formal results."""
import sys

sys.path.insert(0, "/workspace/BasisServe-CALS/benchmarks/system")
import run_v2q8_exp_smoke as smoke

ARM = "lrqk_upstream_kvoff_t9_v128"
for mode, tag, compile_off in (("steady", "nocompile", True), ("breakdown", "nocompile", True),
                               ("breakdown", "trial", False)):
    output = smoke.OUT / "exp3" / f"{ARM}_{tag}_{mode}_b1_t65536"
    script = ["benchmarks/system/bench_v2q8_tp.py", "--arm", ARM, "--mode", mode, "--length", "65536",
              "--batch", "1", "--cohort", "0", "--inputs", str(smoke.INPUTS[4]), "--output", str(output),
              *smoke.COMMON]
    extra = dict(smoke.arm_env(ARM), **({"TORCH_COMPILE_DISABLE": "1"} if compile_off else {}))
    smoke.run(f"exp3_{ARM}_{tag}_{mode}_b1_t65536", smoke.torchrun(script, [0]), [0], output, extra=extra)
