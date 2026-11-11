"""Trial (requester): upstream LRQK with the settings of its paper's throughput table (Table 9: r=16, top-k 1024,
16 lite tokens) at the Exp 3 geometry (TP1, 65536 tokens, B1, GPU 0, serial, cohort 0, 128 output tokens), written
under smoke_v3/exp3: a trial, not a formal result. K+V offload (the paper's placement): request and steady; K-only
offload: request."""
import sys

sys.path.insert(0, "/workspace/BasisServe-CALS/benchmarks/system")
import run_v2q8_exp_smoke as smoke

for arm, mode in (("lrqk_upstream_kvoff_t9_v128", "request"), ("lrqk_upstream_kvoff_t9_v128", "steady"),
                  ("lrqk_upstream_koff_t9_v128", "request")):
    output = smoke.OUT / "exp3" / f"{arm}_trial_{mode}_b1_t65536"
    script = ["benchmarks/system/bench_v2q8_tp.py", "--arm", arm, "--mode", mode, "--length", "65536",
              "--batch", "1", "--cohort", "0", "--inputs", str(smoke.INPUTS[4]), "--output", str(output),
              *smoke.COMMON]
    smoke.run(f"exp3_{arm}_trial_{mode}_b1_t65536", smoke.torchrun(script, [0]), [0], output,
              extra=smoke.arm_env(arm))
