"""Diagnosis: the first formal no-compile LRQK runs (formal settings r=32, top-k 2048, lite 64) decoded at 1.35-3.8
s/step, although the no-compile trial with the Table 9 settings was faster than compile on. Same geometry (TP1, 65536
tokens, B1, GPU 0, cohort 0), breakdown mode, serial: no-compile, then compile on, then no-compile again. Written
under smoke_v3/exp3: diagnosis, not results."""
import sys

sys.path.insert(0, "/workspace/BasisServe-CALS/benchmarks/system")
import run_v2q8_exp_smoke as smoke

for arm, tag in (("lrqk_upstream_kvoff_nocompile_v128", "_diag1"), ("lrqk_upstream_kvoff_v128", "_diag"),
                 ("lrqk_upstream_kvoff_nocompile_v128", "_diag2")):
    output = smoke.OUT / "exp3" / f"{arm}{tag}_breakdown_b1_t65536"
    script = ["benchmarks/system/bench_v2q8_tp.py", "--arm", arm, "--mode", "breakdown", "--length", "65536",
              "--batch", "1", "--cohort", "0", "--inputs", str(smoke.INPUTS[4]), "--output", str(output),
              *smoke.COMMON]
    smoke.run(f"exp3_{arm}{tag}_breakdown_b1_t65536", smoke.torchrun(script, [0]), [0], output,
              extra=smoke.arm_env(arm))
