"""Trial (requester): upstream ShadowKV with one request per prefill forward (arm shadowkv_upstream_g1_v128) at TP4.
A short 8192-token B4 run first, then one 65536-token B16 request run with the formal settings (cohort 0, 128
output tokens), written under smoke_v3/exp2: a trial, not a formal result."""
import sys

sys.path.insert(0, "/workspace/BasisServe-CALS/benchmarks/system")
import run_v2q8_exp_smoke as smoke

ARM, GPUS = "shadowkv_upstream_g1_v128", [0, 1, 2, 3]
for length, batch, short in ((8192, 4, True), (65536, 16, False)):
    name, script, output, extra = smoke.bench("exp2", ARM, "request", length, batch, GPUS, smoke=short)
    record = smoke.run(name, smoke.torchrun(script, GPUS), GPUS, output, extra=extra)
    if record["status"] != "complete":
        break
