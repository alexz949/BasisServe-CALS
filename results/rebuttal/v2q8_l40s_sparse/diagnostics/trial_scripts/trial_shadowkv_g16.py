"""Trial (requester): upstream ShadowKV with all 16 requests in one prefill forward (arm shadowkv_upstream_g16_v128)
at TP4: one 65536-token B16 request run with the formal settings (cohort 0, 128 output tokens), written under
smoke_v3/exp2: a trial, not a formal result."""
import sys

sys.path.insert(0, "/workspace/BasisServe-CALS/benchmarks/system")
import run_v2q8_exp_smoke as smoke

ARM, GPUS = "shadowkv_upstream_g16_v128", [0, 1, 2, 3]
name, script, output, extra = smoke.bench("exp2", ARM, "request", 65536, 16, GPUS, smoke=False)
smoke.run(name, smoke.torchrun(script, GPUS), GPUS, output, extra=extra)
