"""Re-run of the replicated arm's 65536-token B16 smoke probe (the first attempt was stopped by hand to let the
TP1 64K LRQK experiment go first)."""
import sys

sys.path.insert(0, "/workspace/BasisServe-CALS/benchmarks/system")
import run_v2q8_exp_smoke as smoke
import run_shadowkv_svd_modes_smoke as driver

smoke.OUT = driver.OUT
name, script, output, extra = smoke.bench("exp2", "shadowkv_upstream_repl_v128", "request", 65536, 16, driver.GPUS)
smoke.run(name, smoke.torchrun(script, driver.GPUS), driver.GPUS, output, extra=extra)
