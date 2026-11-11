"""Trial (requester's question): what upstream ShadowKV's per-layer cleanup inside get_svd costs. One request run
per arm with the formal settings (cohort 0, 128 output tokens), written under smoke_v3: trials, not formal results.
  tp1: Exp 3 geometry (TP1, 65536 tokens, B1, GPU 0, serial): upstream, then gc.collect off, empty_cache off, both off
  tp4: not run (requester's decision)
"""
import sys

sys.path.insert(0, "/workspace/BasisServe-CALS/benchmarks/system")
import run_v2q8_exp_smoke as smoke

if sys.argv[1] == "tp1":
    for arm in ("shadowkv_upstream_v128", "shadowkv_upstream_nogc_v128", "shadowkv_upstream_noempty_v128",
                "shadowkv_upstream_nohk_v128"):
        output = smoke.OUT / "exp3" / f"{arm}_cleanup_trial_request_b1_t65536"
        script = ["benchmarks/system/bench_v2q8_tp.py", "--arm", arm, "--mode", "request", "--length", "65536",
                  "--batch", "1", "--cohort", "0", "--inputs", str(smoke.INPUTS[4]), "--output", str(output),
                  *smoke.COMMON]
        smoke.run(f"exp3_{arm}_cleanup_trial_request_b1_t65536", smoke.torchrun(script, [0]), [0], output,
                  extra=smoke.arm_env(arm))
else:
    assert sys.argv[1] == "tp4"
    print("tp4 cleanup trial skipped: the requester asked not to run it", flush=True)
