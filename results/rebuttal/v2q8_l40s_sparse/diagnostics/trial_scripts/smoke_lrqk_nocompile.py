"""8192-token TP1 smoke (request / steady / breakdown) of the two LRQK no-compile arms, one job per GPU at a time:
the gate run_v2q8_exp_formal.py asks for before Exp 3. Smoke timings are not results."""
import sys

sys.path.insert(0, "/workspace/BasisServe-CALS/benchmarks/system")
import run_v2q8_exp_smoke as smoke

ARMS = ("lrqk_upstream_kvoff_nocompile_v128", "lrqk_upstream_koff_nocompile_v128")
jobs = [smoke.bench("exp1", arm, mode, 8192, 1, [0]) for arm in ARMS for mode in ("request", "steady", "breakdown")]
smoke.single_gpu(jobs, [0, 1, 2, 3])
