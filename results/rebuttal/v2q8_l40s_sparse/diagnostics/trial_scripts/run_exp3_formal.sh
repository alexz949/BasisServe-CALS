#!/bin/bash
# After the TP1 64K smoke probe: formal Exp 3 for the arms whose probe completed (requester's go-ahead).
while tmux has-session -t v2q8_exp3_probe 2>/dev/null; do sleep 20; done
R=/workspace/BasisServe-CALS/results/rebuttal/v2q8_l40s_sparse
LOG=$R/logs/round2_runner.log
PY=/workspace/basis_artifacts/envs/lowrank/bin/python
cd /workspace/BasisServe-CALS/benchmarks/system
ARMS=$(python3 - <<'PY'
import json, pathlib
root = pathlib.Path("/workspace/BasisServe-CALS/results/rebuttal/v2q8_l40s_sparse/smoke_v3/exp3")
arms = ("basis_koff_v128_reuse_opt", "shadowkv_upstream_v128", "lrqk_upstream_kvoff_v128", "lrqk_upstream_koff_v128")
ok = []
for arm in arms:
    path = root / f"{arm}_request_b1_t65536" / "result.json"
    if path.exists() and json.loads(path.read_text()).get("status") == "complete":
        ok.append(arm)
print(",".join(ok))
PY
)
echo "=== formal_v2 exp3 start (probe passed: ${ARMS:-none}) $(date -u +%FT%TZ)" >> $LOG
if [ -n "$ARMS" ]; then
  $PY run_v2q8_exp_formal.py --experiments 3 --arms "$ARMS" >> $LOG 2>&1
fi
echo "=== formal_v2 exp3 done $(date -u +%FT%TZ)" >> $LOG
