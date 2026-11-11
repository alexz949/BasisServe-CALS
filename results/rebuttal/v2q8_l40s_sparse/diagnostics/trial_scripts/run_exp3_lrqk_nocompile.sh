#!/bin/bash
# LRQK with torch.compile off: 8192-token smoke gate (4 GPUs), then formal Exp 3 for the two arms on GPU 0 (serial).
L=/workspace/BasisServe-CALS/results/rebuttal/v2q8_l40s_sparse/logs/round2_runner.log
PY=/workspace/basis_artifacts/envs/lowrank/bin/python
ARMS=lrqk_upstream_kvoff_nocompile_v128,lrqk_upstream_koff_nocompile_v128
cd /workspace/BasisServe-CALS
echo "=== lrqk nocompile 8192 smoke start $(date -u +%FT%TZ)" >> $L
$PY /workspace/basis_artifacts/smoke_lrqk_nocompile.py >> $L 2>&1
echo "=== lrqk nocompile 8192 smoke done $(date -u +%FT%TZ)" >> $L
cd benchmarks/system
echo "=== formal_v2 exp3 lrqk nocompile start $(date -u +%FT%TZ)" >> $L
$PY run_v2q8_exp_formal.py --experiments 3 --arms $ARMS >> $L 2>&1
echo "=== formal_v2 exp3 lrqk nocompile done $(date -u +%FT%TZ)" >> $L
