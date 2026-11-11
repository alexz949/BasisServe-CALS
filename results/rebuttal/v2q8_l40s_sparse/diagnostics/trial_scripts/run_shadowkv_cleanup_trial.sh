#!/bin/bash
# Waits for the replicated 64K probe, then runs the ShadowKV cleanup trials (TP1 four arms, then TP4 one arm).
L=/workspace/BasisServe-CALS/results/rebuttal/v2q8_l40s_sparse/logs/round2_runner.log
while tmux has-session -t v2q8_svd_repl_probe 2>/dev/null; do sleep 15; done
cd /workspace/BasisServe-CALS
echo "=== shadowkv cleanup trial tp1 start $(date -u +%FT%TZ)" >> $L
/workspace/basis_artifacts/envs/lowrank/bin/python /workspace/basis_artifacts/trial_shadowkv_cleanup.py tp1 >> $L 2>&1
echo "=== shadowkv cleanup trial tp1 done $(date -u +%FT%TZ)" >> $L
/workspace/basis_artifacts/envs/lowrank/bin/python /workspace/basis_artifacts/trial_shadowkv_cleanup.py tp4 >> $L 2>&1
echo "=== shadowkv cleanup trial all done $(date -u +%FT%TZ)" >> $L
