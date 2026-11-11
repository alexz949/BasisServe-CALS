#!/bin/bash
# After the ShadowKV SVD-mode smoke: the TP1 64K smoke probe (Exp 3), four arms, one per GPU.
while tmux has-session -t v2q8_svd_modes 2>/dev/null; do sleep 20; done
LOG=/workspace/BasisServe-CALS/results/rebuttal/v2q8_l40s_sparse/logs/round2_runner.log
cd /workspace/BasisServe-CALS/benchmarks/system
echo "=== smoke_v3 exp3 probe (TP1 64K B1) $(date -u +%FT%TZ)" >> $LOG
/workspace/basis_artifacts/envs/lowrank/bin/python /workspace/basis_artifacts/probe_tp1_64k.py >> $LOG 2>&1
echo "=== smoke_v3 exp3 probe done $(date -u +%FT%TZ)" >> $LOG
