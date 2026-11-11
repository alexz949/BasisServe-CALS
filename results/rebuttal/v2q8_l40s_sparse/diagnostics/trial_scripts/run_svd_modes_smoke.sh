#!/bin/bash
# ShadowKV TP SVD modes: real-model check, teacher-forced check, 8K smoke, one 64K B16 probe per arm.
cd /workspace/BasisServe-CALS/benchmarks/system
LOG=/workspace/BasisServe-CALS/results/rebuttal/v2q8_l40s_sparse/svd_modes/logs/runner.log
PY=/workspace/basis_artifacts/envs/lowrank/bin/python
for stage in model forced smoke probe; do
  echo "=== svd_modes stage $stage $(date -u +%FT%TZ)" >> $LOG
  $PY run_shadowkv_svd_modes_smoke.py --stages $stage >> $LOG 2>&1
done
echo "=== svd_modes done $(date -u +%FT%TZ)" >> $LOG
