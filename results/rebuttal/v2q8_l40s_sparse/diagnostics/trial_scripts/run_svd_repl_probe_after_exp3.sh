#!/bin/bash
while tmux has-session -t v2q8_exp3_formal 2>/dev/null; do sleep 20; done
LOG=/workspace/BasisServe-CALS/results/rebuttal/v2q8_l40s_sparse/svd_modes/logs/runner.log
cd /workspace/BasisServe-CALS/benchmarks/system
echo "=== svd_modes replicated 64K probe, re-run $(date -u +%FT%TZ)" >> $LOG
/workspace/basis_artifacts/envs/lowrank/bin/python /workspace/basis_artifacts/probe_svd_repl_64k.py >> $LOG 2>&1
echo "=== svd_modes replicated 64K probe done $(date -u +%FT%TZ)" >> $LOG
