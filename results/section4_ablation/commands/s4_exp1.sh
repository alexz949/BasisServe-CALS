#!/usr/bin/env bash
# Experiment 1: V -> pre-RoPE K predictability. moments on 8 window shards (32 fit + 16 held-out windows x 131072) -> solve (CPU).
set -uo pipefail; source /home/Ubuntu/l31_router_fit/section4/common.sh; cd $R; OUT=$S4/v_to_k; F=evaluation/section4_v_to_k_information.py
A=(--identity $ID --windows $WIN --output $OUT --sequence-length 131072 --rope native --window-shards 8)
say "=== [exp1] moments: 8 window shards"; run8 exp1_moments $PY -u $F moments "${A[@]}" || { say "exp1 moments FAILED"; exit 1; }
say "=== [exp1] solve"; $PY -u $F solve "${A[@]}" > $L/exp1_solve.log 2>&1 || { say "exp1 solve FAILED"; exit 1; }
tail -n 1 $L/exp1_solve.log; say "=== EXP1 DONE ==="
