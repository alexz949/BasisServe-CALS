#!/usr/bin/env bash
# Routing diagnostics on the 16 held-out calibration windows (131072 tokens, 32 queries each, all 32 layers), no sink, recent 64.
#   exp3a: fixed page-4 B16R16 factors + the page-1/8/32 refits, P {1,4,8,32} x B {256,2048}     (GPU ${GPU_A:-0})
#   exp2 : b16r16 / b16 / r16 / r32 banks at P 4, B 2048                                            (GPU ${GPU_B:-1}; needs the exp-2 banks)
set -uo pipefail; source /home/Ubuntu/l31_router_fit/section4/common.sh; cd $R; F=evaluation/page_granularity_sweep.py
A=(--identity $ID --windows $WIN --sink 0 --sequence-length 131072 --rope native --queries-per-window 32)
WHICH=${WHICH:-"exp3a exp2"}; pids=()
for w in $WHICH; do case $w in
  exp3a) [[ -f $S4/sweep_exp3a_nosink.json ]] || { say "=== [sweep exp3a] P 1,4,8,32 x B 256,2048; banks p4 (fixed) + p1/p8/p32 refits"
         CUDA_VISIBLE_DEVICES=${GPU_A:-0} $PY -u $F "${A[@]}" --banks p4=$W/router_B_p4/ours_b16r16,p1=$W/router_B_p1/ours_b16r16,p8=$W/router_B_p8/ours_b16r16,p32=$W/router_B/ours_b16r16 --page-sizes 1,4,8,32 --budgets 256,2048 --output $S4/sweep_exp3a_nosink.json > $L/sweep_exp3a.log 2>&1 & pids+=("$!"); } ;;
  exp2)  [[ -f $S4/sweep_exp2_p4_b2048.json ]] || { say "=== [sweep exp2] P 4 x B 2048; banks b16r16 / b16 / r16 / r32"
         CUDA_VISIBLE_DEVICES=${GPU_B:-1} $PY -u $F "${A[@]}" --banks b16r16=$W/router_B_p4/ours_b16r16,b16=$W/router_B_p4_b16r0/ours_b16r0,r16=$W/router_B_p4_b0r16/ours_b0r16,r32=$W/router_B_p4_b0r32/ours_b0r32 --page-sizes 4 --budgets 2048 --output $S4/sweep_exp2_p4_b2048.json > $L/sweep_exp2.log 2>&1 & pids+=("$!"); } ;;
esac; done
st=0; for p in "${pids[@]}"; do wait "$p" || st=1; done; [[ $st -eq 0 ]] || { say "sweep FAILED"; exit 1; }; say "=== SWEEPS DONE ($WHICH) ==="
