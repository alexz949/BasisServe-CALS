#!/usr/bin/env bash
# Experiment 2 control (user-approved 2026-09-25): B16-only routing with the first page pinned (sink = 1 page of 4 tokens).
#   1. RULER 330: bank router_B_p4_b16r0 (fitted with pinned 0) run with --pinned-pages 1 --allow-bank-sink-mismatch, budget 2116 = 4 sink + 64 recent + 2048 routed.
#   2. Diagnostic sweep with --sink 4 for all four exp-2 banks (P 4, B 2048) once the RULER arms are done.
#   3. Re-run the exp-2 summary with the control row.
set -uo pipefail; source /home/Ubuntu/l31_router_fit/section4/common.sh; cd $R; E=evaluation/eval_llama_cal128_p1.py; F=evaluation/page_granularity_sweep.py
OUT=$W/eval330_p4_b16sink4; A=(--identity $ID --data $DATA --bank $W/router_B_p4_b16r0/ours_b16r0 --output $OUT --page-size 4 --pinned-pages 1 --allow-bank-sink-mismatch --ours-budget 2116)
mkdir -p $OUT; for f in prompts.json prompts.safetensors; do [[ -f $OUT/$f ]] || cp $W/eval1100_p4/$f $OUT/$f; done
say "=== [b16sink] smoke full (GPU 0) + ours (GPU 1)"
CUDA_VISIBLE_DEVICES=0 $PY -u $E smoke --arm full "${A[@]}" > $L/ruler330_b16sink_smoke_full.log 2>&1 & a=$!
CUDA_VISIBLE_DEVICES=1 $PY -u $E smoke --arm ours "${A[@]}" > $L/ruler330_b16sink_smoke_ours.log 2>&1 & b=$!
st=0; wait $a || st=1; wait $b || st=1; [[ $st -eq 0 ]] || { say "[b16sink] smoke FAILED"; exit 1; }
$PY -u $E audit-smoke "${A[@]}" > $L/ruler330_b16sink_audit.log 2>&1 || { say "[b16sink] audit FAILED"; exit 1; }
say "=== [b16sink] smoke + audit passed; evaluate ours: 8 shards, ordinal < 30"; pids=()
for s in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$s $PY -u $E evaluate --arm ours "${A[@]}" --shard $s --shards 8 --max-ordinal 30 > $L/ruler330_b16sink_shard$s.log 2>&1 & pids+=("$!"); done
st=0; for p in "${pids[@]}"; do wait "$p" || st=1; done; say "=== [b16sink] exit=$st, $(ls $OUT/ours/evaluate 2>/dev/null | wc -l)/330"; [[ $st -eq 0 ]] || exit 1
until grep -q 'RULER330 DONE\|exit=1' $L/ruler330_driver.log 2>/dev/null; do sleep 60; done
say "=== [sweep exp2 sink4] P 4 x B 2048; banks b16r16 / b16 / r16 / r32, sink 4"
CUDA_VISIBLE_DEVICES=7 $PY -u $F --identity $ID --windows $WIN --sink 4 --sequence-length 131072 --rope native --queries-per-window 32 --banks b16r16=$W/router_B_p4/ours_b16r16,b16=$W/router_B_p4_b16r0/ours_b16r0,r16=$W/router_B_p4_b0r16/ours_b0r16,r32=$W/router_B_p4_b0r32/ours_b0r32 --page-sizes 4 --budgets 2048 --output $S4/sweep_exp2_p4_b2048_sink4.json > $L/sweep_exp2_sink4.log 2>&1 || { say "sweep sink4 FAILED"; exit 1; }
until grep -q 'SECTION4 CHAIN DONE' $L/run_all.log 2>/dev/null; do sleep 60; done
say "=== [summary exp2 + control]"; $PY -u evaluation/section4_base_residual_summary.py --prompts $W/eval1100_p4/prompts.json \
  --arms b16=$W/eval330_p4_b16/ours/evaluate b16sink=$OUT/ours/evaluate r16=$W/eval330_p4_r16/ours/evaluate b16r16=$W/eval1100_p4/ours/evaluate r32=$W/eval330_p4_r32/ours/evaluate full=$W/eval1100/full/evaluate \
  --dims b16=16:0 b16sink=16:0 r16=0:16 b16r16=16:16 r32=0:32 --diag $S4/sweep_exp2_p4_b2048.json --extra-diag b16sink=$S4/sweep_exp2_p4_b2048_sink4.json:b16 \
  --page-size 4 --budget 2048 --max-ordinal 30 --output $S4/results/section4_ablation/base_residual > $L/summary_exp2_control.log 2>&1 || { say "summary FAILED"; exit 1; }
sed -n 1,16p $S4/results/section4_ablation/base_residual/summary.md; say "=== B16SINK CONTROL DONE ==="
