#!/usr/bin/env bash
# Experiment 2 downstream: RULER-128K, first 30 prompts of each of the 11 tasks (330), page 4, no sink, 2048 routed + recent 64,
# arms b16 / r16 / r32 (B16R16 and Full-K are taken from the existing 1100-prompt runs on the same frozen prompts).
# Per arm: reuse the frozen prompts of eval1100_p4 -> smoke full + ours -> audit -> evaluate ours (8 shards, --max-ordinal 30).
set -uo pipefail; source /home/Ubuntu/l31_router_fit/section4/common.sh; cd $R; E=evaluation/eval_llama_cal128_p1.py
declare -A BANK=([b16]=$W/router_B_p4_b16r0/ours_b16r0 [r16]=$W/router_B_p4_b0r16/ours_b0r16 [r32]=$W/router_B_p4_b0r32/ours_b0r32)
arm_run(){ local arm=$1 ga=$2 gb=$3; local OUT=$W/eval330_p4_$arm
  local A=(--identity $ID --data $DATA --bank ${BANK[$arm]} --output $OUT --page-size 4 --pinned-pages 0 --ours-budget 2112)
  mkdir -p $OUT; for f in prompts.json prompts.safetensors; do [[ -f $OUT/$f ]] || cp $W/eval1100_p4/$f $OUT/$f; done
  say "=== [ruler330 $arm] smoke full (GPU $ga) + ours (GPU $gb)"
  CUDA_VISIBLE_DEVICES=$ga $PY -u $E smoke --arm full "${A[@]}" > $L/ruler330_${arm}_smoke_full.log 2>&1 & local a=$!
  CUDA_VISIBLE_DEVICES=$gb $PY -u $E smoke --arm ours "${A[@]}" > $L/ruler330_${arm}_smoke_ours.log 2>&1 & local b=$!
  local st=0; wait $a || st=1; wait $b || st=1; [[ $st -eq 0 ]] || { say "[ruler330 $arm] smoke FAILED"; return 1; }
  $PY -u $E audit-smoke "${A[@]}" > $L/ruler330_${arm}_audit.log 2>&1 || { say "[ruler330 $arm] audit FAILED"; return 1; }
  say "=== [ruler330 $arm] smoke + audit passed"; [[ ${STAGE:-all} == smoke ]] && return 0
  say "=== [ruler330 $arm] evaluate ours: 8 shards, ordinal < 30"; local pids=() s
  for s in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$s $PY -u $E evaluate --arm ours "${A[@]}" --shard $s --shards 8 --max-ordinal 30 > $L/ruler330_${arm}_shard$s.log 2>&1 & pids+=("$!"); done
  st=0; local p; for p in "${pids[@]}"; do wait "$p" || st=1; done
  say "=== [ruler330 $arm] exit=$st, $(ls $OUT/ours/evaluate 2>/dev/null | wc -l)/330"; [[ $st -eq 0 ]]; }
ARMS=${ARMS:-"r16 r32 b16"}; set -- $ARMS
# two arms at a time (2 evaluator processes per GPU)
while [[ $# -gt 0 ]]; do a=$1; shift; if [[ $# -gt 0 ]]; then b=$1; shift; arm_run $a 0 1 & pa=$!; arm_run $b 2 3 & pb=$!; wait $pa || exit 1; wait $pb || exit 1; else arm_run $a 0 1 || exit 1; fi; done
say "=== RULER330 DONE ==="
