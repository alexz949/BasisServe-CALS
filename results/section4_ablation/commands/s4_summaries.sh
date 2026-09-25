#!/usr/bin/env bash
# Build the three results/section4_ablation/* directories in the staging area $S4/results (copied into the repo after review).
set -uo pipefail; source /home/Ubuntu/l31_router_fit/section4/common.sh; cd $R; OUT=$S4/results/section4_ablation
say "=== [summaries] experiment 1"; mkdir -p $OUT/v_to_k_information && cp $S4/v_to_k/result.json $S4/v_to_k/summary.md $S4/v_to_k/v_to_k_by_layer.pdf $S4/v_to_k/v_to_k_by_layer_group.pdf $OUT/v_to_k_information/
say "=== [summaries] experiment 2"; $PY -u evaluation/section4_base_residual_summary.py --prompts $W/eval1100_p4/prompts.json \
  --arms b16=$W/eval330_p4_b16/ours/evaluate r16=$W/eval330_p4_r16/ours/evaluate b16r16=$W/eval1100_p4/ours/evaluate r32=$W/eval330_p4_r32/ours/evaluate full=$W/eval1100/full/evaluate \
  --dims b16=16:0 r16=0:16 b16r16=16:16 r32=0:32 --diag $S4/sweep_exp2_p4_b2048.json --page-size 4 --budget 2048 --max-ordinal 30 --output $OUT/base_residual > $L/summary_exp2.log 2>&1 || { say "exp2 summary FAILED"; exit 1; }
say "=== [summaries] experiment 3"; $PY -u evaluation/section4_page_granularity_summary.py --diag $S4/sweep_exp3a_nosink.json --fixed p4 \
  --arms p1=$W/eval1100_p1/ours/evaluate p4=$W/eval1100_p4/ours/evaluate p8=$W/eval1100_p8fit_2task/ours/evaluate p32=$W/eval1100/ours/evaluate full=$W/eval1100/full/evaluate \
  --prompts $W/eval1100_p4/prompts.json --tasks niah_multikey_2,fwe --notes "p1=Page-Fisher refit, no sink, 2048 + recent 64" "p4=Page-Fisher refit, no sink, 2048 + recent 64" "p8=Page-Fisher refit, no sink, 2048 + recent 64" "p32=released bank: pinned sink page (32) + 1952 routed + recent 64 = 2048 total" "full=exact attention" \
  --output $OUT/page_granularity > $L/summary_exp3.log 2>&1 || { say "exp3 summary FAILED"; exit 1; }
say "=== SUMMARIES DONE ==="; ls -R $OUT | head -40
