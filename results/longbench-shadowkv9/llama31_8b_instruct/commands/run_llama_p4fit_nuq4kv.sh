#!/usr/bin/env bash
# Llama-3.1-8B-Instruct V96 (identity B) LongBench-9 with a SIMULATED KVQuant NUQ4 KV cache (pre-RoPE K per-channel, V96 latent per-token,
# 1% outliers, Fisher-weighted signposts; upstream external/KVQuant). Page-4 B16R16 router, no sink, 256 + recent 64.
# Stages: calibrate (capture on GPU 0, fit on 8 CPU shards, merge) -> smoke full+ours -> audit -> evaluate (8 shards each) -> compare with BF16 and FP8.
set -uo pipefail; source /home/Ubuntu/longbench9/common.sh; tag=llama31_8b_instruct_p4fit_nuq4kv; E=evaluation/eval_k_routing_ruler_p16.py; C=evaluation/calibrate_llama_v96_kvquant.py
ID=/home/Ubuntu/l31_retrieval_cal/identity/B.json; QD=/home/Ubuntu/l31_router_fit/kvquant_nuq4_B
if [[ ! -f $QD/quantizers.pt ]]; then
  [[ -f $QD/capture/manifest.json ]] || { say "=== [$tag] calibrate: capture (GPU ${CAL_GPU:-0})"; CUDA_VISIBLE_DEVICES=${CAL_GPU:-0} $PY -u $C capture --identity $ID --output $QD > $L/logs/${tag}_cal_capture.log 2>&1 || { say "capture FAILED"; exit 1; }; }
  say "=== [$tag] calibrate: fit (8 CPU shards)"; pids=(); for s in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=12 $PY -u $C fit --identity $ID --output $QD --layer-shard $s --layer-shards 8 > $L/logs/${tag}_cal_fit_s$s.log 2>&1 & pids+=("$!"); done
  st=0; for p in "${pids[@]}"; do wait "$p" || st=1; done; [[ $st -eq 0 ]] || { say "fit FAILED"; exit 1; }
  say "=== [$tag] calibrate: merge"; $PY -u $C merge --identity $ID --output $QD > $L/logs/${tag}_cal_merge.log 2>&1 || { say "merge FAILED"; exit 1; }
fi
[[ ${STAGE:-all} == calibrate ]] && { say "=== [$tag] calibration done"; exit 0; }
A=(--identity $ID --data $L/llama31_8b_instruct --bank /home/Ubuntu/l31_router_fit/router_B_p4/ours_b16r16 --sequence-length 131072 --rope native --router-fit-count 32 --router-diagnostic-count 0 --chat-template --benchmark longbench --ours-budget 320 --lrqk-topk 256 --shadowkv-budget 256 --page-size 4 --pinned-pages 0 --kv-nuq4 $QD --arms full,ours --output $L/eval_$tag)
say "=== [$tag] smoke full+ours"; CUDA_VISIBLE_DEVICES=${SMOKE_GPU_A:-0} $PY -u $E smoke --arm full "${A[@]}" > $L/logs/${tag}_smoke_full.log 2>&1 & p1=$!; CUDA_VISIBLE_DEVICES=${SMOKE_GPU_B:-1} $PY -u $E smoke --arm ours "${A[@]}" > $L/logs/${tag}_smoke_ours.log 2>&1 & p2=$!
st=0; wait $p1 || st=1; wait $p2 || st=1; [[ $st -eq 0 ]] || { say "smoke FAILED"; exit 1; }
$PY -u $E audit-smoke "${A[@]}" > $L/logs/${tag}_audit.log 2>&1 || { say "audit FAILED"; exit 1; }
say "=== [$tag] smoke + audit passed"; [[ ${STAGE:-all} == smoke ]] && exit 0
say "=== [$tag] evaluate full + ours: 8 shards each (2 procs per GPU)"; pids=()
for arm in full ours; do for s in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$s $PY -u $E evaluate --arm $arm "${A[@]}" --shard-index $s --num-shards 8 > $L/logs/${tag}_${arm}_shard$s.log 2>&1 & pids+=("$!"); done; done
st=0; for p in "${pids[@]}"; do wait "$p" || st=1; done; say "=== [$tag] evaluate exit=$st: full $(ls $L/eval_$tag/full/evaluate 2>/dev/null | wc -l) ours $(ls $L/eval_$tag/ours/evaluate 2>/dev/null | wc -l) /1543"; [[ $st -eq 0 ]] || exit 1
{ echo "# Llama-3.1-8B-Instruct LongBench-9: BF16 vs FP8 (E4M3) vs KVQuant NUQ4 KV cache, Full-K and B16R16 page-4 (256 + recent 64); $(TZ=America/New_York date '+%Y-%m-%d %H:%M %Z')"; $PY $L/compare_arms.py full_bf16=$L/eval_llama31_8b_instruct/full/evaluate ours_bf16=$L/eval_llama31_8b_instruct_p4fit/ours/evaluate full_fp8=$L/eval_llama31_8b_instruct_p4fit_fp8kv/full/evaluate ours_fp8=$L/eval_llama31_8b_instruct_p4fit_fp8kv/ours/evaluate full_nuq4=$L/eval_$tag/full/evaluate ours_nuq4=$L/eval_$tag/ours/evaluate; } > $L/eval_${tag}_summary.txt 2>&1; cat $L/eval_${tag}_summary.txt
say "=== [$tag] DONE ==="
