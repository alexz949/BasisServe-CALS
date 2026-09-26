#!/usr/bin/env bash
# Llama-3.1-8B-Instruct V96 (identity B) LongBench-9 with a SIMULATED FP8 (E4M3) KV cache: K per-token scales, V96 latent per-page-4 scales,
# routing sidecar BF16 (Base from the stored V latent, residual code from BF16 K). Page-4 B16R16 router (router_B_p4), no sink, 256 + recent 64.
# Arms full + ours (both on FP8 KV); smoke both -> audit -> evaluate (8 shards each, concurrent) -> paired comparison with the BF16 runs.
set -uo pipefail; source /home/Ubuntu/longbench9/common.sh; tag=llama31_8b_instruct_p4fit_fp8kv; E=evaluation/eval_k_routing_ruler_p16.py
A=(--identity /home/Ubuntu/l31_retrieval_cal/identity/B.json --data $L/llama31_8b_instruct --bank /home/Ubuntu/l31_router_fit/router_B_p4/ours_b16r16 --sequence-length 131072 --rope native --router-fit-count 32 --router-diagnostic-count 0 --chat-template --benchmark longbench --ours-budget 320 --lrqk-topk 256 --shadowkv-budget 256 --page-size 4 --pinned-pages 0 --kv-fp8 --arms full,ours --output $L/eval_$tag)
say "=== [$tag] smoke full+ours"; CUDA_VISIBLE_DEVICES=${SMOKE_GPU_A:-0} $PY -u $E smoke --arm full "${A[@]}" > $L/logs/${tag}_smoke_full.log 2>&1 & p1=$!; CUDA_VISIBLE_DEVICES=${SMOKE_GPU_B:-1} $PY -u $E smoke --arm ours "${A[@]}" > $L/logs/${tag}_smoke_ours.log 2>&1 & p2=$!
st=0; wait $p1 || st=1; wait $p2 || st=1; [[ $st -eq 0 ]] || { say "smoke FAILED"; exit 1; }
$PY -u $E audit-smoke "${A[@]}" > $L/logs/${tag}_audit.log 2>&1 || { say "audit FAILED"; exit 1; }
say "=== [$tag] smoke + audit passed"; [[ ${STAGE:-all} == smoke ]] && exit 0
say "=== [$tag] evaluate full + ours: 8 shards each (2 procs per GPU)"; pids=()
for arm in full ours; do for s in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$s $PY -u $E evaluate --arm $arm "${A[@]}" --shard-index $s --num-shards 8 > $L/logs/${tag}_${arm}_shard$s.log 2>&1 & pids+=("$!"); done; done
st=0; for p in "${pids[@]}"; do wait "$p" || st=1; done; say "=== [$tag] evaluate exit=$st: full $(ls $L/eval_$tag/full/evaluate 2>/dev/null | wc -l) ours $(ls $L/eval_$tag/ours/evaluate 2>/dev/null | wc -l) /1543"; [[ $st -eq 0 ]] || exit 1
{ echo "# Llama-3.1-8B-Instruct LongBench-9: BF16 KV vs simulated FP8 (E4M3) KV, Full-K and B16R16 page-4 (256 + recent 64); $(TZ=America/New_York date '+%Y-%m-%d %H:%M %Z')"; $PY $L/compare_arms.py full_bf16=$L/eval_llama31_8b_instruct/full/evaluate ours_bf16=$L/eval_llama31_8b_instruct_p4fit/ours/evaluate full_fp8=$L/eval_$tag/full/evaluate ours_fp8=$L/eval_$tag/ours/evaluate; } > $L/eval_${tag}_summary.txt 2>&1; cat $L/eval_${tag}_summary.txt
say "=== [$tag] DONE ==="
