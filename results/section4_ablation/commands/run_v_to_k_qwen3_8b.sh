#!/usr/bin/env bash
# Section 4 experiment 1 on Qwen3-8B (post, V96 identity, retrieval-mix calibration, YaRN x4): V -> pre-RoPE K predictability.
set -uo pipefail; source /home/Ubuntu/q3_8b_post_128k/common.sh; F=evaluation/section4_v_to_k_information.py; OUT=$W/section4_v_to_k
A=(--identity $ID --windows $W/c4_retrieval_50_50 --output $OUT --sequence-length $SEQ --rope $ROPE --window-shards 8)
say "=== [q3 v2k] moments: 8 window shards"; pids=(); for s in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$s $PY -u $F moments "${A[@]}" --window-shard $s > $L/v2k_moments_s$s.log 2>&1 & pids+=("$!"); done
st=0; for p in "${pids[@]}"; do wait "$p" || st=1; done; [[ $st -eq 0 ]] || { say "moments FAILED"; exit 1; }
say "=== [q3 v2k] solve"; $PY -u $F solve "${A[@]}" > $L/v2k_solve.log 2>&1 || { say "solve FAILED"; exit 1; }
tail -n 1 $L/v2k_solve.log; say "=== Q3 V2K DONE ==="
