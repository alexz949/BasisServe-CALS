#!/bin/bash
# Stage 4: RULER-128K, 11 tasks x 100 prompts (Table 2 reports 10; niah_single_3 dropped), chat template with
# enable_thinking=false and the official answer prefix, greedy. ours = B16R16 Page4, 2048 routed + 64 recent (hard 2112), no sink.
# Usage: 04_ruler.sh smoke|evaluate|summarize [arm...]   (evaluate: 4 replicas x 2 GPUs per arm, run arms sequentially)
set -eu; cd /home/sesterce/BasisServe-CALS; source runs/q3-32b-t2/scripts/common.sh
STAGE=$1; shift; ARMS=${@:-full ours}
# Table-2 tasks (niah_single_3 dropped); SAMPLES=10 -> first 10 prompts per task (100), SAMPLES=100 -> the full 1000.
TASKS=niah_single_1,niah_single_2,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,qa_1,qa_2,vt,fwe
SAMPLES=${SAMPLES:-10}
COMMON="--identity $IDENTITY --data $DATA --bank $ROUTER/ours_b16r16 --output $R/ruler --sequence-length 131072 --rope yarn4 \
  --tasks $TASKS --samples-per-task $SAMPLES --router-fit-count 32 --router-diagnostic-count 0 --ours-budget 2112 --page-size 4 --pinned-pages 0 \
  --chat-template --prompt-layout chat_answer_prefix --chat-template-kwargs {\"enable_thinking\":false} --arms full,ours"
for arm in $ARMS; do
  if [ "$STAGE" = evaluate ]; then
    for s in 0 1 2 3; do g=$((2*s)); CUDA_VISIBLE_DEVICES=$g,$((g+1)) $P evaluation/eval_k_routing_ruler_p16.py evaluate --arm $arm $COMMON \
      --shard-index $s --num-shards 4 > $R/logs/04_ruler_${arm}_s$s.log 2>&1 & done; wait
  else
    CUDA_VISIBLE_DEVICES=0,1 $P evaluation/eval_k_routing_ruler_p16.py $STAGE --arm $arm $COMMON 2>&1 | tee $R/logs/04_ruler_${STAGE}_$arm.log
  fi
done
