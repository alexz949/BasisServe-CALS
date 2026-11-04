#!/bin/bash
# Stage 4 chain: smoke full + ours in parallel (GPUs 0-1, 2-3) -> audit-smoke gate -> evaluate full -> evaluate ours
# (4 replicas x 2 GPUs each) -> summarize. Table-2 tasks, SAMPLES prompts per task (default 10).
set -euo pipefail; cd /home/sesterce/BasisServe-CALS; source runs/q3-32b-t2/scripts/common.sh
TASKS=niah_single_1,niah_single_2,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,qa_1,qa_2,vt,fwe
SAMPLES=${SAMPLES:-10}
# The evaluator requires --samples-per-task to equal the dataset manifest; SAMPLES=10 uses the first-10-rows subset.
DATA_DIR=$DATA; [ "$SAMPLES" = 10 ] && DATA_DIR=$R/ruler_data_10
OUT=$R/ruler_n$SAMPLES
COMMON="--identity $IDENTITY --data $DATA_DIR --bank $ROUTER/ours_b16r16 --output $OUT --sequence-length 131072 --rope yarn4 \
  --tasks $TASKS --samples-per-task $SAMPLES --router-fit-count 32 --router-diagnostic-count 0 --ours-budget 2112 --page-size 4 --pinned-pages 0 \
  --chat-template --prompt-layout chat_answer_prefix --chat-template-kwargs {\"enable_thinking\":false} --arms full,ours"
echo "smoke $(date)"
CUDA_VISIBLE_DEVICES=0,1 $P evaluation/eval_k_routing_ruler_p16.py smoke --arm full $COMMON > $R/logs/04_ruler_smoke_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 $P evaluation/eval_k_routing_ruler_p16.py smoke --arm ours $COMMON > $R/logs/04_ruler_smoke_ours.log 2>&1 &
for pid in $(jobs -p); do wait $pid; done
echo "audit $(date)"
$P evaluation/eval_k_routing_ruler_p16.py audit-smoke $COMMON 2>&1 | tee $R/logs/04_ruler_audit.log | grep -E "ALL ARM SMOKES VERIFIED"
for arm in full ours; do
  echo "evaluate $arm $(date)"
  for s in 0 1 2 3; do g=$((2*s)); CUDA_VISIBLE_DEVICES=$g,$((g+1)) $P evaluation/eval_k_routing_ruler_p16.py evaluate --arm $arm $COMMON \
    --shard-index $s --num-shards 4 > $R/logs/04_ruler_${arm}_s$s.log 2>&1 & done
  for pid in $(jobs -p); do wait $pid; done
done
echo "summarize $(date)"
$P evaluation/eval_k_routing_ruler_p16.py summarize $COMMON 2>&1 | tee $R/logs/04_ruler_summarize.log | tail -20
echo "done $(date)"
