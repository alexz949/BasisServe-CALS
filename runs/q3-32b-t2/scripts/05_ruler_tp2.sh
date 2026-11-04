#!/bin/bash
# Table-2 RULER-128K on all 1000 prompts (10 tasks x 100; niah_single_3 dropped), head-sharded TP2, 4 replicas.
# BasisKV = B16R16 Page4, 2048 routed + recent 64 per KV group (hard 2112, GQA max), no sink. Both arms are smoked and
# audited up front so one gate covers them; the arms given as arguments are then evaluated in order.
# Usage: 05_ruler_tp2.sh full [ours]
set -euo pipefail; cd /home/sesterce/BasisServe-CALS; source runs/q3-32b-t2/scripts/common.sh
TORCHRUN=$HOME/miniforge3/envs/lowrank/bin/torchrun
TASKS=niah_single_1,niah_single_2,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,qa_1,qa_2,vt,fwe
OUT=$R/ruler_tp2
COMMON="--identity $IDENTITY --data $DATA --bank $ROUTER/ours_b16r16 --output $OUT --sequence-length 131072 --rope yarn4 \
  --tasks $TASKS --samples-per-task 100 --router-fit-count 32 --router-diagnostic-count 0 --ours-budget 2112 --page-size 4 --pinned-pages 0 \
  --chat-template --prompt-layout chat_answer_prefix --chat-template-kwargs {\"enable_thinking\":false} --arms full,ours --tensor-parallel 2"
replica() {  # $1 = replica index (GPU pair), rest = evaluator arguments
  local s=$1; shift; local g=$((2*s))
  CUDA_VISIBLE_DEVICES=$g,$((g+1)) OMP_NUM_THREADS=4 $TORCHRUN --nnodes 1 --nproc-per-node 2 --master-port $((29640+s)) \
    evaluation/eval_k_routing_ruler_p16.py "$@"
}
echo "smoke $(date)"
replica 0 smoke --arm full $COMMON >> $R/logs/05_smoke_full.log 2>&1 &
replica 1 smoke --arm ours $COMMON >> $R/logs/05_smoke_ours.log 2>&1 &
for pid in $(jobs -p); do wait $pid; done
echo "audit $(date)"
$P evaluation/eval_k_routing_ruler_p16.py audit-smoke $COMMON 2>&1 | tee -a $R/logs/05_audit.log | grep -E "ALL ARM SMOKES VERIFIED"
for arm in "$@"; do
  echo "evaluate $arm (1000) $(date)"
  for s in 0 1 2 3; do replica $s evaluate --arm $arm $COMMON --shard-index $s --num-shards 4 >> $R/logs/05_${arm}_s$s.log 2>&1 & done
  for pid in $(jobs -p); do wait $pid; done
done
echo "done $(date)"
