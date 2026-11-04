#!/bin/bash
# Finish the 100-prompt Table-2 RULER run with head-sharded TP2 replicas (requested 2026-11-03): Full-K only for the 13
# prompts the device_map pipeline run did not reach, BasisKV for all 100. TP2 results are a separate protocol
# (tensor_parallel=2, new evaluator hash) in ruler_n10_tp2/; scores are combined by runs/q3-32b-t2/combine_n10.py.
set -euo pipefail; cd /home/sesterce/BasisServe-CALS; source runs/q3-32b-t2/scripts/common.sh
TORCHRUN=$HOME/miniforge3/envs/lowrank/bin/torchrun
TASKS=niah_single_1,niah_single_2,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,qa_1,qa_2,vt,fwe
FULL_MISSING=84,88,89,90,91,92,93,94,95,96,97,98,99
OUT=$R/ruler_n10_tp2
COMMON="--identity $IDENTITY --data $R/ruler_data_10 --bank $ROUTER/ours_b16r16 --output $OUT --sequence-length 131072 --rope yarn4 \
  --tasks $TASKS --samples-per-task 10 --router-fit-count 32 --router-diagnostic-count 0 --ours-budget 2112 --page-size 4 --pinned-pages 0 \
  --chat-template --prompt-layout chat_answer_prefix --chat-template-kwargs {\"enable_thinking\":false} --arms full,ours --tensor-parallel 2"
replica() {  # $1 = replica index (GPU pair), rest = evaluator arguments
  local s=$1; shift; local g=$((2*s))
  CUDA_VISIBLE_DEVICES=$g,$((g+1)) OMP_NUM_THREADS=4 $TORCHRUN --nnodes 1 --nproc-per-node 2 --master-port $((29610+s)) \
    evaluation/eval_k_routing_ruler_p16.py "$@"
}
echo "smoke $(date)"
replica 0 smoke --arm full $COMMON > $R/logs/04tp_smoke_full.log 2>&1 &
replica 1 smoke --arm ours $COMMON > $R/logs/04tp_smoke_ours.log 2>&1 &
for pid in $(jobs -p); do wait $pid; done
echo "audit $(date)"
$P evaluation/eval_k_routing_ruler_p16.py audit-smoke $COMMON 2>&1 | tee $R/logs/04tp_audit.log | grep -E "ALL ARM SMOKES VERIFIED"
echo "evaluate full (13 missing) $(date)"
for s in 0 1 2 3; do replica $s evaluate --arm full $COMMON --sample-indices $FULL_MISSING --shard-index $s --num-shards 4 \
  > $R/logs/04tp_full_s$s.log 2>&1 & done
for pid in $(jobs -p); do wait $pid; done
echo "evaluate ours (100) $(date)"
for s in 0 1 2 3; do replica $s evaluate --arm ours $COMMON --shard-index $s --num-shards 4 > $R/logs/04tp_ours_s$s.log 2>&1 & done
for pid in $(jobs -p); do wait $pid; done
echo "done $(date)"
