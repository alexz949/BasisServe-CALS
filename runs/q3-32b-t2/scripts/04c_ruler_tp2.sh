#!/bin/bash
# Stage 4 with head-sharded TP2 replicas (evaluation/routing_tensor_parallel.py): each replica is one torchrun group on a
# GPU pair, so both GPUs work on every layer instead of alternating as in the device_map pipeline.
# smoke full + ours (pairs 0-1, 2-3) -> audit-smoke -> evaluate full -> evaluate ours (4 replicas) -> summarize.
set -euo pipefail; cd /home/sesterce/BasisServe-CALS; source runs/q3-32b-t2/scripts/common.sh
TORCHRUN=$HOME/miniforge3/envs/lowrank/bin/torchrun
TASKS=niah_single_1,niah_single_2,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,qa_1,qa_2,vt,fwe
SAMPLES=${SAMPLES:-10}
DATA_DIR=$DATA; [ "$SAMPLES" = 10 ] && DATA_DIR=$R/ruler_data_10
OUT=$R/ruler_n${SAMPLES}_tp2
COMMON="--identity $IDENTITY --data $DATA_DIR --bank $ROUTER/ours_b16r16 --output $OUT --sequence-length 131072 --rope yarn4 \
  --tasks $TASKS --samples-per-task $SAMPLES --router-fit-count 32 --router-diagnostic-count 0 --ours-budget 2112 --page-size 4 --pinned-pages 0 \
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
for arm in full ours; do
  echo "evaluate $arm $(date)"
  for s in 0 1 2 3; do replica $s evaluate --arm $arm $COMMON --shard-index $s --num-shards 4 > $R/logs/04tp_${arm}_s$s.log 2>&1 & done
  for pid in $(jobs -p); do wait $pid; done
done
echo "summarize $(date)"
$P evaluation/eval_k_routing_ruler_p16.py summarize $COMMON 2>&1 | tee $R/logs/04tp_summarize.log | tail -20
echo "done $(date)"
