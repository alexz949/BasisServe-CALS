#!/bin/bash
# Table-2 LongBench-v1 (ShadowKV 9 tasks, official prompts > 4096 tokens: 1549 rows), head-sharded TP2, 4 replicas.
# Same protocol as Qwen3-8B Page4: chat template (enable_thinking=false; official no-chat tasks stay raw), greedy, official
# caps and metrics; BasisKV = B16R16 Page4, 256 routed + recent 64 per KV group (hard 320, GQA max), no sink.
# Both arms are smoked and audited up front so one gate covers them; the arms given as arguments are then evaluated.
# Usage: 06_longbench_tp2.sh full [ours]
set -euo pipefail; cd /home/sesterce/BasisServe-CALS; source runs/q3-32b-t2/scripts/common.sh
TORCHRUN=$HOME/miniforge3/envs/lowrank/bin/torchrun
OUT=$R/longbench_tp2
COMMON="--benchmark longbench --identity $IDENTITY --data $R/longbench_data --bank $ROUTER/ours_b16r16 --output $OUT \
  --sequence-length 131072 --rope yarn4 --router-fit-count 32 --router-diagnostic-count 0 --ours-budget 320 --page-size 4 \
  --pinned-pages 0 --chat-template --chat-template-kwargs {\"enable_thinking\":false} --arms full,ours --tensor-parallel 2"
replica() {  # $1 = replica index (GPU pair), rest = evaluator arguments
  local s=$1; shift; local g=$((2*s))
  CUDA_VISIBLE_DEVICES=$g,$((g+1)) OMP_NUM_THREADS=4 $TORCHRUN --nnodes 1 --nproc-per-node 2 --master-port $((29650+s)) \
    evaluation/eval_k_routing_ruler_p16.py "$@"
}
echo "smoke $(date)"
replica 0 smoke --arm full $COMMON >> $R/logs/06_smoke_full.log 2>&1 &
replica 1 smoke --arm ours $COMMON >> $R/logs/06_smoke_ours.log 2>&1 &
for pid in $(jobs -p); do wait $pid; done
echo "audit $(date)"
$P evaluation/eval_k_routing_ruler_p16.py audit-smoke $COMMON 2>&1 | tee -a $R/logs/06_audit.log | grep -E "ALL ARM SMOKES VERIFIED"
for arm in "$@"; do
  echo "evaluate $arm (1549) $(date)"
  for s in 0 1 2 3; do replica $s evaluate --arm $arm $COMMON --shard-index $s --num-shards 4 >> $R/logs/06_${arm}_s$s.log 2>&1 & done
  for pid in $(jobs -p); do wait $pid; done
done
echo "done $(date)"
