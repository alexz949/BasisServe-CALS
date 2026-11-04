#!/bin/bash
# Per-query-head selection for BasisKV on the 100-prompt Table-2 RULER subset (requested 2026-11-03): every query head
# keeps its own 2048 routed + recent 64 tokens (Page4, no sink); the KV group's physical support is the union over its 8
# query heads, measured (`physical_union_per_kv_group_mean`), not capped. New evaluator source -> new protocol, so the
# Full-K smoke is rerun here only as the audit's first-token reference; the Full-K scores come from ruler_n10*/.
set -euo pipefail; cd /home/sesterce/BasisServe-CALS; source runs/q3-32b-t2/scripts/common.sh
TORCHRUN=$HOME/miniforge3/envs/lowrank/bin/torchrun
TASKS=niah_single_1,niah_single_2,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,qa_1,qa_2,vt,fwe
OUT=$R/ruler_n10_tp2_qh
COMMON="--identity $IDENTITY --data $R/ruler_data_10 --bank $ROUTER/ours_b16r16 --output $OUT --sequence-length 131072 --rope yarn4 \
  --tasks $TASKS --samples-per-task 10 --router-fit-count 32 --router-diagnostic-count 0 --ours-budget 2112 --page-size 4 --pinned-pages 0 \
  --ours-selection head --chat-template --prompt-layout chat_answer_prefix --chat-template-kwargs {\"enable_thinking\":false} \
  --arms full,ours --tensor-parallel 2"
replica() {  # $1 = replica index (GPU pair), rest = evaluator arguments
  local s=$1; shift; local g=$((2*s))
  CUDA_VISIBLE_DEVICES=$g,$((g+1)) OMP_NUM_THREADS=4 $TORCHRUN --nnodes 1 --nproc-per-node 2 --master-port $((29630+s)) \
    evaluation/eval_k_routing_ruler_p16.py "$@"
}
echo "smoke $(date)"
replica 0 smoke --arm full $COMMON >> $R/logs/04qh_smoke_full.log 2>&1 &
replica 1 smoke --arm ours $COMMON >> $R/logs/04qh_smoke_ours.log 2>&1 &
for pid in $(jobs -p); do wait $pid; done
echo "audit $(date)"
$P evaluation/eval_k_routing_ruler_p16.py audit-smoke $COMMON 2>&1 | tee -a $R/logs/04qh_audit.log | grep -E "ALL ARM SMOKES VERIFIED"
[ "${SMOKE_ONLY:-0}" = 1 ] && { echo "smoke only, stop $(date)"; exit 0; }
echo "evaluate ours per-query-head (100) $(date)"
for s in 0 1 2 3; do replica $s evaluate --arm ours $COMMON --shard-index $s --num-shards 4 > $R/logs/04qh_ours_s$s.log 2>&1 & done
for pid in $(jobs -p); do wait $pid; done
echo "done $(date)"
