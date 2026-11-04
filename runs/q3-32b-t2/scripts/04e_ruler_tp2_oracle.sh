#!/bin/bash
# Oracle arm for the 100-prompt Table-2 RULER run (requested 2026-11-03): exact_sparse = FP32 exact-QK Page4 mass with the
# same GQA max rule, 2048 routed + recent 64 per KV group, no sink. Same protocol and output root as 04d (the spec does
# not depend on the arm list). The audit checks the exact_sparse smokes against Full-K; with arms full,exact_sparse its
# record (4 verified) is identical to the existing smoke_audit.json, which completed artifacts never overwrite.
set -euo pipefail; cd /home/sesterce/BasisServe-CALS; source runs/q3-32b-t2/scripts/common.sh
TORCHRUN=$HOME/miniforge3/envs/lowrank/bin/torchrun
TASKS=niah_single_1,niah_single_2,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,qa_1,qa_2,vt,fwe
OUT=$R/ruler_n10_tp2
COMMON="--identity $IDENTITY --data $R/ruler_data_10 --bank $ROUTER/ours_b16r16 --output $OUT --sequence-length 131072 --rope yarn4 \
  --tasks $TASKS --samples-per-task 10 --router-fit-count 32 --router-diagnostic-count 0 --ours-budget 2112 --page-size 4 --pinned-pages 0 \
  --chat-template --prompt-layout chat_answer_prefix --chat-template-kwargs {\"enable_thinking\":false} --arms full,exact_sparse --tensor-parallel 2"
replica() {  # $1 = replica index (GPU pair), rest = evaluator arguments
  local s=$1; shift; local g=$((2*s))
  CUDA_VISIBLE_DEVICES=$g,$((g+1)) OMP_NUM_THREADS=4 $TORCHRUN --nnodes 1 --nproc-per-node 2 --master-port $((29620+s)) \
    evaluation/eval_k_routing_ruler_p16.py "$@"
}
echo "smoke $(date)"
replica 0 smoke --arm exact_sparse $COMMON >> $R/logs/04tp_smoke_oracle.log 2>&1
echo "audit $(date)"
$P evaluation/eval_k_routing_ruler_p16.py audit-smoke $COMMON 2>&1 | tee -a $R/logs/04tp_audit_oracle.log | grep -E "ALL ARM SMOKES VERIFIED"
[ "${SMOKE_ONLY:-0}" = 1 ] && { echo "smoke only, stop $(date)"; exit 0; }
echo "evaluate exact_sparse (100) $(date)"
for s in 0 1 2 3; do replica $s evaluate --arm exact_sparse $COMMON --shard-index $s --num-shards 4 \
  > $R/logs/04tp_oracle_s$s.log 2>&1 & done
for pid in $(jobs -p); do wait $pid; done
echo "done $(date)"
