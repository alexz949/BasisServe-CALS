#!/bin/bash
# Stage 3: Base16 + page-Fisher Residual16 router (Table-2 recipe): deployed V96 teacher, 32 fit windows (16 C4 + 16 synthetic),
# no diagnostic windows, page 4, no pinned sink, recent 64 excluded, 64 fit queries (4 strata x 16), ALS 40 / PCG 100.
# moments and fisher replays are window-sharded: 4 workers x 2 GPUs. Bank: $ROUTER/ours_b16r16
set -eu; cd /home/sesterce/BasisServe-CALS; source runs/q3-32b-t2/scripts/common.sh
COMMON="--identity $IDENTITY --windows $WINDOWS --output $ROUTER --sequence-length 131072 --fit-count 32 --diagnostic-count 0 \
  --page-size 4 --pinned-pages 0 --base-rank 16 --residual-rank 16 --sweeps 40 --pcg-iterations 100 --rope yarn4 --teacher deployed"
replay() {  # $1 = stage
  for s in 0 1 2 3; do g=$((2*s)); CUDA_VISIBLE_DEVICES=$g,$((g+1)) $P evaluation/fit_k_routing_windowed.py $1 $COMMON \
    --window-shard $s --window-shards 4 > $R/logs/03_router_$1_s$s.log 2>&1 & done; wait
}
replay moments
$P evaluation/fit_k_routing_windowed.py merge $COMMON --window-shards 4 2>&1 | tee $R/logs/03_router_merge.log
$P evaluation/fit_k_routing_windowed.py base  $COMMON --window-shards 4 2>&1 | tee $R/logs/03_router_base.log
replay fisher
for i in 0 1 2 3 4 5 6 7; do  # residual ALS per layer, 8 layer shards
  LAYERS=$(seq -s, $i 8 63)
  CUDA_VISIBLE_DEVICES=$i $P evaluation/fit_k_routing_windowed.py fit $COMMON --window-shards 4 --layers $LAYERS > $R/logs/03_router_fit_s$i.log 2>&1 &
done; wait
