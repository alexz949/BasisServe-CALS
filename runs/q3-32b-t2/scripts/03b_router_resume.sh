#!/bin/bash
# Resume stage 3 after moments+merge: base -> fisher (4 workers x 2 GPUs) -> fit (8 layer shards), all with --window-shards 4.
set -eu; cd /home/sesterce/BasisServe-CALS; source runs/q3-32b-t2/scripts/common.sh
COMMON="--identity $IDENTITY --windows $WINDOWS --output $ROUTER --sequence-length 131072 --fit-count 32 --diagnostic-count 0 \
  --page-size 4 --pinned-pages 0 --base-rank 16 --residual-rank 16 --sweeps 40 --pcg-iterations 100 --rope yarn4 --teacher deployed --window-shards 4"
$P evaluation/fit_k_routing_windowed.py base $COMMON 2>&1 | tee $R/logs/03_router_base.log
for s in 0 1 2 3; do g=$((2*s)); CUDA_VISIBLE_DEVICES=$g,$((g+1)) $P evaluation/fit_k_routing_windowed.py fisher $COMMON \
  --window-shard $s > $R/logs/03_router_fisher_s$s.log 2>&1 & done; wait
for i in 0 1 2 3 4 5 6 7; do LAYERS=$(seq -s, $i 8 63)
  CUDA_VISIBLE_DEVICES=$i $P evaluation/fit_k_routing_windowed.py fit $COMMON --layers $LAYERS > $R/logs/03_router_fit_s$i.log 2>&1 &
done; wait
