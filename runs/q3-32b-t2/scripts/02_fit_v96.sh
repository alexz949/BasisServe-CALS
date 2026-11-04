#!/bin/bash
# Stage 2: uniform V96 joint V/O ALS fit from the covariance snapshot (Table-2 recipe: ALS 6 sweeps, CG 16 fixed, covariance
# damping 1e-7, FP32 work, BF16 factors), 8 layer shards on 8 GPUs, merge, then package for the routing fitter/evaluator.
set -eu; cd /home/sesterce/BasisServe-CALS; source runs/q3-32b-t2/scripts/common.sh
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $P scripts/fit_value.py fit-shard --snapshot-dir $R/snapshot --output-dir $R/v96 --rank 96 \
    --encoder-sweeps 6 --encoder-cg-mode fixed --encoder-cg-fixed-iterations 16 --covariance-damping 1e-7 \
    --work-dtype float32 --factor-dtype bfloat16 --layer-shard-index $i --layer-shard-count 8 --device cuda:0 \
    > $R/logs/02_fit_v96_shard$i.log 2>&1 &
done; wait
$P scripts/fit_value.py merge --output-dir $R/v96 2>&1 | tee $R/logs/02_fit_v96_merge.log
$P evaluation/package_c1_uniform_checkpoint.py --factor-bank-dir $R/v96 --output-dir $R/ckpt_v96 --model $MODEL \
  --identity $IDENTITY --rank 96 --format basisserve.qwen3_32b.c1_uniform_v96_128k_als6_retrievalmix.package.v1 \
  2>&1 | tee $R/logs/02_package.log
