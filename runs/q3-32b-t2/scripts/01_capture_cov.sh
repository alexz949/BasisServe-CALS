#!/bin/bash
# Stage 1: attention o_proj covariance snapshot over the 48 mixed 128K windows (32 fit + 16 held-out), dense teacher,
# YaRN factor 4 (original 32768). Two window shards in parallel (GPUs 0-3 and 4-7, 24 windows each; smoke: 42 GiB peak
# per GPU, ~2.5 min per window), then a row-weighted merge. Output: $R/snapshot
set -eu; cd /home/sesterce/BasisServe-CALS; source runs/q3-32b-t2/scripts/common.sh
COMMON="--model-path $MODEL --windows $WINDOWS --expected-attention gqa --fit-windows 32 --heldout-windows 16 --rope yarn4 \
  --device-map balanced --max-memory-per-gpu-gib 70 --torch-num-threads 8 --window-shards 2"
CUDA_VISIBLE_DEVICES=0,1,2,3 $P evaluation/capture_attention_o_proj_covariances.py $COMMON --window-shard 0 --output-dir $R/snapshot_s0 > $R/logs/01_capture_cov_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 $P evaluation/capture_attention_o_proj_covariances.py $COMMON --window-shard 1 --output-dir $R/snapshot_s1 > $R/logs/01_capture_cov_s1.log 2>&1 &
wait
$P evaluation/merge_attention_o_proj_covariances.py --shards $R/snapshot_s0 $R/snapshot_s1 --output-dir $R/snapshot 2>&1 | tee $R/logs/01_capture_cov_merge.log
