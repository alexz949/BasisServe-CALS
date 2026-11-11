#!/usr/bin/env bash
set -euo pipefail
source scripts/gh200_env.sh
c4_snapshot=/work/hdd/bido/azhang35/hf/.cache/huggingface/hub/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
set -x
python -u -m evaluation.prepare_qwen35_hybrid_data \
  --model-path "$HF_HUB_CACHE/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a" \
  --c4-dir "$c4_snapshot" \
  --output-dir "$BASISSERVE_RUN_ROOT/cache/qwen35_9b/data" --seed 20260909
python -u -m evaluation.prepare_qwen35_hybrid_data \
  --model-path "$HF_HUB_CACHE/models--nvidia--Nemotron-H-8B-Base-8K/snapshots/94ea861e008c2dfced3e8e1302094024077aa04e" \
  --c4-dir "$c4_snapshot" \
  --output-dir "$BASISSERVE_RUN_ROOT/cache/nemotron_h_8b/data" --seed 20260909
