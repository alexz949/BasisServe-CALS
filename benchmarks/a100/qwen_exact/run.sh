#!/usr/bin/env bash
set -euo pipefail
source /mnt/basiskv/ruler/setup/env.sh
cd /mnt/basiskv/ruler/code
pids=()
for gpu in {0..7}; do
    CUDA_VISIBLE_DEVICES="$gpu" python -u /mnt/basiskv/qwen-exact/setup/worker.py evaluate --worker "$gpu" >> "/mnt/basiskv/qwen-exact/logs/worker${gpu}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
exit "$status"
