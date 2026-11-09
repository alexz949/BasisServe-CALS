#!/usr/bin/env bash
set -euo pipefail
source /mnt/basiskv/ruler/setup/env.sh
cd /mnt/basiskv/ruler/code
mkdir -p /mnt/basiskv/ruler/logs
pids=()
for gpu in {0..7}; do
    CUDA_VISIBLE_DEVICES="$gpu" python -u /mnt/basiskv/ruler/setup/worker.py validate --worker "$gpu" \
        >> "/mnt/basiskv/ruler/logs/validate${gpu}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
exit "$status"
