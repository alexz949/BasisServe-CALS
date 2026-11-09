#!/usr/bin/env bash
# Active environment: lowrank. Each visible GPU loads one complete model.
set -euo pipefail
cd "$(dirname "$0")/.."
exact_output="${1:-runs/l31-exact-page4}"
exact_logs="$exact_output/logs"
mkdir -p "$exact_logs"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s tests/test_exact_page_routing.py > "$exact_logs/unit-tests.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s tests/test_exact_page4_integration.py > "$exact_logs/integration-tests.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python evaluation/eval_llama_exact_sparse_page4.py prepare --output "$exact_output" > "$exact_logs/prepare.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python evaluation/eval_llama_exact_sparse_page4.py smoke --output "$exact_output" --resume > "$exact_logs/smoke.log" 2>&1
exact_arms=(exact_sparse_page4)
if python - "$exact_output" <<'PY'
import json,sys
from pathlib import Path
gate=json.loads((Path(sys.argv[1])/'smoke.json').read_text())
sys.exit(0 if gate['require_matched_proxy_rerun'] else 1)
PY
then
    exact_arms+=(proxy_page4)
fi
for exact_arm in "${exact_arms[@]}"; do
    exact_pids=()
    for exact_gpu in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES="$exact_gpu" python evaluation/eval_llama_exact_sparse_page4.py evaluate \
            --output "$exact_output" --arm "$exact_arm" --shard "$exact_gpu" --shards 4 --resume \
            > "$exact_logs/${exact_arm}_${exact_gpu}.log" 2>&1 &
        exact_pids+=("$!")
    done
    exact_failed=0
    for exact_pid in "${exact_pids[@]}"; do
        if ! wait "$exact_pid"; then exact_failed=1; fi
    done
    if [ "$exact_failed" -ne 0 ]; then
        echo "A worker failed. Partial outputs are retained; inspect the named worker log before resuming."
        exit 1
    fi
done
python evaluation/eval_llama_exact_sparse_page4.py summarize --output "$exact_output" > "$exact_logs/summary.log" 2>&1
python evaluation/export_exact_page4_tables.py --output "$exact_output" > "$exact_logs/export.log" 2>&1
