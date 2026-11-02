#!/usr/bin/env bash
set -euo pipefail
cd /home/lz299/BasisServe-CALS
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate basis
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONPATH=.
run=results/llama31i_v96_64k
srun --exclusive --exact -N1 -n1 -c2 --gres=gpu:1 \
    python -m pytest -q tests/test_c1_strict_recent_decode.py \
    > "$run/logs/ruler_kernel_tests.log" 2>&1
srun --exclusive --exact -N1 -n1 -c2 --gres=gpu:2 \
    python -u evaluation/llama31_instruct_ruler_64k.py smoke \
    > "$run/logs/ruler_smoke.log" 2>&1
pids=()
for shard in 0 1; do
    srun --exclusive --exact -N1 -n1 -c2 --gres=gpu:2 \
        python -u evaluation/llama31_instruct_ruler_64k.py evaluate --shard "$shard" \
        > "$run/logs/ruler_${shard}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
python -u evaluation/llama31_instruct_ruler_64k.py summarize \
    > "$run/logs/ruler_summary.log" 2>&1
