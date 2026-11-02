#!/usr/bin/env bash
set -euo pipefail
cd /home/lz299/BasisServe-CALS
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate basis
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONPATH=.
run=results/llama31i_v96_64k
mkdir -p "$run/logs"
python -u evaluation/llama31_instruct_c1_64k.py prepare > "$run/logs/prepare.log" 2>&1
pids=()
for shard in 0 1 2 3; do
    srun --exclusive --exact -N1 -n1 -c2 --gres=gpu:1 \
        python -u evaluation/llama31_instruct_c1_64k.py capture --shard "$shard" \
        > "$run/logs/capture_${shard}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
python -u evaluation/llama31_instruct_c1_64k.py merge > "$run/logs/merge_capture.log" 2>&1
pids=()
for shard in 0 1 2 3; do
    srun --exclusive --exact -N1 -n1 -c2 --gres=gpu:1 \
        python -u scripts/fit_value.py fit-shard \
        --snapshot-dir "$run/covariance" --output-dir "$run/c1" \
        --fit-windows 64 --validation-windows 16 --rank 96 \
        --work-dtype float32 --factor-dtype float32 \
        --layer-shard-index "$shard" --layer-shard-count 4 --torch-num-threads 2 --resume \
        > "$run/logs/fit_${shard}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
python -u scripts/fit_value.py merge --output-dir "$run/c1" \
    > "$run/logs/merge_c1.log" 2>&1
