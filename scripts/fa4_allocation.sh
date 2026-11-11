#!/usr/bin/env bash
# Keep one GPU allocation available for FA4 correctness checks and follow-ups.
set -uo pipefail
cd /u/azhang35/BasisServe-CALS
source scripts/gh200_tp4_env.sh
result_root=results/rebuttal/gh200_tp4/fa4
date -Is
hostname
nvidia-smi --query-gpu=name,uuid --format=csv
python -u -m evaluation.benchmark_fa4_diffkv --stage smoke \
    --output "$result_root/smoke-$SLURM_JOB_ID.json"
initial_code=$?
if [[ $initial_code == 0 ]]; then
    python -u -m evaluation.benchmark_fa4_diffkv --stage bench \
        --output "$result_root/bench-$SLURM_JOB_ID.json"
    initial_code=$?
fi
printf '%s\n' "$initial_code" > "$result_root/allocation-$SLURM_JOB_ID.initial_exit"
echo "Initial tests exited $initial_code; allocation retained for follow-up steps."
while [[ ! -f "$result_root/allocation-$SLURM_JOB_ID.release" ]]; do
    sleep 5
done
echo 'Allocation released after experiment review.'
