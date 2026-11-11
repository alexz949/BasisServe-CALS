#!/usr/bin/env bash
# Hold one allocation across validation, fixes and fresh-process benchmarks.
set -uo pipefail
cd /u/azhang35/BasisServe-CALS
source scripts/gh200_tp4_env.sh
root=results/rebuttal/kcuu_gh200_tp4_8k
date -Is
hostname
nvidia-smi topo -m
python -u -m evaluation.run_kcuu_experiments --stage all
smoke_code=$?
printf '%s\n' "$smoke_code" > "$root/allocation-$SLURM_JOB_ID.smoke_exit"
echo "Experiment driver exited $smoke_code; allocation retained for follow-up steps."
while [[ ! -f "$root/allocation-$SLURM_JOB_ID.release" ]]; do
    sleep 10
done
echo 'Allocation released after experiment review.'
