#!/usr/bin/env bash
set -euo pipefail
cd /home/lz299/BasisServe-CALS
v128_wait_pid=${1:-0}
if (( v128_wait_pid > 0 )); then
  echo "Waiting for V128 pipeline PID $v128_wait_pid"
  while kill -0 "$v128_wait_pid" 2>/dev/null; do
    sleep 30
  done
fi
vbank=results/q35_hybrid/banks_v128/c1_twosided_v128.pt
wobank=results/q35_hybrid/wo_v128_g768_f512/wo_bank.pt
if [[ ! -f "$vbank" || ! -f "$wobank" ]]; then
  echo 'PPL not started: the V128 pipeline did not produce both required banks.'
  exit 1
fi
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
export PYTHONPATH=results/q35_hybrid/deps:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
date -Is
CUDA_VISIBLE_DEVICES=6 python -u -m evaluation.run_qwen35_hybrid evaluate \
  --bank "$vbank" --wo-bank "$wobank" \
  --output results/q35_hybrid/ppl/twosided128_g768_f512.json
date -Is
echo 'V128 mixed-Wo PPL completed.'
