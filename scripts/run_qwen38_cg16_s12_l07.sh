#!/usr/bin/env bash
set -euo pipefail
cd /home/lz299/BasisServe-CALS
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
export PYTHONPATH=results/q35_hybrid/deps:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
root=results/q38_hybrid
model=/home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
test ! -e "$root/cg16_s12/l07_r128.pt"
printf '%s Waiting for a GPU with >=45000 MiB free and <=10%% utilization.\n' "$(date -Is)"
while true; do
  gpu=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '$2 >= 45000 && $3 <= 10 {print $1; exit}')
  if [[ -n "$gpu" ]]; then
    break
  fi
  sleep 30
done
export CUDA_VISIBLE_DEVICES="$gpu"
printf '%s Starting layer 7 comparison on GPU %s.\n' "$(date -Is)" "$gpu"
python -u -m evaluation.run_qwen35_hybrid fit \
  --model-path "$model" --data "$root/data" --capture-dir "$root/capture" \
  --layers 7 --ranks 128 --chunk-rows 2048 \
  --linear-max-iter 16 --encoder-sweeps 12 \
  --encoder-preconditioner separable --output "$root/cg16_s12"
printf '%s Comparison fit completed.\n' "$(date -Is)"
