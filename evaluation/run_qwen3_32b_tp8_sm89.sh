#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
output=results/vllm_32b_tp8_sm89_4k_budget8k
test ! -e "$output/dense.json"
test ! -e "$output/c1.json"
mkdir -p "$output"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/usr/local/cuda
export OMP_NUM_THREADS=1
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.9
export VLLM_WORKER_MULTIPROC_METHOD=spawn

runner=(/workspace/miniforge3/bin/conda run --no-capture-output -n basis python)
common=(
  --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
  --factor-dir /workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-32b/c1/factor-banks/R64-S6
  --factor-validation structure
  --output-dir "$output"
  --batch-sizes 1 2 4 8 16 32 64 128 256
  --prefill-tokens 4096
  --decode-tokens 128
  --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.8
  --warmups 1
  --repeats 3
  --profile-batches 1 32 256
)

for arm in dense c1; do
  printf 'Starting %s: %s\n' "$arm" "$(date -u +%FT%TZ)"
  "${runner[@]}" evaluation/benchmark_vllm_qwen3_8b_c1.py \
    --arm "$arm" "${common[@]}" > "$output/$arm.log" 2>&1
  printf 'Completed %s: %s\n' "$arm" "$(date -u +%FT%TZ)"
done

"${runner[@]}" evaluation/summarize_vllm_qwen3_8b_c1.py \
  --input-dir "$output" > "$output/summary.log" 2>&1
