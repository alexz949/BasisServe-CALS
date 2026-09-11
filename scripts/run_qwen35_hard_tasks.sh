#!/usr/bin/env bash
set -euo pipefail
cd /home/lz299/BasisServe-CALS
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ALLOW_CODE_EVAL=1
mode=${1:?Use smoke or full}
[[ "$mode" == smoke || "$mode" == full ]]
out=results/q35_hybrid/hard_tasks
mkdir -p "$out/$mode" "$out/logs"
# CPU-only reference check is shared by all three model arms.
if [[ "$mode" == smoke ]]; then
  python -u -m evaluation.validate_mbpp_plus_full > "$out/logs/mbpp_reference.log" 2>&1
fi
run_arm() {
  local gpu=$1 arm=$2
  shift 2
  local task
  local limit=()
  if [[ "$mode" == smoke ]]; then limit=(--limit 2); fi
  for task in minerva_math500 mbpp_plus_full ifeval; do
    local output="$out/$mode/${arm}_${task}.json"
    if [[ -f "$output" ]]; then
      echo "Existing output requires review before rerunning: $output"
      return 1
    fi
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm \
      --task "$task" --max-num-seqs 32 --max-num-batched-tokens 4096 \
      --max-model-len 8192 --gpu-memory-utilization 0.40 --kv-cache-gib 6 \
      --seed 20260909 --output "$output" "${limit[@]}" "$@" \
      >> "$out/logs/${mode}_${arm}_${task}.log" 2>&1
  done
}
pids=()
run_arm 2 dense &
pids+=("$!")
run_arm 5 dense_wo \
  --bank results/q35_hybrid/banks/c1_uniform_v256.pt \
  --wo-bank results/q35_hybrid/wo_dense_g768_f512/wo_bank.pt &
pids+=("$!")
run_arm 6 v128_wo \
  --bank results/q35_hybrid/banks_v128/c1_twosided_v128.pt \
  --wo-bank results/q35_hybrid/wo_v128_g768_f512/wo_bank.pt &
pids+=("$!")
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
