#!/usr/bin/env bash
set -euo pipefail
cd /home/lz299/BasisServe-CALS
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
export PYTHONPATH=results/q35_hybrid/deps:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 VLLM_WORKER_MULTIPROC_METHOD=spawn
logdir=results/q35_hybrid/logs
vbank=results/q35_hybrid/banks_v128/c1_twosided_v128.pt
moments=results/q35_hybrid/wo_v128_moments
wo=results/q35_hybrid/wo_v128_g768_f512
pids=()
gpus=(5 6)
layer_shards=(3,11,19,27 7,15,23,31)
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${gpus[$shard]}" python -u -m evaluation.run_qwen35_hybrid fit \
    --layers "${layer_shards[$shard]}" --ranks 160,192,224 --chunk-rows 2048 \
    --linear-max-iter 200 --encoder-preconditioner separable \
    --output results/q35_hybrid/factors >> "$logdir/v128_v_fit_$shard.log" 2>&1 &
  pids+=("$!")
done
fit_status=0
for fit_pid in "${pids[@]}"; do
  wait "$fit_pid" || fit_status=1
done
if (( fit_status != 0 )); then exit "$fit_status"; fi
pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${gpus[$shard]}" python -u -m evaluation.qwen35_hybrid_banks profile \
    --anchor 128 --num-shards 2 --shard-index "$shard" --output results/q35_hybrid/kl \
    >> "$logdir/v128_kl_$shard.log" 2>&1 &
  pids+=("$!")
done
fit_status=0
for fit_pid in "${pids[@]}"; do
  wait "$fit_pid" || fit_status=1
done
if (( fit_status != 0 )); then exit "$fit_status"; fi
python -u -m evaluation.qwen35_hybrid_banks assemble --anchor 128 --target-average-rank 128 \
  --output results/q35_hybrid/banks_v128 >> "$logdir/v128_assemble.log" 2>&1
CUDA_VISIBLE_DEVICES=6 python -u -m evaluation.qwen35_hybrid_wo capture \
  --bank "$vbank" --output "$moments" >> "$logdir/v128_wo_capture.log" 2>&1
pids=()
gpus=(5 6)
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${gpus[$shard]}" python -u -m evaluation.qwen35_hybrid_wo fit \
    --bank "$vbank" --moments "$moments" --output "$wo" \
    --gdn-rank 768 --full-rank 512 --work-dtype float64 --num-shards 2 --shard-index "$shard" \
    >> "$logdir/v128_wo_fit_$shard.log" 2>&1 &
  pids+=("$!")
done
fit_status=0
for fit_pid in "${pids[@]}"; do
  wait "$fit_pid" || fit_status=1
done
if (( fit_status != 0 )); then exit "$fit_status"; fi
python -u -m evaluation.qwen35_hybrid_wo assemble --device cpu \
  --bank "$vbank" --moments "$moments" --output "$wo" \
  --gdn-rank 768 --full-rank 512 --work-dtype float64 \
  >> "$logdir/v128_wo_assemble.log" 2>&1
conda activate lowrankarena
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=6 python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm \
  --bank "$vbank" --wo-bank "$wo/wo_bank.pt" --wo-scope all \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --max-new-tokens 1024 --max-model-len 8192 \
  --gpu-memory-utilization 0.40 --kv-cache-gib 6 --seed 20260909 \
  --output results/q35_hybrid/gsm8k_vllm/result_twosided128_g768_f512_wo.json \
  >> "$logdir/gsm8k_vllm_result_twosided128_g768_f512_wo.log" 2>&1
conda activate lowrank
python -u -m evaluation.summarize_qwen35_gsm8k_v96 --v128-wo \
  --output results/q35_hybrid/gsm8k_vllm/v128_g768_f512_wo_summary.json \
  >> "$logdir/v128_gsm8k_audit.log" 2>&1
