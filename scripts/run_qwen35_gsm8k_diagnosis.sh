#!/usr/bin/env bash
# Four paired 128-question diagnostics. Invoke only after launch approval.
set -euo pipefail
cd /home/lz299/BasisServe-CALS
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena
export CUDA_VISIBLE_DEVICES="${1:-2}"
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 VLLM_WORKER_MULTIPROC_METHOD=spawn
out=results/q35_hybrid/gsm8k_diagnosis
mkdir -p "$out"
common=(--limit 128 --max-num-seqs 32 --max-num-batched-tokens 4096
        --max-new-tokens 1024 --max-model-len 8192 --gpu-memory-utilization 0.40
        --kv-cache-gib 6 --seed 20260909)
for family in full_attention gdn; do
    python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm "${common[@]}" \
        --bank results/q35_hybrid/banks/c1_twosided_v64.pt \
        --wo-bank results/q35_hybrid/wo_twosided_v64/wo_bank.pt --wo-scope "$family" \
        --output "$out/v64_wo_${family}.json" >> "$out/v64_wo_${family}.log" 2>&1
done
for rank in 80 96; do
    python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm "${common[@]}" \
        --bank "results/q35_hybrid/banks/c1_twosided_v${rank}.pt" \
        --output "$out/v${rank}.json" >> "$out/v${rank}.log" 2>&1
done
