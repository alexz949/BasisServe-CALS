#!/usr/bin/env bash
# Dense-V calibration and Wo-only evaluation; launch after user confirmation.
set -euo pipefail
cd /home/lz299/BasisServe-CALS
source /home/lz299/miniconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES="${1:-6}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p results/q35_hybrid/logs
conda activate lowrank
export PYTHONPATH=.
python -u -m evaluation.qwen35_hybrid_banks assemble --uniform --anchor 256 \
  --output results/q35_hybrid/banks >> results/q35_hybrid/logs/dense_v_bank.log 2>&1
export PYTHONPATH=results/q35_hybrid/deps:.
python -u -m evaluation.qwen35_hybrid_wo capture \
  --bank results/q35_hybrid/banks/c1_uniform_v256.pt \
  --output results/q35_hybrid/wo_dense_moments \
  >> results/q35_hybrid/logs/wo_dense_capture.log 2>&1
python -u -m evaluation.qwen35_hybrid_wo fit \
  --bank results/q35_hybrid/banks/c1_uniform_v256.pt \
  --moments results/q35_hybrid/wo_dense_moments \
  --work-dtype float64 --output results/q35_hybrid/wo_dense \
  >> results/q35_hybrid/logs/wo_dense_fit.log 2>&1
conda activate lowrankarena
export PYTHONPATH=.
common=(--bank results/q35_hybrid/banks/c1_uniform_v256.pt
        --wo-bank results/q35_hybrid/wo_dense/wo_bank.pt --wo-scope all
        --max-num-seqs 32 --max-num-batched-tokens 4096 --max-new-tokens 1024
        --max-model-len 8192 --gpu-memory-utilization 0.40 --kv-cache-gib 6 --seed 20260909)
python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm "${common[@]}" --smoke \
  --output results/q35_hybrid/gsm8k_vllm/smoke_dense_wo.json \
  >> results/q35_hybrid/logs/gsm8k_smoke_dense_wo.log 2>&1
conda activate lowrank
export PYTHONPATH=results/q35_hybrid/deps:.
python -u -m evaluation.check_qwen35_vllm_reference \
  --smoke results/q35_hybrid/gsm8k_vllm/smoke_dense_wo.json \
  --output results/q35_hybrid/gsm8k_vllm/reference_dense_wo.json \
  >> results/q35_hybrid/logs/gsm8k_reference_dense_wo.log 2>&1
# This is a short-prefix smoke gate, not a proof of arbitrary-length parity.
python -u - >> results/q35_hybrid/logs/gsm8k_reference_dense_wo.log 2>&1 <<'PY'
import json
from pathlib import Path
rows = json.loads(Path('results/q35_hybrid/gsm8k_vllm/reference_dense_wo.json').read_text())['rows']
assert len(rows) == 4
assert all(r['top1_agreement'] >= 0.95 and r['max_abs_chosen_logprob_difference'] < 0.3 for r in rows)
print('Short-prefix HF/vLLM smoke gate passed', flush=True)
PY
conda activate lowrankarena
export PYTHONPATH=.
python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm "${common[@]}" \
  --output results/q35_hybrid/gsm8k_vllm/result_dense_wo.json \
  >> results/q35_hybrid/logs/gsm8k_vllm_result_dense_wo.log 2>&1
