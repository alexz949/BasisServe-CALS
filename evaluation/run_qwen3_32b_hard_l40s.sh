#!/usr/bin/env bash
set -euo pipefail
cd /deac/csc/yangGrp/zhangal/BasisServe-CALS
export CONDA_DEFAULT_ENV=basis
export PATH=/deac/csc/yangGrp/zhangal/.cache/vllm/cu128-v0.18.0/venv/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_ALLOW_CODE_EVAL=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
arms=(Q3-32B-C1-R80 Q3-32B-PALUM-R80 Q3-32B-PALUG2-R80 Q3-32B-PALUG4-R80)
tasks=(minerva_math500 mbpp_plus_full bbh_cot_fewshot)
group=${SLURM_ARRAY_TASK_ID:?}
phase=${1:-full}
extra=()
if [[ $phase == smoke ]]; then
    phase=smoke-l40s-m85
    extra=(--limit 2)
fi
for arm in "${arms[$((group * 2))]}" "${arms[$((group * 2 + 1))]}"; do
    for task in "${tasks[@]}"; do
        cmd=(python evaluation/eval_qwen3_32b_hard_l40s_vllm.py
            --run-id "$arm"
            --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
            --checkpoint-dir "ICLR-results/qwen3-32b/checkpoints/$arm"
            --output-dir "ICLR-results/qwen3-32b/hard-r80/$phase/$arm/$task"
            --task "$task" --max-length 8192 --max-num-seqs 64
            --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85
            --torch-num-threads 4 --confirm-run-unsafe-code "${extra[@]}")
        printf '%q ' "${cmd[@]}"
        printf '\n'
        "${cmd[@]}"
    done
done
