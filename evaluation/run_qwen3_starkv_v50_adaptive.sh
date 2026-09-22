#!/usr/bin/env bash
set -euo pipefail
cd /deac/csc/yangGrp/zhangal/BasisServe-CALS
export CONDA_DEFAULT_ENV=basis
export PATH=/home/zhangal/.conda/envs/basis/bin:$PATH
export HF_HOME=/deac/csc/yangGrp/zhangal/.cache/huggingface
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
phase=${1:?smoke, smoke8k, or full}
test "$phase" = smoke || test "$phase" = smoke8k || test "$phase" = full
extra=()
if [[ $phase == smoke || $phase == smoke8k ]]; then
    python tests/test_starkv_v50_adaptive.py
    extra=(--smoke)
fi
if [[ $phase == smoke8k ]]; then extra+=(--smoke-seq-len 8192); fi
root=ICLR-results/qwen3-8b/star-v50-adaptive
cmd=(python evaluation/train_qwen3_starkv_v50_adaptive.py
    --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4
    --output-dir "$root/$phase"
    "${extra[@]}")
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
