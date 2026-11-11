# Source this file in Delta-GH jobs before using Hugging Face or experiment scripts.
source /u/azhang35/miniforge3/etc/profile.d/conda.sh
conda activate lowrank
export HF_HOME=/projects/bido/azhang35/hf/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN_PATH=/u/azhang35/.cache/huggingface/token
export HF_DATASETS_CACHE=/work/hdd/bido/azhang35/hf/.cache/huggingface/datasets
export BASISSERVE_RUN_ROOT=/work/hdd/bido/azhang35/BasisServe-CALS
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
