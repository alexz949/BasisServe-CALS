# Source this file in Delta-GH jobs before running the Qwen3-32B TP4 serving scripts.
source /u/azhang35/miniforge3/etc/profile.d/conda.sh
conda activate basis
export CUDA_HOME=/projects/bido/azhang35/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MAX_JOBS=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_NCCL_SYMM_MEM=1
export TRITON_CACHE_DIR=/projects/bido/azhang35/triton/gh200
export TORCH_EXTENSIONS_DIR=/projects/bido/azhang35/torch_extensions/gh200
export VLLM_CACHE_ROOT=/projects/bido/azhang35/vllm_cache
export FLASHINFER_WORKSPACE_BASE=/projects/bido/azhang35/flashinfer
export TORCHINDUCTOR_CACHE_DIR=/projects/bido/azhang35/torchinductor
export HF_HUB_CACHE=/projects/bido/azhang35/hf/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
# The default cudatoolkit module puts CUDA 12.9 headers ahead of CUDA 13 through CPATH and the Cray cc/CC wrappers.
unset CPATH
export CC=gcc CXX=g++
