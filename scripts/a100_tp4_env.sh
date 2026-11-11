# Source from the repository root on the authorized standalone A100 host.
source /home/sesterce/miniforge3/etc/profile.d/conda.sh
conda activate basis
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4
export KCUU_PLATFORM=a100
export TORCH_CUDA_ARCH_LIST=8.0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MAX_JOBS=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_NCCL_SYMM_MEM=1 VLLM_USE_V2_MODEL_RUNNER=0
export HF_HUB_CACHE=/mnt/basis-data/hf
export TRITON_CACHE_DIR=/mnt/basis-data/cache/triton
export TORCH_EXTENSIONS_DIR=/mnt/basis-data/cache/torch_extensions
export VLLM_CACHE_ROOT=/mnt/basis-data/cache/vllm
export FLASHINFER_WORKSPACE_BASE=/mnt/basis-data/cache/flashinfer
export TORCHINDUCTOR_CACHE_DIR=/mnt/basis-data/cache/torchinductor
export TMPDIR=/mnt/basis-data/tmp
export CC=gcc CXX=g++
