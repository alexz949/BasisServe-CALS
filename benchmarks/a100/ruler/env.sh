export PATH=/mnt/basiskv/miniforge3/envs/lowrank/bin:$PATH
export CONDA_PREFIX=/mnt/basiskv/miniforge3/envs/lowrank
export CONDA_DEFAULT_ENV=lowrank
export HF_HOME=/mnt/basiskv/cache/huggingface
export LD_LIBRARY_PATH=/mnt/basiskv/cuda-compat/usr/local/cuda-13.0/compat
export TRITON_CACHE_DIR=/mnt/basiskv/ruler/cache/triton
export TORCH_EXTENSIONS_DIR=/mnt/basiskv/ruler/cache/extensions
export PYTHONPATH=/mnt/basiskv/ruler/code
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
