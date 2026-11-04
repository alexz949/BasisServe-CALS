#!/bin/bash
# CUDA_HOME for building BasisServe CUDA extensions against the `basis` env (torch 2.13.0+cu130) on the A100 node.
# The env's pip toolchain mixes nvcc 13.4 with CUDA 13.0 runtime headers and CCCL 13.3, which CCCL rejects
# ("CUDA compiler and CUDA toolkit headers are incompatible"); there is no system CUDA toolkit. This installs one
# consistent CUDA 13.0 compiler set outside the env and exposes it in the CUDA_HOME layout torch expects
# (bin, include, nvvm, lib64/libcudart.so). Host compiler: /usr/bin/g++-11 (the system gcc-12 lacks cc1plus).
set -euo pipefail
PYTHON=$HOME/miniforge3/envs/basis/bin/python
TARGET=$HOME/cuda-13.0-pip
CUDA_HOME=$HOME/cuda-13.0
# cuRAND 10.4.0.35 (= the env's) is needed by FlashInfer's JIT sampling kernels that vLLM builds at startup.
$PYTHON -m pip install --no-deps --upgrade --target "$TARGET" nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 \
  nvidia-nvvm==13.0.88 nvidia-cuda-cccl==13.0.85 nvidia-cuda-runtime==13.0.96 nvidia-curand==10.4.0.35
SOURCE=$TARGET/nvidia/cu13
mkdir -p "$CUDA_HOME/lib64"
ln -sfn "$SOURCE/bin" "$CUDA_HOME/bin"
ln -sfn "$SOURCE/include" "$CUDA_HOME/include"
ln -sfn "$SOURCE/nvvm" "$CUDA_HOME/nvvm"
for library in "$SOURCE"/lib/*; do ln -sfn "$library" "$CUDA_HOME/lib64/"; done
ln -sfn "$SOURCE/lib/libcudart.so.13" "$CUDA_HOME/lib64/libcudart.so"
"$CUDA_HOME/bin/nvcc" --version | tail -1
# Build environment: CUDA_HOME=$HOME/cuda-13.0 CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 TORCH_CUDA_ARCH_LIST=8.0
# with $HOME/miniforge3/envs/basis/bin on PATH (ninja).
