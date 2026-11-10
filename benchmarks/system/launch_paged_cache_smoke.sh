#!/usr/bin/env bash
set -euo pipefail
cd /zpool-00/home/yl1139/BasisServe-CALS
output_dir=$1
test_module=${2:-benchmarks.system.smoke_paged_joint_cache}
IFS=, read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
selected_gpus=()
for gpu in "${allocated_gpus[@]}"; do
    if [[ "${SLURMD_NODENAME:-}" == node6 && "$gpu" == 1 ]]; then continue; fi
    selected_gpus+=("$gpu")
    if [[ ${#selected_gpus[@]} == 4 ]]; then break; fi
done
test "${#selected_gpus[@]}" = 4
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${selected_gpus[*]}")
export NCCL_P2P_DISABLE=1 NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=2
export CUDA_HOME=/home/yl1139/basis_artifacts/cuda/12.9
export PATH="$CUDA_HOME/bin:/home/yl1139/basis_artifacts/envs/vllm/basis/bin:$PATH"
export LD_LIBRARY_PATH=/home/yl1139/basis_artifacts/envs/vllm/basis/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
printf 'node=%s gpus=%s\n' "${SLURMD_NODENAME:-unknown}" "$CUDA_VISIBLE_DEVICES"
set -x
exec torchrun --standalone --nproc-per-node=4 -m "$test_module" --output "$output_dir"
