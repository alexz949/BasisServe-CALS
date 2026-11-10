#!/usr/bin/env bash
set -euo pipefail
cd /zpool-00/home/yl1139/BasisServe-CALS
tp_size=$1
output_dir=$2
arm=${3:-dense}
test "$tp_size" = 4 || test "$tp_size" = 8
IFS=, read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
selected_gpus=()
for gpu in "${allocated_gpus[@]}"; do
    if [[ "${SLURMD_NODENAME:-}" == node6 && "$gpu" == 1 ]]; then
        continue
    fi
    selected_gpus+=("$gpu")
    if [[ ${#selected_gpus[@]} == "$tp_size" ]]; then break; fi
done
test "${#selected_gpus[@]}" = "$tp_size"
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${selected_gpus[*]}")
export NCCL_P2P_DISABLE=1 NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1
export PATH=/home/yl1139/basis_artifacts/envs/vllm/basis/bin:$PATH
export LD_LIBRARY_PATH=/home/yl1139/basis_artifacts/envs/vllm/basis/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_HOME=/home/yl1139/basis_artifacts/cuda/12.9
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9
printf 'node=%s tp=%s gpus=%s\n' "${SLURMD_NODENAME:-unknown}" "$tp_size" "$CUDA_VISIBLE_DEVICES"
mkdir -p "$output_dir"
python -m pip check
python -c 'import sqlite3; print("sqlite", sqlite3.sqlite_version)'
nvcc --version
python -m pip freeze > "$output_dir/packages.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv > "$output_dir/devices.csv"
common=(
    --model /home/yl1139/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
    --windows /home/yl1139/basis_artifacts/calibration/llama31-8b-instruct-128k/c4-48x128k/windows.safetensors
    --tp "$tp_size"
    --arm "$arm"
    --factors /home/yl1139/basis_artifacts/checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix
)
set -x
python -m benchmarks.system.smoke_llama_vllm "${common[@]}" --graph eager --output "$output_dir/eager.json"
python -m benchmarks.system.smoke_llama_vllm "${common[@]}" --graph full --output "$output_dir/graph.json" --reference "$output_dir/eager.json"
