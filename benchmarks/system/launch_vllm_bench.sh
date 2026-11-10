#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# BASIS_ARTIFACTS holds envs/vllm/basis, cuda/12.9, checkpoints and calibration; BENCH_MODEL is the model snapshot.
artifacts=${BASIS_ARTIFACTS:-/home/yl1139/basis_artifacts}
model=${BENCH_MODEL:-/home/yl1139/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}
tp_size=$1
output_dir=$2
arm=$3
prompt=$4
batch=$5
experiment=$6
shift 6
test "$tp_size" = 4 || test "$tp_size" = 8
# Slurm renumbers CUDA_VISIBLE_DEVICES inside the job; physical ids are in SLURM_STEP_GPUS.
IFS=, read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:?Slurm allocation required}"
IFS=, read -r -a physical_gpus <<< "${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-$CUDA_VISIBLE_DEVICES}}"
selected_gpus=()
for index in "${!allocated_gpus[@]}"; do
    # Known faulty devices: node6 physical GPU 1 (ECC history), node5 physical GPU 2 (uncorrectable ECC).
    if [[ "${SLURMD_NODENAME:-}" == node6 && "${physical_gpus[$index]}" == 1 ]]; then continue; fi
    if [[ "${SLURMD_NODENAME:-}" == node5 && "${physical_gpus[$index]}" == 2 ]]; then continue; fi
    selected_gpus+=("${allocated_gpus[$index]}")
    if [[ ${#selected_gpus[@]} == "$tp_size" ]]; then break; fi
done
test "${#selected_gpus[@]}" = "$tp_size"
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${selected_gpus[*]}")
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1} NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1
export CUDA_HOME="$artifacts/cuda/12.9"
export PATH="$CUDA_HOME/bin:$artifacts/envs/vllm/basis/bin:$PATH"
export LD_LIBRARY_PATH="$artifacts/envs/vllm/basis/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9
mkdir -p "$output_dir"
printf 'node=%s tp=%s gpus=%s\n' "${SLURMD_NODENAME:-unknown}" "$tp_size" "$CUDA_VISIBLE_DEVICES"
python -m pip check
nvidia-smi topo -m > "$output_dir/topology.txt"
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used --format=csv > "$output_dir/devices.csv"
lscpu > "$output_dir/cpu.txt"
python -c 'from pathlib import Path; import json; print(json.dumps({str(p): p.read_text() for pattern in ("node*/cpulist", "node*/meminfo", "node*/distance") for p in Path("/sys/devices/system/node").glob(pattern)}, indent=2))' > "$output_dir/numa.json"
set -x
exec python -m benchmarks.system.bench_llama_vllm \
    --model "$model" \
    --factors "${BENCH_FACTORS:-$artifacts/checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix}" \
    --windows "$artifacts/calibration/llama31-8b-instruct-128k/c4-48x128k/windows.safetensors" \
    --tp "$tp_size" --output "$output_dir" --arm "$arm" \
    --prompt "$prompt" --batch "$batch" --experiment "$experiment" "$@"
