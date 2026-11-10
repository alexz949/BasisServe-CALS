#!/usr/bin/env bash
set -euo pipefail
cd /zpool-00/home/yl1139/BasisServe-CALS

tp_size=$1
prompt_tokens=$2
batch_size=$3
output_dir=$4
test "$tp_size" = 4 || test "$tp_size" = 8

# Select only allocated devices; exclude node6's known faulty physical GPU1.
IFS=, read -r -a assigned_gpus <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
selected_gpus=()
for gpu in "${assigned_gpus[@]}"; do
    if [[ "${SLURMD_NODENAME:-}" == node6 && "$gpu" == 1 ]]; then
        continue
    fi
    selected_gpus+=("$gpu")
    if [[ ${#selected_gpus[@]} == "$tp_size" ]]; then
        break
    fi
done
test "${#selected_gpus[@]}" = "$tp_size"
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${selected_gpus[*]}")
export NCCL_P2P_DISABLE=1 CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 OMP_NUM_THREADS=2
printf 'node=%s tp=%s gpus=%s\n' "${SLURMD_NODENAME:-unknown}" "$tp_size" "$CUDA_VISIBLE_DEVICES"

set -x
exec /home/yl1139/miniforge3/envs/basis/bin/torchrun \
    --standalone --nproc-per-node="$tp_size" -m benchmarks.system.smoke_tp4_joint_model \
    --model /home/yl1139/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    --windows /home/yl1139/basis_artifacts/calibration/llama31-8b-instruct-128k/c4-48x128k/windows.safetensors \
    --factors /home/yl1139/basis_artifacts/checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix \
    --router /home/yl1139/basis_artifacts/checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix/router_b16r16 \
    --output "$output_dir" --arm basis_joint --prompt-tokens "$prompt_tokens" \
    --batch "$batch_size" --cohort 0 --conditioning-steps 16 --profile-steps 32
