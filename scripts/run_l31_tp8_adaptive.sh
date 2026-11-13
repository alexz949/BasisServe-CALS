#!/usr/bin/env bash
# Llama-3.1-8B-Instruct TP8: Dense, adaptive BasisKV V96, adaptive PaLU G-LRD4 V96 replicated (A) and sharded (B).
# Usage: run_l31_tp8_adaptive.sh <hardware|reference|validate|smoke|formal|summary>
# Activate the basis (or lowrank) conda environment first. BASIS_ARTIFACTS holds the checkpoints and
# prompt windows; BENCH_MODEL is the model snapshot. Every arm runs eager because the sharded arm does.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
phase=$1
artifacts=${BASIS_ARTIFACTS:?}
export BENCH_MODEL=${BENCH_MODEL:?}
checkpoints=$artifacts/ICLR-results/llama31-8b-instruct/checkpoints
basis=$checkpoints/L31-8B-I-C1-R96
palu=$checkpoints/L31-8B-I-PALUG4-R96
windows=$artifacts/calibration/llama31-8b-instruct-128k/c4-48x128k/windows.safetensors
out=${BENCH_OUTPUT:-results/rebuttal/l31_tp8_adaptive}
check=benchmarks/system/validate_palu_g4_vllm.py
declare -A arm_of=([dense]=dense [basiskv]=als_full [palu_a]=palu_g4 [palu_b]=palu_g4_sharded)
declare -A factors_of=([dense]=$basis [basiskv]=$basis [palu_a]=$palu [palu_b]=$palu)
declare -A variant_of=([dense]=dense [basiskv]=basis [palu_a]=palu [palu_b]=palu)
methods=(dense basiskv palu_a palu_b)
# NCCL P2P is off for every arm: with it on, an 8-GPU all-reduce deadlocks on this node (hardware/nccl_p2p_check).
export PYTHONPATH="$PWD" NCCL_P2P_DISABLE=1 NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
mkdir -p "$out/logs"
exec > >(tee -a "$out/logs/$phase.log") 2>&1
echo "$(date -u +%FT%TZ) phase=$phase host=$(hostname) env=$CONDA_PREFIX gpus=$CUDA_VISIBLE_DEVICES"

snapshot() {  # machine state before a phase
    local dir=$out/hardware/$1
    mkdir -p "$dir"
    nvidia-smi > "$dir/nvidia-smi.txt"
    nvidia-smi topo -m > "$dir/topo_m.txt"
    nvidia-smi topo -p2p r > "$dir/topo_p2p_r.txt"
    nvidia-smi topo -p2p w > "$dir/topo_p2p_w.txt"
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu --format=csv > "$dir/devices.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$dir/processes.csv"
    env | grep -E '^(NCCL_|CUDA_|VLLM_|OMP_|NVIDIA_|TORCH_|TRITON_|CONDA_|PYTHONPATH|HF_HUB_OFFLINE)' | sort > "$dir/environment.txt"
    python -c 'import json, torch; from importlib.metadata import version
print(json.dumps(dict(cuda=torch.version.cuda, nccl=".".join(map(str, torch.cuda.nccl.version())),
    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    **{n: version(n) for n in ("torch", "vllm", "transformers", "triton")}), indent=2))' > "$dir/software.json"
    git rev-parse HEAD > "$dir/git_head.txt"
    git diff > "$dir/source.patch"
    echo "other GPU processes: $(($(wc -l < "$dir/processes.csv") - 1))"
}

bench() {  # bench <output dir> <method> <prompt> <batch> [extra flags]; the exact command is logged by set -x
    local dir=$1 method=$2 prompt=$3 batch=$4
    shift 4
    mkdir -p "$(dirname "$dir")"
    (set -x; python -m benchmarks.system.bench_llama_vllm --model "$BENCH_MODEL" --factors "${factors_of[$method]}" \
        --windows "$windows" --tp 8 --output "$dir" --arm "${arm_of[$method]}" --prompt "$prompt" --batch "$batch" \
        --experiment request --graph eager --repetitions 1 --gpu-memory-utilization 0.8 "$@") > "$dir.log" 2>&1 \
        && echo "ok $dir" || echo "FAILED $dir"
}

if [[ $phase == hardware ]]; then
    snapshot hardware
elif [[ $phase == reference ]]; then  # single-GPU HF references of the same factors
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
    for variant in dense basis palu; do
        factors=$basis; [[ $variant == palu ]] && factors=$palu
        python "$check" reference --variant "$variant" --factors "$factors" --windows "$windows" \
            --output "$out/validation/reference_$variant.json"
    done
elif [[ $phase == validate ]]; then  # TP8 serving against the reference of the same factors
    for method in "${methods[@]}"; do
        python "$check" serve --arm "${arm_of[$method]}" --tp 8 --gpu-memory-utilization 0.8 \
            --factors "${factors_of[$method]}" --windows "$windows" --output "$out/validation/vllm_$method.json" \
            > "$out/validation/vllm_$method.log" 2>&1 || echo "FAILED serve $method"
        python "$check" compare --reference "$out/validation/reference_${variant_of[$method]}.json" \
            --served "$out/validation/vllm_$method.json" --output "$out/validation/compare_$method.json" \
            || echo "FAILED compare $method"
    done
elif [[ $phase == smoke ]]; then  # short prompts, small batch, NCCL transport logged
    snapshot smoke
    export NCCL_DEBUG=INFO VLLM_COMPUTE_NANS_IN_LOGITS=1
    for method in "${methods[@]}"; do bench "$out/smoke/$method" "$method" 512 4 --smoke; done
elif [[ $phase == formal ]]; then
    snapshot formal
    for batch in 32 64; do
        for method in "${methods[@]}"; do bench "$out/formal/$method/b$batch" "$method" 4096 "$batch"; done
    done
elif [[ $phase == summary ]]; then
    python benchmarks/system/summarize_l31_tp8_adaptive.py --root "$out"
fi
echo "$(date -u +%FT%TZ) completed phase=$phase"
