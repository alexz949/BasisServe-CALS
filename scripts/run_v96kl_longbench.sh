#!/usr/bin/env bash
# Two CPU-light workers; select a currently idle GPU before each GPU process.
set -euo pipefail
cd /home/lz299/BasisServe-CALS
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
mkdir -p results/logs/v96kl

gpu_run() {
    local label="$1"
    shift
    local attempt gpu total free util lockfd code acquired slot slotfd slot_acquired max_util
    slot_acquired=0
    while (( slot_acquired == 0 )); do
        for slot in 0 1; do
            exec {slotfd}>"results/logs/v96kl/worker_${slot}.lock"
            if flock -n "$slotfd"; then slot_acquired=1; break; fi
            exec {slotfd}>&-
        done
        if (( slot_acquired == 0 )); then sleep 30; fi
    done
    for attempt in 1 2 3; do
        acquired=0
        while (( acquired == 0 )); do
            # Prefer idle devices. When both are busy, use the largest free-memory
            # allocation allowed by AGENTS.md; our own workers never share a GPU.
            for max_util in 10 100; do
                while IFS=, read -r gpu total free util; do
                    gpu="${gpu// /}"; free="${free// /}"; util="${util// /}"
                    if [[ "$gpu" != 2 && "$gpu" != 4 ]]; then continue; fi
                    if (( free < 60000 || util > max_util )); then continue; fi
                    exec {lockfd}>"results/logs/v96kl/gpu_${gpu}.lock"
                    if flock -n "$lockfd"; then acquired=1; break; fi
                    exec {lockfd}>&-
                done < <(nvidia-smi --query-gpu=index,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits | sort -t, -k3,3nr -k4,4n)
                if (( acquired )); then break; fi
            done
            if (( acquired == 0 )); then
                printf '%s waiting for GPU 2 or 4 with >=60000 MiB free: %s\n' "$(date -Is)" "$label"
                sleep 30
            fi
        done
        printf '%s GPU=%s free_MiB=%s utilization=%s attempt=%s command=' "$(date -Is)" "$gpu" "$free" "$util" "$attempt" >>"results/logs/v96kl/${label}.log"
        printf '%q ' "$@" >>"results/logs/v96kl/${label}.log"
        printf '\n' >>"results/logs/v96kl/${label}.log"
        if CUDA_VISIBLE_DEVICES="$gpu" "$@" >>"results/logs/v96kl/${label}.log" 2>&1; then code=0; else code=$?; fi
        flock -u "$lockfd"
        exec {lockfd}>&-
        if (( code == 0 )); then
            flock -u "$slotfd"
            exec {slotfd}>&-
            return 0
        fi
        printf '%s failed %s exit=%s; completed artifacts remain reusable\n' "$(date -Is)" "$label" "$code"
        sleep 10
    done
    flock -u "$slotfd"
    exec {slotfd}>&-
    return "$code"
}

two_shards() {
    local label="$1"
    shift
    local first second code=0
    gpu_run "${label}_0" "$@" --shard-index 0 --num-shards 2 & first=$!
    gpu_run "${label}_1" "$@" --shard-index 1 --num-shards 2 & second=$!
    wait "$first" || code=$?
    wait "$second" || code=$?
    return "$code"
}

python -u evaluation/prepare_v96kl_data.py windows >>results/logs/v96kl/windows.log 2>&1 & windows_pid=$!
python -u evaluation/prepare_v96kl_data.py longbench >>results/logs/v96kl/data.log 2>&1 & data_pid=$!
wait "$windows_pid"
gpu_run calibration python -u evaluation/calibrate_v96kl_router.py & calibration_pid=$!
wait "$data_pid"
for arm in full lrqk; do
    gpu_run "smoke_${arm}" python -u evaluation/eval_v96kl_longbench.py smoke --arm "$arm"
done
two_shards full python -u evaluation/eval_v96kl_longbench.py evaluate --arm full & full_pid=$!
two_shards lrqk python -u evaluation/eval_v96kl_longbench.py evaluate --arm lrqk & lrqk_pid=$!
wait "$calibration_pid"
gpu_run smoke_b16r16 python -u evaluation/eval_v96kl_longbench.py smoke --arm b16r16
two_shards b16r16 python -u evaluation/eval_v96kl_longbench.py evaluate --arm b16r16 & router_pid=$!
wait "$full_pid"
wait "$lrqk_pid"
wait "$router_pid"
python -u evaluation/eval_v96kl_longbench.py summarize >>results/logs/v96kl/summary.log 2>&1
printf '%s COMPLETE results/evaluation/v96kl_longbench/summary.md\n' "$(date -Is)"
