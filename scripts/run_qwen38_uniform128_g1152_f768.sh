#!/usr/bin/env bash
set -euo pipefail
cd /home/lz299/BasisServe-CALS
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
export PYTHONPATH=results/q35_hybrid/deps:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
root=results/q38_hybrid
model=/home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
common=(--model-path "$model" --data "$root/data")
vbank="$root/banks/c1_uniform_v128.pt"
moments="$root/wo_uniform128_moments"
wo="$root/wo_uniform128_g1152_f768"
test -f "$root/capture/manifest.json"
test -f "$root/baseline_summary.json"
test -f "$root/checks/smoke.json"
run() {
  local gpu=$1 log=$2
  shift 2
  printf 'GPU=%s command=' "$gpu"
  printf '%q ' "$@"
  printf '\n'
  CUDA_VISIBLE_DEVICES="$gpu" "$@" >> "$root/logs/uniform128_$log.log" 2>&1
}
wait_all() {
  local status=0 pid
  for pid in "$@"; do wait "$pid" || status=1; done
  return "$status"
}

printf 'STAGE fit_uniform_v128\n'
pids=()
gpus=(2 5 6)
layer_shards=(3,15,27,39,51,63 7,19,31,43,55 11,23,35,47,59)
for shard in 0 1 2; do
  run "${gpus[$shard]}" "v_fit_$shard" python -u -m evaluation.run_qwen35_hybrid fit \
    "${common[@]}" --capture-dir "$root/capture" --layers "${layer_shards[$shard]}" \
    --ranks 128 --chunk-rows 2048 --linear-max-iter 200 \
    --encoder-preconditioner separable --output "$root/factors" &
  pids+=("$!")
done
wait_all "${pids[@]}"
if [[ ! -f "$vbank" ]]; then
  run '' assemble python -u -m evaluation.qwen35_hybrid_banks assemble "${common[@]}" \
    --model-manifest "$root/baseline_summary.json" --factors "$root/factors" \
    --anchor 128 --target-average-rank 128 --uniform --output "$root/banks"
fi
printf 'STAGE uniform_v_wo_capture\n'
if [[ ! -f "$moments/manifest.json" ]]; then
  run 6 wo_capture python -u -m evaluation.qwen35_hybrid_wo capture "${common[@]}" \
    --bank "$vbank" --moment-device cpu --output "$moments"
fi
printf 'STAGE fit_wo\n'
pids=()
for shard in 0 1 2; do
  run "${gpus[$shard]}" "wo_fit_$shard" python -u -m evaluation.qwen35_hybrid_wo fit \
    "${common[@]}" --bank "$vbank" --moments "$moments" --output "$wo" \
    --gdn-rank 1152 --full-rank 768 --work-dtype float64 --num-shards 3 --shard-index "$shard" &
  pids+=("$!")
done
wait_all "${pids[@]}"
if [[ ! -f "$wo/wo_bank.pt" ]]; then
  run '' wo_assemble python -u -m evaluation.qwen35_hybrid_wo assemble "${common[@]}" \
    --device cpu --bank "$vbank" --moments "$moments" --output "$wo" \
    --gdn-rank 1152 --full-rank 768 --work-dtype float64
fi
printf 'COMPLETE uniform V128 and Wo banks; downstream benchmarks have not been run\n'
