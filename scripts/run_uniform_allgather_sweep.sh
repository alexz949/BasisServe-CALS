#!/usr/bin/env bash
set -euo pipefail

TP_SIZE="${TP_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
WARMUP="${WARMUP:-100}"
ITERS="${ITERS:-2000}"
SWEEP_CHANNELS="${SWEEP_CHANNELS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/uniform_allgather}"
mkdir -p "$OUTPUT_DIR"

if (( TP_SIZE != 2 && TP_SIZE != 4 && TP_SIZE != 8 )); then
  echo "TP_SIZE must be 2, 4, or 8" >&2
  exit 2
fi
if (( 32 % TP_SIZE != 0 )); then
  echo "Qwen3-8B query-head count 32 is not divisible by TP_SIZE=$TP_SIZE" >&2
  exit 2
fi

# Qwen3-8B local wire width = local query heads * compact-V rank.
local_query_heads=$((32 / TP_SIZE))
ranks=(32 48 64 80 96 112)
tokens=(1 8 32 128)
algorithms=(fanout fanout_warp recursive_doubling ring)

for rank in "${ranks[@]}"; do
  local_width=$((local_query_heads * rank))
  for token_count in "${tokens[@]}"; do
    torchrun --standalone --nproc-per-node="$TP_SIZE" \
      benchmarks/bench_uniform_allgather.py \
      --local-width "$local_width" \
      --tokens "$token_count" \
      --dtype "$DTYPE" \
      --backends feature_direct,uniform_nccl,uniform_nccl_graph \
      --warmup "$WARMUP" --iters "$ITERS" \
      --output-json "$OUTPUT_DIR/nccl_r${rank}_t${token_count}.json"

    for algorithm in "${algorithms[@]}"; do
      torchrun --standalone --nproc-per-node="$TP_SIZE" \
        benchmarks/bench_uniform_allgather.py \
        --local-width "$local_width" \
        --tokens "$token_count" \
        --dtype "$DTYPE" \
        --backends uniform_ipc \
        --ipc-algorithm "$algorithm" \
        --warmup "$WARMUP" --iters "$ITERS" \
        --output-json \
          "$OUTPUT_DIR/ipc_${algorithm}_r${rank}_t${token_count}.json"
    done

    if [[ "$SWEEP_CHANNELS" == "1" ]]; then
      for algorithm in fanout ring; do
        for channels in 1 2 4 8; do
          torchrun --standalone --nproc-per-node="$TP_SIZE" \
            benchmarks/bench_uniform_allgather.py \
            --local-width "$local_width" \
            --tokens "$token_count" \
            --dtype "$DTYPE" \
            --backends uniform_ipc \
            --ipc-algorithm "$algorithm" \
            --ipc-channels "$channels" \
            --warmup "$WARMUP" --iters "$ITERS" \
            --output-json \
              "$OUTPUT_DIR/ipc_${algorithm}_c${channels}_r${rank}_t${token_count}.json"
        done
      done
    fi
  done
done
