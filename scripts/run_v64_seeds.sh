#!/usr/bin/env bash
# Nominal-V64 calibration-seed sensitivity (BasisKV, PaLU M-LRD, PaLU G-LRD4, Dense).
#
#   scripts/run_v64_seeds.sh MODEL launch "42 43 44"   # whole experiment on this node's GPUs
#   scripts/run_v64_seeds.sh MODEL STAGE [ARGS]        # one stage, on CUDA_VISIBLE_DEVICES
#
# MODEL is qwen3_8b (Qwen3-8B-Base) or llama31_8b_instruct. Each model follows the
# calibration protocol of its Table-1 row in the paper. Stages skip finished
# outputs, so rerunning the launch command resumes a failed run in place.
set -euo pipefail

cd "$(dirname "$0")/.."
SELF=scripts/run_v64_seeds.sh
PY=${BASIS_PYTHON:-$HOME/miniforge3/envs/basis/bin/python}
profile=$1
stage=$2
shift 2

case $profile in
  qwen3_8b)
    ROOT=results/rebuttal/q3_v64_seeds
    MODEL=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4
    RUN_PREFIX=Q3-8B
    # The paper's Qwen banks only regenerate under datasets 4.4.2.
    PY_WINDOWS=$HOME/miniforge3/envs/basis-win/bin/python
    KL_BANK=separate   # 640 x 4096-token bank drawn with calibration seed + 7
    DAMPING=1e-5
    KL_SHARDS=4
    WHITENING_SCRIPT=evaluation/build_qwen3_8b_section3_palu.py
    ;;
  llama31_8b_instruct)
    ROOT=results/rebuttal/l31i_v64_seeds
    MODEL=/data/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
    RUN_PREFIX=L31-8B-I
    # The paper's Llama banks regenerate under datasets 5.0.0.
    PY_WINDOWS=$PY
    KL_BANK=extended   # the calibration stream continued to 512 x 2048-token windows
    DAMPING=0
    KL_SHARDS=8
    WHITENING_SCRIPT=evaluation/build_llama31_8b_instruct_iclr_v_checkpoint.py
    ;;
  *)
    echo "unknown model: $profile" >&2
    exit 2
    ;;
esac
ROOT=${V64_ROOT:-$ROOT}
GPUS=${V64_GPUS:-"0 1 2 3 4 5 6 7"}
LOGS=$ROOT/logs
EVAL_WINDOWS=$ROOT/eval_inputs/c4_validation_128x2048/windows.safetensors
EVAL_SEED=20260828          # fixed C4-validation PPL bank of the paper
RANKS="32 48 64 80 96 112"  # fitted candidate banks; rank 128 is the exact dense endpoint
# Paper window counts; a smoke test overrides them with small values.
FIT_WINDOWS=${FIT_WINDOWS:-256}                        # PaLU whitening/Fisher and C1 fit
HELDOUT_WINDOWS=${HELDOUT_WINDOWS:-64}                 # C1 held-out error
KL_PROFILE_WINDOWS=${KL_PROFILE_WINDOWS:-128}          # rows 320..447 of the KL bank
KL_CONFIRMATION_WINDOWS=${KL_CONFIRMATION_WINDOWS:-16} # rows 448..463

seed=${1:-}
calib=$ROOT/calib/s$seed
basiskv=$ROOT/basiskv/s$seed

factor_dirs() {
  for rank in $RANKS; do printf -- "--factor-dir %s=%s/banks/R%s " "$rank" "$basiskv" "$rank"; done
}

kl_args() {
  # Two-Sided-KL settings of the paper's C1-R64 allocation for this model.
  echo "--model $MODEL --windows $calib/windows_kl/windows.safetensors \
    --snapshot-dir $calib/cov --profile-dir $basiskv/kl_profile $(factor_dirs) \
    --anchor-rank 64 --candidate-ranks 32,48,64,80,96,112,128 --window-start 320 \
    --profile-windows $KL_PROFILE_WINDOWS --confirmation-windows $KL_CONFIRMATION_WINDOWS --sequence-length 2048 --batch-size 16 \
    --covariance-damping $DAMPING --vocab-chunk-size 8192 --torch-num-threads 4 \
    --device-map balanced --max-memory-per-gpu-gib 44 --model-dtype bfloat16 \
    --attn-implementation sdpa --local-files-only --profile-shard-count $KL_SHARDS \
    --factorized-probe-rank 96 --factorized-compression-probe-rank 32 \
    --local-error-split heldout --factorized-exponent 1 --profile-backend sampled_suffix \
    --terminal-position-counts 64,128,256,512,1024 --terminal-selection-position-count 1024 \
    --terminal-position-seed 20260903"
}

on_gpu() {  # run one stage alone on the first free GPU
  local gpu status
  while true; do
    for gpu in $GPUS; do
      ( flock -n 9 || exit 99; CUDA_VISIBLE_DEVICES=$gpu "$SELF" "$profile" "$@" ) 9> "$LOGS/.gpu$gpu.lock" && return 0
      status=$?
      [ "$status" -eq 99 ] || return "$status"
    done
    sleep 5
  done
}

basiskv_chain() {  # SEED: statistics, candidate banks, Two-Sided-KL allocation, quality
  local seed=$1 pids=() pid shard rank
  on_gpu cov "$seed"
  for rank in $RANKS; do
    on_gpu bank "$seed" "$rank" > "$LOGS/s$seed.bank_R$rank.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  pids=()
  for shard in $(seq 0 $((KL_SHARDS - 1))); do
    on_gpu kl-profile "$seed" "$shard" > "$LOGS/s$seed.kl_profile_$shard.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  on_gpu kl-finalize "$seed"
  on_gpu eval "basiskv/s$seed" "$RUN_PREFIX-C1-R64"
}

palu_chain() {  # SEED: whitening, Fisher, M-LRD and G-LRD4 factors, quality
  local seed=$1 pids=() pid
  on_gpu whitening "$seed"
  on_gpu fisher "$seed"
  "$SELF" "$profile" palu "$seed" 1
  "$SELF" "$profile" palu "$seed" 4
  on_gpu eval "palu_m/s$seed" "$RUN_PREFIX-PALUM-R64" > "$LOGS/s$seed.palu_m_eval.log" 2>&1 &
  pids+=($!)
  on_gpu eval "palu_g4/s$seed" "$RUN_PREFIX-PALUG4-R64" > "$LOGS/s$seed.palu_g4_eval.log" 2>&1 &
  pids+=($!)
  for pid in "${pids[@]}"; do wait "$pid"; done
}

case $stage in
  launch)
    # All GPUs form one pool; every GPU stage holds one GPU alone (one model per GPU).
    seeds=${1:-"42 43 44"}
    mkdir -p "$LOGS"
    "$SELF" "$profile" eval-windows > "$LOGS/eval_windows.log" 2>&1
    pids=()
    for seed in $seeds; do
      "$SELF" "$profile" windows "$seed" > "$LOGS/s$seed.windows.log" 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
    for seed in $seeds; do
      basiskv_chain "$seed" > "$LOGS/s$seed.basiskv.log" 2>&1 &
      pids+=($!)
      palu_chain "$seed" > "$LOGS/s$seed.palu.log" 2>&1 &
      pids+=($!)
    done
    ( "$SELF" "$profile" dense && on_gpu eval dense "$RUN_PREFIX-Dense" ) > "$LOGS/dense.log" 2>&1 &
    pids+=($!)
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    echo "[launch] finished with status $status"
    exit $status
    ;;
  eval-windows)
    [ -f "$EVAL_WINDOWS" ] || $PY_WINDOWS "evaluation/prepare_${profile}_c4_windows.py" --model "$MODEL" \
      --output-dir "$(dirname "$EVAL_WINDOWS")" --dataset-split validation \
      --samples 128 --sequence-length 2048 --seed $EVAL_SEED
    ;;
  windows)
    [ -d "$calib/windows_fit" ] || $PY_WINDOWS "evaluation/prepare_${profile}_c4_windows.py" --model "$MODEL" \
      --output-dir "$calib/windows_fit" --samples "$FIT_WINDOWS" --sequence-length 2048 --seed "$seed"
    [ -d "$calib/windows_fit_heldout" ] || $PY_WINDOWS "evaluation/prepare_${profile}_c1_windows.py" --model "$MODEL" \
      --output-dir "$calib/windows_fit_heldout" --prefix-windows "$calib/windows_fit/windows.safetensors" \
      --fit-windows "$FIT_WINDOWS" --heldout-windows "$HELDOUT_WINDOWS" --sequence-length 2048 --seed "$seed" \
      --dataset-cache "$HOME/.cache/huggingface/datasets"
    if [ ! -d "$calib/windows_kl" ]; then
      if [ "$KL_BANK" = separate ]; then
        $PY_WINDOWS "evaluation/prepare_${profile}_c4_windows.py" --model "$MODEL" \
          --output-dir "$calib/windows_kl" --samples 640 --sequence-length 4096 --seed $((seed + 7))
      else
        calibration=$((FIT_WINDOWS + HELDOUT_WINDOWS))
        $PY_WINDOWS "evaluation/prepare_${profile}_c1_windows.py" --model "$MODEL" \
          --output-dir "$calib/windows_kl" --prefix-windows "$calib/windows_fit_heldout/windows.safetensors" \
          --fit-windows "$calibration" --heldout-windows $((512 - calibration)) --sequence-length 2048 --seed "$seed" \
          --dataset-cache "$HOME/.cache/huggingface/datasets"
      fi
    fi
    ;;
  cov)
    if [ ! -f "$calib/cov/manifest.json" ]; then
      rm -rf "$calib/cov" "$calib/cov.partial"  # an interrupted capture leaves a directory it will not overwrite
      $PY evaluation/capture_attention_o_proj_covariances.py --model-path "$MODEL" \
        --windows "$calib/windows_fit_heldout/windows.safetensors" --output-dir "$calib/cov" \
        --expected-attention gqa --fit-windows "$FIT_WINDOWS" --heldout-windows "$HELDOUT_WINDOWS" --device cuda:0
    fi
    ;;
  bank)
    rank=$2
    if [ ! -f "$basiskv/banks/R$rank/results.json" ]; then
      $PY scripts/fit_value.py fit-shard --snapshot-dir "$calib/cov" --output-dir "$basiskv/banks/R$rank" \
        --rank "$rank" --covariance-damping "$DAMPING" --device cuda:0 --resume
      $PY scripts/fit_value.py merge --output-dir "$basiskv/banks/R$rank"
    fi
    ;;
  kl-profile)
    shard=$2
    # shellcheck disable=SC2046
    $PY "evaluation/run_${profile}_c1_two_sided_factorized_kl_sharded.py" profile $(kl_args) --profile-shard-index "$shard"
    ;;
  kl-finalize)
    if [ ! -f "$basiskv/checkpoint/manifest.json" ]; then
      if [ ! -f "$basiskv/checkpoint/result.json" ]; then
        rm -rf "$basiskv/checkpoint"  # same for an interrupted allocation
        # shellcheck disable=SC2046
        $PY "evaluation/run_${profile}_c1_two_sided_factorized_kl_sharded.py" finalize $(kl_args) \
          --output-dir "$basiskv/checkpoint" --target-average-rank 64 --force-selected-candidate two_sided_factorized_kl
      fi
      $PY evaluation/build_c1_iclr_checkpoint.py --profile "$profile" --model "$MODEL" --allocation-dir "$basiskv/checkpoint"
    fi
    ;;
  whitening)
    [ -d "$calib/whitening" ] || $PY -m torch.distributed.run --standalone --nproc-per-node=1 \
      "$WHITENING_SCRIPT" prepare-whitening --model "$MODEL" \
      --output-dir "$calib/whitening" --windows "$calib/windows_fit/windows.safetensors" --samples "$FIT_WINDOWS"
    ;;
  fisher)
    [ -d "$calib/fisher" ] || $PY evaluation/collect_gqa_palu_fisher.py --profile "$profile" --target v --model "$MODEL" \
      --windows "$calib/windows_fit/windows.safetensors" --output-dir "$calib/fisher" \
      --samples "$FIT_WINDOWS" --sequence-length 2048 --retained-ratio 0.5 --device cuda:0
    ;;
  palu)
    group=$2
    name=$([ "$group" = 1 ] && echo palu_m || echo "palu_g$group")
    [ -d "$ROOT/$name/s$seed/checkpoint" ] || $PY "evaluation/build_${profile}_iclr_v_checkpoint.py" --method palu-fisher \
      --equivalent-rank 64 --head-group-size "$group" --model "$MODEL" \
      --fisher-result "$calib/fisher/fisher.json" --whitening-dir "$calib/whitening" \
      --output-dir "$ROOT/$name/s$seed/checkpoint" --torch-num-threads 8
    ;;
  dense)
    [ -d "$ROOT/dense/checkpoint" ] || $PY "evaluation/build_${profile}_iclr_v_checkpoint.py" --method dense \
      --model "$MODEL" --output-dir "$ROOT/dense/checkpoint"
    ;;
  eval)
    # eval ARM_DIR RUN_ID, e.g. eval basiskv/s42 Q3-8B-C1-R64; MCQ_LIMIT=10 for a smoke test
    arm=$1
    run_id=$2
    [ -f "$ROOT/$arm/quality/result.json" ] || $PY "evaluation/eval_${profile}_iclr_quality.py" --run-id "$run_id" \
      --model "$MODEL" --checkpoint-dir "$ROOT/$arm/checkpoint" --c4-windows "$EVAL_WINDOWS" \
      --output-dir "$ROOT/$arm/quality" --batch-size 2 --lm-eval-batch-size 8 \
      --max-memory-per-gpu-gib 44 --torch-num-threads 8 --expected-gpu-count 1 \
      ${MCQ_LIMIT:+--mcq-limit "$MCQ_LIMIT"}
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
