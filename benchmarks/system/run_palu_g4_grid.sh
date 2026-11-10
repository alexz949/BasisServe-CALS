#!/usr/bin/env bash
# PaLU G-LRD4 TP4 deployment grid: Dense, BasisKV V96/V64, PaLU-A and PaLU-B at R96/R64.
# Usage: run_palu_g4_grid.sh <phase smoke|formal> <graph full|eager> [arm ...]
# GRID_BATCHES overrides the formal batch list; GRID_ROOT overrides the phase directory (other hardware).
# Each cell is one fresh vLLM process (warmup trial plus three repetitions inside it); cells
# whose summary.json is already complete are skipped so a stopped grid can resume.
set -u
cd /zpool-00/home/yl1139/BasisServe-CALS
phase=$1; graph=$2; shift 2
checkpoints=/home/yl1139/basis_artifacts/checkpoints
declare -A arm_of=(
    [dense]=dense [basis_v96]=als_full [basis_v64]=als_full
    [palu_g4_r96]=palu_g4 [palu_g4_r64]=palu_g4
    [palu_g4_sharded_r96]=palu_g4_sharded [palu_g4_sharded_r64]=palu_g4_sharded)
declare -A factors_of=(
    [dense]=$checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix
    [basis_v96]=$checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix
    [basis_v64]=$checkpoints/attention_c1/llama31_8b_instruct_uniform_v64_factor_bank_r64_s6_d0
    [palu_g4_r96]=$checkpoints/palu_g4/L31-8B-I-PALUG4-R96 [palu_g4_r64]=$checkpoints/palu_g4/L31-8B-I-PALUG4-R64
    [palu_g4_sharded_r96]=$checkpoints/palu_g4/L31-8B-I-PALUG4-R96
    [palu_g4_sharded_r64]=$checkpoints/palu_g4/L31-8B-I-PALUG4-R64)
arms=("$@"); [[ ${#arms[@]} == 0 ]] && arms=(dense basis_v96 basis_v64 palu_g4_r96 palu_g4_r64 palu_g4_sharded_r96 palu_g4_sharded_r64)
if [[ $phase == smoke ]]; then batches=(32); experiments=(request); extra=(--smoke); else read -r -a batches <<< "${GRID_BATCHES:-32 64 128}"; experiments=(request decode); extra=(); fi
root=${GRID_ROOT:-results/rebuttal/v2q8_palu_g4_tp4/$phase}/$graph
mkdir -p "$root"
echo "grid start $(date -u +%FT%TZ) node=${SLURMD_NODENAME:-?} physical_gpus=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-?}} phase=$phase graph=$graph arms=${arms[*]}"
for batch in "${batches[@]}"; do
  for name in "${arms[@]}"; do
    for experiment in "${experiments[@]}"; do
      out=$root/$name/p4096_b$batch/$experiment
      if grep -q '"status": "complete"' "$out/summary.json" 2>/dev/null; then echo "skip $out"; continue; fi
      echo "=== $name b$batch $experiment $(date -u +%T)"
      mkdir -p "$(dirname "$out")"
      BENCH_FACTORS=${factors_of[$name]} bash benchmarks/system/launch_vllm_bench.sh 4 "$out" "${arm_of[$name]}" 4096 "$batch" "$experiment" --graph "$graph" "${extra[@]}" > "$out.log" 2>&1
      echo "exit=$? $(grep -oE '"status": "[a-z_]+"' "$out/summary.json" 2>/dev/null | tail -1)"
    done
  done
done
echo "grid done $(date -u +%FT%TZ)"
