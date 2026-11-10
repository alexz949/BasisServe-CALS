# Running the PaLU G-LRD4 TP4 deployment grid on 4x L40S

This is a self-contained instruction for reproducing the grid on another machine. It measures how
PaLU G-LRD4 behaves when deployed under tensor parallelism, next to Dense and BasisKV, for the
rebuttal request "a unified accuracy-memory-latency comparison, including measured PaLU latency".

Everything below was already run end to end on 4x RTX A5000 (batches 32 and 64). The L40S run is
the same code with batch 128 added. Do not change any benchmark setting; if something does not
fit, record it and report back instead of adapting the protocol.

## What is measured

Llama-3.1-8B-Instruct, BF16, vLLM 0.30, TP4, 4096-token prompts, 128 generated tokens, greedy.

| Arm | Meaning |
|---|---|
| `dense` | Unmodified model. |
| `basis_v96`, `basis_v64` | BasisKV, uniform Value rank 96 / 64 per head. |
| `palu_g4_r96`, `palu_g4_r64` | PaLU G-LRD4, group latent replicated on every rank (Scheme A). Standard fused attention, no extra communication, more Value memory. |
| `palu_g4_sharded_r96`, `palu_g4_sharded_r64` | PaLU G-LRD4, group latent split across the two ranks that share a group (Scheme B). Less memory, but each decode step exchanges attention weights and partial outputs between the rank pair. Eager only. |

Each cell is one fresh vLLM process: one warmup trial, then three measured repetitions.
`request` measures end-to-end latency, TTFT and TPOT. `decode` measures the pure decode step.
The grid is 7 arms x batches 32/64/128 x 2 experiments in eager mode, plus the 5 non-sharded
arms again with CUDA Graphs.

## Requirements

- One node with 4 healthy L40S (48 GB) that nothing else is using. NVIDIA driver 550 or newer.
- About 30 GB of disk: 15 GB model, 2 GB factors, 9 GB Python environment.
- 10 CPU cores and 160 GB RAM were requested for the reference run.
- A Hugging Face account with access to `meta-llama/Llama-3.1-8B-Instruct`.

Before starting, check the GPUs for ECC problems and keep the output with the results:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.used,ecc.errors.uncorrected.aggregate.total --format=csv
```

## 1. Code and paths

Two absolute paths are all the scripts need. `BASIS_ARTIFACTS` holds the environment, the CUDA
toolkit, the factors and the prompt windows. `BENCH_MODEL` is the model snapshot directory.

```bash
export BASIS_ARTIFACTS=/abs/path/to/basis_artifacts      # choose; must be absolute
git clone https://github.com/alexz949/BasisServe-CALS.git
cd BasisServe-CALS                                        # run every command below from here
git log -1 --oneline                                      # record this
mkdir -p results/rebuttal/v2q8_palu_g4_tp4/logs
```

## 2. Environment

The environment directory must be named `basis` (the benchmark asserts it).

```bash
conda create -y -p "$BASIS_ARTIFACTS/envs/vllm/basis" -c conda-forge python=3.11
command -v uv || pip install uv
uv pip install --python "$BASIS_ARTIFACTS/envs/vllm/basis/bin/python" \
  'https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  --torch-backend=cu129
bash benchmarks/system/install_vllm_toolkit.sh            # CUDA 12.9 nvcc/cudart/curand into $BASIS_ARTIFACTS/cuda/12.9
"$BASIS_ARTIFACTS/envs/vllm/basis/bin/python" -m pip check
```

Reference versions: vLLM 0.30.0+cu129, torch 2.13.0+cu129, Triton 3.7.1, Transformers 5.17.0.
Every cell records the versions it actually ran with in its `config.json`.

## 3. Assets

All factors and prompt windows are public in the Hugging Face model repo `alexz949/BasisServe-CALS`.
The loaders verify SHA-256 against the manifests, so a bad download fails at model load.

```bash
HF="$BASIS_ARTIFACTS/envs/vllm/basis/bin/hf"; STAGE="$BASIS_ARTIFACTS/hf_stage"
"$HF" download alexz949/BasisServe-CALS --revision 57d90e28dd1d63245a028b9d376deec156758068 --local-dir "$STAGE" \
  --include "ICLR-results/llama31-8b-instruct/checkpoints/L31-8B-I-PALUG4-R96/*" \
  --include "ICLR-results/llama31-8b-instruct/checkpoints/L31-8B-I-PALUG4-R64/*" \
  --include "ICLR-results/llama31-8b-instruct/c1/factor-banks/R64-S6-D0/layer_*.safetensors" \
  --include "checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix/layer_*.safetensors" \
  --include "checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix/manifest.json" \
  --include "calibration/llama31-8b-instruct-128k/c4-48x128k/*"        # 71 files, 1.8 GB

C="$BASIS_ARTIFACTS/checkpoints"; V64="$C/attention_c1/llama31_8b_instruct_uniform_v64_factor_bank_r64_s6_d0"
mkdir -p "$C/palu_g4" "$C/attention_c1" "$BASIS_ARTIFACTS/calibration/llama31-8b-instruct-128k"
mv "$STAGE"/ICLR-results/llama31-8b-instruct/checkpoints/L31-8B-I-PALUG4-R96 "$C/palu_g4/"
mv "$STAGE"/ICLR-results/llama31-8b-instruct/checkpoints/L31-8B-I-PALUG4-R64 "$C/palu_g4/"
mv "$STAGE"/checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix "$C/attention_c1/"
mv "$STAGE"/ICLR-results/llama31-8b-instruct/c1/factor-banks/R64-S6-D0 "$V64"
mv "$STAGE"/calibration/llama31-8b-instruct-128k/c4-48x128k "$BASIS_ARTIFACTS/calibration/llama31-8b-instruct-128k/"
# The V64 bank has no manifest on Hugging Face; the one the loader needs is kept in this repo.
cp results/rebuttal/v2q8_palu_g4_tp4/assets/llama31_8b_instruct_uniform_v64_factor_bank_r64_s6_d0.manifest.json "$V64/manifest.json"

# Model (gated: log in first with `"$HF" auth login`). The command prints the snapshot directory.
export BENCH_MODEL=$("$HF" download meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 --exclude "original/*" | tail -1 | sed 's/^path=//')
ls "$BENCH_MODEL/config.json"
```

Keep the directory names exactly as above. The summary script reads the rank from them.

## 4. GPUs

The launcher takes the first four devices of `CUDA_VISIBLE_DEVICES`. Under Slurm that variable is
already set. Without Slurm, set it yourself:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 5. Numerical validation (about 10 minutes, must pass before any timing)

This compares each vLLM deployment against a single-GPU Hugging Face reference on the same prompts.

```bash
(
set -e
export NCCL_P2P_DISABLE=1 NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1
export CUDA_HOME="$BASIS_ARTIFACTS/cuda/12.9" MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 PYTHONPATH="$PWD"
export PATH="$CUDA_HOME/bin:$BASIS_ARTIFACTS/envs/vllm/basis/bin:$PATH"
export LD_LIBRARY_PATH="$BASIS_ARTIFACTS/envs/vllm/basis/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
V=benchmarks/system/validate_palu_g4_vllm.py
OUT=results/rebuttal/v2q8_palu_g4_tp4/validation/tp4_l40s
W="$BASIS_ARTIFACTS/calibration/llama31-8b-instruct-128k/c4-48x128k/windows.safetensors"
F="$BASIS_ARTIFACTS/checkpoints/palu_g4/L31-8B-I-PALUG4-R96"
python $V reference --variant dense --windows "$W" --output $OUT/reference_dense.json
python $V reference --variant palu --factors "$F" --windows "$W" --output $OUT/reference_palu_r96.json
python $V serve --arm dense --tp 4 --windows "$W" --output $OUT/vllm_dense.json
python $V serve --arm palu_g4 --tp 4 --factors "$F" --windows "$W" --output $OUT/vllm_palu_r96.json
python $V serve --arm palu_g4_sharded --tp 4 --factors "$F" --windows "$W" --output $OUT/vllm_palu_sharded_r96.json
python $V compare --reference $OUT/reference_dense.json --served $OUT/vllm_dense.json --output $OUT/compare_dense.json
python $V compare --reference $OUT/reference_palu_r96.json --served $OUT/vllm_palu_r96.json --output $OUT/compare_palu_r96.json
python $V compare --reference $OUT/reference_palu_r96.json --served $OUT/vllm_palu_sharded_r96.json --output $OUT/compare_palu_sharded_r96.json
) 2>&1 | tee results/rebuttal/v2q8_palu_g4_tp4/logs/validate_tp4_l40s.log
```

Pass criterion for all three `compare_*.json` files: `teacher_forced_top1_agreement >= 0.97` and
`teacher_forced_logprob_mean_abs_difference < 0.05`. Values on the A5000 run:

| Comparison | Top-1 agreement | Mean abs log-prob difference |
|---|---|---|
| dense vLLM vs dense reference | 1.000 | 0.018 |
| PaLU-A vLLM vs PaLU reference | 1.000 | 0.021 |
| PaLU-B vLLM vs PaLU reference | 0.992 | 0.020 |

The dense row is the ordinary Hugging Face versus vLLM BF16 gap; the PaLU rows must sit in the same
range. Greedy continuation agreement is reported too but is not a criterion, because one early
near-tie flips the rest of the sequence. If a row fails, stop and report; do not run the grid.

## 6. Smoke (7 short cells, about 30 minutes)

```bash
bash benchmarks/system/run_palu_g4_grid.sh smoke eager 2>&1 | tee results/rebuttal/v2q8_palu_g4_tp4/logs/smoke_l40s.log
```

Every cell must print `exit=0 "status": "complete"`. The first cell compiles kernels and prints a
60-second shared-memory wait message; that is expected.

## 7. Formal grid (estimated 10 to 12 hours in total)

Run the two passes one after the other, never at the same time, inside `tmux` or with `nohup`.

```bash
L=results/rebuttal/v2q8_palu_g4_tp4/logs
bash benchmarks/system/run_palu_g4_grid.sh formal eager > $L/formal_eager_l40s.log 2>&1
bash benchmarks/system/run_palu_g4_grid.sh formal full dense basis_v96 basis_v64 palu_g4_r96 palu_g4_r64 > $L/formal_full_l40s.log 2>&1
```

- Progress: each cell prints a `=== <arm> b<batch> <experiment>` line and then its exit status
  into the log above. The per-cell vLLM log is next to the cell directory, as `<experiment>.log`.
- Output: `results/rebuttal/v2q8_palu_g4_tp4/formal/{eager,full}/<arm>/p4096_b<batch>/<experiment>/`.
- Resume: rerun the same command. Cells whose `summary.json` is complete are skipped. If a cell
  failed, rerun with the same settings; keep the failed cell's log and mention it in the report.
- Batch 128 has not been run anywhere yet. If a batch-128 cell runs out of memory, the grid
  continues with the next cell. Report which cells failed; do not lower the batch or change
  `gpu_memory_utilization`.

## 8. Profiles (optional, about 20 minutes)

These split the decode step into kernel time and communication time. They are optional because, at
the time of writing, the profile experiment had not yet been run for the PaLU arms.

```bash
C="$BASIS_ARTIFACTS/checkpoints"; P=results/rebuttal/v2q8_palu_g4_tp4/formal/profile_eager
for n in dense:dense:$C/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix \
         palu_g4_r96:palu_g4:$C/palu_g4/L31-8B-I-PALUG4-R96 \
         palu_g4_sharded_r96:palu_g4_sharded:$C/palu_g4/L31-8B-I-PALUG4-R96; do
  IFS=: read -r name arm factors <<< "$n"; out=$P/$name/p4096_b32; mkdir -p "$(dirname "$out")"
  BENCH_FACTORS=$factors bash benchmarks/system/launch_vllm_bench.sh 4 "$out" "$arm" 4096 32 profile --graph eager > "$out.log" 2>&1
  echo "$name exit=$?"
done
```

## 9. Summaries

```bash
python3 benchmarks/system/summarize_palu_g4_tp4.py --root results/rebuttal/v2q8_palu_g4_tp4/formal/eager
python3 benchmarks/system/summarize_palu_g4_tp4.py --root results/rebuttal/v2q8_palu_g4_tp4/formal/full
```

Each writes `SUMMARY.md` and `cells.csv` into its root.

## 10. What to send back

1. The whole `results/rebuttal/v2q8_palu_g4_tp4/` tree from that machine: `formal/`, `validation/tp4_l40s/`
   and `logs/`. Upload it to the Hugging Face repo under `system_benchmarks/v2q8_palu_g4_tp4/l40s/`.
2. The two `SUMMARY.md` files, the three validation rows, the `git log -1` line, the GPU check from
   the Requirements section, and the list of any failed cells.

## Sanity checks against the A5000 run

Memory layout does not depend on the GPU, so these must match exactly (column "KV B/token/rank"):

| Arm | KV bytes per token per rank | Mean Value width per rank |
|---|---|---|
| `dense` | 32768 | 256 |
| `basis_v96` | 28672 | 192 |
| `basis_v64` | 24576 | 128 |
| `palu_g4_r96` | 40960 | 384 |
| `palu_g4_r64` | 32896 | 258 |
| `palu_g4_sharded_r96` | 28672 | 192 |
| `palu_g4_sharded_r64` | 24640 | 129 |

Timing will differ on L40S, but the ordering should hold. A5000, eager, decode step p50 in ms:

| Arm | batch 32 | batch 64 |
|---|---|---|
| `dense` | 37.15 | 55.38 |
| `basis_v96` | 32.77 | 49.18 |
| `basis_v64` | 31.01 | 46.92 |
| `palu_g4_r96` | 41.23 | 61.96 |
| `palu_g4_r64` | 38.62 | 59.47 |
| `palu_g4_sharded_r96` | 122.18 (superseded) | 198.18 |
| `palu_g4_sharded_r64` | 118.76 (superseded) | 193.65 |

The two superseded numbers were measured before per-layer device synchronizations were removed from
the sharded decode path. The batch-64 numbers and the code in this repository are after that fix.

## Known caveats

- BasisKV uses the uniform V96 and V64 factors here, not the per-layer allocated factors behind the
  paper's Table 1 quality numbers.
- The sharded arm cannot use CUDA Graphs, so it appears only in the eager pass. Compare it with the
  eager numbers of the other arms.
- vLLM pre-allocates its KV pool up to `gpu_memory_utilization=0.8`, so peak allocated memory says
  little. Use bytes per token per rank and the pool capacity in tokens from the summaries.
- A performance advisory about `OMP_NUM_THREADS=2` and a forced kill of the engine core at process
  teardown are both harmless.
