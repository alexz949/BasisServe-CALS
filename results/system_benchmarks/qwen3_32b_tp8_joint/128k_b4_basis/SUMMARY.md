# Qwen3-32B TP8 Joint V96: 130048/B4 supplement

## Setup

Environment: `basis`, 8 L40S GPUs, TP8, BF16. Full-scan B16R16 routing,
2048-token physical support, historical exact K on pinned host memory.
No two-stage routing, MLP changes, SHA256 checks, or generated-token equivalence checks.
This is a single-trial full-model steady-decode pilot, not vLLM or E2E request latency.

The primary run uses 16 conditioning forwards and 128 measured forwards.
All eight ranks completed successfully, without OOM. Dense comes from the
completed Flash SDPA rerun; it was not rerun concurrently with this supplement.

## Primary Results

| Metric | Flash Dense | Basis Joint V96 |
|---|---:|---:|
| Decode ms/step | 71.391 | 56.710 |
| Aggregate tokens/s | 56.09 | 70.57 |
| Prefill peak allocated GiB, maximum rank | 40.308 | 36.304 |
| Decode resident allocated GiB, maximum rank | 24.905 | 20.899 |
| Host persistent exact K GiB, sum of ranks | 0 | 63.570 |

Latency speedup: **1.259x**. Basis prefill wall time on rank 0 was 150.943 s;
this is not included in steady-decode latency. Rank 0's last-step aggregate
K-slot hit rate was 50.09%; this is not a whole-run average.

Raw Basis results: `smoke_basis_joint_p130048_b4_r0/rank{0..7}.json`.
Dense source: `../../tp8_dense_flash_rerun/qwen_p130048/b4/dense/`.
The combined comparison is updated in `../../tp8_dense_flash_rerun/SUMMARY.md`.

## Diagnostic Profile

A separate run used 16 conditioning forwards and 32 measured forwards with
`--profile-components`. All eight ranks completed. The instrumented total was
107.335 ms/step versus 56.710 ms/step in the primary run. This is substantial
perturbation; the runs also have different measurement lengths. Do not use
the profiled total for speedup, or treat its components as exact attribution
of the uninstrumented 56.710 ms.

The following are medians across ranks of CUDA-event intervals summed over
64 layers per step. Parent/child intervals overlap; rank medians need not add
up. Intervals can include launch gaps, synchronization, and rank waiting,
not just kernel execution or wire transfer time.

| Component | Instrumented ms/step |
|---|---:|
| Attention total, parent interval | 88.003 |
| QKV projection / RoPE / append | 7.824 |
| Full-scan router | 17.575 |
| Selection / packing / K-slot refresh | 6.416 |
| Selected exact attention | 7.897 |
| Output AllGather interval | 38.879 |
| Output decoder | 6.145 |
| MLP block | 15.595 |
| Pre-attention norm | 1.599 |
| Attention residual | 0.243 |

The full-scan router and output-collective path merit investigation. The
AllGather interval is not evidence of 38.879 ms of pure communication in the
primary run. A lower-perturbation measurement is needed to establish the
dominant normal-run bottleneck. These measurements do not establish that the
implementation is optimal.

Reducing selected attention to 2048 tokens does not reduce all model work by
64x: routing still scans the full history, missing K must be retrieved, and
projection, decoder, MLP, and TP communication remain. The current execution
path is an eager custom harness, not a CUDA-graph vLLM serving path.

Profile raw data and logs: `../128k_b4_basis_profile/`.

## Commands

Working directory: `/workspace/BasisServe-CALS-opt`. Each launcher saves
`basis_joint.log`, `trials.json` with the exact child command, and rank logs.

Primary:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_qwen3_32b_joint_smoke.py \
  --arms basis_joint --length 130048 --batch 4 \
  --conditioning-steps 16 --measure-steps 128 \
  --tokens /workspace/runs/qwen3-32b-densev-ruler30/calibration/windows.safetensors \
  --output results/system_benchmarks/qwen3_32b_tp8_joint/128k_b4_basis
```

Diagnostic:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_qwen3_32b_joint_smoke.py \
  --arms basis_joint --length 130048 --batch 4 \
  --conditioning-steps 16 --measure-steps 32 --profile-components \
  --tokens /workspace/runs/qwen3-32b-densev-ruler30/calibration/windows.safetensors \
  --output results/system_benchmarks/qwen3_32b_tp8_joint/128k_b4_basis_profile
```
