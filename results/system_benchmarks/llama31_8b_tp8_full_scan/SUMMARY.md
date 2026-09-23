# Frozen TP8 Full-Scan Decode Grid

Conda environment: `basis`; direct execution on 8 x NVIDIA L40S.

## Validation

```json
{
  "trials": 144,
  "successful_trials": 132,
  "failed_trials": 12,
  "validated_rank_results": 1056,
  "frozen_source_files_unchanged": 1629,
  "source_verification": "direct byte comparison; no SHA256",
  "routing_mode": "full_scan_b16r16_persistent_slots",
  "conda_environment": "basis",
  "model": "/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
  "software": {
    "pytorch": "2.13.0+cu130",
    "cuda": "13.0",
    "nccl": "[2, 29, 7]"
  }
}
```

Every successful trial has 8 complete rank files, identical generated tokens and synchronized step arrays across ranks,
128 measured steps after 16 conditioning steps, and 145 generated tokens per sequence including the prefill prediction.
All frozen source files were compared directly against the pre-run archive. No SHA256 was computed.

## Three-Cohort Medians

A row receives median metrics only when all three cohorts completed. Speed ratios compare median mean-step latencies
and are omitted if either arm lacks a complete three-cohort result. P95 is the median of per-cohort P95 values.

| Prompt | Batch | Arm | Status | Mean ms | P50 ms | P95 ms | tokens/s | vs Dense | vs ALS | Prefill GPU GiB | Resident GPU GiB | Host K GiB | Slot hit |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | dense | complete | 29.399 | 29.141 | 31.915 | 34.278 | 1.000 | 0.695 | 2.959 | 2.801 | 0.000 | - |
| 4096 | 1 | als_full | complete | 20.439 | 19.970 | 22.915 | 49.462 | 1.438 | 1.000 | 3.898 | 3.740 | 0.000 | - |
| 4096 | 1 | basis_joint | complete | 24.205 | 23.729 | 27.321 | 41.523 | 1.215 | 0.844 | 3.883 | 3.725 | 0.259 | 0.828 |
| 4096 | 8 | dense | complete | 31.066 | 30.672 | 33.754 | 258.327 | 1.000 | 0.668 | 4.513 | 3.260 | 0.000 | - |
| 4096 | 8 | als_full | complete | 20.745 | 20.424 | 23.696 | 387.007 | 1.498 | 1.000 | 5.472 | 4.220 | 0.000 | - |
| 4096 | 8 | basis_joint | complete | 24.639 | 24.123 | 27.764 | 326.164 | 1.261 | 0.842 | 5.398 | 4.146 | 2.070 | 0.839 |
| 4096 | 32 | dense | complete | 31.076 | 30.879 | 32.782 | 1031.122 | 1.000 | 0.670 | 9.896 | 4.892 | 0.000 | - |
| 4096 | 32 | als_full | complete | 20.824 | 20.830 | 21.625 | 1538.823 | 1.492 | 1.000 | 10.907 | 5.904 | 0.000 | - |
| 4096 | 32 | basis_joint | complete | 24.696 | 24.643 | 25.671 | 1297.725 | 1.258 | 0.843 | 10.551 | 5.547 | 8.281 | 0.832 |
| 4096 | 128 | dense | complete | 39.326 | 39.304 | 39.762 | 3255.773 | 1.000 | 0.859 | 31.158 | 11.151 | 0.000 | - |
| 4096 | 128 | als_full | complete | 33.776 | 33.778 | 33.982 | 3790.768 | 1.164 | 1.000 | 32.540 | 12.533 | 0.000 | - |
| 4096 | 128 | basis_joint | complete | 51.364 | 51.348 | 52.304 | 2492.467 | 0.766 | 0.658 | 31.207 | 11.200 | 33.125 | 0.829 |
| 16384 | 1 | dense | complete | 29.730 | 29.274 | 32.305 | 33.717 | 1.000 | 0.848 | 3.433 | 2.988 | 0.000 | - |
| 16384 | 1 | als_full | complete | 25.220 | 24.974 | 26.768 | 39.650 | 1.179 | 1.000 | 4.349 | 3.904 | 0.000 | - |
| 16384 | 1 | basis_joint | complete | 24.482 | 23.824 | 27.978 | 41.044 | 1.214 | 1.030 | 4.257 | 3.812 | 1.009 | 0.788 |
| 16384 | 4 | dense | complete | 31.172 | 30.874 | 33.334 | 128.499 | 1.000 | 0.848 | 5.506 | 3.748 | 0.000 | - |
| 16384 | 4 | als_full | complete | 26.430 | 26.381 | 26.783 | 151.353 | 1.179 | 1.000 | 6.359 | 4.601 | 0.000 | - |
| 16384 | 4 | basis_joint | complete | 24.292 | 24.053 | 26.142 | 165.469 | 1.283 | 1.088 | 6.004 | 4.245 | 4.035 | 0.708 |
| 16384 | 8 | dense | complete | 30.676 | 30.616 | 32.192 | 260.955 | 1.000 | 0.863 | 8.270 | 4.761 | 0.000 | - |
| 16384 | 8 | als_full | complete | 26.462 | 26.448 | 26.752 | 302.346 | 1.159 | 1.000 | 9.043 | 5.534 | 0.000 | - |
| 16384 | 8 | basis_joint | complete | 24.211 | 24.078 | 25.824 | 331.662 | 1.267 | 1.093 | 8.320 | 4.811 | 8.070 | 0.709 |
| 16384 | 16 | dense | complete | 32.096 | 32.035 | 32.753 | 498.585 | 1.000 | 0.871 | 13.798 | 6.788 | 0.000 | - |
| 16384 | 16 | als_full | complete | 27.945 | 27.884 | 28.135 | 572.668 | 1.149 | 1.000 | 14.411 | 7.400 | 0.000 | - |
| 16384 | 16 | basis_joint | complete | 24.517 | 24.233 | 26.464 | 653.293 | 1.309 | 1.140 | 12.981 | 5.971 | 16.141 | 0.731 |
| 65536 | 1 | dense | complete | 85.649 | 85.661 | 86.129 | 11.675 | 1.000 | 0.863 | 5.340 | 3.739 | 0.000 | - |
| 65536 | 1 | als_full | complete | 73.903 | 73.906 | 74.297 | 13.531 | 1.159 | 1.000 | 6.172 | 4.570 | 0.000 | - |
| 65536 | 1 | basis_joint | complete | 23.955 | 23.794 | 25.777 | 41.865 | 3.575 | 3.085 | 5.766 | 4.165 | 4.009 | 0.572 |
| 65536 | 4 | dense | complete | 87.011 | 87.068 | 87.338 | 45.971 | 1.000 | 0.868 | 13.058 | 6.749 | 0.000 | - |
| 65536 | 4 | als_full | complete | 75.549 | 75.534 | 75.927 | 52.946 | 1.152 | 1.000 | 13.554 | 7.239 | 0.000 | - |
| 65536 | 4 | basis_joint | complete | 26.759 | 26.641 | 27.547 | 149.969 | 3.252 | 2.823 | 11.936 | 5.623 | 16.035 | 0.651 |
| 65536 | 8 | dense | complete | 87.412 | 87.461 | 87.669 | 91.522 | 1.000 | 0.864 | 23.350 | 10.764 | 0.000 | - |
| 65536 | 8 | als_full | complete | 75.546 | 75.548 | 75.749 | 105.897 | 1.157 | 1.000 | 23.393 | 10.799 | 0.000 | - |
| 65536 | 8 | basis_joint | complete | 23.846 | 23.834 | 24.338 | 336.555 | 3.666 | 3.168 | 20.099 | 7.505 | 32.070 | 0.656 |
| 65536 | 16 | dense | gpu_oom_prefill | - | - | - | - | - | - | - | - | - | - |
| 65536 | 16 | als_full | complete | 78.461 | 78.452 | 78.633 | 203.931 | - | 1.000 | 43.073 | 17.918 | 0.000 | - |
| 65536 | 16 | basis_joint | complete | 30.597 | 30.592 | 31.090 | 523.013 | - | 2.564 | 36.469 | 11.314 | 64.141 | 0.673 |
| 130048 | 1 | dense | complete | 160.567 | 160.583 | 161.203 | 6.228 | 1.000 | 0.860 | 7.877 | 4.736 | 0.000 | - |
| 130048 | 1 | als_full | complete | 138.067 | 138.094 | 138.371 | 7.243 | 1.163 | 1.000 | 8.608 | 5.465 | 0.000 | - |
| 130048 | 1 | basis_joint | complete | 23.869 | 23.599 | 25.590 | 42.154 | 6.727 | 5.784 | 7.773 | 4.632 | 7.946 | 0.635 |
| 130048 | 4 | dense | complete | 162.187 | 162.231 | 162.581 | 24.663 | 1.000 | 0.862 | 23.115 | 10.742 | 0.000 | - |
| 130048 | 4 | als_full | complete | 139.801 | 139.780 | 140.145 | 28.612 | 1.160 | 1.000 | 23.147 | 10.773 | 0.000 | - |
| 130048 | 4 | basis_joint | complete | 23.977 | 23.888 | 24.657 | 167.595 | 6.764 | 5.831 | 19.791 | 7.417 | 31.785 | 0.661 |
| 130048 | 8 | dense | complete | 162.620 | 162.669 | 162.928 | 49.194 | 1.000 | 0.861 | 43.321 | 18.643 | 0.000 | - |
| 130048 | 8 | als_full | complete | 140.004 | 140.012 | 140.198 | 57.141 | 1.162 | 1.000 | 42.439 | 17.756 | 0.000 | - |
| 130048 | 8 | basis_joint | complete | 27.850 | 27.812 | 28.262 | 287.274 | 5.839 | 5.027 | 35.768 | 11.085 | 63.570 | 0.687 |
| 130048 | 16 | dense | gpu_oom_prefill | - | - | - | - | - | - | - | - | - | - |
| 130048 | 16 | als_full | gpu_oom_prefill | - | - | - | - | - | - | - | - | - | - |
| 130048 | 16 | basis_joint | gpu_oom_prefill | - | - | - | - | - | - | - | - | - | - |

## Failures

- P=65536, B=16, dense, cohort 0: gpu_oom_prefill; `launcher_dense_p65536_b16_c0.log`.
- P=65536, B=16, dense, cohort 1: gpu_oom_prefill; `launcher_dense_p65536_b16_c1.log`.
- P=65536, B=16, dense, cohort 2: gpu_oom_prefill; `launcher_dense_p65536_b16_c2.log`.
- P=130048, B=16, dense, cohort 0: gpu_oom_prefill; `launcher_dense_p130048_b16_c0.log`.
- P=130048, B=16, als_full, cohort 0: gpu_oom_prefill; `launcher_als_full_p130048_b16_c0.log`.
- P=130048, B=16, basis_joint, cohort 0: gpu_oom_prefill; `launcher_basis_joint_p130048_b16_c0.log`.
- P=130048, B=16, dense, cohort 1: gpu_oom_prefill; `launcher_dense_p130048_b16_c1.log`.
- P=130048, B=16, als_full, cohort 1: gpu_oom_prefill; `launcher_als_full_p130048_b16_c1.log`.
- P=130048, B=16, basis_joint, cohort 1: gpu_oom_prefill; `launcher_basis_joint_p130048_b16_c1.log`.
- P=130048, B=16, dense, cohort 2: gpu_oom_prefill; `launcher_dense_p130048_b16_c2.log`.
- P=130048, B=16, als_full, cohort 2: gpu_oom_prefill; `launcher_als_full_p130048_b16_c2.log`.
- P=130048, B=16, basis_joint, cohort 2: gpu_oom_prefill; `launcher_basis_joint_p130048_b16_c2.log`.

## Method and Limits

- TP=8, DP=1, PP=1; one rank per L40S, no NVLink. Captured topology: `freeze/nvidia_topology.txt`.
- BF16, TF32 disabled, eager execution. MLP and Dense were not optimized for this rerun.
- Dense uses this repository's DenseTP8Attention / gpu_paged_attention backend with HF TP8. Ratios are against this matched control, not tuned vLLM, SGLang, or TensorRT-LLM deployments.
- Basis-joint: V96, all-history B16R16 Page32 routing, 62 selected pages plus recent64, 2048 support slots. No coarse screening or 512-candidate limit.
- Dense and ALS-full cache on GPU; Basis keeps historical exact K in pinned host memory with persistent GPU K slots.
- GPU memory is max-rank PyTorch allocated memory; host K is summed allocation capacity, not measured host peak RSS.
- Latency covers the full-model decode step, including MLP and collectives. It is not divided by batch size. Throughput uses maximum-rank wall time.
- Slot hit rate describes the final measured step only. Profiling instrumentation was disabled.
- Container restrictions prevent strict NUMA host-memory binding (`set_mempolicy: Operation not permitted`); CPU affinity is applied.
- NCCL emits current-device inference warnings for barriers. Any failures remain in the manifest and logs.
- Prefill OOM is not a decode-cache-only capacity limit. This is not a capacity search, TTFT/request benchmark, or task-quality evaluation.
- Cross-rank agreement and finite logits do not establish task accuracy or equivalence to Dense.
- Historical two-stage measurements were preserved and are not combined with these full-scan results.

## Frozen Source and Command

Base commit: `bae46c1409d3fa00030bab88df8b359e20b886bf` plus the working-tree source in `freeze/source.tar.gz`.
`freeze/manifest.json` lists archived files, inputs, environment and command; `freeze/source.patch` records tracked changes.
The archive, rather than the base commit alone, identifies the tested implementation.

```bash
CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_llama31_8b_tp8_decode_grid.py --arms dense als_full basis_joint --contexts 4096 16384 65536 130048 --cohorts 0 1 2 --conditioning-steps 16 --measure-steps 128 --tag full_scan_decode --output-root results/system_benchmarks/llama31_8b_tp8_full_scan
```

Summary command: `conda run --no-capture-output -n basis python benchmarks/system/summarize_tp8_full_scan.py`.
Exact per-trial torchrun commands, timestamps and return codes are in `decode_grid_trials.json`.
Top-level progress is in `run.log`; per-trial launcher and rank logs remain alongside raw JSON results.
