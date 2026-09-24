# LRQK GPU-local TP1 Supplement

Llama-3.1-8B-Instruct, L40S GPU 0, TP1/B1, basis environment. Three fixed prompt cohorts.
Exact K/V and routing state remain on GPU. Rank 32, active 2048, lite 64, fitting and hit/miss logic unchanged.
CPU gather is replaced by GPU index_select. Upstream 1.5x allocation growth remains unchanged.
Same frozen request protocol: same-shape warmup, fresh request state, greedy fixed-length output.
Request includes prefill/build and N-1 decode calls; excludes model loading, tokenization and input transfer.
Construction is a separate synchronized profile nested inside prefill, not additive to request latency.
No post-prefill restoration is needed. This is a storage control, not a quality evaluation.

| Context | Output | Complete | Request s | Decode tail ms/token | Build profile s | Peak allocated GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 128 | 3/3 | 28.659 | 187.501 | 0.910 | 28.669 |
| 65536 | 128 | 0/3 | - | - | - | - |
| 130048 | 128 | 0/3 | - | - | - | - |
| 130048 | 32 | 0/3 | - | - | - | - |
| 130048 | 512 | 0/3 | - | - | - | - |

## Failures

- 65536 / 128 outputs / cohort 0: gpu_oom (same_shape_runtime_warmup).
- 130048 / 128 outputs / cohort 0: gpu_oom (same_shape_runtime_warmup).
- 130048 / 32 outputs / cohort 0: gpu_oom (same_shape_runtime_warmup).
- 130048 / 512 outputs / cohort 0: gpu_oom (same_shape_runtime_warmup).
- 65536 / 128 outputs / cohort 1: gpu_oom (same_shape_runtime_warmup).
- 130048 / 128 outputs / cohort 1: gpu_oom (same_shape_runtime_warmup).
- 130048 / 32 outputs / cohort 1: gpu_oom (same_shape_runtime_warmup).
- 130048 / 512 outputs / cohort 1: gpu_oom (same_shape_runtime_warmup).
- 65536 / 128 outputs / cohort 2: gpu_oom (same_shape_runtime_warmup).
- 130048 / 128 outputs / cohort 2: gpu_oom (same_shape_runtime_warmup).
- 130048 / 32 outputs / cohort 2: gpu_oom (same_shape_runtime_warmup).
- 130048 / 512 outputs / cohort 2: gpu_oom (same_shape_runtime_warmup).

GPU OOM is specific to this runtime/allocation policy, not proof of an algorithmic limit.
Original CPU-offload results are unchanged in ../tp1_offline/SUMMARY.md.
Raw logs retain graph-break/recompilation warnings. No claim of optimal LRQK implementation.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_tp1_lrqk_local.py
```

## Archived Artifacts

Raw JSON, failure logs, correctness checks and frozen supplement source are published in [the HF artifact directory](https://huggingface.co/alexz949/BasisServe-CALS/tree/main/system_benchmarks/tp1_lrqk_local).
Immutable raw-data revision: `86c03092c663dc6654584143130b5e5abf2fedaa`.
See [the original grid's artifact instructions](../tp1_offline/SUMMARY.md#archived-artifacts) for downloading both datasets and restoring shared dependencies, prompts and factors.
GitHub retains the benchmark code, summaries and small outcome/verification manifests.
