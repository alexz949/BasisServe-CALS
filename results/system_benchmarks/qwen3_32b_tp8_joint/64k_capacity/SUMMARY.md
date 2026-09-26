# Qwen3-32B TP8 Joint V96: 64K Capacity Pilot

IMPORTANT: Dense decode uses this repository's `gpu_paged_attention` CUDA kernel.
Earlier TP1 Dense measurements used Flash SDPA. These speedups are NOT measured
against FlashAttention/vLLM or an independently optimized Dense serving stack.
A matched Flash SDPA Dense control is needed before attributing the full speedup
to sparsity or comparing its magnitude with TP1.

Single trial per point. These are not three-cohort formal results.

Eight L40S GPUs, TP8/BF16, `basis` environment. 65536 prompt tokens,
16 conditioning forwards, 128 measured forwards. Fixed active batch,
custom Transformers execution, not vLLM or complete request timing.

All arms use the same first B rows of the Qwen C4 calibration bank.
Inputs are not a held-out quality evaluation. Matching 128K V96/B16R16
factors, static YaRN4, full-scan Page32, 62 pages plus recent64.
Basis offloads historical exact K to pinned host memory, retaining GPU
K slots and V96. Dense/ALS-full retain K/V on GPU. No two-stage routing.

## Measurements

| Batch | Arm | Status | Decode ms/step | Tokens/s | Prefill peak GiB/rank | Decode resident GiB/rank | Host K GiB total |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | dense | complete | 145.714 | 6.863 | 12.904 | 10.915 | 0.000 |
| 1 | als_full | complete | 146.367 | 6.832 | 17.205 | 15.215 | 0.000 |
| 1 | basis_joint | complete | 57.503 | 17.436 | 16.384 | 14.394 | 8.018 |
| 2 | dense | complete | 150.133 | 13.321 | 16.872 | 12.924 | 0.000 |
| 2 | als_full | complete | 148.300 | 13.486 | 20.945 | 16.996 | 0.000 |
| 2 | basis_joint | complete | 57.006 | 35.163 | 19.301 | 15.351 | 16.035 |
| 4 | dense | complete | 148.176 | 26.995 | 24.807 | 16.943 | 0.000 |
| 4 | als_full | complete | 149.169 | 26.815 | 28.426 | 20.558 | 0.000 |
| 4 | basis_joint | complete | 56.562 | 70.836 | 25.185 | 17.317 | 32.070 |
| 8 | dense | complete | 149.236 | 53.607 | 40.676 | 24.981 | 0.000 |
| 8 | als_full | gpu_oom | - | - | - | - | - |
| 8 | basis_joint | complete | 58.843 | 135.988 | 36.801 | 21.097 | 64.141 |
| 16 | dense | gpu_oom | - | - | - | - | - |
| 16 | als_full | not_attempted_after_oom | - | - | - | - | - |
| 16 | basis_joint | gpu_oom | - | - | - | - | - |

## Scope and Failures

Decode ms is the mean of per-step maximum-rank CUDA-event durations;
throughput uses maximum-rank wall time. GPU values are maximum-rank
PyTorch allocated memory, not total device usage. Host K is summed
allocation capacity, not peak host RSS.

- B8 als_full: GPU OOM; last recorded phase(s): prefill.
- B16 dense: GPU OOM; last recorded phase(s): prefill.
- B16 basis_joint: GPU OOM; last recorded phase(s): prefill.

Larger points skipped after an arm's first OOM were NOT tested.
Prefill OOM is not a decode-cache capacity limit. Successful runs check
finite logits and cross-rank token agreement, not cross-arm quality
equivalence. Host NUMA memory binding is unverified because the container
rejects set_mempolicy. No SHA256 checks. No global kernel optimality claim.

## Reproduction

Working directory: `/workspace/BasisServe-CALS-opt`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_qwen3_32b_joint_capacity.py --batches 2 4 8 16 --output results/system_benchmarks/qwen3_32b_tp8_joint/64k_capacity
```

The prior B1 results are included without rerunning them. Exact child commands,
statuses and timestamps are in `manifest.json`; per-arm logs and rank
JSONs are under `b<batch>/<arm>/`. See [B1 summary](../64k_b1/SUMMARY.md)
and [initial validation](../SMOKE_SUMMARY.md).
