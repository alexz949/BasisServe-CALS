# Qwen3-8B TP8 dense / C1 benchmark

> **Historical implementation baseline.** These measurements are valid for the
> first-version unified Triton DiffKV implementation used by this run. They were
> collected before the SM89 QK128/V64 prefill specialization and therefore do
> not represent the optimized C1 performance ceiling. Keep them as the
> pre-optimization baseline and rerun both arms before using a final main-table
> comparison. See `docs/c1_benchmark_archive.md`.

Environment: `basis`, eight NVIDIA L40S (PCIe), PyTorch `2.13.0+cu130`, vLLM `0.29.0`.

Model: Qwen3-8B-Base BF16; C1 uses R64-S6 factors. Fixed cohorts of 4096 prompt tokens and exactly 128 output tokens per request. One full warmup and three measured runs per batch; table entries are medians over runs. Both arms use chunked prefill (8192-token budget), no prefix caching, synchronous scheduling, FULL_DECODE_ONLY CUDA Graphs, and compilation mode NONE. This is the matched first-version serving configuration, not a claim of optimal production vLLM tuning. Dense uses FlashAttention 2; C1 uses native Triton DiffKV.

| Batch | Dense E2E s | C1 E2E s | E2E speedup | Dense output tok/s | C1 output tok/s | Dense TPOT ms | C1 TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.106 | 1.091 | 1.013x | 115.7 | 117.3 | 6.561 | 6.985 |
| 2 | 1.432 | 1.366 | 1.048x | 178.8 | 187.3 | 6.906 | 7.485 |
| 4 | 2.070 | 1.835 | 1.128x | 247.4 | 278.9 | 9.643 | 9.416 |
| 8 | 3.260 | 2.694 | 1.210x | 314.1 | 380.1 | 13.519 | 12.046 |
| 16 | 5.763 | 4.554 | 1.266x | 355.3 | 449.7 | 23.792 | 19.582 |
| 32 | 10.796 | 8.337 | 1.295x | 379.4 | 491.3 | 45.205 | 35.668 |
| 64 | 20.880 | 15.821 | 1.320x | 392.3 | 517.8 | 88.001 | 67.192 |
| 128 | 41.128 | 30.351 | 1.355x | 398.4 | 539.8 | 171.495 | 125.893 |
| 256 | 83.668 | 59.108 | 1.415x | 391.6 | 554.4 | 340.947 | 235.609 |

E2E throughput includes prefill. TPOT is computed separately for each request as `(last_token_ts - first_token_ts) / 127`, then averaged over requests. It includes scheduler interleaving and is not isolated decode-kernel latency. TTFT uses engine queue-to-first-token timestamps and includes admission-barrier wait. Raw request timestamps, output token IDs, TTFT, run ranges, and worker statistics are retained in the JSON/CSV files. GPU peak allocation is cumulative and includes vLLM's preallocated KV pool; it is not live per-request KV usage.

## Rank-zero GPU profiles

Profiles are separate full-cohort runs after all timed measurements. They include prefill and decode. Kernel durations are summed, not wall time; profiling overhead can affect communication wait. Categories are inferred from kernel names, so consult raw traces for attribution.

| Profile | AllReduce % | AllGather % | GEMM/GEMV % | Attention % | Other % |
| --- | ---: | ---: | ---: | ---: | ---: |
| c1_b1.kernels | 22.9 | 9.8 | 51.3 | 9.0 | 7.0 |
| c1_b256.kernels | 52.5 | 14.3 | 19.5 | 7.0 | 6.7 |
| c1_b32.kernels | 49.6 | 13.8 | 22.4 | 7.4 | 6.7 |
| dense_b1.kernels | 43.9 | 0.7 | 41.5 | 6.6 | 7.3 |
| dense_b256.kernels | 72.7 | 0.6 | 11.2 | 11.1 | 4.4 |
| dense_b32.kernels | 74.6 | 0.6 | 14.2 | 5.7 | 4.9 |

### Batch-one decode graph attribution

The following values use the 127 post-prefill graph replays only. Decoder attribution follows the per-layer graph order: after AllGather for C1, before the attention-output AllReduce for dense. All 36 layer boundaries are checked per replay. Values are mean GPU kernel microseconds per layer, not isolated operator benchmarks or unprofiled wall latency.

| Arm | Principal O/decoder kernel us | Output collective us | Attention kernels us | Pre-AllGather copy us |
| --- | ---: | ---: | ---: | ---: |
| c1 | 25.37 | 12.73 | 17.07 | 0.90 |
| dense | 7.91 | 22.98 | 13.34 | 0.00 |

The first-version path is functional but is not established as optimal. Retain one decoder GEMM. Prioritize realistic full-model/cold-cache small-batch decoder layout and GEMM/GEMV selection, then paged compact-attention tuning. The C1 replicated decoder is [2048,4096] per rank versus a dense local O projection of [512,4096]: four times the BF16 weight bytes and matrix-product work per rank. Repeated resident-weight microbenchmarks do not reproduce the full model's cache working set. Cold-weight traffic is a hypothesis to test, not a measured DRAM-bandwidth diagnosis. Source-local copy fusion is a lower priority at batch one given the measured sub-microsecond copy.

Warnings: native custom AllReduce variants are unsupported for eight PCIe-only GPUs, so vLLM uses PyNccl. C1's first long-prefill DiffKV JIT compilation occurred during warmup. vLLM process teardown can emit forced EngineCore cleanup and Python shared-memory/semaphore resource-tracker warnings; retain logs. Completed runs validate exact output lengths, non-corrupted request status, and all 36 C1 V layers plus graph-capture counters on every rank.

## Reproduction

Use the `basis` environment with `CUDA_HOME=/usr/local/cuda`, `OMP_NUM_THREADS=1`, and `VLLM_WORKER_MULTIPROC_METHOD=spawn`. No Slurm is available; the runs execute sequentially on all eight local GPUs.

```bash
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_qwen3_8b_c1.py --arm dense --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir /workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-8b/c1/factor-banks/R64-S6 --output-dir results/vllm_tp8 --batch-sizes 1 2 4 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --profile-batches 1 32 256
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_qwen3_8b_c1.py --arm c1 --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir /workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-8b/c1/factor-banks/R64-S6 --output-dir results/vllm_tp8 --batch-sizes 1 2 4 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --profile-batches 1 32 256
/workspace/miniforge3/envs/basis/bin/python evaluation/summarize_vllm_qwen3_8b_c1.py
```

C1 manifest SHA256: `66e38669d248b0013e44f61fadec9cefb3d8180fb0b532675f2a6ec49ea0b510`.

Model startup, JIT compilation, profiling, and trace export are excluded from timed measurements. This is a synthetic fixed-length throughput test, not a quality evaluation or an online arrival-rate benchmark.
