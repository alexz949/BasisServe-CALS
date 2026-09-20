# Qwen3-32B TP8 dense / C1 benchmark

> **Historical implementation baseline.** These measurements are valid for the
> first-version unified Triton DiffKV implementation used by this run. They were
> collected before the SM89 QK128/V64 prefill specialization and therefore do
> not represent the optimized C1 performance ceiling. Keep them as the
> pre-optimization baseline and rerun both arms before using a final main-table
> comparison. See `docs/c1_benchmark_archive.md`.

Environment: `basis`, eight NVIDIA L40S (PCIe), PyTorch `2.13.0+cu130`, vLLM `0.29.0`.

Model: Qwen3-32B BF16; C1 uses R64-S6 factors. Fixed cohorts of 4096 prompt tokens and exactly 128 output tokens per request. 1 full warmup(s) and 3 measured runs per batch; table entries are medians over runs. Both arms use chunked prefill (8192-token budget), no prefix caching, synchronous scheduling, FULL_DECODE_ONLY CUDA Graphs, and compilation mode NONE. This is the matched first-version serving configuration, not a claim of optimal production vLLM tuning. Dense uses FlashAttention 2; C1 uses native Triton DiffKV.

| Batch | Dense E2E s | C1 E2E s | E2E speedup | Dense output tok/s | C1 output tok/s | Dense TPOT ms | C1 TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.124 | 3.297 | 0.948x | 41.0 | 38.8 | 19.179 | 21.323 |
| 2 | 3.944 | 4.001 | 0.986x | 64.9 | 64.0 | 20.075 | 22.082 |
| 4 | 5.489 | 5.311 | 1.033x | 93.3 | 96.4 | 26.461 | 27.362 |
| 8 | 8.450 | 7.818 | 1.081x | 121.2 | 131.0 | 35.831 | 35.081 |
| 16 | 14.600 | 13.077 | 1.116x | 140.3 | 156.6 | 60.321 | 55.797 |
| 32 | 26.946 | 23.604 | 1.142x | 152.0 | 173.5 | 111.357 | 98.973 |
| 64 | 51.784 | 44.385 | 1.167x | 158.2 | 184.6 | 215.327 | 184.362 |
| 128 | 102.756 | 85.179 | 1.206x | 159.4 | 192.3 | 424.362 | 345.860 |
| 256 | 215.228 | 166.899 | 1.290x | 152.2 | 196.3 | 845.646 | 661.435 |

## Scheduler preemptions

Counts are summed over measured runs only, excluding warmup and profiling. Nonzero values mark capacity-affected results that must not be attributed solely to communication/kernel changes. Zero preemptions do not by themselves rule out cache-capacity-limited admission; also inspect KV utilization logs.

| Batch | Dense preemptions | C1 preemptions |
| ---: | ---: | ---: |
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 4 | 0 | 0 |
| 8 | 0 | 0 |
| 16 | 0 | 0 |
| 32 | 0 | 0 |
| 64 | 0 | 0 |
| 128 | 0 | 0 |
| 256 | 3 | 0 |

E2E throughput includes prefill. TPOT is computed separately for each request as `(last_token_ts - first_token_ts) / 127`, then averaged over requests. It includes scheduler interleaving and is not isolated decode-kernel latency. Batch denotes the number of requests submitted together, not a constant GPU execution batch: continuous batching and chunked prefill remain active. TTFT uses engine queue-to-first-token timestamps and includes admission-barrier wait. Raw request timestamps, output token IDs, TTFT, run ranges, and worker statistics are retained in the JSON/CSV files. GPU peak allocation is cumulative and includes vLLM's preallocated KV pool; it is not live per-request KV usage.

## Rank-zero GPU profiles

Profiles are separate full-cohort runs after all timed measurements. They include prefill and decode. Kernel durations are summed, not wall time; profiling overhead can affect communication wait. Categories are inferred from kernel names, so consult raw traces for attribution.

| Profile | AllReduce % | AllGather % | GEMM/GEMV % | Attention % | Other % |
| --- | ---: | ---: | ---: | ---: | ---: |
| c1_b1.kernels | 18.2 | 6.9 | 64.6 | 5.8 | 4.4 |
| c1_b256.kernels | 41.0 | 16.8 | 30.1 | 6.1 | 6.1 |
| c1_b32.kernels | 38.3 | 15.8 | 34.0 | 6.1 | 5.8 |
| dense_b1.kernels | 37.3 | 0.2 | 53.8 | 4.1 | 4.6 |
| dense_b256.kernels | 62.8 | 0.2 | 19.2 | 13.2 | 4.6 |
| dense_b32.kernels | 65.7 | 0.2 | 24.1 | 5.0 | 4.9 |

### Batch-one decode graph attribution

The following values use post-prefill graph replays only. Decoder attribution follows the per-layer graph order: after AllGather for C1, before the attention-output AllReduce for dense. All 64 layer boundaries are checked per replay. Values are mean GPU kernel microseconds per layer, not isolated operator benchmarks or unprofiled wall latency.

| Arm | Principal O/decoder kernel us | Output collective us | Attention kernels us | Pre-AllGather copy us |
| --- | ---: | ---: | ---: | ---: |
| c1 | 64.78 | 13.43 | 19.15 | 0.91 |
| dense | 16.72 | 37.26 | 13.44 | 0.00 |

The first-version path is functional but is not established as optimal. Retain one decoder GEMM. Prioritize realistic full-model/cold-cache small-batch decoder layout and GEMM/GEMV selection, then paged compact-attention tuning. The C1 replicated decoder is [4096,5120] per rank versus a dense local O projection of [1024,5120]: four times the BF16 weight bytes and matrix-product work per rank. Repeated resident-weight microbenchmarks do not reproduce the full model's cache working set. Cold-weight traffic is a hypothesis to test, not a measured DRAM-bandwidth diagnosis. Compare the measured source-local copy cost with decoder and attention costs before prioritizing copy fusion.

Warnings: native custom AllReduce variants are unsupported for eight PCIe-only GPUs, so vLLM uses PyNccl. C1's first long-prefill DiffKV JIT compilation occurred during warmup. vLLM process teardown can emit forced EngineCore cleanup and Python shared-memory/semaphore resource-tracker warnings; retain logs. Completed runs validate exact output lengths, non-corrupted request status, and all 64 C1 V layers plus graph-capture counters on every rank.

## Reproduction

Use the `basis` environment with `CUDA_HOME=/usr/local/cuda`, `OMP_NUM_THREADS=1`, and `VLLM_WORKER_MULTIPROC_METHOD=spawn`. No Slurm is available; the runs execute sequentially on all eight local GPUs.

```bash
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_qwen3_8b_c1.py --arm dense --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 --factor-dir /workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-32b/c1/factor-banks/R64-S6 --output-dir results/vllm_32b_tp8 --batch-sizes 1 2 4 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --profile-batches 1 32 256
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_qwen3_8b_c1.py --arm c1 --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 --factor-dir /workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-32b/c1/factor-banks/R64-S6 --output-dir results/vllm_32b_tp8 --batch-sizes 1 2 4 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --profile-batches 1 32 256
/workspace/miniforge3/envs/basis/bin/python evaluation/summarize_vllm_qwen3_8b_c1.py --input-dir results/vllm_32b_tp8
```

C1 manifest SHA256: `26bafc0674362208b05d56b16f8b5a74a22756738cf355af39bf80b859195b3f`.

Model startup, JIT compilation, profiling, and trace export are excluded from timed measurements. This is a synthetic fixed-length throughput test, not a quality evaluation or an online arrival-rate benchmark.
