# Qwen3-8B-Base TP8 fixed-batch context-length sweep

> **Historical implementation baseline.** These measurements are valid for the
> first-version unified Triton DiffKV implementation used by this run. They were
> collected before the SM89 QK128/V64 prefill specialization and therefore do
> not represent the optimized C1 performance ceiling. In particular, the
> long-context regression diagnosed here motivated the new prefill kernel. Keep
> these data as the pre-optimization baseline and rerun both arms for the final
> comparison. See `docs/c1_benchmark_archive.md`.

Batch 32, 128 output tokens, native context only (YaRN disabled). Environment: `basis`, eight NVIDIA L40S (PCIe), PyTorch `2.13.0+cu130`, vLLM `0.29.0`. Each point has 1 full warmup and 3 measured runs; entries are medians.

Both arms use one engine with max model length 32768, chunked prefill with an 8192-token scheduler budget, no prefix caching, synchronous scheduling, compilation mode NONE, and FULL_DECODE_ONLY CUDA Graphs. Dense uses FlashAttention 2; C1 uses native Triton DiffKV, prepared NCCL, and one decoder GEMM.

| Prefill | Dense E2E s | C1 E2E s | E2E speedup | Dense TTFT ms | C1 TTFT ms | Dense TPOT ms | C1 TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2.620 | 2.184 | 1.200x | 823.06 | 605.63 | 14.068 | 12.353 |
| 1024 | 3.781 | 3.031 | 1.247x | 1408.88 | 1036.39 | 18.508 | 15.553 |
| 2048 | 6.084 | 4.756 | 1.279x | 2588.21 | 1916.11 | 27.226 | 22.095 |
| 4096 | 10.786 | 8.331 | 1.295x | 4976.90 | 3731.49 | 45.133 | 35.645 |
| 8192 | 20.382 | 15.798 | 1.290x | 9872.87 | 7558.36 | 81.413 | 63.620 |
| 16384 | 40.393 | 31.892 | 1.267x | 19832.14 | 15641.20 | 158.586 | 124.980 |
| 32640 | 83.388 | 69.355 | 1.202x | 40924.14 | 34359.92 | 325.601 | 267.645 |

All measured dense and C1 runs recorded zero scheduler preemptions. Batch means 32 requests submitted together, not a constant execution batch: continuous batching and chunked prefill remain active. E2E throughput includes prefill. TTFT includes admission-barrier wait; TPOT is per request `(last-first)/127` and includes scheduler interleaving, so it is not isolated decode-kernel latency.

## Rank-zero full-cohort profiles

Profiles are separate unmeasured runs after all timed points. Values are summed GPU kernel durations, not wall time; communication durations can include waiting and categories are inferred from kernel names.

| Profile | Total kernel s | AllReduce s | AllGather s | GEMM/GEMV s | Attention s | Other s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c1_s16384 | 31.409 | 14.915 | 3.896 | 5.839 | 4.824 | 1.936 |
| c1_s32640 | 68.270 | 29.289 | 7.577 | 11.046 | 16.583 | 3.775 |
| c1_s4096 | 8.129 | 4.034 | 1.118 | 1.827 | 0.604 | 0.546 |
| c1_s512 | 2.012 | 0.848 | 0.305 | 0.645 | 0.074 | 0.140 |
| dense_s16384 | 39.840 | 29.273 | 0.074 | 4.753 | 3.946 | 1.794 |
| dense_s32640 | 82.667 | 57.675 | 0.088 | 9.145 | 12.269 | 3.490 |
| dense_s4096 | 10.546 | 7.877 | 0.066 | 1.496 | 0.597 | 0.511 |
| dense_s512 | 2.452 | 1.621 | 0.064 | 0.534 | 0.098 | 0.134 |

The E2E speedup rises from short context to a maximum around 4K–8K, then declines as long-context prefill occupies a larger share of the fixed 128-token generation workload. This is a serving-path comparison, not an isolated attention-kernel test or a model-quality evaluation.

## Reproduction

Use the `basis` environment with `CUDA_HOME=/usr/local/cuda`, `OMP_NUM_THREADS=1`, and `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

```bash
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_qwen3_8b_c1_seqlen.py --arm dense --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir /workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-8b/c1/factor-banks/R64-S6 --output-dir results/vllm_8b_tp8_seqlen --batch-size 32 --prefill-lengths 512 1024 2048 4096 8192 16384 32640 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --gpu-memory-utilization 0.8 --profile-prefill-lengths 512 4096 16384 32640
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_qwen3_8b_c1_seqlen.py --arm c1 --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir /workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-8b/c1/factor-banks/R64-S6 --output-dir results/vllm_8b_tp8_seqlen --batch-size 32 --prefill-lengths 512 1024 2048 4096 8192 16384 32640 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --gpu-memory-utilization 0.8 --profile-prefill-lengths 512 4096 16384 32640
/workspace/miniforge3/envs/basis/bin/python evaluation/summarize_vllm_qwen3_8b_c1_seqlen.py --input-dir results/vllm_8b_tp8_seqlen
```

C1 factor manifest SHA256: `66e38669d248b0013e44f61fadec9cefb3d8180fb0b532675f2a6ec49ea0b510`.

Non-fatal runtime warnings about unavailable custom PCIe all-reduce backends, initial Triton JIT, profiler export waits, and vLLM process cleanup are retained in the arm logs.
