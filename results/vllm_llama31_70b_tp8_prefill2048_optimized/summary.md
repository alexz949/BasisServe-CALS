# Llama-3.1-70B TP8 dense / uniform V64 / uniform V96

Environment: `basis`, eight NVIDIA L40S PCIe GPUs; PyTorch `2.13.0+cu130`, vLLM `0.29.0`.

Every request has 2048 prompt tokens and exactly 128 generated tokens. Results are medians of 3 measured fixed-cohort runs after 1 full warmup run(s). V64 and V96 use matched uniform per-layer/per-KV-head checkpoints.

| Batch | Dense s | V64 s | V64 speedup | V96 s | V96 speedup | Dense TPOT ms | V64 TPOT ms | V96 TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.649 | 5.115 | 0.909x | 5.775 | 0.805x | 31.194 | 35.985 | 40.751 |
| 2 | 5.633 | 5.885 | 0.957x | 6.568 | 0.858x | 33.475 | 37.776 | 42.197 |
| 4 | 7.113 | 7.062 | 1.007x | 7.866 | 0.904x | 34.254 | 38.515 | 43.027 |
| 8 | 10.363 | 9.597 | 1.080x | 10.673 | 0.971x | 48.506 | 49.375 | 55.130 |
| 16 | 16.790 | 14.569 | 1.152x | 16.171 | 1.038x | 74.170 | 68.846 | 76.545 |
| 32 | 29.704 | 24.602 | 1.207x | 27.170 | 1.093x | 129.971 | 111.494 | 123.128 |
| 64 | 53.730 | 43.383 | 1.239x | 47.816 | 1.124x | 229.326 | 188.848 | 208.093 |
| 128 | 102.081 | 80.787 | 1.264x | 89.083 | 1.146x | 428.506 | 340.383 | 375.012 |

## Scheduler preemptions

| Batch | Dense | V64 | V96 |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 |
| 16 | 0 | 0 | 0 |
| 32 | 0 | 0 | 0 |
| 64 | 0 | 0 | 0 |
| 128 | 0 | 0 | 0 |

## Rank-zero GPU profiles

Profiles are separate full-cohort runs after timing. Percentages sum GPU kernel durations and include prefill and decode; communication durations can include profiler-inflated wait time.

| Profile | AllReduce % | AllGather % | GEMM/GEMV % | Attention % | Other % |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense_b1.kernels | 22.3 | 0.1 | 71.9 | 2.6 | 3.1 |
| dense_b128.kernels | 67.9 | 0.2 | 23.4 | 3.6 | 4.9 |
| dense_b32.kernels | 63.7 | 0.2 | 29.0 | 2.5 | 4.6 |
| v64_b1.kernels | 10.4 | 4.3 | 81.1 | 1.5 | 2.6 |
| v64_b128.kernels | 43.5 | 11.7 | 35.9 | 2.5 | 6.3 |
| v64_b32.kernels | 39.3 | 10.5 | 42.4 | 2.2 | 5.5 |
| v96_b1.kernels | 9.2 | 5.7 | 81.0 | 1.6 | 2.5 |
| v96_b128.kernels | 39.6 | 15.5 | 36.4 | 2.8 | 5.7 |
| v96_b32.kernels | 35.6 | 13.9 | 43.1 | 2.4 | 5.0 |

### Batch-one decode graph attribution

Values are mean GPU kernel microseconds per layer over verified 80-layer CUDA Graph replays. The pre-AllGather kernel is the direct segment reduction for the optimized V64/V96 decode path; no layout-copy kernel remains.

| Arm | Decoder/O-projection us | Output collective us | Attention us | Pre-AllGather reduction us |
| --- | ---: | ---: | ---: | ---: |
| dense | 25.33 | 24.58 | 10.66 | 0.00 |
| v64 | 98.24 | 13.57 | 6.92 | 1.63 |
| v96 | 147.05 | 21.17 | 8.21 | 1.95 |

V64/V96 keep dense K128 and compress only V. Decode uses the SM89 8-local-head specialization and writes segment reduction directly into the feature-major AllGather source slot. V64 and V96 prefill both use the SM89 specialization.

Warnings: TP8 over PCIe does not support vLLM native custom AllReduce, so dense collectives use PyNccl. Inspect nonzero preemptions before attributing large-batch changes solely to kernels.

## Reproduction

```bash
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_llama31_70b_c1.py --arm dense --model /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac --output-dir results/vllm_llama31_70b_tp8_prefill2048_optimized --batch-sizes 1 2 4 8 16 32 64 128 --prefill-tokens 2048 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --gpu-memory-utilization 0.8 --profile-batches 1 32 128
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_llama31_70b_c1.py --arm v64 --model /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac --factor-dir /workspace/.cache/huggingface/models--alexz949--BasisServe-CALS/snapshots/68c16cfb9bfb14906b36fa2719073e99e26e10c8/ICLR-results/llama31-70b/checkpoints/L31-70B-C1U-R64 --output-dir results/vllm_llama31_70b_tp8_prefill2048_optimized --batch-sizes 1 2 4 8 16 32 64 128 --prefill-tokens 2048 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --gpu-memory-utilization 0.8 --profile-batches 1 32 128
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_llama31_70b_c1.py --arm v96 --model /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac --factor-dir /workspace/.cache/huggingface/models--alexz949--BasisServe-CALS/snapshots/68c16cfb9bfb14906b36fa2719073e99e26e10c8/ICLR-results/llama31-70b/checkpoints/L31-70B-C1U-R96 --output-dir results/vllm_llama31_70b_tp8_prefill2048_optimized --batch-sizes 1 2 4 8 16 32 64 128 --prefill-tokens 2048 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --gpu-memory-utilization 0.8 --profile-batches 1 32 128
/workspace/miniforge3/envs/basis/bin/python evaluation/summarize_vllm_llama31_70b_c1.py --input-dir results/vllm_llama31_70b_tp8_prefill2048_optimized
```

V64 manifest SHA256: `a54c7e4ade1f263e990337ab6969a60359f49718a03f81735fdc6602560fa67a`.
V96 manifest SHA256: `f44ee70c1f54fca4012d527a22ee5ba214f8dade20bbde93b2c3d271a34269ed`.

This is a synthetic fixed-length serving benchmark, not a quality evaluation.
