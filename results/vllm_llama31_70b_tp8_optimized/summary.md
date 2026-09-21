# Llama-3.1-70B TP8 dense / uniform V64 / uniform V96

Environment: `basis`, eight NVIDIA L40S PCIe GPUs; PyTorch `2.13.0+cu130`, vLLM `0.29.0`.

Every request has 4096 prompt tokens and exactly 128 generated tokens. Results are medians of 3 measured fixed-cohort runs after 1 full warmup run(s). V64 and V96 use matched uniform per-layer/per-KV-head checkpoints.

| Batch | Dense s | V64 s | V64 speedup | V96 s | V96 speedup | Dense TPOT ms | V64 TPOT ms | V96 TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.384 | 5.708 | 0.943x | 6.427 | 0.838x | 31.481 | 36.292 | 41.041 |
| 2 | 7.091 | 7.029 | 1.009x | 7.818 | 0.907x | 34.018 | 38.112 | 42.526 |
| 4 | 10.038 | 9.376 | 1.071x | 10.428 | 0.963x | 45.796 | 47.522 | 52.955 |
| 8 | 16.145 | 14.084 | 1.146x | 15.693 | 1.029x | 66.148 | 62.916 | 70.183 |
| 16 | 28.291 | 23.624 | 1.198x | 26.100 | 1.084x | 114.219 | 100.286 | 110.893 |
| 32 | 52.458 | 42.592 | 1.232x | 47.077 | 1.114x | 214.386 | 177.888 | 196.725 |
| 64 | 99.981 | 79.635 | 1.255x | 87.855 | 1.138x | 405.987 | 326.213 | 360.162 |

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

## Rank-zero GPU profiles

Profiles are separate full-cohort runs after timing. Percentages sum GPU kernel durations and include prefill and decode; communication durations can include profiler-inflated wait time.

| Profile | AllReduce % | AllGather % | GEMM/GEMV % | Attention % | Other % |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense_b1.kernels | 28.7 | 0.1 | 64.7 | 3.0 | 3.6 |
| dense_b32.kernels | 65.8 | 0.1 | 26.0 | 3.1 | 4.9 |
| dense_b64.kernels | 67.6 | 0.1 | 23.6 | 3.7 | 5.0 |
| v64_b1.kernels | 13.9 | 5.0 | 75.9 | 2.0 | 3.2 |
| v64_b32.kernels | 41.6 | 10.9 | 38.6 | 2.8 | 6.1 |
| v64_b64.kernels | 43.3 | 11.5 | 36.0 | 2.9 | 6.3 |
| v96_b1.kernels | 12.2 | 6.8 | 76.0 | 2.0 | 2.9 |
| v96_b32.kernels | 37.6 | 14.5 | 39.2 | 3.0 | 5.5 |
| v96_b64.kernels | 39.1 | 15.2 | 36.7 | 3.2 | 5.8 |

### Batch-one decode graph attribution

Values are mean GPU kernel microseconds per layer over verified 80-layer CUDA Graph replays. The pre-AllGather kernel is the direct segment reduction for the optimized V64/V96 decode path; no layout-copy kernel remains.

| Arm | Decoder/O-projection us | Output collective us | Attention us | Pre-AllGather reduction us |
| --- | ---: | ---: | ---: | ---: |
| dense | 25.37 | 24.68 | 13.53 | 0.00 |
| v64 | 98.27 | 13.92 | 8.98 | 1.60 |
| v96 | 147.08 | 21.72 | 10.59 | 1.96 |

V64/V96 keep dense K128 and compress only V. Decode uses the SM89 8-local-head specialization and writes segment reduction directly into the feature-major AllGather source slot. V64 and V96 prefill both use the SM89 specialization.

Warnings: TP8 over PCIe does not support vLLM native custom AllReduce, so dense collectives use PyNccl. Inspect nonzero preemptions before attributing large-batch changes solely to kernels.

## Reproduction

```bash
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_llama31_70b_c1.py --arm dense --model /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac --output-dir results/vllm_llama31_70b_tp8_optimized --batch-sizes 1 2 4 8 16 32 64 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --gpu-memory-utilization 0.8 --profile-batches 1 32 64
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_llama31_70b_c1.py --arm v64 --model /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac --factor-dir /workspace/.cache/huggingface/models--alexz949--BasisServe-CALS/snapshots/68c16cfb9bfb14906b36fa2719073e99e26e10c8/ICLR-results/llama31-70b/checkpoints/L31-70B-C1U-R64 --output-dir results/vllm_llama31_70b_tp8_optimized --batch-sizes 1 2 4 8 16 32 64 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --gpu-memory-utilization 0.8 --profile-batches 1 32 64
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_llama31_70b_c1.py --arm v96 --model /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac --factor-dir /workspace/.cache/huggingface/models--alexz949--BasisServe-CALS/snapshots/68c16cfb9bfb14906b36fa2719073e99e26e10c8/ICLR-results/llama31-70b/checkpoints/L31-70B-C1U-R96 --output-dir results/vllm_llama31_70b_tp8_optimized --batch-sizes 1 2 4 8 16 32 64 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --gpu-memory-utilization 0.8 --profile-batches 1 32 64
/workspace/miniforge3/envs/basis/bin/python evaluation/summarize_vllm_llama31_70b_c1.py --input-dir results/vllm_llama31_70b_tp8_optimized
```

V64 manifest SHA256: `a54c7e4ade1f263e990337ab6969a60359f49718a03f81735fdc6602560fa67a`.
V96 manifest SHA256: `f44ee70c1f54fca4012d527a22ee5ba214f8dade20bbde93b2c3d271a34269ed`.

This is a synthetic fixed-length serving benchmark, not a quality evaluation.
