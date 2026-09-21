# Llama-3.1-8B-Instruct TP8 dense / uniform V64 / uniform V96

Environment: `basis`, eight NVIDIA L40S PCIe GPUs; PyTorch `2.13.0+cu130`, vLLM `0.29.0`.

Every request has 4096 prompt tokens and exactly 128 generated tokens. Results are medians of 3 measured fixed-cohort runs after 1 full warmup run(s). V64 and V96 use matched uniform per-layer/per-KV-head checkpoints.

| Batch | Dense s | V64 s | V64 speedup | V96 s | V96 speedup | Dense TPOT ms | V64 TPOT ms | V96 TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.026 | 0.973 | 1.055x | 1.059 | 0.969x | 6.122 | 6.210 | 6.749 |
| 2 | 1.319 | 1.207 | 1.093x | 1.338 | 0.985x | 6.447 | 6.591 | 7.332 |
| 4 | 1.892 | 1.634 | 1.158x | 1.805 | 1.048x | 8.897 | 8.404 | 9.302 |
| 8 | 2.967 | 2.421 | 1.225x | 2.668 | 1.112x | 12.392 | 10.919 | 12.039 |
| 16 | 5.228 | 4.083 | 1.280x | 4.498 | 1.162x | 21.672 | 17.710 | 19.504 |
| 32 | 9.762 | 7.437 | 1.312x | 8.187 | 1.192x | 40.942 | 31.955 | 35.106 |
| 64 | 18.830 | 14.075 | 1.338x | 15.547 | 1.211x | 79.339 | 59.769 | 66.034 |
| 128 | 37.087 | 27.125 | 1.367x | 30.017 | 1.236x | 154.608 | 112.602 | 124.784 |
| 256 | 75.383 | 53.091 | 1.420x | 58.764 | 1.283x | 307.148 | 212.567 | 235.308 |

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
| 256 | 0 | 0 | 0 |

## Rank-zero GPU profiles

Profiles are separate full-cohort runs after timing. Percentages sum GPU kernel durations and include prefill and decode; communication durations can include profiler-inflated wait time.

| Profile | AllReduce % | AllGather % | GEMM/GEMV % | Attention % | Other % |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense_b1.kernels | 42.5 | 0.6 | 44.8 | 6.4 | 5.6 |
| dense_b256.kernels | 71.9 | 0.6 | 12.4 | 10.9 | 4.2 |
| dense_b32.kernels | 73.5 | 0.6 | 15.6 | 5.7 | 4.6 |
| v64_b1.kernels | 23.4 | 10.0 | 57.0 | 4.3 | 5.4 |
| v64_b256.kernels | 52.2 | 14.1 | 21.3 | 5.9 | 6.4 |
| v64_b32.kernels | 49.7 | 13.7 | 24.5 | 5.8 | 6.3 |
| v96_b1.kernels | 21.3 | 10.4 | 59.0 | 4.4 | 4.9 |
| v96_b256.kernels | 47.0 | 18.7 | 21.6 | 6.8 | 5.9 |
| v96_b32.kernels | 45.0 | 18.1 | 24.8 | 6.3 | 5.8 |

### Batch-one decode graph attribution

Values are mean GPU kernel microseconds per layer over verified 32-layer CUDA Graph replays. The pre-AllGather kernel is the direct segment reduction for the optimized V64/V96 decode path; no layout-copy kernel remains.

| Arm | Decoder/O-projection us | Output collective us | Attention us | Pre-AllGather reduction us |
| --- | ---: | ---: | ---: | ---: |
| dense | 7.91 | 23.11 | 13.31 | 0.00 |
| v64 | 25.34 | 13.08 | 7.95 | 1.62 |
| v96 | 40.50 | 13.23 | 8.76 | 1.79 |

V64/V96 keep dense K128 and compress only V. Decode uses the SM89 4-local-head specialization and writes segment reduction directly into the feature-major AllGather source slot. V64 and V96 prefill both use the SM89 specialization.

Warnings: TP8 over PCIe does not support vLLM native custom AllReduce, so dense collectives use PyNccl. Inspect nonzero preemptions before attributing large-batch changes solely to kernels.

## Reproduction

```bash
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_llama31_8b_instruct_c1.py --arm dense --model /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 --output-dir results/vllm_llama31_8b_tp8_optimized --batch-sizes 1 2 4 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --gpu-memory-utilization 0.8 --profile-batches 1 32 256
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_llama31_8b_instruct_c1.py --arm v64 --model /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 --factor-dir ICLR-results/llama31-8b-instruct/c1/factor-banks/R64-S6-D0 --output-dir results/vllm_llama31_8b_tp8_optimized --batch-sizes 1 2 4 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --gpu-memory-utilization 0.8 --profile-batches 1 32 256
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_llama31_8b_instruct_c1.py --arm v96 --model /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 --factor-dir ICLR-results/llama31-8b-instruct/c1/factor-banks/R96-S6-D0 --output-dir results/vllm_llama31_8b_tp8_optimized --batch-sizes 1 2 4 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --gpu-memory-utilization 0.8 --profile-batches 1 32 256
/workspace/miniforge3/envs/basis/bin/python evaluation/summarize_vllm_llama31_8b_instruct_c1.py --input-dir results/vllm_llama31_8b_tp8_optimized
```

V64 manifest SHA256: `2d0a3396503f90ebc5cb36d5871b9e18753495cd767473f953844355a3f0848e`.
V96 manifest SHA256: `bb5eac0d14f20973c1be2812429ba571629b0b1bbdf616548164272c411048b0`.

This is a synthetic fixed-length serving benchmark, not a quality evaluation.
