# Qwen3-32B TP8 dense / C1 benchmark

## Published artifacts

Raw Dense/C1 JSON, logs, complete profiler traces, and the run-start source snapshot are archived in [Hugging Face](https://huggingface.co/alexz949/BasisServe-CALS/tree/f1ad6c3c6d0144d4f27a127ee194bb2da320bac1/system_benchmarks/vllm_32b_tp8_sm89_4k_budget8k). Immutable artifact revision: `f1ad6c3c6d0144d4f27a127ee194bb2da320bac1`. GitHub retains this report, summary CSV, parsed profile summary, and per-kernel CSV tables. The archived `summary_source.py` is the final report generator; `source.tar` preserves the run-start sources. Model weights and the R64-S6 factor bank are not duplicated in this archive.

Reproduction command (conda environment `basis`): `bash evaluation/run_qwen3_32b_tp8_sm89.sh`. The runner records both arm logs, uses structural factor validation without SHA256 checks, and refuses to overwrite existing arm JSON. Model and factor paths are specified in the runner.

Environment: `basis`, eight NVIDIA L40S (PCIe), PyTorch `2.13.0+cu130`, vLLM `0.29.0`.

Model: Qwen3-32B BF16; C1 uses R64-S6 factors. Fixed cohorts of 4096 prompt tokens and exactly 128 output tokens per request. 1 full warmup(s) and 3 measured runs per batch; table entries are medians over runs. Both arms use chunked prefill (8192-token budget), no prefix caching, synchronous scheduling, FULL_DECODE_ONLY CUDA Graphs, and compilation mode NONE. This is a matched serving configuration, not a claim of globally optimal production vLLM tuning. Dense uses FlashAttention 2; C1 uses the SM89-specialized QK128/V64 Triton DiffKV prefill kernel and the recorded decode path(s): `sm89_qk128_v64_h8`.

| Batch | Dense E2E s | C1 E2E s | E2E speedup | Dense output tok/s | C1 output tok/s | Dense TPOT ms | C1 TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.120 | 3.199 | 0.975x | 41.0 | 40.0 | 19.154 | 20.652 |
| 2 | 3.944 | 3.891 | 1.014x | 64.9 | 65.8 | 20.074 | 21.405 |
| 4 | 5.487 | 5.195 | 1.056x | 93.3 | 98.6 | 26.456 | 26.758 |
| 8 | 8.448 | 7.688 | 1.099x | 121.2 | 133.2 | 35.852 | 34.626 |
| 16 | 14.597 | 12.837 | 1.137x | 140.3 | 159.5 | 60.331 | 55.023 |
| 32 | 26.892 | 23.032 | 1.168x | 152.3 | 177.8 | 111.178 | 96.994 |
| 64 | 51.770 | 43.652 | 1.186x | 158.2 | 187.7 | 215.505 | 181.688 |
| 128 | 102.881 | 83.702 | 1.229x | 159.3 | 195.7 | 425.037 | 340.825 |
| 256 | 215.311 | 164.768 | 1.307x | 152.2 | 198.9 | 846.518 | 653.657 |

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
| c1_b1.kernels | 18.8 | 7.2 | 66.9 | 2.8 | 4.3 |
| c1_b256.kernels | 41.7 | 17.0 | 30.5 | 4.6 | 6.1 |
| c1_b32.kernels | 39.2 | 16.2 | 34.4 | 4.3 | 5.9 |
| dense_b1.kernels | 37.1 | 0.2 | 53.9 | 4.1 | 4.6 |
| dense_b256.kernels | 62.8 | 0.2 | 19.2 | 13.2 | 4.6 |
| dense_b32.kernels | 65.8 | 0.2 | 24.0 | 5.0 | 4.9 |

### Batch-one decode graph attribution

The following values use post-prefill graph replays only. Decoder attribution follows the per-layer graph order: after AllGather for C1, before the attention-output AllReduce for dense. All 64 layer boundaries are checked per replay. Values are mean GPU kernel microseconds per layer, not isolated operator benchmarks or unprofiled wall latency.

The last kernel before AllGather can be a segment reduction rather than a layout copy; consult trace names. A reduction classified as attention is already included in attention time and must not be added again.

| Arm | Principal O/decoder kernel us | Output collective us | Attention kernels us | Pre-AllGather kernel us |
| --- | ---: | ---: | ---: | ---: |
| c1 | 64.73 | 13.64 | 8.71 | 1.63 |
| dense | 16.71 | 36.94 | 13.45 | 0.00 |

The path is functional and offline-tuned for the measured SM89 prefill shapes, but is not established as globally optimal. Retain one decoder GEMM. The remaining priorities are realistic full-model/cold-cache small-batch decoder layout and GEMM/GEMV selection, followed by broader-shape prefill tuning. The C1 replicated decoder is [4096,5120] per rank versus a dense local O projection of [1024,5120]: four times the BF16 weight bytes and matrix-product work per rank. Repeated resident-weight microbenchmarks do not reproduce the full model's cache working set. Cold-weight traffic is a hypothesis to test, not a measured DRAM-bandwidth diagnosis. Compare the measured output preparation cost with decoder and attention costs before prioritizing fusion.

Warnings: native custom AllReduce variants are unsupported for eight PCIe-only GPUs, so vLLM uses PyNccl. Consult each arm's log for startup and JIT messages. vLLM process teardown can emit forced EngineCore cleanup and Python shared-memory/semaphore resource-tracker warnings; retain logs. Completed runs validate exact output lengths, non-corrupted request status, and all 64 C1 V layers plus graph-capture counters on every rank.

## Reproduction

Use the `basis` environment with `CUDA_HOME=/usr/local/cuda`, `OMP_NUM_THREADS=1`, and `VLLM_WORKER_MULTIPROC_METHOD=spawn`. No Slurm is available; the runs execute sequentially on all eight local GPUs.

```bash
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_qwen3_8b_c1.py --arm dense --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 --factor-dir /workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-32b/c1/factor-banks/R64-S6 --factor-validation structure --output-dir results/vllm_32b_tp8_sm89_4k_budget8k --batch-sizes 1 2 4 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8 --warmups 1 --repeats 3 --profile-batches 1 32 256
/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_qwen3_8b_c1.py --arm c1 --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 --factor-dir /workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-32b/c1/factor-banks/R64-S6 --factor-validation structure --output-dir results/vllm_32b_tp8_sm89_4k_budget8k --batch-sizes 1 2 4 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8 --warmups 1 --repeats 3 --profile-batches 1 32 256
/workspace/miniforge3/envs/basis/bin/python evaluation/summarize_vllm_qwen3_8b_c1.py --input-dir results/vllm_32b_tp8_sm89_4k_budget8k
```

Factor validation: configuration, manifest structure and tensor shapes only; no SHA256 checks were performed. Factor contents were not authenticated.

Model startup, JIT compilation, profiling, and trace export are excluded from timed measurements. This is a synthetic fixed-length throughput test, not a quality evaluation or an online arrival-rate benchmark.
