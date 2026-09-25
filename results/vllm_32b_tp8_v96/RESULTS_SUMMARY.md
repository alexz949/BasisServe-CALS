# Qwen3-32B TP8 V96 results

Completed on 2026-09-25 UTC. Environment: `basis`, 8 x NVIDIA L40S,
PyTorch `2.13.0+cu130`, CUDA `13.0`, vLLM `0.29.0`.

This publication adds V96 results only. Serving code and unrelated experiments
are not changed by the publication commit. Measured source snapshots remain
in the raw-data archive for reproducibility; do not assume the published
GitHub code includes the local V96 integration changes.

## Protocol

- Qwen3-32B BF16, TP8, uniform R96-S6 factors, QK128/V96 SM89 Triton DiffKV.
- 128 output tokens per request, 8192-token scheduler budget, GPU memory
  utilization 0.8, one full warmup and three measured repetitions per point.
- Continuous batching, chunked prefill, no prefix caching, synchronous
  scheduling, FULL_DECODE_ONLY CUDA Graphs, compilation mode NONE.
- Batch is the submitted request cohort, not a fixed active GPU batch.
  E2E latency covers the entire cohort, including prefill and scheduling.
  Output throughput is `batch * 128 / cohort wall seconds`.
- Factor validation checked configuration, manifest structure and tensor
  shapes. No SHA256 checks were performed; factor contents are not authenticated.

Model snapshot: `Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137`.
Factor snapshot: `alexz949/BasisServe-CALS@0d0b7a569eb3adcfbf5d63d78a8beb767980fcad`,
`ICLR-results/qwen3-32b/c1/factor-banks/R96-S6`.

## 4K Prefill

All entries are medians of three complete runs. Dense is reused from the
previous matched-configuration 4K benchmark; V96 is newly measured here.
The interrupted local Dense rerun is NOT used as the baseline.

| Submitted batch | Dense E2E s | V96 E2E s | Dense / V96 |
| ---: | ---: | ---: | ---: |
| 1 | 3.120 | 3.515 | 0.888x |
| 2 | 3.944 | 4.341 | 0.908x |
| 4 | 5.487 | 5.818 | 0.943x |
| 8 | 8.448 | 8.653 | 0.976x |
| 16 | 14.597 | 14.457 | 1.010x |
| 32 | 26.892 | 26.124 | 1.029x |
| 64 | 51.770 | 49.215 | 1.052x |
| 128 | 102.881 | 94.646 | 1.087x |
| 256 | 215.311 | 187.678 | 1.147x |

No OOM. B1-B128: zero preemptions in all runs. At B256, Dense and V96 each
had one preemption per repetition (three total per arm), so this point is
capacity/scheduling affected. See [full-precision CSV](formal/summary.csv).

Dense provenance: the existing
[4K baseline summary](../vllm_32b_tp8_sm89_4k_budget8k/summary.md), published in
GitHub commit `dace81d`; original raw data at HF revision
`f1ad6c3c6d0144d4f27a127ee194bb2da320bac1`,
`system_benchmarks/vllm_32b_tp8_sm89_4k_budget8k/dense.json`.
Configuration and PyTorch/vLLM versions match V96. The stopped Dense rerun
also reproduced B1-B32 medians within 0.15%; its partial records are archived
only for audit and must not be mixed into the complete baseline.

## 8K Prefill

Dense and V96 were both measured at each point with the same 8192-token
scheduler budget. No kernel profiling runs were requested for these points.

| Submitted batch | Dense E2E s | V96 E2E s | Speedup | Dense output tok/s | V96 output tok/s | Preemptions per run, Dense / V96 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 212.535 | 189.113 | 1.124x | 77.09 | 86.64 | 0 / 1 |
| 256 | 438.371 | 373.031 | 1.175x | 74.75 | 87.84 | 0 / 0 |

All six runs at each batch completed without OOM. At B256, sampled logs
reached 106 active requests for both arms; this is not simultaneous GPU
execution of 256 requests. Zero preemptions do not imply unlimited KV capacity:
waiting/admission still affects the result.

V96 B256 raw times: 372.996173, 373.031065, 373.316921 seconds.
Dense B256 raw times: 438.560426, 438.370570, 438.182270 seconds.
All 1536 measured requests in the B256 comparison returned exactly 128 tokens.

Full-precision CSVs: [B128](8k_b128/summary.csv), [B256](8k_b256/summary.csv).
The 8K B256 median latency reduction is 14.91%. These are serving outcomes,
not isolated decoder/attention speedups or quality evaluations.

## 4K Profiles and Interpretation

Separate rank-zero profiling runs completed at B1/B32/B256 after timed runs.
See [profile summary](formal/profile_summary.json) and the kernel CSVs under
`formal/profiles/`. Summed GPU kernel durations are not unprofiled wall time.
For B32/B256, graph replay batch sizes can change as requests finish.

B1 post-prefill graph attribution, mean microseconds per layer:

| Component | Dense, previous profile | V96 |
| --- | ---: | ---: |
| Principal O projection / decoder kernel | 16.71 | 87.57 |
| Attention-output collective | 36.94 | 20.38 |
| Attention kernels | 13.45 | 10.04 |

V96 uses a replicated [6144,5120] decoder per rank, versus Dense's local
[1024,5120] O projection. This is 6x weight bytes and matrix-product work for
that projection, not 6x full-model work. It is a measured cost tradeoff, not
proof of a DRAM-bandwidth bottleneck or globally optimal implementation.

V96 is slower than Dense at small 4K batches and faster at larger cohorts.
Increasing prefill to 8K improved the tested E2E ratios, but these experiments
do not isolate decoder amortization from prefill, attention, and scheduling.

## Raw Data and Reproduction

[Raw data on Hugging Face](https://huggingface.co/alexz949/BasisServe-CALS/tree/main/system_benchmarks/vllm_32b_tp8_v96)
contains complete JSON records, logs, compressed traces, source archives,
smoke/kernel checks, and original run-time notes. The pinned upload revision
and exact file sizes are recorded in [UPLOAD_MANIFEST.json](UPLOAD_MANIFEST.json).
The archived root `README.md` describes the earlier smoke stage; this file
is the current summary of completed formal runs.

Exact Python commands are stored in each raw JSON's `command` field.
The 8K executable commands are also retained as `8k_b128/run.sh` and
`8k_b256/run.sh` in the raw archive. Use the `basis` environment and
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, `CUDA_HOME=/usr/local/cuda`,
`OMP_NUM_THREADS=1`, `MAX_JOBS=2`, `TORCH_CUDA_ARCH_LIST=8.9`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`.

Each arm runs `evaluation/benchmark_vllm_qwen3_8b_c1.py` with
`--factor-validation structure --decode-tokens 128
--max-num-batched-tokens 8192 --gpu-memory-utilization 0.8 --warmups 1 --repeats 3`.
The 4K V96 grid uses `--arm c1 --prefill-tokens 4096
--batch-sizes 1 2 4 8 16 32 64 128 256 --profile-batches 1 32 256`.
The 8K pairs use `--arm c1` then `--arm dense`, `--prefill-tokens 8192`,
`--batch-sizes 128` or `256`, and empty `--profile-batches`.

Startup/JIT and separate profiling are excluded from timed results. vLLM
uses PyNccl on these eight PCIe GPUs. Teardown logs can include forced
EngineCore cleanup after workers exit; all formal JSONs reported completion.
Initial smoke/test failure logs are retained alongside successful reruns.
No claim of model quality or kernel optimality follows from these timings.
