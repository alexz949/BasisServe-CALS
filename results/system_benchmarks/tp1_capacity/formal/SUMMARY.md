# TP1 capacity and K-offload throughput

Historical grid: the later chunked-RoPE 128K/B2 result is in [the formal supplement](../rope_chunked/formal/SUMMARY.md).
Raw JSON, logs and source snapshots: [immutable HF archive](https://huggingface.co/alexz949/BasisServe-CALS/tree/0d0b7a569eb3adcfbf5d63d78a8beb767980fcad/system_benchmarks/tp1_capacity/formal).

Completed trials: 42; successful: 36; GPU OOM: 6. OOM is an observed outcome, not a missing throughput estimate.

Llama-3.1-8B-Instruct BF16, one L40S, environment `basis`. Full V128 remains on GPU in every arm. Basis uses full-scan B16R16 Page32 routing, 1,984 routed tokens plus 64 recent tokens, and persistent exact-K slots. No two-stage routing. Sparse and dense attention are different algorithms; this is not a quality evaluation.

Prompts are admitted sequentially into independent preallocated KV rows. All B requests then decode together with no EOS stopping or batch reduction. Reported throughput excludes prefill, includes logits/argmax, finite checks and synchronization. The default 128 measured decode forwards exclude the first token produced by prefill. Warmup lengths and K-slot state are reset. This is a fixed-active-batch benchmark, not scheduler throughput or E2E latency.

All methods use the same prompt rows. Repeats are fresh processes; table values are medians across successful repeats. GPU peak is PyTorch allocated memory, not total device usage. Host peak RSS includes model loading; explicit pinned K bytes are separate in CSV. OOM during allocation or prefill is not labelled decode OOM. Other errors are retained as failures, not OOM.

| Context | Method | Batch | Status | Repeats | Decode ms/step | Aggregate tok/s | GPU peak GiB |
|---:|---|---:|---|---:|---:|---:|---:|
| 64K | Dense-local | 1 | success | 3 | 36.16 | 27.66 | 26.76 |
| 64K | Dense-local | 2 | success | 3 | 48.64 | 41.12 | 34.78 |
| 64K | Dense-local | 4 | gpu_oom (cache_allocation) | 0 | - | - | - |
| 64K | Dense-K-offload | 1 | success | 3 | 196.89 | 5.08 | 22.88 |
| 64K | Dense-K-offload | 2 | success | 3 | 369.12 | 5.42 | 27.01 |
| 64K | Dense-K-offload | 4 | success | 3 | 712.04 | 5.62 | 35.31 |
| 64K | Dense-K-offload | 8 | gpu_oom (cache_allocation) | 0 | - | - | - |
| 64K | BasisKV-K-offload | 1 | success | 3 | 34.06 | 29.36 | 24.03 |
| 64K | BasisKV-K-offload | 2 | success | 3 | 43.38 | 46.10 | 29.29 |
| 64K | BasisKV-K-offload | 4 | success | 3 | 58.66 | 68.18 | 39.83 |
| 64K | BasisKV-K-offload | 8 | gpu_oom (cache_allocation) | 0 | - | - | - |
| 128K | Dense-local | 1 | success | 3 | 47.74 | 20.94 | 38.55 |
| 128K | Dense-local | 2 | gpu_oom (cache_allocation) | 0 | - | - | - |
| 128K | Dense-K-offload | 1 | success | 3 | 367.75 | 2.72 | 30.79 |
| 128K | Dense-K-offload | 2 | success | 3 | 710.76 | 2.81 | 39.05 |
| 128K | Dense-K-offload | 4 | gpu_oom (cache_allocation) | 0 | - | - | - |
| 128K | BasisKV-K-offload | 1 | success | 3 | 40.66 | 24.59 | 32.94 |
| 128K | BasisKV-K-offload | 2 | gpu_oom (prefill) | 0 | - | - | - |

## Largest successful tested batch

Only configurations completing all requested repeats count. These are tested powers of two, not exact maximum capacities.

| Context | Dense-local | Dense-K-offload | BasisKV-K-offload |
|---:|---:|---:|---:|
| 64K | 2 | 4 | 4 |
| 128K | 1 | 2 | 1 |

## Prefill capacity caveat

The 128K/B2 Basis run failed during prefill in `apply_rotary_pos_emb`, while allocating a 1 GiB temporary for `rotate_half(q) * sin`. It did not reach decode. The CUDA diagnostic reported 42.31 GiB allocated by PyTorch and 1.28 GiB reserved but unallocated, with 277 MiB device memory free. This result does not establish a fundamental decode-resident capacity limit: reducing prefill temporaries or allocator fragmentation would require a separately validated run. Dense-K-offload completed 128K/B2; the current data do not demonstrate a Basis capacity advantage there.

All other observed OOMs occurred during cache allocation. The 64K/B4 comparison is successful for both offload methods while Dense-local OOMs. Do not extrapolate the 64K result to an unmeasured successful Basis 128K/B2 decode.

## Reproduction

Environment: `basis`. Raw per-trial commands, logs, tokens and memory statistics are retained in trial directories. Source snapshot: `source.tar.gz`; run settings: `manifest.json`; end-of-run byte comparison: `source_check.json`. The final report generator (`summary_source.py`) adds the observed OOM caveat and figure formatting after the run. No runtime source changed during the run. No SHA256 checks.

```bash
/workspace/miniforge3/envs/basis/bin/python benchmarks/system/run_tp1_capacity.py --contexts 65536 131072 --batches 1 2 4 8 16 32 --repeats 3 --warmup 8 --steps 128 --output results/system_benchmarks/tp1_capacity/formal
```
