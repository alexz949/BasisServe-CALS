# V2Q8 sparse baselines on L40S node 2: V128, preliminary

> **Round 2 is in progress.** This README describes round 1 (`formal/`). Round 2 reproduces both baselines from
> their unmodified upstream runtimes and fixes the issues listed below (`smoke_v3/`, `formal_v2/`). See
> `audit/COMPATIBILITY.md` sections 12 and 12.1. This README will be rewritten when round 2 finishes.

2026-11-10, Vast.ai 4x L40S node 2, the same machine as `v2q8_palu_g4_tp4/l40s_node2`. The repository is at
`abf8550`. The benchmark code for these runs is uncommitted, in the working tree: `bench_v2q8_tp.py`,
`v2q8_basis.py`, `basisserve/core/lrqk_tp.py`, the ShadowKV TP adapter, `run_v2q8_exp_*.py` and the checks.

**Status: preliminary.** The formal V128 grid is complete. Four known issues, listed below, affect the prefill
numbers and the ShadowKV decode comparison. None of the fixes has been applied yet. Decode and memory numbers of
BasisKV are not affected.

## Design

| | Exp 1 | Exp 2 |
|---|---|---|
| Parallelism | TP1 (GPU 0) | TP4 |
| Workload | 130048 prompt + 128 output tokens, B1 | 65536 + 128, B16 |
| Prefill | one request | one request at a time into its cache row; decode runs all rows together |

Arms (Llama-3.1-8B-Instruct, BF16):

| Arm | Method | Placement |
|---|---|---|
| `basis_koff_v128` | BasisKV, B16R16 router, Page32 | K in mapped pinned host memory with GPU slots; V on GPU |
| `basis_koff_v128_reuse` | Same; the Base code is read from the rotated V | Same |
| `shadowkv_voff_v128` | ShadowKV rank 160, budget 2048; TP adapter around the official `ShadowKVCache_CPU` | V on pinned host |
| `lrqk_koff_v128` | LRQK rank 32, top-2048 + 64 lite | K in the CPU pool; V on GPU |

Modes:
- `request`: 3 cohorts.
- `steady`: 3 cohorts; 16 conditioning steps, then 128 measured.
- `breakdown`: cohort 0, with synchronized component timers. Its component sums exceed the steady step time.

All arms use the same runtime settings: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 24 threads on the
GPU's NUMA node, and chunked MLP. Adapter checks, criteria and every change made before the formal runs are in
`audit/COMPATIBILITY.md` (section 11).

## Outcome

- 42 runs complete.
- 14 runs failed. All of them are LRQK, and all are CUDA out of memory:
  - TP1 130K: about 50 MiB short, at the ALS step `new_A_K = K @ solve(...)`.
  - TP4 B16: about 38.5 GiB per rank is resident before prefill.

### Exp 1: TP1, 130048 + 128, B1

Medians over 3 cohorts.

| | BasisKV | BasisKV reuse | ShadowKV |
|---|---|---|---|
| TTFT | 43.5 s (issue 1) | 43.6 s (issue 1) | 44.6 s, then 1.1 s placement |
| Request E2E | 48.5 s | 48.6 s | 50.1 s |
| Decode step, steady median | 39.3 ms | 39.4 ms | 36.0 ms |
| GPU resident / prefill peak | 25.2 / 32.7 GiB | 24.2 / 31.7 GiB | 17.6 / 22.5 GiB |
| Host offload | 7.95 GiB | 7.95 GiB | 7.94 GiB |

### Exp 2: TP4, 65536 + 128, B16

Slowest rank; medians over 3 cohorts.

| | BasisKV | BasisKV reuse | ShadowKV |
|---|---|---|---|
| All 16 requests prefilled | 132.7 s (issue 1) | 131.7 s (issue 1) | 176.5 s (issue 2) |
| Prefill per request | 8.27 s | 8.23 s | 11.05 s |
| Decode step, steady median | 48.3 ms | 48.8 ms | 51.3 ms (cohorts 48.4 to 51.7) |
| Decode p95 | 50 to 55 ms | 54 to 57 ms | 52 to 75 ms |
| GPU resident / peak per rank | 25.4 / 28.0 GiB | 23.4 / 26.0 GiB | 17.9 / 18.6 GiB |
| Host, 4 ranks | 64.1 GiB | 64.1 GiB | 64.0 GiB |

Per-cohort numbers and breakdowns are in `diagnostics/formal_summary.txt`; raw files are under `formal/`.

ShadowKV's U is shared by all 8 KV heads, so under TP it is replicated on every rank. At TP4 it takes 10.0 GiB
per rank (inventory), against 16 GiB for the raw K of the rank's 2 heads. At TP1 it is 1.24 GiB against 7.94 GiB.

## Known issues

1. **BasisKV prefill Key offload uses a slow copy path.** `v2q8_basis.py` `_write_prefill` writes the prompt
   Keys with `host_key[...].copy_(key)`. Both tensors are non-contiguous, so PyTorch stages the copy through a
   pageable buffer.
   - The archived Table 3 harness used `router_extension.append` (`cudaMemcpy2DAsync`) instead.
   - Measured per layer at 130K: 226 ms with `copy_`, 10.2 ms with `append`, with bit-identical results
     (`diagnostics/k_offload_copy_bench.*`).
   - Cost: about 7.2 s of TTFT at TP1 130K, and about 1 s per request at TP4.
   - Estimate after the fix (not measured): TTFT about 37 s at TP1; about 7.2 s per request at TP4.
2. **ShadowKV TP4 prefill serializes the SVD across ranks.** Each layer's owner (`layer % 4`) runs the global
   SVD inside the forward and broadcasts immediately, so the other ranks wait.
   - Rank 0 per request: SVD 0.95 s, waiting 2.89 s, out of 11.05 s.
   - The factors are only needed for decode, so the SVD could run on all ranks in parallel after the forward. Not
     applied.
3. **The ShadowKV adapter differs from the upstream runtime.** The upstream source is unmodified
   (`external/ShadowKV` at `e51904cd`). Landmarks, outliers, V offload, selection, K reconstruction, V fetch and
   the kernels are upstream calls. The SVD (`get_svd`) and the model around attention are reimplemented:
   - the SVD runs without upstream's sync, `gc.collect()` and `empty_cache()`;
   - U and SV are placed on the GPU directly;
   - the model is the HF model shared with BasisKV.

   Against the earlier pure-upstream TP1 run (`system_benchmarks/tp1_offline`), the adapter is:
   - faster in prefill: 44.4 s against 53.5 s;
   - slower in decode: 36.0 ms against 32.6 ms per step. The cause has not been located yet.
4. **LRQK placement.** The requested placement is K-only offload, with V on the GPU. Its GPU state is then about
   the size of the dense KV cache, hence the OOMs.
   - Upstream LRQK writes K to its CPU pool with the same kind of non-contiguous `copy_` as issue 1.

## Consistency with earlier results

- `table3_k_offload_128k`: `basis_k_offload_nobase` decodes at 39.48 ms with 24.29 GiB resident. The reuse arm
  here measures 39.40 ms and 24.23 GiB.
- `tp1_offline`:
  - BasisKV with GPU-resident Keys had a prefill of 38.15 s, consistent with the issue-1 estimate.
  - LRQK with CPU K and V took about 380 to 400 ms per token at 32K and 64K.

## Files

| Path | Content |
|---|---|
| `formal/exp1`, `formal/exp2` | Per-run `result.json` (per-rank timings, memory, inventory, environment); LRQK directories are empty (OOM) |
| `formal/logs`, `logs/` | Per-run logs, runner and chain logs, GPU checks before the runs |
| `formal/plan_*.json`, `formal/runner_*.json` | Gates, adapter checks and the run list |
| `smoke/` | Two smoke rounds at 8192 tokens, build logs, adapter checks (`smoke/checks`). Smoke timings are not results |
| `audit/` | `COMPATIBILITY.md` and working notes |
| `reuse_accounting/` | State accounting for the value-reuse arm |
| `tp4_64k/inputs/` | TP4 prompts (3 cohorts) |
| `diagnostics/` | Formal summary, K offload copy microbenchmark |
