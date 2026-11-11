# LRQK TP1 retest after the prefill offload stream-order fix

Runs complete: 12 of 12 present (arms: lrqk_upstream_koff_v128, lrqk_upstream_kvoff_v128).
The three cohorts are three different prompts (row 0 of cohort 0, 1, 2 of the TP4 64K inputs), not repetitions of one prompt.

## Setup

- Llama-3.1-8B-Instruct, BF16, original V128/Wo, revision `0e9e39f249a16976918f6564b8830bc894c89659`; TP1, B1, 65536 prompt tokens.
- LRQK: rank 32, top-k 2048, 64 lite tokens, iterations (2, 2), tolerance (0.01, 0.01); upstream `tenghuilee/LRQK@caf16293`.
- Repository `be7c8a2` (contains the fix `be7c8a2`: True), with uncommitted entry-point changes (diff sha256 `236dac4a93df`).
- GPU: NVIDIA L40S, driver 550.54.15, host `node6.athena`, one GPU.
- CPU: Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz; 24 threads on 12 physical cores (both hyperthreads) of NUMA node 1, `OMP_NUM_THREADS=24`.
- Software: conda env `basis`, Python 3.11.16, PyTorch 2.13.0+cu129, CUDA 12.9, Transformers 5.17.0, NCCL 2.29.7; PyTorch default (PYTORCH_CUDA_ALLOC_CONF empty), as the archived upstream runs.
- Each run: fresh process, same-shape warmup request, fresh state, then the measurement. Request = 128 tokens; steady = 16 conditioning + 128 measured steps. No component timers.

## Correctness

- Stream-order check (`check_lrqk_offload_order.py`, read from its `result.json`): ordered copy wrong elements {'K': 0, 'V': 0, 'A_K': 0}; passed: True. The unordered upstream copy in the same probe: {'K': 17301504, 'V': 17301504, 'A_K': 16777216}. Source tensors are kept alive by the script: it isolates the producer-to-copy stream ordering and is not a full allocator-lifetime regression test.
- 8192-token smoke, `lrqk_upstream_koff_v128_request_b1_t8192_smoke`: status complete, 8 tokens, log-probabilities finite: True, imported files match the reviewed hashes: True, ordered prefill class in use: True; passed: True.
- 8192-token smoke, `lrqk_upstream_kvoff_v128_request_b1_t8192_smoke`: status complete, 8 tokens, log-probabilities finite: True, imported files match the reviewed hashes: True, ordered prefill class in use: True; passed: True.
- 65536-token probe, `lrqk_upstream_koff_v128_request_b1_t65536_smoke`: status complete, 8 tokens, log-probabilities finite: True, imported files match the reviewed hashes: True, ordered prefill class in use: True; passed: True.
- 65536-token probe, `lrqk_upstream_kvoff_v128_request_b1_t65536_smoke`: status complete, 8 tokens, log-probabilities finite: True, imported files match the reviewed hashes: True, ordered prefill class in use: True; passed: True.

Placement consistency (8192 tokens). Fixed K-only (lrqk_upstream_koff_v128) decodes 128 greedy tokens from an 8192-token prompt; fixed K+V (lrqk_upstream_kvoff_v128) is fed exactly those tokens. Compared at every position on the same prefix: the K+V argmax against the K-only token (position 0 is the prefill prediction), and the log-probability each placement gives the same fed token (127 decode steps).

| Cohort | Steps | K+V argmax = K-only token | Max abs log-prob difference | Mean abs difference |
|---|---|---|---|---|
| 0 | 128 | 128 / 128 | 0.0 | 0.0 |
| 1 | 128 | 128 / 128 | 0.0 | 0.0 |
| 2 | 128 | 128 / 128 | 0.0 | 0.0 |

Exact agreement (top-1 100 %, maximum difference 0): True. Implementation and placement consistency only; not a quality comparison with Dense.

## Latency and memory

Median of the three cohorts, with the range in brackets. Request and steady step times are different metrics and are not merged.

| | K offload, V on GPU | K and V offload |
|---|---|---|
| Decode-ready (prefill until all rows placed) | 15.73 s (15.70 to 15.76; n=3) | 18.07 s (17.98 to 18.36; n=3) |
| Request end to end, 128 tokens | 54.23 s (54.09 to 55.87; n=3) | 65.45 s (64.84 to 66.48; n=3) |
| Request: mean decode step (127 steps) | 302.9 ms (302.2 to 316.1; n=3) | 370.8 ms (369.0 to 381.1; n=3) |
| Steady: mean step | 325.5 ms (314.0 to 330.0; n=3) | 342.0 ms (337.3 to 350.4; n=3) |
| Steady: p50 step | 314.4 ms (312.2 to 320.3; n=3) | 332.5 ms (326.5 to 337.2; n=3) |
| Steady: p95 step | 390.1 ms (372.7 to 393.3; n=3) | 401.3 ms (400.1 to 414.6; n=3) |
| Steady: tokens per second | 3.07 tok/s (3.03 to 3.18; n=3) | 2.92 tok/s (2.85 to 2.96; n=3) |
| GPU resident, allocated | 26.04 GiB (26.04 to 26.04; n=3) | 22.03 GiB (22.03 to 22.03; n=3) |
| Prefill peak allocated | 26.20 GiB (26.20 to 26.20; n=3) | 22.31 GiB (22.31 to 22.31; n=3) |
| Prefill peak reserved | 37.51 GiB (37.51 to 37.51; n=3) | 31.08 GiB (31.08 to 31.08; n=3) |
| Decode peak allocated | 26.20 GiB (26.20 to 26.20; n=3) | 22.19 GiB (22.19 to 22.19; n=3) |
| Decode peak reserved | 30.39 GiB (30.39 to 30.39; n=3) | 22.44 GiB (22.44 to 22.44; n=3) |
| Pinned host memory holding offloaded state | 6.00 GiB (6.00 to 6.00; n=3) | 12.00 GiB (12.00 to 12.00; n=3) |
| Process RSS | 19.99 GiB (19.99 to 20.00; n=3) | 27.93 GiB (27.93 to 27.93; n=3) |

Per cohort:

| Arm | Mode | Cohort | Status | Decode-ready s | E2E s | Request step ms | Steady mean / p50 / p95 ms | Others on pinned CPUs, max |
|---|---|---|---|---|---|---|---|---|
| `lrqk_upstream_koff_v128` | request | 0 | complete | 15.70 | 54.09 | 302.2 |  | 1.9 |
| `lrqk_upstream_kvoff_v128` | request | 0 | complete | 18.07 | 66.48 | 381.1 |  | 3.2 |
| `lrqk_upstream_koff_v128` | request | 1 | complete | 15.73 | 55.87 | 316.1 |  | 3.3 |
| `lrqk_upstream_kvoff_v128` | request | 1 | complete | 18.36 | 65.45 | 370.8 |  | 3.2 |
| `lrqk_upstream_koff_v128` | request | 2 | complete | 15.76 | 54.23 | 302.9 |  | 3.4 |
| `lrqk_upstream_kvoff_v128` | request | 2 | complete | 17.98 | 64.84 | 369.0 |  | 2.2 |
| `lrqk_upstream_koff_v128` | steady | 0 | complete |  |  |  | 330.0 / 314.4 / 390.1 | 4.3 |
| `lrqk_upstream_kvoff_v128` | steady | 0 | complete |  |  |  | 342.0 / 332.5 / 401.3 | 3.7 |
| `lrqk_upstream_koff_v128` | steady | 1 | complete |  |  |  | 325.5 / 320.3 / 393.3 | 4.3 |
| `lrqk_upstream_kvoff_v128` | steady | 1 | complete |  |  |  | 337.3 / 326.5 / 400.1 | 1.3 |
| `lrqk_upstream_koff_v128` | steady | 2 | complete |  |  |  | 314.0 / 312.2 / 372.7 | 3.5 |
| `lrqk_upstream_kvoff_v128` | steady | 2 | complete |  |  |  | 350.4 / 337.2 / 414.6 | 3.4 |

## Warnings, failures and interference

- No formal run failed.
- Earlier attempt, kept in `runs.jsonl`: `lrqk_upstream_koff_v128_request_b1_t8192_smoke` started 2026-11-11T20:25:25Z: failed (exit 1); torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
- Effective settings (torch.compile disabled, float32 matmul precision, cuda.matmul.allow_tf32), read after each measurement: [(False, 'high', True)].
- Warning lines per run (torch.compile recompiles, autotune, deprecations) are counted in `runs.csv` and listed in each `run.json`.
- `load_pinned_busy_by_others_max`: the most logical CPUs of the 24 pinned ones that other processes kept busy in any 5 s sample; `load.jsonl` in each run directory has the samples.

## Limits

- This is the absolute TP1 latency of the fixed LRQK on this host. The earlier TP1/64K numbers (K+V about 369 ms/step, K-only about 430 ms/step) were measured before the fix, on another host (Xeon Platinum 8558, driver 570, Transformers 5.22.2) and under CPU contention; differences from them are not attributable to the stream fix.
- BasisKV and ShadowKV were not rerun here; any table that puts these numbers next to theirs compares different hosts and sessions, not a matched pair.
- The correctness checks cover implementation and placement consistency, not quality against Dense.
