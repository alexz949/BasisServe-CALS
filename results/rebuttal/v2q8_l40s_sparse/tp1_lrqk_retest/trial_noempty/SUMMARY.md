# LRQK TP1 retest after the prefill offload stream-order fix

Runs complete: 4 of 4 present (arms: lrqk_upstream_koff_noempty_v128, lrqk_upstream_kvoff_noempty_v128).
The three cohorts are three different prompts (row 0 of cohort 0, 1, 2 of the TP4 64K inputs), not repetitions of one prompt.

## Setup

- Llama-3.1-8B-Instruct, BF16, original V128/Wo, revision `0e9e39f249a16976918f6564b8830bc894c89659`; TP1, B1, 65536 prompt tokens.
- LRQK: rank 32, top-k 2048, 64 lite tokens, iterations (2, 2), tolerance (0.01, 0.01); upstream `tenghuilee/LRQK@caf16293`.
- Repository `be7c8a2` (contains the fix `be7c8a2`: True), with uncommitted entry-point changes (diff sha256 `6d90c5d8fe75`).
- GPU: NVIDIA L40S, driver 550.54.15, host `node6.athena`, one GPU.
- CPU: Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz; 24 threads on 12 physical cores (both hyperthreads) of NUMA node 1, `OMP_NUM_THREADS=24`.
- Software: conda env `basis`, Python 3.11.16, PyTorch 2.13.0+cu129, CUDA 12.9, Transformers 5.17.0, NCCL 2.29.7; PyTorch default (PYTORCH_CUDA_ALLOC_CONF empty), as the archived upstream runs.
- Each run: fresh process, same-shape warmup request, fresh state, then the measurement. Request = 128 tokens; steady = 16 conditioning + 128 measured steps. No component timers.

## Correctness

- 8192-token smoke, `lrqk_upstream_koff_noempty_v128_request_b1_t8192_smoke`: status complete, 8 tokens, log-probabilities finite: True, imported files match the reviewed hashes: True, ordered prefill class in use: True; passed: True.
- 8192-token smoke, `lrqk_upstream_kvoff_noempty_v128_request_b1_t8192_smoke`: status complete, 8 tokens, log-probabilities finite: True, imported files match the reviewed hashes: True, ordered prefill class in use: True; passed: True.
## Latency and memory

Median of the three cohorts, with the range in brackets. Request and steady step times are different metrics and are not merged.

| | K offload, no decode empty_cache (trial) | K and V offload, no decode empty_cache (trial) |
|---|---|---|
| Decode-ready (prefill until all rows placed) | 15.68 s (15.68 to 15.68; n=1) | 18.45 s (18.45 to 18.45; n=1) |
| Request end to end, 128 tokens | 45.49 s (45.49 to 45.49; n=1) | 49.30 s (49.30 to 49.30; n=1) |
| Request: mean decode step (127 steps) | 234.7 ms (234.7 to 234.7; n=1) | 242.9 ms (242.9 to 242.9; n=1) |
| Steady: mean step | 231.0 ms (231.0 to 231.0; n=1) | 243.9 ms (243.9 to 243.9; n=1) |
| Steady: p50 step | 230.2 ms (230.2 to 230.2; n=1) | 242.1 ms (242.1 to 242.1; n=1) |
| Steady: p95 step | 238.3 ms (238.3 to 238.3; n=1) | 255.6 ms (255.6 to 255.6; n=1) |
| Steady: tokens per second | 4.33 tok/s (4.33 to 4.33; n=1) | 4.10 tok/s (4.10 to 4.10; n=1) |
| GPU resident, allocated | 26.04 GiB (26.04 to 26.04; n=1) | 22.03 GiB (22.03 to 22.03; n=1) |
| Prefill peak allocated | 26.20 GiB (26.20 to 26.20; n=1) | 22.31 GiB (22.31 to 22.31; n=1) |
| Prefill peak reserved | 39.08 GiB (39.08 to 39.08; n=1) | 36.64 GiB (36.64 to 36.64; n=1) |
| Decode peak allocated | 26.20 GiB (26.20 to 26.20; n=1) | 22.19 GiB (22.19 to 22.19; n=1) |
| Decode peak reserved | 39.08 GiB (39.08 to 39.08; n=1) | 36.64 GiB (36.64 to 36.64; n=1) |
| Pinned host memory holding offloaded state | 6.00 GiB (6.00 to 6.00; n=1) | 12.00 GiB (12.00 to 12.00; n=1) |
| Process RSS | 19.95 GiB (19.95 to 19.95; n=1) | 27.93 GiB (27.93 to 27.93; n=1) |

Per cohort:

| Arm | Mode | Cohort | Status | Decode-ready s | E2E s | Request step ms | Steady mean / p50 / p95 ms | Others on pinned CPUs, max |
|---|---|---|---|---|---|---|---|---|
| `lrqk_upstream_koff_noempty_v128` | request | 0 | complete | 15.68 | 45.49 | 234.7 |  | 0.9 |
| `lrqk_upstream_kvoff_noempty_v128` | request | 0 | complete | 18.45 | 49.30 | 242.9 |  | 1.6 |
| `lrqk_upstream_koff_noempty_v128` | steady | 0 | complete |  |  |  | 231.0 / 230.2 / 238.3 | 0.6 |
| `lrqk_upstream_kvoff_noempty_v128` | steady | 0 | complete |  |  |  | 243.9 / 242.1 / 255.6 | 1.9 |

## Warnings, failures and interference

- No formal run failed.
- Effective settings (torch.compile disabled, float32 matmul precision, cuda.matmul.allow_tf32), read after each measurement: [(False, 'high', True)].
- Warning lines per run (torch.compile recompiles, autotune, deprecations) are counted in `runs.csv` and listed in each `run.json`.
- `load_pinned_busy_by_others_max`: the most logical CPUs of the 24 pinned ones that other processes kept busy in any 5 s sample; `load.jsonl` in each run directory has the samples.

## Limits

- This is the absolute TP1 latency of the fixed LRQK on this host. The earlier TP1/64K numbers (K+V about 369 ms/step, K-only about 430 ms/step) were measured before the fix, on another host (Xeon Platinum 8558, driver 570, Transformers 5.22.2) and under CPU contention; differences from them are not attributable to the stream fix.
- BasisKV and ShadowKV were not rerun here; any table that puts these numbers next to theirs compares different hosts and sessions, not a matched pair.
- The correctness checks cover implementation and placement consistency, not quality against Dense.
