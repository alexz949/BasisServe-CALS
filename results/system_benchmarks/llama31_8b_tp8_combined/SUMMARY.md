# Llama-3.1-8B-Instruct TP8 Joint-ALS + key-routing decode benchmark

## Outcome

The frozen Experiment B grid completed with **132/144 successful trials** and **12 preserved failures**. Each summary row is the median of three preselected real-text cohorts; each successful trial contains 16 conditioning steps followed by 128 measured autoregressive decode steps.

ALS-full is faster than Dense at **14/14** valid matched workloads; Basis-joint is faster than Dense at **6/14**. A speedup is omitted wherever Dense did not complete.

This report covers the pure fixed-batch decode grid only. It is not a complete request/TTFT benchmark, capacity search, or profiler breakdown.

## Configuration

- Model: Llama-3.1-8B-Instruct, local snapshot `0e9e39f249a16976918f6564b8830bc894c89659`.
- Replica: TP=8, DP=1, PP=1 on 8x NVIDIA L40S (46,068 MiB each); PCIe-only topology with cross-socket `SYS` links and no NVLink.
- Software: PyTorch 2.13.0+cu130, CUDA 13.0, NCCL 2.29.7.
- Precision: BF16; TF32 disabled; eager execution.
- Basis route: V96 + Base16/Residual16, Page32, two-stage512, 62 routed pages + recent64 = hard physical support 2048, persistent exact-K GPU slots.
- Basis placement: V96/R16/metadata/K slots on GPU; historical exact K in pinned host memory. ALS-full and Dense keep their complete cache on GPU.
- Factor manifest SHA256: `a61d334e5f93e8a0d7a9ccbabc47099dce228b46964f9613df878fba3e0e7e2d`.
- Router layer-bank SHA256: `3c1523ed1fca0db608c7e47d74031f50a959f834538a1cc8e276e3c0f23d47c5`.
- Implementation commit recorded by the trials: `caf34fcbf111326895711e1c5d8605d28c5016b5` (the worktree was dirty; per-trial source hashes are retained in raw JSON).
- Command: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 conda run --no-capture-output -n basis python benchmarks/system/run_llama31_8b_tp8_decode_grid.py`.

## Median results

`GPU prefill` and `GPU resident` are max-rank PyTorch allocated bytes, not the sum across the replica. `Host K` is the sum of rank-private exact-K capacity.

### 4K

| B | Arm | Status | Mean ms | P50 ms | P95 ms | tokens/s | vs Dense | GPU prefill GiB | GPU resident GiB | Host K GiB | slot hit |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Dense | OK | 29.86 | 29.29 | 33.36 | 33.72 | 1.000x | 2.96 | 2.80 | 0.00 | - |
| 1 | ALS-full V96 | OK | 25.85 | 25.51 | 28.72 | 38.90 | 1.155x | 3.74 | 3.58 | 0.00 | - |
| 1 | Basis-joint V96+B16R16 | OK | 37.29 | 36.64 | 40.98 | 26.91 | 0.801x | 3.73 | 3.57 | 0.26 | 86.5% |
| 8 | Dense | OK | 30.95 | 30.66 | 33.53 | 259.33 | 1.000x | 4.51 | 3.26 | 0.00 | - |
| 8 | ALS-full V96 | OK | 26.93 | 26.45 | 30.67 | 298.06 | 1.149x | 5.31 | 4.06 | 0.00 | - |
| 8 | Basis-joint V96+B16R16 | OK | 40.55 | 39.18 | 44.79 | 197.71 | 0.763x | 5.26 | 4.01 | 2.07 | 96.8% |
| 32 | Dense | OK | 30.83 | 30.70 | 32.20 | 1039.26 | 1.000x | 9.90 | 4.89 | 0.00 | - |
| 32 | ALS-full V96 | OK | 27.00 | 26.96 | 27.98 | 1186.91 | 1.142x | 10.75 | 5.75 | 0.00 | - |
| 32 | Basis-joint V96+B16R16 | OK | 42.06 | 42.85 | 44.93 | 761.41 | 0.733x | 10.48 | 5.48 | 8.28 | 96.6% |
| 128 | Dense | OK | 39.32 | 39.29 | 39.70 | 3256.61 | 1.000x | 31.16 | 11.15 | 0.00 | - |
| 128 | ALS-full V96 | OK | 35.08 | 35.09 | 35.23 | 3649.97 | 1.121x | 32.38 | 12.38 | 0.00 | - |
| 128 | Basis-joint V96+B16R16 | OK | 41.42 | 41.23 | 43.08 | 3090.90 | 0.949x | 31.41 | 11.40 | 33.12 | 96.5% |

### 16K

| B | Arm | Status | Mean ms | P50 ms | P95 ms | tokens/s | vs Dense | GPU prefill GiB | GPU resident GiB | Host K GiB | slot hit |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Dense | OK | 29.77 | 29.28 | 32.94 | 33.70 | 1.000x | 3.43 | 2.99 | 0.00 | - |
| 1 | ALS-full V96 | OK | 26.58 | 26.14 | 29.38 | 37.65 | 1.120x | 4.19 | 3.75 | 0.00 | - |
| 1 | Basis-joint V96+B16R16 | OK | 37.94 | 37.20 | 41.29 | 26.48 | 0.785x | 4.11 | 3.66 | 1.01 | 78.2% |
| 4 | Dense | OK | 30.59 | 30.37 | 32.69 | 130.89 | 1.000x | 5.51 | 3.75 | 0.00 | - |
| 4 | ALS-full V96 | OK | 28.40 | 28.09 | 30.80 | 140.94 | 1.077x | 6.20 | 4.44 | 0.00 | - |
| 4 | Basis-joint V96+B16R16 | OK | 39.36 | 38.35 | 42.95 | 101.77 | 0.777x | 5.88 | 4.12 | 4.04 | 91.8% |
| 8 | Dense | OK | 30.76 | 30.65 | 32.86 | 260.28 | 1.000x | 8.27 | 4.76 | 0.00 | - |
| 8 | ALS-full V96 | OK | 27.83 | 27.69 | 28.24 | 287.52 | 1.106x | 8.89 | 5.38 | 0.00 | - |
| 8 | Basis-joint V96+B16R16 | OK | 39.64 | 38.29 | 43.28 | 202.05 | 0.776x | 8.26 | 4.75 | 8.07 | 93.6% |
| 16 | Dense | OK | 32.08 | 31.81 | 32.75 | 498.82 | 1.000x | 13.80 | 6.79 | 0.00 | - |
| 16 | ALS-full V96 | OK | 29.21 | 29.11 | 29.72 | 547.90 | 1.098x | 14.25 | 7.24 | 0.00 | - |
| 16 | Basis-joint V96+B16R16 | OK | 38.47 | 37.96 | 41.00 | 416.17 | 0.834x | 12.96 | 5.95 | 16.14 | 90.3% |

### 64K

| B | Arm | Status | Mean ms | P50 ms | P95 ms | tokens/s | vs Dense | GPU prefill GiB | GPU resident GiB | Host K GiB | slot hit |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Dense | OK | 85.63 | 85.66 | 86.17 | 11.68 | 1.000x | 5.34 | 3.74 | 0.00 | - |
| 1 | ALS-full V96 | OK | 75.00 | 75.01 | 75.34 | 13.33 | 1.142x | 6.02 | 4.41 | 0.00 | - |
| 1 | Basis-joint V96+B16R16 | OK | 37.03 | 36.69 | 39.09 | 27.07 | 2.313x | 5.63 | 4.03 | 4.01 | 70.3% |
| 4 | Dense | OK | 87.01 | 87.07 | 87.26 | 45.97 | 1.000x | 13.06 | 6.75 | 0.00 | - |
| 4 | ALS-full V96 | OK | 77.26 | 77.26 | 77.60 | 51.77 | 1.126x | 13.40 | 7.08 | 0.00 | - |
| 4 | Basis-joint V96+B16R16 | OK | 38.28 | 37.94 | 41.42 | 104.67 | 2.273x | 11.90 | 5.58 | 16.04 | 83.3% |
| 8 | Dense | OK | 87.42 | 87.47 | 87.81 | 91.51 | 1.000x | 23.35 | 10.76 | 0.00 | - |
| 8 | ALS-full V96 | OK | 76.72 | 76.71 | 76.89 | 104.28 | 1.140x | 23.24 | 10.64 | 0.00 | - |
| 8 | Basis-joint V96+B16R16 | OK | 39.86 | 38.27 | 43.73 | 201.00 | 2.193x | 20.18 | 7.59 | 32.07 | 70.0% |
| 16 | Dense | gpu_oom_prefill | - | - | - | - | - | - | - | - | - |
| 16 | ALS-full V96 | OK | 79.68 | 79.67 | 79.82 | 200.81 | - | 42.92 | 17.76 | 0.00 | - |
| 16 | Basis-joint V96+B16R16 | OK | 42.69 | 42.64 | 43.87 | 375.22 | - | 36.81 | 11.66 | 64.14 | 79.4% |

### ~128K (P=130048)

| B | Arm | Status | Mean ms | P50 ms | P95 ms | tokens/s | vs Dense | GPU prefill GiB | GPU resident GiB | Host K GiB | slot hit |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Dense | OK | 160.59 | 160.63 | 161.01 | 6.23 | 1.000x | 7.88 | 4.74 | 0.00 | - |
| 1 | ALS-full V96 | OK | 139.04 | 139.00 | 139.43 | 7.19 | 1.155x | 8.45 | 5.31 | 0.00 | - |
| 1 | Basis-joint V96+B16R16 | OK | 37.72 | 37.03 | 39.69 | 26.59 | 4.257x | 7.67 | 4.53 | 7.95 | 69.3% |
| 4 | Dense | OK | 162.19 | 162.25 | 162.61 | 24.66 | 1.000x | 23.12 | 10.74 | 0.00 | - |
| 4 | ALS-full V96 | OK | 141.57 | 141.56 | 141.87 | 28.26 | 1.146x | 22.99 | 10.62 | 0.00 | - |
| 4 | Basis-joint V96+B16R16 | OK | 38.01 | 37.86 | 39.04 | 105.42 | 4.267x | 19.87 | 7.50 | 31.79 | 81.2% |
| 8 | Dense | OK | 162.60 | 162.65 | 162.87 | 49.20 | 1.000x | 43.32 | 18.64 | 0.00 | - |
| 8 | ALS-full V96 | OK | 141.18 | 141.19 | 141.42 | 56.67 | 1.152x | 42.28 | 17.60 | 0.00 | - |
| 8 | Basis-joint V96+B16R16 | OK | 42.55 | 42.48 | 43.59 | 188.37 | 3.821x | 36.10 | 11.42 | 63.57 | 79.8% |
| 16 | Dense | gpu_oom_prefill | - | - | - | - | - | - | - | - | - |
| 16 | ALS-full V96 | gpu_oom_prefill | - | - | - | - | - | - | - | - | - |
| 16 | Basis-joint V96+B16R16 | gpu_oom_prefill | - | - | - | - | - | - | - | - | - |

## Capacity observations

- P=65536, B=16: Dense failed all three cohorts during prefill; ALS-full and Basis-joint completed all three. This is the clean measured point where Dense cannot run but both compressed paths can.
- P=130048, B=8: all three arms completed all cohorts.
- P=130048, B=16: all three arms failed during full-context prefill. The immediate allocation was a 15.88 GiB prompt/hidden-state tensor, so this identifies the current one-shot fixed-runner prefill peak, not a decode-cache-only limit.
- These are workload-grid observations, not Experiment C `B_max`: no integer capacity search or three-trial boundary verification was performed beyond the grid.

## Validation and caveats

- All 132 successful trials contain all eight rank JSON files. Synchronized 128-step timing arrays and generated token IDs are identical across ranks within each TP8 replica.
- Step latency is full-model replica latency and includes projections, routing, K-slot planning/fetch, attention, TP collectives, decoder, MLP, LM head, and greedy token choice. It is not divided by batch size.
- CPU affinity was bound to the GPU-local NUMA node, but `set_mempolicy` returned `Operation not permitted`; strict NUMA placement of pinned host pages was therefore not enforceable in this container.
- Host K is allocation capacity, not sampled physical RSS. NVML phase samples, active-cache byte itemization, host peak, and mean/P95/max unique selected-token counts were not collected by this runner. The slot metric is the weighted hit rate from the final measured step only.
- `prefill_cuda_ms` is retained in CSV/raw JSON for diagnostic context, but this Experiment B report does not present it as TTFT or request E2E.

## Artifacts

- `summary.csv`: one median row per arm/P/B (48 rows).
- `decode_trial_summary.csv`: all 144 cohort trials, including failure stage/log.
- `decode_grid_trials.json`: launcher commands, timestamps, and return codes.
- `formal_decode_*`: per-rank raw JSON, 128 step samples, token IDs, memory, and logs.
