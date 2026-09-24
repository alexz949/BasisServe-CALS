# TP1 full-scan K-offload: 8-warp versus 16-warp router

## Outcome

All eight paired trials completed successfully. The 16-warp router reduces
pooled full-model steady decode CUDA-event median latency by **1.05% at 64K**
and **1.44% at 128K**. Both candidate repeat medians are below both baseline
repeat medians at each context. GPU memory allocation is unchanged.

This extends the [earlier GPU-local result](../SUMMARY.md) to K-offload.
It is a modest, narrow-workload improvement, not a claim of universal speedup
or statistical significance across prompts. The candidate remains isolated;
production source and previous results have not been overwritten. No further
optimization experiments were launched after this run.

## Configuration

- Environment: `basis`, NVIDIA L40S on physical GPU 0, TP1/B1, BF16, TF32 off.
  PyTorch 2.13.0+cu130; extension compiler NVCC 12.8.93, SM89.
- Model: Llama-3.1-8B-Instruct; Dense V128, full B16R16 Page32 routing,
  2,048-token support. No two-stage shortlist and no MLP changes.
- Storage: historical exact K in mapped pinned CPU memory, Dense V on GPU,
  existing persistent 2,048-token GPU K-slot cache with reuse enabled.
- Only experimental change: router block configuration, from 8 warps / 4 pages
  per block to 16 warps / 8 pages per block. All eligible pages are still scored.
- K-slot refresh is the preceding frozen implementation. Neither rejected
  candidate from the [slot-refresh experiment](../../tp1_slot_refresh/SUMMARY.md)
  is used here.
- Each context runs baseline r0 / candidate r0 / candidate r1 / baseline r1,
  sequentially, with a fresh process per trial. Direct local execution follows
  the user's authorization; no Slurm allocation was used.
- Each process runs 16 warmup + 128 measured teacher-forced steps. Component
  instrumentation is disabled. CUDA events and synchronized wall timing remain.

## Main Results

Pooled p50 combines 256 measured steps from two repeats; it is not the mean
of the two per-run medians.

| Context | 8-warp p50 ms/token | 16-warp p50 ms/token | Saved ms/token | Reduction | Peak GPU GiB, both arms |
|---:|---:|---:|---:|---:|---:|
| 65,536 | 34.232 | 33.871 | 0.361 | 1.05% | 24.033 |
| 131,072 | 40.615 | 40.031 | 0.584 | 1.44% | 32.939 |

Per-trial medians, in execution order:

| Context | Baseline r0 | Candidate r0 | Candidate r1 | Baseline r1 |
|---:|---:|---:|---:|---:|
| 65,536 | 34.214928 | 33.890335 | 33.837440 | 34.248671 |
| 131,072 | 40.699936 | 39.957201 | 40.238672 | 40.553202 |

The 128K candidate repeats differ by about 0.28 ms. Both are reported;
the faster repeat is not used alone to calculate the main improvement.
Peak memory means PyTorch peak allocated GPU memory across the benchmark,
not KV-only memory or total device usage.

## Correctness and Cache Observations

All eight trials have finite logits, 32/32 exact router-score audits, and
32/32 attention validations. On the first warmup decode step, each layer's
router output is compared exactly against the 8-warp reference. The harness
also checks full-scan selection/support packing and selected attention against
its reference. The checks occur in warmup; both arms use the same wrapper.

At each context, all 144 recorded argmax tokens (warmup plus measurement)
match the first baseline for both arms and both repeats. This does not claim
bitwise equality of all logits or a downstream quality evaluation.

Last-step K-slot hit percentages are close, but not identical:

| Context | Baseline r0 | Candidate r0 | Candidate r1 | Baseline r1 |
|---:|---:|---:|---:|---:|
| 65,536 | 52.81098% | 52.79234% | 52.78965% | 52.80099% |
| 131,072 | 55.75734% | 55.75427% | 55.74238% | 55.76982% |

The spread is below 0.03 percentage points at either context, and baseline
repeats also differ. The unchanged slot planner uses atomic reservations, so
physical free-slot assignment order is not fixed. This is a plausible source
of small hit-rate differences when spare residents remain, but this experiment
did not collect a full per-step cache trace to establish the cause. Do not
describe cache behavior as bitwise identical. The reported hit values are for
the last step, not average hit rates across the measured sequence.

## Scope

These are full-model steady decode timings on one frozen teacher-forced prompt,
not request E2E or the final three-cohort greedy protocol. Greedy argmax
selection is outside the timed interval. The longer context is 131,072 rather
than 130,048. Dense, external baselines, and downstream quality were not rerun.
No hardware-counter result or explanation of achieved SM utilization is claimed.

## Commands and Artifacts

Run from the repository root, using:

```bash
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9
```

Preceding 4K smoke (all 32 layers passed):

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_router_config.py --phase decode --config w16u1 \
  --mode optimized --storage offload --routing full --key-reuse \
  --length 4096 --warmup-steps 1 --measure-steps 2 --validate \
  --tag offload-smoke --output-root results/system_benchmarks/tp1_router_config/offload \
  > results/system_benchmarks/tp1_router_config/offload/smoke.log 2>&1
```

Eight paired trials:

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_router_config.py --phase paired \
  --config w16u1 --basis-storage offload \
  > results/system_benchmarks/tp1_router_config/offload/paired.log 2>&1
```

The benchmark reuses `../../tp1_sparse_local/frozen_source/` and the router
variants preserved in `../source/w8u1/` and `../source/w16u1/`. Model/factor/token
paths are recorded in each raw result. No SHA256 checks were performed.

- `paired.json`: all eight outcomes and raw-result paths.
- `decode_summary.json`: pooled and per-run medians, peak memory, last-step
  hit fractions, validation flags, and argmax comparisons.
- `decode/*.command.json` and `decode/*.log`: exact child commands and launcher
  logs, including all 32 layer audits per trial.
- `decode/*/benchmark.json` and `decode/*/run.log`: complete raw results and
  per-step output/timing logs.

All artifacts are local; nothing from this run has been committed or uploaded.
