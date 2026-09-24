# TP1 K-slot refresh: bounded optimization experiment

## Decision

**Stop this optimization round and retain the existing production refresh.**
Neither candidate provided a meaningful, consistent microbenchmark benefit.
No candidate advanced to a full-model run, and no production kernel, routing,
cache budget, MLP, or preceding frozen result was changed. The earlier 16-warp
router candidate is a separate experiment and was not combined with this test.

This is a negative result for two specific implementation changes, not proof
that the existing implementation has reached a theoretical limit.

## Setup

- Environment: `basis`, one NVIDIA L40S, physical GPU 0, SM89; NVCC 12.8.93,
  PyTorch 2.13.0+cu130.
- Benchmark source: `benchmarks/system/bench_tp1_slot_refresh.py`.
- Baseline: the exact `persistent_key_slots.cu` from
  `../tp1_sparse_local/frozen_source/`, with diagnostic plan-only/fetch-only
  entry points added identically to the candidates. No source hashes used.
- Timed shape: batch 1, eight KV heads, 2,048 K slots/head, 128 BF16 values/K.
  CPU K uses the existing mapped pinned-host allocator. K values remain exact.
- Cache capacities: 65,536 and 131,072 tokens. Sorted Page32 supports are spread
  across the context; two alternating supports give controlled steady hit
  fractions of 0%, 50%, 75%, or 100%.
- GPU 0 was idle before testing. Runs were direct and serial under the user's
  prior authorization; no Slurm allocation was used.

## Candidates

1. **warp:** aggregate the slot planner's counters/reservations at warp level,
   retaining hit slots and assigning misses to available slots. Physical free
   slot order can differ from the atomic-order baseline. The lookup/resident
   validity checks and reuse rules remain the same; selected tokens do not
   change. Fetch remains the existing 256-thread `uint4` kernel.
2. **fetch128:** keep the planner unchanged and use 128 rather than 256 threads
   per fetch block. Vector width and total selected-token work remain unchanged.

The previous scalar-to-vector improvement and rejected single-block fused
planning/fetch prototype were not repeated. No new eviction policy, enlarged
slot cache, approximate routing, or speculative prefetch was introduced.

## Correctness

Both smoke and main microbenchmark completed 180 stateful checks each:
three implementations, five budgets (1/31/33/127/2048), 12 sequential steps,
batch 2 and three KV heads. Cases cover cold cache, overlapping/revisited
supports, negative and out-of-range IDs, partial budgets, and reuse disabled.

Checks verify:

- Hit/valid counts against the implementation's pre-refresh lookup/resident
  state, and preservation of already-hit slot assignments.
- The exact miss mask, invalid slot handling and resident token IDs.
- Every valid selected K against the mapped CPU source with zero tolerance.

All passed. Physical slot IDs are intentionally not compared across variants.
For partial-valid supports, arbitrary free-slot assignment can leave different
unrequested residents in otherwise spare slots; the tests validate each
implementation's state, not universal cross-variant hit-trace equivalence.
The timed supports are fully valid and have matched, checked hit fractions.

## Timing Method

The 4K smoke used eager CUDA-event timing with 20 calls/sample. It showed no
useful refresh improvement. The final microbenchmark uses CUDA Graph replay to
reduce Python-launch gaps for the short planning kernels: 20 eager warmups,
capture 100 calls, then time one graph replay with CUDA events. Each candidate
and phase uses three ABBA rounds, giving six samples per arm.

Plan and full-refresh calls alternate the two supports. Fetch-only replays a
fixed missing-token/slot list prepared by two initial refreshes. Buffers and
input addresses remain stable. This is warmed synthetic microbenchmarking,
not a model decode measurement. CUDA Graph was used only for measurement;
it was not added to the serving path. Independent component medians need not
sum exactly to the full-refresh median.

## Results

Full-refresh latency, microseconds per layer across eight KV heads. Each
candidate has its own matched baseline. Positive change means lower latency.

| Capacity | Hit rate | Warp baseline us | Warp candidate us | Reduction | Fetch128 baseline us | Fetch128 candidate us | Reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 65,536 | 0% | 165.181 | 165.637 | -0.276% | 165.173 | 165.136 | +0.023% |
| 65,536 | 50% | 87.169 | 87.540 | -0.426% | 87.251 | 87.205 | +0.052% |
| 65,536 | 75% | 48.241 | 48.359 | -0.244% | 48.239 | 48.209 | +0.060% |
| 65,536 | 100% | 8.796 | 8.869 | -0.835% | 8.817 | 9.364 | -6.202% |
| 131,072 | 0% | 165.317 | 165.643 | -0.197% | 165.225 | 165.149 | +0.046% |
| 131,072 | 50% | 87.165 | 87.418 | -0.290% | 87.177 | 87.201 | -0.028% |
| 131,072 | 75% | 48.230 | 48.348 | -0.245% | 48.246 | 48.213 | +0.068% |
| 131,072 | 100% | 8.817 | 8.865 | -0.552% | 8.804 | 9.370 | -6.432% |

At 50% hits, separate baseline component medians from the warp comparison are:

| Capacity | Plan us | Fetch us | Full refresh us |
|---:|---:|---:|---:|
| 65,536 | 7.330 | 79.896 | 87.169 |
| 131,072 | 7.351 | 79.892 | 87.165 |

Fetch accounts for roughly 92% of the measured refresh duration in this
synthetic 50%-hit case. Warp planning itself is about 3% slower than baseline
here. Fetch128's sub-0.1% non-all-hit changes are not a meaningful improvement,
and its all-hit case regresses. Therefore neither candidate merits promotion
or a further model trial in this bounded round.

Hardware counter access remains unavailable. These results do not measure
PCIe bus utilization, achieved occupancy, or stall causes, and cannot prove
bandwidth saturation. The preceding model's roughly 2.2 ms/token refresh
profile is a separate measurement with real, varying hit patterns; the
synthetic per-layer numbers above should not replace it.

## Commands and Artifacts

Run from the repository root with:

```bash
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9
```

Smoke:

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_slot_refresh.py --phase micro --smoke \
  > results/system_benchmarks/tp1_slot_refresh/smoke.log 2>&1
```

Final microbenchmark:

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_slot_refresh.py --phase micro --cuda-graph \
  > results/system_benchmarks/tp1_slot_refresh/micro.log 2>&1
```

`smoke.json` and `micro.json` retain all raw timing samples and check counts.
`smoke.log` includes compiler resource reports; `source/` retains all three
generated C++/CUDA variants. The candidate planner is
`benchmarks/system/warp_slot_plan.cuh`. Python syntax and scoped whitespace
checks passed. No full-model trial, quality evaluation, or new throughput
claim was produced. All artifacts remain local, uncommitted and not uploaded.
