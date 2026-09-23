# TP8 Decode Path Correctness and Optimization

Historical report: these measurements used two-stage routing. The active TP8
path subsequently removed coarse screening and the 512-candidate limit; see
[full-scan routing](tp8_full_scan_routing.md). Do not attribute the timings below
to the new full-scan path. The original result JSONs and
`results/system_benchmarks/tp8_path_opt/source.patch` remain unchanged.

Date: 2026-09-23 UTC. Baseline source: commit `bae46c1`.
Environment: conda `basis`, PyTorch `2.13.0+cu130`, NVIDIA L40S.
Work directory: `/workspace/BasisServe-CALS-opt`.
Hardware topology: [L40S TP8 machine](hardware/l40s_tp8_topology.md).

## Changes

- Respect the output tensor's batch, head, and feature strides in the
  slot-attention merge kernel. This supports the feature-major AllGather
  arena directly, without a transpose/copy kernel.
- Allocate coarse-score storage once. Reuse the contiguous output view
  within a page, and rebuild only the view when the page count changes.
- When there are at most 512 pages, skip coarse scoring and reuse ascending
  candidate IDs until the page count changes. All pages already survived
  the original candidate selection at these lengths.
- Keep the existing fine router, selected support, precision, and TP8
  communication policy.
- Use one shared fine-router build directory, compiled by rank 0 before
  other ranks load it. This removes rank-dependent source paths from the
  extension cache and reduces repeated startup compilation; it does not
  change measured decode operations.
- Add batch-subset and component-profile options to the grid launcher.
  Record commands and source diffs without computing source or input hashes.

## Correctness

The original merge kernel assumed contiguous output. For feature-major
output, the new numerical test failed at B=2 and B=8 before the fix:
743/768 and 2968/3072 elements respectively exceeded the test tolerance.
Maximum absolute errors were 0.18550 and 0.20124. B=1 and all contiguous
output cases passed before the fix.

The corrected path passes numerical comparison against selected dense
attention, including invalid token IDs, invalid slots, empty splits,
nonzero output offsets, and guard regions around the output arena.
Routing tests check exact candidate-ID agreement across the 512-page
boundary and coarse-score correctness after page growth.
The final regression run passed all 16 cases in 7.66 seconds.

The TP8 integration smoke checks actual prepared uniform NCCL AllGather
and a `3072 -> 4096` decoder following slot attention:

| Batch | Status | Decoder maximum absolute error across ranks |
|---:|---|---:|
| 1 | Passed | 0 |
| 2 | Passed | 0.00048828125 |
| 8 | Passed | 0.00048828125 |

Historical B>1 Basis-joint generation and quality results from the affected
layout must be revalidated. Cross-rank agreement alone cannot detect this
layout error. Archived reports are retained as historical artifacts.

## Candidate-Path Measurements

Single GPU, synthetic BF16 inputs, median of seven trials of 64 calls.
The cache length is prompt length + 17, corresponding to the first measured
position after 16 conditioning steps. The baseline reproduces the original
score allocation, coarse kernel, and candidate-selection operations.
All eight workloads produced exactly the same candidate IDs.

| Prompt | Batch | Cache pages | Before wall us/call | After wall us/call |
|---:|---:|---:|---:|---:|
| 4096 | 1 | 129 | 18.98 | 0.69 |
| 4096 | 8 | 129 | 25.31 | 0.68 |
| 16384 | 1 | 513 | 21.85 | 20.97 |
| 16384 | 8 | 513 | 21.26 | 20.80 |
| 65536 | 1 | 2049 | 21.45 | 20.99 |
| 65536 | 8 | 2049 | 23.16 | 23.07 |
| 130048 | 1 | 4065 | 21.84 | 21.30 |
| 130048 | 8 | 4065 | 29.63 | 29.62 |

These are eager candidate-path timings, including Python/launch overhead
and synchronization amortized over each trial. CUDA-event samples retained
in JSON also include stream idle time between launches. The short path
returns cached IDs and performs no GPU work at steady state, so its sub-us
measurement is mostly host overhead, not a GPU kernel latency.

Long-context differences are small and do not establish a robust speedup.
Measurements exclude page-boundary updates, fine routing, K fetch,
attention, output projection, and communication. These microbenchmarks do
not establish a full-model speedup. A 16K prompt exceeds the 512-page
threshold once decode begins. The separate full-model pilot is below.

## Commands

Regression tests:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python -m pytest tests/test_slot_indexed_attention.py tests/test_tp8_candidate_path.py -q
```

Candidate-path measurement:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp8_candidate_path.py \
  --lengths 4096 16384 65536 130048 --batches 1 8 \
  --iterations 64 --trials 7 --decode-offset 17 \
  --output results/system_benchmarks/tp8_path_opt/candidates_decode.json
```

TP8 integration smoke:

```bash
CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
  /workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  torchrun --standalone --nproc-per-node=8 tests/distributed_tp8_slot_output_smoke.py
```

The first integration launch failed before computation because `CUDA_HOME`
was unset. Setting it to the installed toolkit resolved the failure.
Both attempts remain in `results/system_benchmarks/tp8_path_opt/tp8_output.log`.
The toolkit at `/usr/local/cuda` reports nvcc 12.8; the PyTorch build reports
CUDA 13.0. The integration smoke passed in this environment.

All jobs ran directly on this machine as requested. No model training,
factor fitting, full decode grid, or SHA256 verification was run.

## Full-Model Pilot

Results and validation details: [pilot summary](../results/system_benchmarks/tp8_path_opt/SUMMARY.md).
The pilot uses cohort 0 only, not the complete three-cohort matrix.
Dense, ALS-full, and corrected Basis-joint use the same prompts, greedy
decode loop, precision, and measurement settings. This compares the arms;
it does not isolate the incremental effect of the candidate-path change.

All launcher commands below use this environment prefix:

```bash
CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis
```

Full-model smoke:

```bash
python benchmarks/system/run_llama31_8b_tp8_decode_grid.py \
  --arms basis_joint --contexts 4096 65536 --batches 1 8 --cohorts 0 \
  --conditioning-steps 4 --measure-steps 8 --tag smoke \
  --output-root results/system_benchmarks/tp8_path_opt/smoke
```

Uninstrumented timing:

```bash
python benchmarks/system/run_llama31_8b_tp8_decode_grid.py \
  --arms dense als_full basis_joint --contexts 4096 16384 65536 130048 \
  --batches 1 8 --cohorts 0 --conditioning-steps 16 --measure-steps 128 \
  --tag timing --output-root results/system_benchmarks/tp8_path_opt/timing
```

Separate component profiling:

```bash
python benchmarks/system/run_llama31_8b_tp8_decode_grid.py \
  --arms basis_joint --contexts 4096 16384 65536 130048 --batches 1 8 \
  --cohorts 0 --conditioning-steps 16 --measure-steps 16 \
  --tag profile --profile-components \
  --output-root results/system_benchmarks/tp8_path_opt/profile
```

Each phase retains its top-level log, per-trial launcher logs, per-rank
logs/JSON, and a manifest containing exact torchrun commands and exit codes.
The smoke preceded the shared-build-directory startup fix; timing and
profiling use that fix. Numerical decode operations are the same.

Summary generation:

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/summarize_tp8_path_run.py \
  --root results/system_benchmarks/tp8_path_opt
```

Component CUDA events include eager launch gaps and collective arrival
skew; they are not isolated kernel or PCIe transfer timings. The parent
`attention_total` overlaps its subcomponents. Instrumented latency must
not be substituted into the uninstrumented speed table.

CUDA Graph work must handle evolving cache length, page count, metadata
position, ring slot, and input-buffer addresses. Capturing the current
Python-stateful forward unchanged would freeze some launch arguments.
Any graph implementation needs multi-step reference checks across page
boundaries and ring wraparound before full-model performance claims.

## Pilot Findings and Next Priority

All 4 full-model smoke, 24 uninstrumented timing, and 8 component-profile
trials completed. All 288 rank result files have complete measurements
and matching generated tokens across ranks. The final regression run,
including two new stream-timing cases, passed 18 tests in 8.50 seconds.

At B=8, corrected Basis-joint latency is 25.13, 26.02, 25.78, and 28.48 ms
at 4K, 16K, 64K, and 130048 prompt tokens. The corresponding speed ratios
versus Dense are 1.225x, 1.197x, 3.390x, and 5.710x. At 4K, ALS-full is
still faster (20.93 ms). At 130048/B=8, Basis uses 11.59 GiB GPU-resident
memory per rank versus Dense's 18.64 GiB, while retaining 63.57 GiB of
exact K in host memory across all eight ranks.

At 130048/B=8, instrumented rank-0 per-step intervals are 0.799 ms for
sparse attention, 2.561 ms for fine routing, 2.409 ms for postprocessing,
14.404 ms for output AllGather, and 10.659 ms for the MLP block. The latter
includes its existing TP operations. These intervals include launch gaps
and must not be treated as isolated device-kernel durations.

The existing standalone AllGather benchmark had a stream mismatch: events
could be recorded on the capture stream while graph replay launched on
the caller's current stream. The helper now runs warmup, replay, and
timing events on the requested stream and restores the caller's stream.
Tests cover explicit and inherited streams.

After that fix, short TP8 collective smoke measurements at local width
384 BF16 produced these median latencies:

| Batch | NCCL us | NCCL single-collective graph us | IPC auto us |
|---:|---:|---:|---:|
| 1 | 18.16 | 18.35 | 23.36 |
| 8 | 35.38 | 27.71 | 48.37 |

All six backend/batch cases passed exact gathered-output comparison.
These use 20 warmups and 100 measurements, without the full-model rank
affinity settings. They are diagnostic smoke results, not a repeated
backend-selection study. IPC auto was not faster in these samples.
A graph containing one collective does not establish the benefit of
capturing a whole model block.

Next work should prioritize CPU/GPU timeline attribution around the
output collective and MLP, followed by static-block launch reduction or
graph capture. Keep the existing NCCL backend until a matched end-to-end
comparison justifies changing it. Long-context fine routing and host-K
refresh are secondary targets; the sparse-attention kernel is not the
largest observed interval.

Generation trajectories diverged between B=1 and B=8 in 8 of 12
arm/context comparisons, including Dense. This is not by itself proof of
a kernel error or of quality equivalence. Teacher-forced logit comparison
and task-quality evaluation remain necessary before quality claims.

Additional regression command:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python -m pytest tests/test_slot_indexed_attention.py tests/test_tp8_candidate_path.py \
  tests/test_uniform_allgather_timing.py -q
```

Collective smoke commands use the same CUDA/conda environment prefix as
the full-model launcher, with a 120-second timeout for each invocation:

```bash
torchrun --standalone --nproc-per-node=8 benchmarks/bench_uniform_allgather.py \
  --local-width 384 --tokens 1 --dtype bfloat16 \
  --backends uniform_nccl,uniform_nccl_graph,uniform_ipc --warmup 20 --iters 100 \
  --output-json results/system_benchmarks/tp8_path_opt/collective_b1.json

torchrun --standalone --nproc-per-node=8 benchmarks/bench_uniform_allgather.py \
  --local-width 384 --tokens 8 --dtype bfloat16 \
  --backends uniform_nccl,uniform_nccl_graph,uniform_ipc --warmup 20 --iters 100 \
  --output-json results/system_benchmarks/tp8_path_opt/collective_b8.json
```

Logs: `tests_pilot.log`, `collective_b1.log`, and `collective_b8.log` in the
pilot result directory. Detailed metrics and generation samples are in
the linked pilot summary and its CSV/JSON companions.
