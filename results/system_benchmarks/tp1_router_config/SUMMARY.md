# TP1 full-scan router execution-configuration experiment

## Outcome

A 16-warp block is a promising, modest GPU-local decode improvement over the
current 8-warp block. It reduces pooled full-model CUDA-event median latency
by 1.13% at 64K and 1.49% at 128K in this experiment. Eight paired model trials
completed successfully. The candidate remains isolated; the production kernel
and preceding frozen results were not overwritten. No offload speedup or
downstream quality result is claimed.

Environment: `basis`, one NVIDIA L40S (GPU 0), TP1/B1, BF16, TF32 disabled,
PyTorch 2.13.0+cu130, NVCC 12.8.93, SM89. Direct local execution followed the
user's authorization. Hardware performance counters remain unavailable;
these are CUDA-event timings, not Nsight counter measurements.

## Scope and Candidates

Sources come from `../tp1_sparse_local/frozen_source/`. Model:
Llama-3.1-8B-Instruct, Dense V128, B16R16, Page32, full-scan routing,
2,048-token support. K/V stay on GPU. MLP and all serving policies are unchanged.

The [earlier register-router experiment](../register_router/RESULTS.md) already
found 8 warps preferable to 4; that comparison was not repeated. The current
baseline already contains the register-consumed MMA implementation.

| Name | Warps/block | Threads/block | Pages/block | Feature-loop unroll |
|:---|---:|---:|---:|---:|
| w8u1, baseline | 8 | 256 | 4 | 1 |
| w16u1 | 16 | 512 | 8 | 1 |
| w8u2 | 8 | 256 | 4 | 2 |

`w16u1` changes only the two matching warp-count constants in the kernel body
and launch site. `w8u2` changes only the feature-loop unroll pragma. All pages
are still scored. Neither candidate changes BF16 rounding boundaries, scoring,
selection, support, rank, or cache capacity. This is not two-stage routing.

Compiler output reports 48 registers/thread and zero spill loads/stores for
all three configurations. Static shared memory is 11,648 bytes for w8u1/w8u2
and 17,792 bytes for w16u1. Fewer, larger blocks change execution organization;
the specific hardware bottleneck responsible for the observed improvement has
not been established without counters.

## Synthetic Screening

Three seeds (127, 311, 509), lengths 1/31/32/33/127/128/129/255/256/257/2047/
4096/16384/65536/131072. Short cases include batch 2 and three KV heads;
long cases use batch 1 and eight KV heads. All 90 candidate checks had
**exactly equal FP32 page scores and selected page IDs** versus w8u1.
An earlier short smoke completed 72 checks, also exact.

Timings use seed 127, preallocated outputs, 20 warmup calls, 200 calls per
sample and three ABBA rounds per candidate/context (six samples per arm).
Each call includes residual-query projection and full router scoring, not
selection or attention. These are warmed, synthetic, single-layer timings.

| Tokens | Candidate | Matched baseline ms | Candidate ms | Latency reduction |
|---:|:---|---:|---:|---:|
| 16,384 | w16u1 | 0.062994 | 0.057962 | 7.99% |
| 65,536 | w16u1 | 0.208289 | 0.187478 | 9.99% |
| 131,072 | w16u1 | 0.410682 | 0.384906 | 6.28% |
| 16,384 | w8u2 | 0.062886 | 0.061936 | 1.51% |
| 65,536 | w8u2 | 0.208097 | 0.202571 | 2.66% |
| 131,072 | w8u2 | 0.409461 | 0.402448 | 1.71% |

Only w16u1 advanced to model testing. w8u2 is not rejected for correctness;
it had a smaller screening benefit and was not further evaluated.

## Paired Full-Model Decode

Per context: baseline r0, candidate r0, candidate r1, baseline r1, each in a
fresh process on GPU 0. Each process runs 16 warmup + 128 measured steps.
Component profiling is off. Both arms use the same validation wrapper:
the first decode call of each of 32 layers checks router scores exactly
against w8u1, followed by the harness's selector/support and attention checks.
These numerical checks occur during warmup, not the measured 128 steps.

| Context | Baseline pooled p50 ms | Candidate pooled p50 ms | Saved ms/token | Reduction |
|---:|---:|---:|---:|---:|
| 65,536 | 32.163 | 31.800 | 0.363 | 1.13% |
| 131,072 | 38.448 | 37.877 | 0.571 | 1.49% |

Pooled p50 is the median of 256 measured steps from two repeats, not the mean
of two per-run medians. The per-run medians expose run-to-run variation:

| Context | Baseline r0 | Candidate r0 | Candidate r1 | Baseline r1 |
|---:|---:|---:|---:|---:|
| 65,536 | 32.154320 | 31.796480 | 31.806560 | 32.230463 |
| 131,072 | 38.427967 | 37.900721 | 37.858751 | 38.520334 |

Both candidate repeat medians beat both baseline repeat medians at each
context. This is a small improvement on one machine and prompt, not a broad
statistical claim. Do not replace it with the larger microbenchmark percentage.

All eight trials have finite logits, 32/32 exact router-score audits, and
32/32 attention validations. All 144 recorded argmax tokens (including warmup)
match the first baseline at the same context, for both arms and both repeats.
A preceding 4K candidate model smoke also passed all 32 layers.

These are full-model steady, teacher-forced decode timings on one frozen
prompt. Greedy token selection is outside the timed interval. This is not
request E2E or the final three-cohort greedy protocol. Context 128K here means
131,072, not 130,048. No new RULER evaluation, TP8 measurement, offload trial,
or full-model w8u2 trial was run.

## Reproduction

Run from the repository root. Every GPU command below uses:

```bash
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9
```

Synthetic smoke and screening:

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_router_config.py --phase micro --smoke \
  > results/system_benchmarks/tp1_router_config/smoke.log 2>&1
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_router_config.py --phase micro \
  > results/system_benchmarks/tp1_router_config/micro.log 2>&1
```

Model smoke:

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_router_config.py --phase decode --config w16u1 \
  --mode optimized --storage local --routing full --length 4096 \
  --warmup-steps 1 --measure-steps 2 --validate --tag config-smoke \
  --output-root results/system_benchmarks/tp1_router_config \
  > results/system_benchmarks/tp1_router_config/decode_smoke.log 2>&1
```

Eight paired model trials:

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_router_config.py --phase paired --config w16u1 \
  > results/system_benchmarks/tp1_router_config/paired.log 2>&1
```

Artifacts: `source/` preserves generated CUDA/C++ variants; `smoke.log` contains
compiler resource reports; `micro.json` includes correctness and raw timing
samples; `paired.json` records all eight outcomes; `decode_summary.json`
contains pooled/per-repeat metrics and output checks. `decode/` retains exact
child commands, launcher logs, per-step logs and raw benchmark JSON.
No SHA256 checks were performed. Nothing from this experiment has been uploaded
or committed.
