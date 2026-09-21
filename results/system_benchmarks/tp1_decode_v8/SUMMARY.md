# TP1 sparse-decode v8: fused postprocess and vectorized K fetch

Complete request-level E2E results and per-component decode attribution are in
[E2E_AND_BREAKDOWN.md](E2E_AND_BREAKDOWN.md).

## Outcome

On one NVIDIA L40S, the optimized BasisKV path stays close to 30 ms/token from
16K through 128K. It crosses Dense-local by 32K and the paper-faithful
ShadowKV-default runtime between 32K and 64K.

| Context | BasisKV v8 | Dense local | ShadowKV default | Previous Basis K-offload | LRQK default |
|---:|---:|---:|---:|---:|---:|
| 16K | **29.964 ms** | 27.106 ms | 28.542 ms | 32.689 ms | 222.507 ms |
| 32K | **29.806 ms** | 30.081 ms | 28.798 ms | 34.260 ms | 226.200 ms |
| 64K | **30.636 ms** | 35.944 ms | 30.966 ms | 37.149 ms | 268.235 ms |
| 128K | **31.331 ms** | 47.473 ms | 33.055 ms | 43.592 ms | OOM |

At 16K and 32K, v8 latency is respectively 4.98% and 3.50% higher than
ShadowKV. At 64K and 128K, its throughput-equivalent speedup over ShadowKV is
1.011x and 1.055x. The corresponding speedup over Dense-local is 1.173x and
1.515x. Against the previous full-router, direct-PCIe Basis K-offload path, v8
speedup is 1.091x, 1.149x, 1.213x, and 1.391x from 16K through 128K.

These are full-model B1 decode measurements, not isolated attention-kernel
numbers. ShadowKV uses its paper-faithful rank160/chunk8 CPU-V runtime with
2,496 prompt support tokens; BasisKV uses B16R16 Page32 routing, 2,048 physical
support tokens, CPU-pinned exact K, Dense V128, and original Wo. The comparison
is therefore default-vs-default rather than matched-support.

### Prefill

Prefill was measured in the same fresh-process repeats. BasisKV v8 includes
exact FlashAttention prefill plus exact-K storage, Dense-V storage, B16R16 code
construction, and two-stage page-metadata construction. It does not defer a
cache-construction phase into decode.

| Context | Dense local | BasisKV v8 | ShadowKV default | ShadowKV placement | LRQK default |
|---:|---:|---:|---:|---:|---:|
| 16K | 2.337 s | **2.475 s** | 13.642 s | +0.525 s | 10.448 s |
| 32K | 4.906 s | **4.987 s** | 16.658 s | +0.606 s | 17.586 s |
| 64K | 12.562 s | **12.410 s** | 28.405 s | +2.581 s | 39.140 s |
| 128K | 36.759 s | **36.761 s** | 57.136 s | +1.310 s | not retained; decode OOM |

BasisKV v8 prefill stays within 5.92% of Dense-local at every context and is
effectively identical at 64K and 128K. Excluding ShadowKV's separate placement
step, v8 is 5.51x, 3.34x, 2.29x, and 1.55x faster from 16K through 128K.
Including placement only increases that advantage. LRQK completed 128K
prefill, but its process OOMed while restoring the official low-rank decode
state before it could persist the timing result.

ShadowKV's `batch_prefill` time and subsequent `H2D()` placement are reported
separately because that is how its official runtime exposes the two phases.
BasisKV and Dense build their decode-ready caches inside the reported prefill
interval and have no analogous post-prefill placement step.

### Current v8 scaling

| Context | CUDA median | Mean | P95 | Throughput | Peak GPU | Last-step K-slot hit |
|---:|---:|---:|---:|---:|---:|---:|
| 16K | 29.964 ms | 30.041 ms | 30.739 ms | 33.26 tok/s | 17.47 GiB | 60.45% |
| 32K | 29.806 ms | 29.858 ms | 30.327 ms | 33.47 tok/s | 19.74 GiB | 55.37% |
| 64K | 30.636 ms | 30.703 ms | 31.372 ms | 32.55 tok/s | 24.28 GiB | 52.89% |
| 128K | 31.331 ms | 31.374 ms | 31.983 ms | 31.86 tok/s | 33.43 GiB | 55.80% |

Each row pools two fresh-process repeats with 128 measured decode steps per
repeat after 16 warmup steps. The two repeats produced identical argmax token
sequences, all logits were finite, and all 32 layers passed the end-to-end
validation path at every context. From 16K to 128K, median latency grows by
only 4.56%.

### Improvement over v7

At the two contexts previously measured for v7, v8 reduces latency by about 2%
while preserving the existing two-stage router's selected pages exactly.

| Context | v7 median | v8 median | Median improvement | v7 p95 | v8 p95 | P95 improvement |
|---:|---:|---:|---:|---:|---:|---:|
| 64K | 31.265 ms | 30.636 ms | 0.629 ms (2.01%) | 31.942 ms | 31.372 ms | 0.570 ms (1.79%) |
| 128K | 31.944 ms | 31.331 ms | 0.613 ms (1.92%) | 32.588 ms | 31.983 ms | 0.605 ms (1.86%) |

## Changes

- Added an SM89-specialized CUDA postprocess kernel that performs fixed-prefix
  group-max page selection, candidate-to-physical-page mapping, and direct
  Page32/recent-token support packing in one launch.
- Reworked persistent K-slot refresh into a small planning kernel followed by a
  globally parallel 16-byte (`uint4`) mapped-host fetch kernel.
- Kept the resident cache fixed at the 2048-token attention budget. Experiments
  with 4096 and 8192 slots did not improve end-to-end latency and were removed.
- Added validation metrics for page-set recall and selected-score-mass recall
  against a full router scan.

The final 64K component profile (32 steps) reports:

| Component, whole 32-layer model | Median |
|---|---:|
| Router scan | 3.364 ms |
| Page selection and support packing | 0.415 ms |
| Persistent K planning and PCIe fetch | 2.168 ms |
| Selected sparse attention | 0.798 ms |
| Decode cache append | 0.357 ms |

The prior v7 profile measured about 0.99 ms for page selection/packing and
2.34 ms for K refresh. The fused postprocess therefore supplies most of the
repeatable end-to-end gain. A single-block-per-KV-head fused planning/fetch
prototype was rejected because it reduced PCIe copy parallelism and regressed
64K decode to 32.68 ms.

## Routing approximation audit

The fused postprocess consumes the same FP32 fine-router score tensor as the old
selector. It matched the old selector's integer page IDs exactly at candidate
counts 62, 63, 129, and 512, and the 32-layer benchmark validates this equality
before timing.

The existing 512-page coarse shortlist remains approximate relative to a full
scan:

| Context | Page-set recall, mean/min layer | Selected-score-mass recall, mean/min layer |
|---:|---:|---:|
| 16K | 100.00% / 100.00% | 100.00% / 100.00% |
| 32K | 98.64% / 92.94% | 99.16% / 93.80% |
| 64K | 95.00% / 85.89% | 97.30% / 87.27% |
| 128K | 91.34% / 80.85% | 94.86% / 83.17% |

These numbers are an audit of the pre-existing two-stage approximation, not a
quality-equivalence claim. The speed optimization does not add any further
selection difference. A separate paired six-prompt 128K RULER regression pilot
scored 83.33 for both the full router and v8, with no per-prompt score
regression; all 32 layers passed validation. That pilot is a regression gate,
not a replacement for a full downstream quality evaluation. The lowest
coverage occurs in a few early layers; if more quality margin is needed, the
next experiment should test larger shortlists only there rather than doubling
fine routing for every layer.

## Commands

Environment: `basis` conda environment; direct execution (no Slurm available).

Formal runs, with `<length>` set to `16384`, `32768`, `65536`, and `131072` and
`<repeat>` set to `0` and `1`:

```bash
CUDA_VISIBLE_DEVICES=<gpu> CUDA_HOME=/usr/local/cuda \
conda run --no-capture-output -n basis \
python benchmarks/system/bench_tp1_sparse_full_v6.py \
  --mode optimized --storage offload --routing two-stage --key-reuse \
  --length <length> --warmup-steps 16 --measure-steps 128 \
  --validate --repeat <repeat> --tag formal-two-stage-reuse \
  --output-root results/system_benchmarks/tp1_decode_v8
```

Component profile:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda \
conda run --no-capture-output -n basis \
python benchmarks/system/bench_tp1_sparse_full_v6.py \
  --mode optimized --storage offload --routing two-stage --key-reuse \
  --length 65536 --warmup-steps 8 --measure-steps 32 \
  --profile-components --validate --repeat 0 \
  --tag profile-two-stage-reuse \
  --output-root results/system_benchmarks/tp1_decode_v8
```

The raw `benchmark.json` and `run.log` files are stored beside this summary.
