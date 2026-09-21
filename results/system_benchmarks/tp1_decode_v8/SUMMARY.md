# TP1 sparse-decode v8: fused postprocess and vectorized K fetch

## Outcome

On one NVIDIA L40S, v8 reduces the full Llama-3.1-8B TP1 teacher-forced decode
latency by about 2% relative to v7 while preserving the existing two-stage
router's selected pages exactly.

| Context | v7 median | v8 median | Median improvement | v7 p95 | v8 p95 | P95 improvement |
|---:|---:|---:|---:|---:|---:|---:|
| 64K | 31.265 ms | 30.636 ms | 0.629 ms (2.01%) | 31.942 ms | 31.372 ms | 0.570 ms (1.79%) |
| 128K | 31.944 ms | 31.331 ms | 0.613 ms (1.92%) | 32.588 ms | 31.983 ms | 0.605 ms (1.86%) |

Each v8 row pools two independent repeats with 128 measured decode steps per
repeat after 16 warmup steps. The two repeats produced identical argmax token
sequences, all logits were finite, and all 32 layers passed the end-to-end
validation path.

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
| 64K | 95.00% / 85.89% | 97.30% / 87.27% |
| 128K | 91.34% / 80.85% | 94.86% / 83.17% |

These numbers are an audit of the pre-existing two-stage approximation, not a
quality-equivalence claim. The speed optimization does not add any further
selection difference, but RULER or another downstream evaluation is still
required before claiming model-quality parity. The lowest coverage occurs in a
few early layers; a future quality/speed experiment should test larger
shortlists only there rather than doubling fine routing for every layer.

## Commands

Environment: `basis` conda environment; direct execution (no Slurm available).

Formal runs, with `<length>` set to `65536` and `131072` and `<repeat>` set to
`0` and `1`:

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
