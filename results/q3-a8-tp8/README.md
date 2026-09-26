# Fused Latent A8 and Real TP8 Decoder Benchmark

Environment: `basis`, eight NVIDIA L40S GPUs on one node, TP8 across both NUMA
nodes. This experiment measures actual NCCL transport plus a replicated global
C1 decoder. It does not run the full model, the encoder, attention, cache I/O,
MLP, or a serving scheduler in the measured region. No packed NUQ4 deployment
or full-model speedup is claimed.

## Formal Outcome

The approved grid completed with exit code zero. All 216 round/execution
records and 151200 critical-rank latency samples passed integrity checks.
There are 72 summarized layer/row/execution cases. Captured source files match
the tested files byte-for-byte. Six kernel tests passed; every distributed
configuration validated payloads and fused/unfused decoder outputs on all ranks.

The table below is **the ratio of summed median boundary latencies over the
three tested layers**, not a full-model speedup. Each layer first takes the
median of three round medians. All tested layers, including slow ones, are kept.

| Rows | R64 eager | R64 graph | R96 eager | R96 graph |
|---:|---:|---:|---:|---:|
| 1 | 0.369x | 1.027x | 0.639x | 1.852x |
| 4 | 0.516x | 1.094x | 0.674x | 1.622x |
| 16 | 0.556x | 1.316x | 0.704x | 1.494x |
| 64 | 0.923x | 1.224x | 1.334x | 1.852x |
| 128 | 1.208x | 1.534x | 1.374x | 1.480x |
| 256 | 1.301x | 1.389x | 1.391x | 1.400x |

These compare the complete fused A8/W8 pipeline against the matched BF16
boundary, including send packing in both. Graph execution is important for
small inputs; the eager FP8 path has extra launch/dispatch overhead and does
not consistently beat BF16. Do not extrapolate these ratios to full-model
speed or continuous batching.

Examples from layer 18, using CUDA Graph and including communication:

| Configuration | BF16 ms | Fused A8/W8 ms | Speedup vs BF16 | Speedup vs unfused A8/W8 |
|---|---:|---:|---:|---:|
| R64 / 128 rows | 0.097200 | 0.061680 | 1.576x | 1.171x |
| R96 / 64 rows | 0.113152 | 0.060704 | 1.864x | 1.133x |

Not every fused case improves over the unfused path. In particular R64 layer
35 graph results at rows 16/64 show slower fused totals despite the general
trend. R64 layer 35 / graph / rows 4 also has >25% max/min variation across
the three BF16 trial medians. First small-row eager BF16 trials for R64 layer
0 and R96 layers 0/18 are slower than later trials. Raw samples and all such
cases are retained; this run does not establish machine-independent or optimal
performance. Stable whole-model integration remains unmeasured.

[Complete per-layer results](formal/SUMMARY.md) include both eager and graph,
and `formal/results.json` retains component timings and every latency sample.
The original full-PPL quality results are unchanged and were not rerun here.

## Frozen Numerical Configuration

Qwen3-8B-Base adaptive C1 R64/R96, same factors and static per-layer scales as
`../q3-a8-dec`. Encoder stays BF16. Fixed NUQ4 is used only when producing real
latent fixtures from the first 256 tokens of the first training calibration
window. No evaluation tokens or new scale fitting are used. Each fixture
contains three layers (0/18/35), compact decoder weights, latents, source widths
and the frozen scalar scale. One KV group and four query heads belong to each
TP rank. Outputs are fully replicated, so there is no decoder-output all-reduce.
R64/R96 are model-wide average ranks; individual layers have different widths.

Fixtures `r64.pt` and `r96.pt` are raw data, not new model checkpoints. Original
PPL code/results are unmodified. Validation is structural and direct source-byte
comparison only, never SHA256. The tokenizer warning about the joined training
corpus length does not mean a full-corpus model forward: export forwards only
256 tokens.

## Kernel Changes

- Fused send: BF16 token-major latent -> scaled/clamped E4M3 bytes written
  directly into the feature-major source slot. No FP32 intermediate tensor,
  separate clamp/cast kernels or extra transpose/copy allocation.
- Fused receive: feature-major bytes -> token-major FP8 codes for W8A8, or
  transpose/dequantize directly to BF16 for the A8/BF16 decoder control.
- Weight quantization remains offline and unchanged. Actual W8A8 uses
  `torch._scaled_mm`, E4M3 operands, BF16 output, `use_fast_accum=False`.
- Existing prepared NCCL AllGather is reused for uniform source widths. The
  existing exact-width NCCL ring handles ragged widths without padding.

Five pipeline controls: BF16; original A8/BF16 and A8/W8A8; fused A8/BF16 and
A8/W8A8. BF16 also includes source packing. A serving attention kernel that
already writes into the BF16 send slot may avoid that copy, so this control
must not be described as an optimized full-model baseline.

Tests require byte-exact FP8 codes, exact restored BF16 values, strided inputs,
clipping, non-tile-aligned shapes and CUDA Graph replay. Distributed validation
checks actual gathered payloads and exact fused/unfused outputs on every rank.
No token-generation equivalence or additional PPL run is performed.

## Measurement

Formal grid: R64/R96 x layers 0/18/35 x rows 1/4/16/64/128/256 x three rounds,
both eager and CUDA Graph. Each operation gets 10 warmups and 50 measurements.
Record every sample's maximum CUDA-event time across all eight ranks, then its
median. Final tables use the median of the three round medians. Rotate operation
order between rounds. No concurrent GPU jobs are launched.

Record full pipeline and separate pack, gather, receive-conversion plus decoder
timings. Component medians are not additive because measurement boundaries and
launch gaps differ. Graph capture, offline weight conversion, fixture loading
and correctness checks are outside timing. Graph inputs/shapes are fixed;
rows represent GEMM shapes, not independently generated serving requests.
Communication byte counts describe application payload, not measured bandwidth.

Smoke uses only layer 18, rows 1/16, one round, two warmups and five samples.
Six CUDA kernel unit tests passed. Formal launch was explicitly approved
conditional on successful smoke. Slurm has no configuration; use the approved
direct single-node execution.

## Smoke Correction

The first smoke and its diagnostic rerun stopped at the payload equality check
for 16 rows. The error was a double-rounding difference in Triton's generic
FP32 -> FP16 -> E4M3 lowering, not a communication layout error. The sender now
uses direct `cvt.rn.satfinite.e4m3x2.f32` on Ada, with a regression test for a
value near a rounding boundary. All six exported layer fixtures (256 rows
each) match the frozen PyTorch quantizer byte-for-byte after the fix. The full
TP8 smoke then passed for both ranks and both row counts, eager and graph.

The original kernel and benchmark are preserved as `smoke/preflight_*.py`;
partial failed measurements are retained in `smoke/preflight_progress.json`.
`smoke.log` retains both failures and the final successful run. Only the
successful run's `smoke/results.json` and final formal results are summarized.

## Commands

Working directory: `/workspace/BasisServe-CALS`.

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m pytest -q tests/test_latent_a8_pack.py > results/q3-a8-tp8/tests.log 2>&1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/prepare_qwen3_a8_tp8.py --output results/q3-a8-tp8 > results/q3-a8-tp8/prepare.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m torch.distributed.run --standalone --nproc_per_node=8 evaluation/benchmark_qwen3_a8_tp8.py --phase smoke --output results/q3-a8-tp8 > results/q3-a8-tp8/smoke.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m torch.distributed.run --standalone --nproc_per_node=8 evaluation/benchmark_qwen3_a8_tp8.py --phase formal --output results/q3-a8-tp8 > results/q3-a8-tp8/formal.log 2>&1
```

Each phase saves source snapshots, a manifest, topology, sample-level raw JSON,
progress, compact summary JSON and `SUMMARY.md`. All artifacts remain local;
no GitHub or HF upload is authorized for this experiment yet.
