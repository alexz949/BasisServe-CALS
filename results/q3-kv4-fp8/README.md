# Qwen3-8B-Base KV4 and FP8 Projection Quality

Single L40S, `basis` environment, R64 and R96 original adaptive C1 checkpoints.
This experiment measures quality only, not throughput or packed-cache capacity.
No TP8 validation is included.

## Four Arms Per Checkpoint

| Arm | K/V quantization | Folded V encoder / output decoder |
|---|---|---|
| `bf16` | None | BF16 |
| `kv4` | Official KVQuant NUQ4 with outliers | BF16 |
| `fp8` | None | Actual E4M3 W8A8 GEMM, BF16 output |
| `kv4_fp8` | Official KVQuant NUQ4 with outliers | Actual E4M3 W8A8 GEMM, BF16 output |

R64/R96 describe average rank over layers, not uniform per-layer ranks.
Full attention is retained. Q/K projections, MLP, norms and language-model head
remain BF16. V coordinates are zero-padded into the HF attention interface;
padding is excluded from V quantizer fitting and application. K is quantized
after Qwen's K normalization and before RoPE, with static per-channel thresholds.
V uses dynamic per-token thresholds across the active coordinates of all eight
KV heads. NUQ4 is nonuniform 4-bit codebook quantization, not uniform signed INT4.
Its 0.99 percentile rule preserves outliers; not every element is four bits.
No first-token exclusion or rotation is applied.

## Calibration and Evaluation

Formal calibration uses 16 seed-0 random windows of 2048 tokens from WikiText2
train. NUQ4 fitting uses the official weighted K-means implementation and squared
activation gradients of next-token summed cross entropy in the BF16 C1 model.
The fitted quantizers are frozen across the two KV4 arms of each checkpoint.

FP8 weights have tensorwise absmax scales. Activation scales are per-projection
absmax values collected on the same training windows under the matching KV
setting with BF16 projections, then frozen. They are never calibrated on test
tokens or adjusted using future test activations. Each FP8 result records actual
GEMM call counts and clipped input element counts. Both operands are E4M3;
`torch._scaled_mm` uses `use_fast_accum=False` and returns BF16.

Formal PPL uses all complete non-overlapping 2048-token WikiText2 test windows,
batch one, FP32 summed cross entropy, and 2047 scored next-token targets per
window. The LM head is evaluated in 128-position chunks for memory efficiency.
All arms use the same tokens, factors and loss calculation.

Smoke instead uses one 128-token training window and two 256-token test windows.
**Smoke PPL is diagnostic only**, and its calibration must not be reused for
formal results. Smoke and formal artifacts are separated by directory.

## Inputs

- Model snapshot: `Qwen/Qwen3-8B-Base`, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Factors: `alexz949/BasisServe-CALS`, revision `3df8ff71c718ebcb096f479209baf95cd46abc60`,
  `ICLR-results/qwen3-8b/checkpoints/Q3-8B-C1-R{64,96}`.
- Dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Official source: `external/KVQuant/quant/kvquant/simquant_module_quantizer.py`;
  its Git revision and source bytes are saved with each phase.

The loader validates model geometry, schedules, tensor names/shapes, finite
factors and total rank. It does not calculate or validate SHA256. The phase's
`source/` directory saves the evaluator, runtime, FP8 helpers and upstream
quantizer as run artifacts. Existing result directories require matching
protocols and byte-identical source files before reuse.

## Commands

Working directory: `/workspace/BasisServe-CALS`. The short smoke runs directly;
Slurm configuration is unavailable on this machine. Formal execution requires
the user's separate approval of the command below, which was received after the
smoke launch. The formal run starts only after the smoke and short order audit.

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m pytest -q tests/test_qwen3_kv4_fp8_quality.py > results/q3-kv4-fp8/tests.log 2>&1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/eval_qwen3_kv4_fp8_ppl.py --phase smoke --ranks 64 96 --output results/q3-kv4-fp8 > results/q3-kv4-fp8/smoke.log 2>&1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/eval_qwen3_kv4_fp8_ppl.py --phase formal --ranks 64 96 --output results/q3-kv4-fp8 > results/q3-kv4-fp8/formal.log 2>&1
```

Each phase writes its manifest, per-rank progress, calibration files, per-arm
metrics, final JSON and `SUMMARY.md`. The combined arm's PPL and delta versus
BF16 are the primary requested results. These quality simulations do not support
claims of INT4 cache memory savings or serving speedup.

## Multi-GPU Scheduling

The user approved switching the formal experiment to independent single-GPU
workers across the eight L40S devices. The numerical evaluator and calibration
protocol are unchanged; this is not tensor parallelism. The original serial
process was deliberately terminated after R64 NUQ4 calibration had fully saved
and the KV4 evaluator had completed 3/146 windows. Its exit code 143 in
`formal.log` is the controlled handoff, not an OOM or a numerical failure.
`handoff.log` records the exact boundary. The partial KV4 evaluation is not used;
that arm restarts at window zero. R64 BF16 and its completed codebooks are reused.

| GPU | Task |
|---:|---|
| 0 | R64 KV4 |
| 1 | R96 NUQ4 calibration |
| 2 | R64 FP8 |
| 3 | R64 KV4 + FP8 |
| 4 | R96 BF16 |
| 5 | R96 FP8 |
| 6 | R96 KV4, after R96 calibration |
| 7 | R96 KV4 + FP8, after R96 calibration |

Every worker uses the `basis` environment and two CPU threads. Codebook fitting
remains CPU-based; eight GPUs do not imply an eightfold end-to-end speedup.
Per-arm logs and commands are under `formal/r{64,96}/`; the coordinator records
all tasks and exit codes in `parallel_manifest.json` and `parallel_outcomes.json`.
Concurrent worker wall times are not serving-latency measurements.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/run_qwen3_kv4_fp8_parallel.py --output results/q3-kv4-fp8/formal > results/q3-kv4-fp8/parallel.log 2>&1
```

## Preflight Note

The first smoke attempt stopped before PPL when a path check incorrectly
rejected HF snapshot symlinks pointing to the shared blob directory. The loader
now validates the relative artifact path without rejecting normal HF symlinks.
The original runtime is retained as `smoke/source/preflight_qwen3_kv4_fp8_quality.py`;
the failure remains in `smoke.log`. The same smoke command is rerun with logging
appended, and the standard source snapshot contains the corrected runtime.

The completed R64 two-window smoke has a large apparent PPL improvement under
quantization. This is not reported as a quality gain. The short audit in
`smoke/audit/results.json` verified exactly matching BF16 summed losses before
and after the quantized arms and under identity hooks, with unchanged projection
weights. It rules out the tested order/weight-mutation explanations, not every
possible source of numerical sensitivity. Only the formal full-test evaluation
is intended for the final quality comparison. Audit command (`basis`, GPU 0):

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/audit_qwen3_kv4_fp8_ppl.py > results/q3-kv4-fp8/audit.log 2>&1
```
