# BF16 Encoder, A8 Latent, W8 Decoder

Qwen3-8B-Base, original adaptive C1 R64/R96 checkpoints, `basis` environment.
This experiment isolates the communication-boundary activation quantization
from decoder weight quantization. Encoder, Q/K projections, MLP, norms and LM
head remain BF16. No Hadamard rotation or new quantization fitting is added.

## Completed Formal Results

Both formal jobs completed successfully after explicit user approval. All six
PPL arms scored 298862 targets from 146 windows. Both rerun KV4 baselines match
the previous summed NLL exactly. Every A8/W8 arm executed 5256 actual FP8
decoder GEMMs; every encoder FP8 call count is zero. Frozen input/weight scales
match the previous training calibration, and code snapshots match source bytes.

| Nominal rank | KV4 + BF16 | KV4 + latent A8 | KV4 + latent A8 / decoder W8 | A8 delta vs KV4 | A8/W8 delta vs KV4 |
|---:|---:|---:|---:|---:|---:|
| 64 | 8.277446 | 8.280403 | 8.285759 | +0.002957 | +0.008313 |
| 96 | 7.338056 | 7.342300 | 7.344549 | +0.004244 | +0.006493 |

The additional W8 decoder effect relative to A8-only is +0.005356 / +0.002249
PPL for R64/R96. Relative to the original non-quantized C1 baseline (without
KV4), the complete KV4+A8/W8 changes are +0.027031 / +0.061192. These are
end-to-end PPL differences, not independently additive local error estimates.
The quality result supports testing this narrower precision scope in serving;
it does not establish accuracy on other tasks or context lengths.

All 30 decoder microbenchmark cases completed. For the current eager,
unfused implementation, direct prequantized W8A8 decoder speedup ranges from
0.573x to 2.068x; **including activation quantization, every case is slower**
than BF16 (0.219x to 0.766x). For example, R96 layer 18 / one row takes
0.051712 ms in BF16, 0.025440 ms for prequantized W8A8, and 0.067872 ms for
quantization plus W8A8. These include eager launch effects and do not establish
the hardware limit of FP8. The microbenchmark ran on GPU 2 while PPL workers
ran on GPUs 0/1, so host scheduling contention is another timing limitation.
No real communication or full-model speedup has been measured in this round.

Full tables: [PPL](formal/SUMMARY.md) and [decoder timings](decoder_formal/SUMMARY.md).
Tests: `tests.log` (one focused CUDA test passed), plus both smoke order checks.
There were no formal worker failures or OOMs. All jobs have exited.

## Quality Arms

All arms use the same previous formal NUQ4 K/V codebooks, including outliers.
The encoder is BF16 in every arm.

| Arm | Decoder input latent | Decoder computation |
|---|---|---|
| `bf16` | BF16 | BF16 weights and GEMM |
| `a8` | E4M3 codes, restored to BF16 | BF16 weights and GEMM |
| `w8a8` | E4M3 codes | E4M3 weights, actual FP8 GEMM, BF16 output |

Quantization is applied to the attention output immediately before `o_proj`,
not the encoder input or cached V. One static scalar scale per layer is shared
unchanged across the two A8 arms. We reuse the decoder entries in
`../q3-kv4-fp8/formal/r{64,96}/kv4_fp8_scales.json`: those scales were calibrated
with BF16 encoder/decoder and matching KV4 on 16 x 2048 WT2 train tokens. There
is no new fitting on test or per-arm retuning. Weight scales are tensorwise
absmax, as in the previous baseline. The encoder FP8 call count must be zero.

The single-GPU quality path simulates the numerical effect of quantizing the
wire latent; it does not execute collectives. It is not TP8 quality validation,
and it does not establish TP8 communication or serving speedup. Real TP routing
and per-source scale choices require a separate integration experiment.

Formal PPL uses 146 non-overlapping 2048-token WT2 test windows and 298862
next-token targets, B1, exactly the previous evaluator. Each rank reruns its
KV4/BF16 baseline and checks agreement with the previous full result at 1e-6
relative PPL tolerance. Smoke evaluates only the first two such windows and
checks exact restoration of BF16 output after switching quantization modes.
Smoke PPL is not a quality conclusion.

NUQ4 is a quantize/dequantize quality simulation, not packed INT4 cache storage.
Factor validation is structural. No SHA256 is calculated or validated. Previous
results and code snapshots are preserved; new snapshots use direct byte checks.

## Decoder Timings

The microbenchmark uses real decoder weights and latent activations collected
from 128 WT2 train tokens with BF16 projections and fixed KV4. Inactive padded
latent coordinates are removed. The decoder output width is the full hidden
size 4096: this measures a compact global decoder on one GPU, not a specific
TP8 output-sharded deployment. Rows emulate GEMM batch dimensions, not complete
independent requests. Weights and activations are warm/reused.

Reported CUDA-event medians separate BF16 GEMM, A8 restore + BF16 GEMM, direct
W8A8 GEMM with prequantized inputs, activation quantization alone, and combined
quantization + decoder. Weight quantization is offline and excluded. No debug
clipping counters are timed. No NCCL, communication packing, scheduler or full
model is timed; byte counts are analytical payload sizes, not transfer speed.

Smoke: layer 18 (zero-based), rows 1/16, 3 warmups and 10 measurements.
Prepared formal: layers 0/18/35, rows 1/4/16/64/128, 20 warmups and 100 measurements.
Do not treat smoke timings as final performance measurements.

## Commands

Working directory: `/workspace/BasisServe-CALS`. Slurm configuration remains
unavailable; runs execute directly as previously agreed. The user explicitly
approved both formal commands below before launch. Per-rank PPL jobs are independent
single-GPU evaluations and use two CPU threads each.

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m pytest -q tests/test_qwen3_latent_a8.py > results/q3-a8-dec/tests.log 2>&1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/eval_qwen3_latent_a8.py --phase smoke --ranks 64 96 --gpus 0 1 --output results/q3-a8-dec > results/q3-a8-dec/smoke.log 2>&1
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/benchmark_qwen3_latent_decoder.py --phase smoke --ranks 64 96 --output results/q3-a8-dec > results/q3-a8-dec/decoder_smoke.log 2>&1
```

Executed formal commands:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/eval_qwen3_latent_a8.py --phase formal --ranks 64 96 --gpus 0 1 --output results/q3-a8-dec > results/q3-a8-dec/formal.log 2>&1
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/benchmark_qwen3_latent_decoder.py --phase formal --ranks 64 96 --output results/q3-a8-dec > results/q3-a8-dec/decoder_formal.log 2>&1
```

Smoke results are in `smoke/SUMMARY.md` and `decoder_smoke/SUMMARY.md`; completed
formal summaries are in `formal/SUMMARY.md` and `decoder_formal/SUMMARY.md`. Raw per-arm JSON,
progress, call counts, command manifests, source snapshots and logs are kept.
Nothing from this experiment has been committed or uploaded.
