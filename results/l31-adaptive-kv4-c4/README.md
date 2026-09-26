# Llama Adaptive V96 KV4 C4 Evaluation

Model: Llama-3.1-8B-Instruct. Checkpoint:
`ICLR-results/llama31-8b-instruct/c1/two-sided-kl/R96-D0-S8`.
The saved Two-Sided KL allocation has ranks 32-128 and averages V96; it is not
the uniform R96-S6-D0 checkpoint used in the preceding WT2 KV4 evaluation.

## Protocol

- Environment: `basis`, one NVIDIA L40S, GPU 0, B1/TP1.
- Fresh Fisher-weighted NUQ4 calibration: WT2 train, 16 x 2048 tokens, seed 0.
- K is quantized per channel before RoPE; V per token over the active adaptive
  latent coordinates across all eight KV heads. Padded coordinates are excluded.
- Official NUQ4 uses 16 codebook entries and the 0.99 outlier rule. Outliers
  retain higher precision; this is not a claim of exactly four total bits per
  element after accounting for outliers, scales, and metadata.
- Encoder/decoder and other modules remain BF16. No A8, FP8 GEMM, sparse
  routing, MLP changes, or recalibration on evaluation data.
- Evaluate the frozen C4 validation bank, 128 x 2048 tokens, 262,016 scored
  next-token labels, with no cross-document transitions.
- BF16 and KV4 use the same installed checkpoint, windows, and evaluator.
  Their difference is the primary quantization increment. The historical
  C4 BF16 PPL 10.662169 is a reference, not the denominator for the new increment.
- Quality uses quantize/dequantize simulation, not a packed-cache serving benchmark.
- Check model geometry, factor shapes, and the archived rank schedule. No SHA256.
  Source files are archived and compared byte-for-byte before/after evaluation.

## Validation and Status

The adaptive folding/active-coordinate test passed. The two-window smoke
completed with BF16 PPL 13.097338 and KV4 PPL 13.151366. This smoke used only
one 128-token train window and is not a formal quality measurement.
Historical BF16 on the same first two C4 documents is 13.097335, a difference
of approximately 0.0000025 from the new BF16 evaluator.

The user approved the formal command below. It completed successfully: all 16
Fisher windows, all 64 codebooks, and both 128-window C4 evaluations finished.

| Arm | C4 PPL |
|---|---:|
| Same-checkpoint BF16 | 10.662169 |
| Same-checkpoint KV4 | 10.726644 |

The matched increment is **+0.064475 PPL (+0.605%)**. BF16 differs from the
historical result by only approximately 0.00000018 PPL. There was no OOM or
nonfinite-loss failure. Both arms scored all 262,016 labels; final source-byte
comparisons and post-run PPL/label-count arithmetic checks passed.

See [formal/SUMMARY.md](formal/SUMMARY.md) and `formal/result.json` for the result,
`formal/manifest.json` for the protocol, and `formal.log` for progress.
The new frozen quantizers are in `formal/quantizers.pt`.
Smoke and formal artifacts remain separate, and old experiment files are unchanged.

## Commands

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m pytest -q tests/test_llama_adaptive_nuq4_quality.py > results/l31-adaptive-kv4-c4/tests.log 2>&1

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/eval_llama_adaptive_nuq4_c4.py --phase smoke --output results/l31-adaptive-kv4-c4 > results/l31-adaptive-kv4-c4/smoke.log 2>&1

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/eval_llama_adaptive_nuq4_c4.py --phase formal --output results/l31-adaptive-kv4-c4 > results/l31-adaptive-kv4-c4/formal.log 2>&1
```

Slurm has no usable configuration on this machine; the user approved direct execution.
Tokenization may warn that the entire WT2 train vector exceeds the model context;
only the recorded 2048-token training windows are passed to the model in the formal run.
Nothing has been uploaded to GitHub or HF for this experiment.
