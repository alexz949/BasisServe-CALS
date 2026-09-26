# Llama Two-Sided KL V64/V96: KV4 C4 Comparison

Both completed runs use Llama-3.1-8B-Instruct with adaptive Two-Sided KL
allocations, not uniform per-layer ranks. Environment: `basis`, one L40S,
TP1/B1. Encoder/decoder and other model modules remain BF16; no A8 or FP8 GEMM.

| Equivalent Value rank | BF16 C4 PPL | KV4 C4 PPL | Delta PPL | Relative increase |
|---:|---:|---:|---:|---:|
| 64 | 12.453613 | 12.594977 | +0.141364 | +1.135% |
| 96 | 10.662169 | 10.726644 | +0.064475 | +0.605% |

Each increment uses its own same-checkpoint BF16 control. Calibration uses the
same 16 x 2048 WT2 train windows (seed 0), but fits independent rank-specific
Fisher-weighted NUQ4 codebooks. Both evaluate the same frozen 128 x 2048 C4
validation windows, scoring 262,016 labels without cross-document transitions.
The C4 validation windows are never used to fit quantizers.

K uses static pre-RoPE per-channel ranges. V uses dynamic per-token ranges over
all active latent coordinates across eight KV heads. Both use official NUQ4
and the 0.99 outlier rule, with no rotation or first-token exclusion. Outliers
retain higher precision, so total storage is not exactly four bits per element.
These are quantize/dequantize quality simulations, not packed-cache performance.

## Validation

- Adaptive folding and active-coordinate tests: 2 passed.
- V64 smoke: BF16 15.296446, KV4 15.518771, two C4 windows and one 128-token
  training window. These smoke numbers are not formal quality results.
- V64 formal: all 16 Fisher windows, all 64 K/V codebooks, and both 128-window
  PPL evaluations completed. No OOM or nonfinite-loss failure.
- Formal BF16 differs from historical C4 PPL 12.4536129386 by approximately
  0.00000030, reproducing the original baseline.
- Model/factor geometry, rank schedules, finite codebooks, source-byte equality,
  scored-token counts and PPL arithmetic checked. No SHA256 checks.
- The tokenizer warning concerns the complete WT2 train vector; only the
  recorded 2048-token windows are fed to the model during formal calibration.

## Artifacts

- [V64 summary](formal/SUMMARY.md), `formal/result.json`, `formal/manifest.json`.
- `formal/quantizers.pt`: new V64 quantizers; V96 quantizers were not reused.
- `formal.log`, `smoke.log`, `tests.log`, and phase-specific source snapshots.
- [V96 summary](../l31-adaptive-kv4-c4/formal/SUMMARY.md): original result and
  frozen source remain unchanged. The current driver requires explicit `--rank`.

## Commands

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m pytest -q tests/test_llama_adaptive_nuq4_quality.py > results/l31-adaptive-kv4-c4-r64/tests.log 2>&1

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/eval_llama_adaptive_nuq4_c4.py --phase smoke --rank 64 --output results/l31-adaptive-kv4-c4-r64 > results/l31-adaptive-kv4-c4-r64/smoke.log 2>&1

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/eval_llama_adaptive_nuq4_c4.py --phase formal --rank 64 --output results/l31-adaptive-kv4-c4-r64 > results/l31-adaptive-kv4-c4-r64/formal.log 2>&1
```

The user approved local execution because Slurm is unconfigured. No GitHub/HF
upload, commit, push, or old experiment overwrite was performed.
