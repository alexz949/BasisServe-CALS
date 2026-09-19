# Nemotron-H-56B-Base-8K: C1 V and Mamba Wo compression

## Protocol

- Calibration: C4 train, 256x2048 fit windows and 64x2048 held-out windows.
- Full-attention V: ALS6, fixed encoder CG16, BF16 factors, two-sided terminal-KL layer allocation.
- Mamba2 Wo: the same retained ratio as full attention, ALS6, fixed encoder CG16, FP32 fit and BF16 factors.
- Quality: full WikiText2 test at 2048, 128 disjoint C4 validation windows at 2048, and seven zero-shot MCQ tasks.

## Quality

| Setting | WikiText2 PPL | vs Dense | C4 PPL | vs Dense | MCQ average | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 3.650869 | — | 7.013513 | — | 76.469% | +0.000 pt |
| Retain 50% (mean V64) | 4.135171 | 13.265% | 7.386983 | 5.325% | 76.520% | +0.050 pt |
| Retain 75% (mean V96) | 3.760560 | 3.005% | 7.089277 | 1.080% | 76.526% | +0.057 pt |

## MCQ by task

| Task | Dense | Retain 50% | Retain 75% |
|---|---:|---:|---:|
| arc_easy | 85.101% | 85.396% | 85.564% |
| arc_challenge | 63.652% | 62.457% | 63.481% |
| hellaswag | 87.024% | 86.786% | 87.134% |
| piqa | 85.310% | 84.875% | 85.637% |
| winogrande | 81.452% | 82.005% | 81.215% |
| boolq | 83.945% | 85.719% | 83.853% |
| openbookqa | 48.800% | 48.400% | 48.800% |

## Fit diagnostics

| Setting | Full-attention layer ranks | Uniform V held-out rel-MSE | Selected terminal KL | Mamba Wo held-out rel-MSE |
|---|---|---:|---:|---:|
| Mean V64 | `[32, 32, 48, 80, 112, 80, 64, 32, 64, 96]` | 0.116911 | 0.010740 | 0.051563 |
| Mean V96 | `[32, 64, 80, 128, 128, 128, 96, 48, 128, 128]` | 0.047290 | 0.003283 | 0.012354 |

## Communication accounting

- Mean V64: full-attention private AllGather bytes are reduced by 50.000%; Mamba2 Wo private AllGather bytes are reduced by 50.000%.
- Mean V96: full-attention private AllGather bytes are reduced by 25.000%; Mamba2 Wo private AllGather bytes are reduced by 25.000%.

All persisted factors and result inputs are hash-verified in `summary.json`.
