# Nemotron-H-8B-Base-8K: C1 V and Mamba Wo compression

## Protocol

- Calibration: C4 train, 256x2048 fit windows and 64x2048 held-out windows.
- Full-attention V: ALS6, fixed encoder CG16, BF16 factors, two-sided terminal-KL layer allocation.
- Mamba2 Wo: the same retained ratio as full attention, ALS6, fixed encoder CG16, FP32 fit and BF16 factors.
- Quality: full WikiText2 test at 2048, 128 disjoint C4 validation windows at 2048, and seven zero-shot MCQ tasks.

## Quality

| Setting | WikiText2 PPL | vs Dense | C4 PPL | vs Dense | MCQ average | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 6.099849 | — | 9.360031 | — | 73.692% | +0.000 pt |
| Retain 50% (mean V64) | 6.675828 | 9.443% | 9.845956 | 5.191% | 72.337% | -1.355 pt |
| Retain 75% (mean V96) | 6.218765 | 1.949% | 9.451837 | 0.981% | 73.768% | +0.076 pt |

## MCQ by task

| Task | Dense | Retain 50% | Retain 75% |
|---|---:|---:|---:|
| arc_easy | 83.628% | 82.870% | 84.386% |
| arc_challenge | 60.239% | 59.130% | 60.922% |
| hellaswag | 81.070% | 79.337% | 81.000% |
| piqa | 82.263% | 82.209% | 82.046% |
| winogrande | 75.770% | 74.270% | 76.085% |
| boolq | 85.474% | 84.343% | 85.138% |
| openbookqa | 47.400% | 44.200% | 46.800% |

## Fit diagnostics

| Setting | Full-attention layer ranks | Uniform V held-out rel-MSE | Selected terminal KL | Mamba Wo held-out rel-MSE |
|---|---|---:|---:|---:|
| Mean V64 | `[32, 64, 96, 64]` | 0.103025 | 0.015067 | 0.049651 |
| Mean V96 | `[48, 96, 128, 112]` | 0.042217 | 0.005877 | 0.011372 |

## Communication accounting

- Mean V64: full-attention private AllGather bytes are reduced by 50.000%; Mamba2 Wo private AllGather bytes are reduced by 50.000%.
- Mean V96: full-attention private AllGather bytes are reduced by 25.000%; Mamba2 Wo private AllGather bytes are reduced by 25.000%.

All persisted factors and result inputs are hash-verified in `summary.json`.
