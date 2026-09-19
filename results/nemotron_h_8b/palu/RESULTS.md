# Nemotron-H-8B-Base-8K: PaLU V-only compression

## Protocol

- Calibration: C4 train; 256x2048 windows for PaLU double-shift Fisher weighting and V-input covariance, plus 64x2048 held-out covariance windows.
- Methods: M-LRD (one group per KV head) and G-LRD4 (four KV heads per group), with an exact global mean-rank budget allocated in blocks of 32.
- Targets: full-attention V only. Attention o_proj and every Mamba2 Wo remain dense.
- Quality: full WikiText2 test at 2048, 128 disjoint C4 validation windows at 2048, and seven zero-shot MCQ tasks.

## Quality

| Setting | WikiText2 PPL | vs Dense | C4 PPL | vs Dense | MCQ average | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 6.099849 | — | 9.360031 | — | 73.692% | — |
| M-LRD retain 50% | 6.568105 | 7.677% | 9.655748 | 3.159% | 72.998% | -0.694 pt |
| M-LRD retain 75% | 6.252134 | 2.497% | 9.456699 | 1.033% | 73.555% | -0.136 pt |
| G-LRD4 retain 50% | 6.340837 | 3.951% | 9.516172 | 1.668% | 73.169% | -0.522 pt |
| G-LRD4 retain 75% | 6.195244 | 1.564% | 9.417483 | 0.614% | 73.422% | -0.270 pt |

## Fit diagnostics

| Setting | Layer group ranks | Realized retention | Fit rel-error | Held-out rel-error |
|---|---|---:|---:|---:|
| M-LRD retain 50% | `[[64, 64, 64, 64, 64, 64, 64, 64], [96, 96, 96, 96, 96, 96, 96, 96], [64, 64, 64, 64, 64, 64, 64, 64], [32, 32, 32, 32, 32, 32, 32, 32]]` | 50.000% | 0.538638 | 0.541404 |
| M-LRD retain 75% | `[[96, 96, 96, 96, 96, 96, 96, 96], [128, 128, 128, 128, 128, 128, 128, 128], [96, 96, 96, 96, 96, 96, 96, 96], [64, 64, 64, 64, 64, 64, 64, 64]]` | 75.000% | 0.315834 | 0.317541 |
| G-LRD4 retain 50% | `[[288, 288], [384, 384], [256, 256], [96, 96]]` | 50.000% | 0.372457 | 0.375078 |
| G-LRD4 retain 75% | `[[416, 416], [512, 512], [416, 416], [192, 192]]` | 75.000% | 0.180029 | 0.181298 |

All factors, checkpoint manifests, smoke runs, and quality results are hash-verified in `summary.json`.
