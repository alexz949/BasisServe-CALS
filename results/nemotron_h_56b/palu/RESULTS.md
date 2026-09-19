# Nemotron-H-56B-Base-8K: PaLU V-only compression

## Protocol

- Calibration: C4 train; 256x2048 windows for PaLU double-shift Fisher weighting and V-input covariance, plus 64x2048 held-out covariance windows.
- Methods: M-LRD (one group per KV head) and G-LRD4 (four KV heads per group), with an exact global mean-rank budget allocated in blocks of 32.
- Targets: full-attention V only. Attention o_proj and every Mamba2 Wo remain dense.
- Quality: full WikiText2 test at 2048, 128 disjoint C4 validation windows at 2048, and seven zero-shot MCQ tasks.

## Quality

| Setting | WikiText2 PPL | vs Dense | C4 PPL | vs Dense | MCQ average | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 3.650869 | — | 7.013513 | — | 76.469% | — |
| M-LRD retain 50% | 4.054660 | 11.060% | 7.351668 | 4.821% | 75.955% | -0.514 pt |
| M-LRD retain 75% | 3.741140 | 2.473% | 7.087354 | 1.053% | 76.492% | +0.022 pt |
| G-LRD4 retain 50% | 3.837189 | 5.103% | 7.155669 | 2.027% | 75.909% | -0.560 pt |
| G-LRD4 retain 75% | 3.687038 | 0.991% | 7.036523 | 0.328% | 76.418% | -0.051 pt |

## Fit diagnostics

| Setting | Layer group ranks | Realized retention | Fit rel-error | Held-out rel-error |
|---|---|---:|---:|---:|
| M-LRD retain 50% | `[[32, 32, 32, 32, 32, 32, 32, 32], [128, 128, 128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128, 128, 128], [64, 64, 64, 64, 64, 64, 64, 64], [32, 32, 32, 32, 32, 32, 32, 32], [32, 32, 32, 32, 32, 32, 32, 32], [32, 32, 32, 32, 32, 32, 32, 32], [32, 32, 32, 32, 32, 32, 32, 32], [32, 32, 32, 32, 32, 32, 32, 32]]` | 50.000% | 0.497778 | 0.498790 |
| M-LRD retain 75% | `[[96, 96, 96, 96, 96, 96, 96, 96], [128, 128, 128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128, 128, 128], [96, 96, 96, 96, 96, 96, 96, 96], [64, 64, 64, 64, 64, 64, 64, 64], [64, 64, 64, 64, 64, 64, 64, 64], [64, 64, 64, 64, 64, 64, 64, 64], [64, 64, 64, 64, 64, 64, 64, 64]]` | 75.000% | 0.293269 | 0.294542 |
| G-LRD4 retain 50% | `[[128, 128], [480, 480], [512, 512], [512, 512], [352, 352], [224, 224], [128, 128], [64, 64], [64, 64], [96, 96]]` | 50.000% | 0.366297 | 0.367325 |
| G-LRD4 retain 75% | `[[320, 320], [512, 512], [512, 512], [512, 512], [512, 512], [448, 448], [320, 320], [224, 224], [224, 224], [256, 256]]` | 75.000% | 0.139851 | 0.141830 |

All factors, checkpoint manifests, smoke runs, and quality results are hash-verified in `summary.json`.
