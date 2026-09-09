# Qwen3-32B-Base: complete ICLR result summary

## Scope and status

This report consolidates every completed result in the formal Qwen3-32B
V-side matrix: Dense, Weight-SVD, PaLU M-LRD/G-LRD2/G-LRD4, and C1 at
equivalent ranks R64, R80, and R96. All 16 planned quality runs completed and
passed artifact/protocol consistency checks.

The completed 32B matrix covers full WikiText-2 perplexity, fixed C4
perplexity, and seven zero-shot multiple-choice tasks. IFEval, GSM8K, and
HumanEval were not run for Qwen3-32B and are therefore not reported here.

## Main result

C1 is the strongest compressed method at every tested rank on all three
aggregate measures: WikiText-2 PPL, C4 PPL, and mean MCQ accuracy. C1-R96 is
especially close to Dense: its WikiText-2 PPL is 1.94% higher, C4 PPL is
0.96% higher, and MCQ average is only 0.002 percentage points lower.

| Rank | Best compressed method | WT2 PPL | C4 PPL | MCQ average | MCQ gap to Dense |
| ---: | --- | ---: | ---: | ---: | ---: |
| R96 | C1 | 7.757330 | 10.917164 | 73.5316% | -0.002 pp |
| R80 | C1 | 8.052279 | 11.139824 | 72.7319% | -0.802 pp |
| R64 | C1 | 8.622977 | 11.624270 | 71.6845% | -1.849 pp |

Dense reaches 7.609852 WikiText-2 PPL, 10.813615 C4 PPL, and 73.5338% mean
MCQ accuracy.

## Complete quality matrix

PPL is lower-is-better. Task accuracy and the unweighted seven-task average
are higher-is-better.

| Run ID | Method | Retained V | WT2 PPL | C4 PPL | ARC-E | ARC-C | HellaSwag | PIQA | WinoGrande | BoolQ | OBQA | Average |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q3-32B-Dense | Dense | 1.000000 | 7.609852 | 10.813615 | 0.829966 | 0.605802 | 0.826230 | 0.822089 | 0.738753 | 0.864526 | 0.460000 | 0.735338 |
| Q3-32B-SVD-R96 | Weight-SVD | 0.750000 | 47.891616 | 35.426802 | 0.523569 | 0.374573 | 0.662517 | 0.688792 | 0.554065 | 0.635168 | 0.374000 | 0.544669 |
| Q3-32B-SVD-R80 | Weight-SVD | 0.625000 | 124.252448 | 77.894371 | 0.481061 | 0.342150 | 0.602868 | 0.659956 | 0.528808 | 0.611621 | 0.330000 | 0.508066 |
| Q3-32B-SVD-R64 | Weight-SVD | 0.500000 | 330.367637 | 263.937935 | 0.398148 | 0.319113 | 0.513941 | 0.616975 | 0.522494 | 0.575535 | 0.324000 | 0.467172 |
| Q3-32B-PALUM-R96 | PaLU M-LRD | 0.742188 | 8.475078 | 11.457041 | 0.813552 | 0.610922 | 0.815873 | 0.807943 | 0.715075 | 0.859021 | 0.446000 | 0.724055 |
| Q3-32B-PALUM-R80 | PaLU M-LRD | 0.636719 | 9.077192 | 12.055624 | 0.771886 | 0.562287 | 0.799343 | 0.794342 | 0.719811 | 0.812232 | 0.452000 | 0.701700 |
| Q3-32B-PALUM-R64 | PaLU M-LRD | 0.496094 | 10.290939 | 13.352030 | 0.740320 | 0.521331 | 0.769070 | 0.785092 | 0.710339 | 0.741590 | 0.430000 | 0.671106 |
| Q3-32B-PALUG2-R96 | PaLU G-LRD2 | 0.744141 | 8.340474 | 11.357118 | 0.817340 | 0.616041 | 0.814977 | 0.803047 | 0.743489 | 0.863914 | 0.442000 | 0.728687 |
| Q3-32B-PALUG2-R80 | PaLU G-LRD2 | 0.615234 | 8.887113 | 11.906275 | 0.765152 | 0.575939 | 0.800438 | 0.792165 | 0.735596 | 0.868196 | 0.450000 | 0.712498 |
| Q3-32B-PALUG2-R64 | PaLU G-LRD2 | 0.498047 | 9.780397 | 12.698288 | 0.710017 | 0.541809 | 0.778630 | 0.781828 | 0.749803 | 0.855963 | 0.426000 | 0.692007 |
| Q3-32B-PALUG4-R96 | PaLU G-LRD4 | 0.747070 | 8.238425 | 11.269805 | 0.808081 | 0.623720 | 0.819259 | 0.804679 | 0.736385 | 0.866055 | 0.448000 | 0.729454 |
| Q3-32B-PALUG4-R80 | PaLU G-LRD4 | 0.626953 | 8.628236 | 11.611834 | 0.779461 | 0.614334 | 0.812089 | 0.795974 | 0.719811 | 0.850153 | 0.430000 | 0.714546 |
| Q3-32B-PALUG4-R64 | PaLU G-LRD4 | 0.500977 | 9.163749 | 12.067138 | 0.749158 | 0.574232 | 0.794961 | 0.792709 | 0.739542 | 0.874006 | 0.434000 | 0.708373 |
| Q3-32B-C1-R96 | C1 + Two-Sided KL | 0.750000 | 7.757330 | 10.917164 | 0.831650 | 0.622867 | 0.824736 | 0.808487 | 0.715864 | 0.863609 | 0.480000 | 0.735316 |
| Q3-32B-C1-R80 | C1 + Two-Sided KL | 0.625000 | 8.052279 | 11.139824 | 0.804714 | 0.598123 | 0.820554 | 0.809032 | 0.733228 | 0.867584 | 0.458000 | 0.727319 |
| Q3-32B-C1-R64 | C1 + Two-Sided KL | 0.500000 | 8.622977 | 11.624270 | 0.808502 | 0.593003 | 0.807807 | 0.806855 | 0.715075 | 0.840673 | 0.446000 | 0.716845 |

## Change relative to Dense

Positive PPL changes are regressions; negative MCQ changes are accuracy
regressions. MCQ differences are absolute percentage points.

| Run ID | WT2 PPL change | C4 PPL change | MCQ change |
| --- | ---: | ---: | ---: |
| Q3-32B-SVD-R96 | +529.34% | +227.61% | -19.067 pp |
| Q3-32B-SVD-R80 | +1532.78% | +620.34% | -22.727 pp |
| Q3-32B-SVD-R64 | +4241.31% | +2340.79% | -26.817 pp |
| Q3-32B-PALUM-R96 | +11.37% | +5.95% | -1.128 pp |
| Q3-32B-PALUM-R80 | +19.28% | +11.49% | -3.364 pp |
| Q3-32B-PALUM-R64 | +35.23% | +23.47% | -6.423 pp |
| Q3-32B-PALUG2-R96 | +9.60% | +5.03% | -0.665 pp |
| Q3-32B-PALUG2-R80 | +16.78% | +10.10% | -2.284 pp |
| Q3-32B-PALUG2-R64 | +28.52% | +17.43% | -4.333 pp |
| Q3-32B-PALUG4-R96 | +8.26% | +4.22% | -0.588 pp |
| Q3-32B-PALUG4-R80 | +13.38% | +7.38% | -2.079 pp |
| Q3-32B-PALUG4-R64 | +20.42% | +11.59% | -2.697 pp |
| Q3-32B-C1-R96 | +1.94% | +0.96% | -0.002 pp |
| Q3-32B-C1-R80 | +5.81% | +3.02% | -0.802 pp |
| Q3-32B-C1-R64 | +13.31% | +7.50% | -1.849 pp |

## C1 versus the strongest PaLU geometry

PaLU G-LRD4 is the strongest PaLU variant on all three aggregates at every
rank. The table reports the C1 improvement relative to G-LRD4: relative PPL
reduction and absolute MCQ gain.

| Rank | WT2 PPL reduction | C4 PPL reduction | MCQ gain |
| ---: | ---: | ---: | ---: |
| R96 | 5.84% | 3.13% | +0.586 pp |
| R80 | 6.68% | 4.06% | +1.277 pp |
| R64 | 5.90% | 3.67% | +0.847 pp |

## Method and budget definitions

- **Dense** keeps K and V dense.
- **Weight-SVD** applies ordinary truncated weight SVD independently to each
  physical V head, with uniform per-head rank and no calibration-aware rank
  allocation.
- **PaLU M-LRD** uses one physical V head per low-rank group. **G-LRD2** and
  **G-LRD4** group two and four physical V heads, respectively. Thus G2-R96
  uses nominal group rank 192 and G4-R96 uses nominal group rank 384. PaLU
  uses activation-weighted factorization plus Fisher allocation with rank
  blocks of 32, so its realized retained-V ratios can differ slightly from
  the nominal targets.
- **C1** jointly fits the V encoder and output decoder. It uses
  activation-weighted-SVD initialization, six encoder sweeps, the final
  decoder-refit endpoint, and Two-Sided factorized terminal-KL allocation
  with alpha = 1. K stays dense for every compressed method.

Nominal R64/R80/R96 means equivalent rank per physical KV head. Since the
head dimension is 128, the exact C1 retained-V ratios are 50%, 62.5%, and
75%. Counting dense K plus compressed V, these correspond to total KV-cache
retention of 75%, 81.25%, and 87.5%.

## C1 fitting diagnostics

All trained factor banks use the same 256 C4 fitting contexts, 64 held-out
diagnostic contexts, sequence length 2048, activation-weighted-SVD
initialization, six ALS encoder sweeps, and the fixed final decoder-refit
endpoint. Factors are stored in BF16; fitting work and covariance operations
use FP32 with covariance damping `1e-5`.

| Uniform factor bank | Mean held-out relative MSE |
| ---: | ---: |
| R32 | 0.2809792780 |
| R48 | 0.1983001530 |
| R64 | 0.1376529510 |
| R80 | 0.0906974766 |
| R96 | 0.0531781892 |
| R112 | 0.0230567048 |

R128 is available to the allocator as the dense endpoint; it is not a
separately trained factor bank.

### Two-Sided KL allocation

The allocator uses an R64 profiling anchor independently of the target
average-rank budget. Profiling uses 128 sampled-suffix windows and final
confirmation uses 16 held-out windows. Candidate ranks are
32/48/64/80/96/112/128, the factorized exponent is alpha = 1, and terminal
positions are sampled at counts 64/128/256/512/1024 with selection at 1024.

| Target | Exact rank sum | Retained V | Confirmation KL | Layers changed from R64 anchor |
| ---: | ---: | ---: | ---: | ---: |
| R64 | 32,768 | 50.0% | 0.1071603960 | 57/64 |
| R80 | 40,960 | 62.5% | 0.0494550569 | 59/64 |
| R96 | 49,152 | 75.0% | 0.0173683082 | 56/64 |

The R64 uniform profiling anchor has confirmation KL 0.134761733. This anchor
is a profiling reference, not a same-budget uniform baseline for R80 or R96.

The selected per-layer rank distributions are:

| Target | R32 layers | R48 | R64 | R80 | R96 | R112 | R128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R64 | 30 | 5 | 7 | 4 | 4 | 3 | 11 |
| R80 | 15 | 12 | 5 | 4 | 2 | 6 | 20 |
| R96 | 2 | 10 | 8 | 6 | 7 | 2 | 29 |

Each layer assigns the same rank to all eight physical KV sources, so the
tensor-parallel collective width is uniform within a layer and requires no
ragged padding.

## Evaluation protocol

| Item | Setting |
| --- | --- |
| Model | `Qwen/Qwen3-32B`, revision `9216db5781bf21249d130ec9da846c4624c16137` |
| Environment | `basis` |
| Software | PyTorch 2.6.0+cu124, Transformers 5.15.1, datasets 5.0.0, lm-eval 0.4.11 |
| Hardware | 4 NVIDIA L40S GPUs on Lovelace; two parallel quality shards with 2 GPUs each |
| WikiText-2 | Full test corpus; sequence length 2048; 146 chunks and 298,862 predicted tokens |
| C4 PPL | 128 fixed, document-disjoint validation windows; sequence length 2048; 262,016 predicted tokens |
| MCQ | ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande, BoolQ, OpenBookQA; zero-shot |
| MCQ metric | `acc_norm` when defined, otherwise `acc`; unweighted arithmetic mean |
| PPL batch size | 2 |
| lm-eval batch size | 8 |

Every arm used the same pinned model revision, model config/index hashes, C4
validation-window hash, software environment, and 2-by-2 GPU evaluation
layout. All 16 merged result manifests and source-stage hashes passed the
strict audit.

## Interpretation

1. **C1-R96 is effectively Dense on aggregate MCQ.** The measured difference
   is only 0.002 percentage points, while retaining 75% of V. This is a single
   deterministic evaluation matrix, so the tiny gap should not be interpreted
   as statistically meaningful superiority or inferiority.
2. **C1 degrades gracefully as rank falls.** At R80 it loses 0.802 MCQ points
   and at R64 it loses 1.849 points relative to Dense, with much smaller PPL
   degradation than every baseline.
3. **PaLU benefits consistently from larger groups.** M-LRD -> G-LRD2 ->
   G-LRD4 improves WikiText-2 PPL, C4 PPL, and mean MCQ accuracy at R64, R80,
   and R96. All nine predeclared trend checks pass.
4. **Weight-only SVD is not competitive here.** Even R96 increases
   WikiText-2 PPL by 529% and loses 19.067 MCQ points, indicating that weight
   reconstruction alone is a poor proxy for preserving Qwen3-32B behavior.
5. **The strongest operating point depends on the memory target.** R96 nearly
   preserves Dense quality; R80 retains 62.5% of V with less than one MCQ
   point lost; R64 halves V while remaining clearly ahead of the matched-rank
   PaLU and SVD baselines.

## Machine-readable sources

- `ICLR-results/qwen3-32b/quality-summary.json`: audited 16-row aggregate
  matrix.
- `ICLR-results/qwen3-32b/quality/<run-id>/result.json`: final result for each
  arm.
- `ICLR-results/qwen3-32b/quality-summary.md`: compact generated matrix.
- `ICLR-results/qwen3-32b/checkpoints/Q3-32B-C1-R{64,80,96}/summary.md`: C1
  allocation outcomes and full per-layer schedules.
- `ICLR-results/qwen3-32b/c1/factor-banks/R*-S6/summary.md`: factor-bank
  held-out reconstruction diagnostics.

