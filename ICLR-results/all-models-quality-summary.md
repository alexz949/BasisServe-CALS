# ICLR V-side quality results

## Scope

This report summarizes the complete V-side quality matrices for
Qwen3-8B-Base, Llama-3.1-8B, and Llama-2-7B. Each matrix contains Dense and
four compressed method families at nominal equivalent ranks R96, R80, and
R64, for 16 runs per model and 48 runs in total.

| Model | Complete runs | Matrix status | Expected-trend checks |
| --- | ---: | :---: | ---: |
| Qwen3-8B-Base | 16/16 | Complete | 8/9 |
| Llama-3.1-8B | 16/16 | Complete | 8/9 |
| Llama-2-7B | 16/16 | Complete | 6/9 |
| **Total** | **48/48** | **Complete** | **22/27** |

The expected-trend checks are diagnostic assertions, not completion checks.
Every result is complete. Failed trend checks only indicate that PaLU did not
strictly improve from M-LRD to G-LRD2 to G-LRD4 at every rank and on every
aggregate metric.

## Methods and rank convention

- **Dense:** dense K and dense V.
- **Weight-SVD:** dense K; ordinary truncated SVD is applied independently to
  each physical V head, without calibration activations or Fisher allocation.
- **PaLU M-LRD:** activation-weighted SVD with Fisher rank allocation and one
  physical V head per low-rank group.
- **PaLU G-LRD2/G-LRD4:** activation-weighted SVD with Fisher rank allocation
  and groups of two/four physical V heads.
- **C1 + Two-Sided KL:** activation-weighted-SVD initialization, a fixed sixth
  ALS encoder sweep followed by the final decoder refit, and two-sided KL rank
  allocation with alpha = 1.

R96/R80/R64 denotes equivalent rank per physical head. For grouped PaLU, the
nominal group ranks are therefore 2R for G-LRD2 and 4R for G-LRD4. Fisher block
allocation can make PaLU's realized retained-V ratio differ slightly from the
nominal target; each table reports the realized ratio. C1 uses exact average
rank budgets and retains 75%, 62.5%, and 50% of V at R96, R80, and R64.

The trained C1 factor banks are R32, R48, R64, R80, R96, and R112. R112 is the
highest trained bank needed by the selected R64/R80/R96 layer schedules; no
R128 bank is used in these reported checkpoints.

## Evaluation protocol

| Item | Setting |
| --- | --- |
| Environment | `basis` |
| `datasets` | 5.0.0 |
| `lm-eval` | 0.4.11 |
| Hardware | 4 x NVIDIA L40S on `lovelace` |
| WikiText-2 | Full `test` corpus, sequence length 2048, FP32 loss |
| C4 | 128 fixed document-disjoint `validation` samples, 2048 tokens, FP32 loss |
| MCQ tasks | ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande, BoolQ, OpenBookQA; zero-shot |
| MCQ metric | `acc_norm` when available, otherwise `acc` |
| MCQ average | Unweighted arithmetic mean of the seven task accuracies |
| PPL batch size | 2 |
| lm-eval batch size | 8 |

Pinned model revisions:

- Qwen3-8B-Base: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- Llama-3.1-8B: `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Llama-2-7B: `01c7f73d771dfac7d292323805ebc428287df4f9`

## Main findings

At the same nominal rank, C1 is the best compressed method on all three
aggregate metrics in five of the nine model/rank comparisons. The four
WikiText-2 exceptions are Llama-3.1-8B at every rank and Llama-2-7B at R96,
where PaLU G-LRD4 has lower WikiText-2 PPL. C1 nevertheless has the best C4 PPL
and MCQ average for every model and every rank.

| Model | Rank | Best WT2 compressed | Best C4 compressed | Best MCQ compressed |
| --- | ---: | --- | --- | --- |
| Qwen3-8B | R96 | C1: 7.282905 | C1: 9.344231 | C1: 0.695621 |
| Qwen3-8B | R80 | C1: 7.700490 | C1: 9.584222 | C1: 0.685690 |
| Qwen3-8B | R64 | C1: 8.259953 | C1: 10.059465 | C1: 0.676320 |
| Llama-3.1-8B | R96 | G-LRD4: 6.983324 | C1: 9.034154 | C1: 0.701317 |
| Llama-3.1-8B | R80 | G-LRD4: 7.635286 | C1: 9.670008 | C1: 0.691584 |
| Llama-3.1-8B | R64 | G-LRD4: 9.364653 | C1: 10.547192 | C1: 0.669186 |
| Llama-2-7B | R96 | G-LRD4: 6.038581 | C1: 6.716750 | C1: 0.659456 |
| Llama-2-7B | R80 | C1: 6.401839 | C1: 6.983492 | C1: 0.657998 |
| Llama-2-7B | R64 | C1: 6.987429 | C1: 7.400864 | C1: 0.641642 |

Relative to Dense, C1 preserves MCQ accuracy particularly well. The following
table gives relative PPL increases and absolute MCQ-average changes in
percentage points.

| Model | Rank | WT2 vs Dense | C4 vs Dense | MCQ Avg vs Dense |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B | R96 | +4.00% | +1.92% | -0.87 pp |
| Qwen3-8B | R80 | +9.97% | +4.53% | -1.86 pp |
| Qwen3-8B | R64 | +17.96% | +9.72% | -2.80 pp |
| Llama-3.1-8B | R96 | +17.66% | +8.38% | -0.66 pp |
| Llama-3.1-8B | R80 | +35.44% | +16.01% | -1.63 pp |
| Llama-3.1-8B | R64 | +54.93% | +26.53% | -3.87 pp |
| Llama-2-7B | R96 | +10.57% | +4.50% | -0.72 pp |
| Llama-2-7B | R80 | +16.96% | +8.65% | -0.87 pp |
| Llama-2-7B | R64 | +27.66% | +15.15% | -2.51 pp |

Weight-SVD is the weakest compressed baseline at every rank for every model.
Its PPL degradation is especially severe on Qwen3-8B, demonstrating that
weight-only reconstruction is not a useful proxy for preserving this model's
language-model behavior.

## Qwen3-8B-Base

| Run ID | Method | Retained V | WT2 PPL | C4 PPL | ARC-E | ARC-C | HS | PIQA | WG | BoolQ | OBQA | Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q3-8B-Dense | Dense | 1.000000 | 7.002509 | 9.168594 | 0.801768 | 0.569113 | 0.786497 | 0.793798 | 0.727703 | 0.831193 | 0.420000 | 0.704296 |
| Q3-8B-SVD-R96 | Weight-SVD | 0.750000 | 4530.954327 | 847.913367 | 0.337963 | 0.261092 | 0.381299 | 0.563656 | 0.513812 | 0.576453 | 0.302000 | 0.419468 |
| Q3-8B-SVD-R80 | Weight-SVD | 0.625000 | 90181.147396 | 19263.675824 | 0.311869 | 0.282423 | 0.319259 | 0.537541 | 0.494081 | 0.544343 | 0.294000 | 0.397645 |
| Q3-8B-SVD-R64 | Weight-SVD | 0.500000 | 739229.370152 | 307515.739391 | 0.289141 | 0.269625 | 0.288588 | 0.521219 | 0.483820 | 0.557798 | 0.274000 | 0.383456 |
| Q3-8B-PALUM-R96 | PaLU M-LRD | 0.715278 | 8.663951 | 10.751474 | 0.740320 | 0.522184 | 0.768273 | 0.779108 | 0.719021 | 0.793578 | 0.394000 | 0.673783 |
| Q3-8B-PALUM-R80 | PaLU M-LRD | 0.645833 | 9.224107 | 11.294961 | 0.728956 | 0.500853 | 0.747162 | 0.780740 | 0.711918 | 0.782263 | 0.400000 | 0.664556 |
| Q3-8B-PALUM-R64 | PaLU M-LRD | 0.493056 | 14.839074 | 16.866396 | 0.626263 | 0.420648 | 0.699562 | 0.738847 | 0.670087 | 0.762997 | 0.356000 | 0.610629 |
| Q3-8B-PALUG2-R96 | PaLU G-LRD2 | 0.750000 | 8.298955 | 10.480836 | 0.768519 | 0.536689 | 0.778630 | 0.793798 | 0.707182 | 0.805199 | 0.416000 | 0.686574 |
| Q3-8B-PALUG2-R80 | PaLU G-LRD2 | 0.625000 | 9.713508 | 12.007475 | 0.679293 | 0.490614 | 0.763692 | 0.778020 | 0.709550 | 0.775229 | 0.396000 | 0.656057 |
| Q3-8B-PALUG2-R64 | PaLU G-LRD2 | 0.493056 | 12.386457 | 14.150045 | 0.611532 | 0.456485 | 0.729735 | 0.757889 | 0.696922 | 0.760245 | 0.376000 | 0.626972 |
| Q3-8B-PALUG4-R96 | PaLU G-LRD4 | 0.748264 | 8.244049 | 10.403133 | 0.771044 | 0.556314 | 0.786795 | 0.795974 | 0.718232 | 0.793272 | 0.408000 | 0.689947 |
| Q3-8B-PALUG4-R80 | PaLU G-LRD4 | 0.625000 | 9.567456 | 11.782581 | 0.755471 | 0.537543 | 0.773750 | 0.782372 | 0.709550 | 0.738226 | 0.404000 | 0.671559 |
| Q3-8B-PALUG4-R64 | PaLU G-LRD4 | 0.496528 | 10.650523 | 12.870647 | 0.697811 | 0.501706 | 0.754431 | 0.767138 | 0.709550 | 0.720183 | 0.406000 | 0.650974 |
| Q3-8B-C1-R96 | C1 + Two-Sided KL | 0.750000 | 7.282905 | 9.344231 | 0.788300 | 0.544369 | 0.771858 | 0.792709 | 0.730071 | 0.818043 | 0.424000 | 0.695621 |
| Q3-8B-C1-R80 | C1 + Two-Sided KL | 0.625000 | 7.700490 | 9.584222 | 0.796717 | 0.545222 | 0.759012 | 0.793798 | 0.716654 | 0.780428 | 0.408000 | 0.685690 |
| Q3-8B-C1-R64 | C1 + Two-Sided KL | 0.500000 | 8.259953 | 10.059465 | 0.771044 | 0.518771 | 0.735312 | 0.787813 | 0.719021 | 0.804281 | 0.398000 | 0.676320 |

Qwen3-8B observations:

- C1 is the best compressed method on WT2 PPL, C4 PPL, and MCQ average at
  all three ranks.
- The strict PaLU grouping trend passes at R96 and R64. At R80, G-LRD2 is
  worse than M-LRD, although G-LRD4 is the best PaLU geometry.
- C1-R64 retains half of V while keeping the MCQ average within 2.80 points of
  Dense.

## Llama-3.1-8B

| Run ID | Method | Retained V | WT2 PPL | C4 PPL | ARC-E | ARC-C | HS | PIQA | WG | BoolQ | OBQA | Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L31-8B-Dense | Dense | 1.000000 | 6.239787 | 8.335565 | 0.811027 | 0.536689 | 0.789285 | 0.809576 | 0.735596 | 0.821101 | 0.452000 | 0.707896 |
| L31-8B-SVD-R96 | Weight-SVD | 0.750000 | 56.172585 | 87.709555 | 0.413300 | 0.275597 | 0.396335 | 0.634385 | 0.520916 | 0.489602 | 0.292000 | 0.431734 |
| L31-8B-SVD-R80 | Weight-SVD | 0.625000 | 173.884904 | 223.442079 | 0.372475 | 0.228669 | 0.331010 | 0.589771 | 0.532755 | 0.451070 | 0.272000 | 0.396821 |
| L31-8B-SVD-R64 | Weight-SVD | 0.500000 | 470.468302 | 397.687336 | 0.337542 | 0.233788 | 0.303724 | 0.554407 | 0.514601 | 0.407645 | 0.266000 | 0.373958 |
| L31-8B-PALUM-R96 | PaLU M-LRD | 0.742188 | 7.345675 | 9.537923 | 0.781145 | 0.516212 | 0.770862 | 0.800871 | 0.737174 | 0.774312 | 0.440000 | 0.688654 |
| L31-8B-PALUM-R80 | PaLU M-LRD | 0.656250 | 8.354250 | 10.598922 | 0.739899 | 0.482082 | 0.749751 | 0.793798 | 0.737964 | 0.772171 | 0.402000 | 0.668238 |
| L31-8B-PALUM-R64 | PaLU M-LRD | 0.476562 | 14.175872 | 14.663795 | 0.632576 | 0.395904 | 0.675463 | 0.754625 | 0.707972 | 0.648318 | 0.374000 | 0.598408 |
| L31-8B-PALUG2-R96 | PaLU G-LRD2 | 0.746094 | 7.184943 | 9.306987 | 0.775673 | 0.511945 | 0.778032 | 0.797062 | 0.727703 | 0.793272 | 0.440000 | 0.689098 |
| L31-8B-PALUG2-R80 | PaLU G-LRD2 | 0.628906 | 8.134802 | 10.210429 | 0.719697 | 0.472696 | 0.753734 | 0.784004 | 0.730860 | 0.784404 | 0.420000 | 0.666485 |
| L31-8B-PALUG2-R64 | PaLU G-LRD2 | 0.503906 | 10.842140 | 12.207576 | 0.669613 | 0.406143 | 0.700159 | 0.757889 | 0.709550 | 0.722018 | 0.378000 | 0.620482 |
| L31-8B-PALUG4-R96 | PaLU G-LRD4 | 0.751953 | 6.983324 | 9.076497 | 0.789141 | 0.522184 | 0.783011 | 0.804135 | 0.729282 | 0.800917 | 0.440000 | 0.695524 |
| L31-8B-PALUG4-R80 | PaLU G-LRD4 | 0.626953 | 7.635286 | 9.745016 | 0.750421 | 0.494027 | 0.767975 | 0.789445 | 0.726914 | 0.784404 | 0.424000 | 0.676741 |
| L31-8B-PALUG4-R64 | PaLU G-LRD4 | 0.498047 | 9.364653 | 11.124858 | 0.687290 | 0.445392 | 0.730532 | 0.772579 | 0.721389 | 0.758104 | 0.392000 | 0.643898 |
| L31-8B-C1-R96 | C1 + Two-Sided KL | 0.750000 | 7.341578 | 9.034154 | 0.803872 | 0.527304 | 0.778231 | 0.804135 | 0.730860 | 0.816820 | 0.448000 | 0.701317 |
| L31-8B-C1-R80 | C1 + Two-Sided KL | 0.625000 | 8.450979 | 9.670008 | 0.779882 | 0.512799 | 0.765485 | 0.795430 | 0.737964 | 0.801529 | 0.448000 | 0.691584 |
| L31-8B-C1-R64 | C1 + Two-Sided KL | 0.500000 | 9.667477 | 10.547192 | 0.745370 | 0.478669 | 0.741984 | 0.785637 | 0.722968 | 0.777676 | 0.432000 | 0.669186 |

Llama-3.1-8B observations:

- PaLU G-LRD4 has the lowest compressed WikiText-2 PPL at every rank.
- C1 has the lowest compressed C4 PPL and highest compressed MCQ average at
  every rank.
- The strict PaLU grouping trend passes at R96 and R64. At R80, its average
  accuracy changes 0.668238 -> 0.666485 -> 0.676741, so M-to-G2 is not
  monotonic.

## Llama-2-7B

| Run ID | Method | Retained V | WT2 PPL | C4 PPL | ARC-E | ARC-C | HS | PIQA | WG | BoolQ | OBQA | Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2-7B-Dense | Dense | 1.000000 | 5.473670 | 6.427415 | 0.747896 | 0.463311 | 0.758813 | 0.788357 | 0.688240 | 0.778287 | 0.442000 | 0.666700 |
| L2-7B-SVD-R96 | Weight-SVD | 0.750000 | 71.683667 | 70.427030 | 0.416667 | 0.274744 | 0.475901 | 0.638194 | 0.563536 | 0.555657 | 0.292000 | 0.459528 |
| L2-7B-SVD-R80 | Weight-SVD | 0.625000 | 717.478130 | 398.178761 | 0.320286 | 0.249147 | 0.307309 | 0.555495 | 0.523283 | 0.423853 | 0.286000 | 0.380768 |
| L2-7B-SVD-R64 | Weight-SVD | 0.500000 | 2063.970889 | 1162.636981 | 0.285774 | 0.236348 | 0.275742 | 0.519042 | 0.519337 | 0.387156 | 0.284000 | 0.358200 |
| L2-7B-PALUM-R96 | PaLU M-LRD | 0.757812 | 6.137560 | 7.048691 | 0.707492 | 0.443686 | 0.746465 | 0.781284 | 0.671665 | 0.770336 | 0.426000 | 0.649561 |
| L2-7B-PALUM-R80 | PaLU M-LRD | 0.625000 | 6.916416 | 7.806620 | 0.646465 | 0.410410 | 0.719279 | 0.766594 | 0.661405 | 0.673394 | 0.410000 | 0.612507 |
| L2-7B-PALUM-R64 | PaLU M-LRD | 0.515625 | 8.612951 | 9.167719 | 0.581229 | 0.366041 | 0.668094 | 0.749728 | 0.641673 | 0.654128 | 0.372000 | 0.576128 |
| L2-7B-PALUG2-R96 | PaLU G-LRD2 | 0.746094 | 6.114865 | 6.946324 | 0.704125 | 0.447099 | 0.743278 | 0.775299 | 0.684294 | 0.756269 | 0.430000 | 0.648623 |
| L2-7B-PALUG2-R80 | PaLU G-LRD2 | 0.613281 | 7.246017 | 7.865628 | 0.638468 | 0.406143 | 0.703047 | 0.749184 | 0.659826 | 0.649541 | 0.404000 | 0.601459 |
| L2-7B-PALUG2-R64 | PaLU G-LRD2 | 0.492188 | 9.372558 | 9.458851 | 0.558923 | 0.349829 | 0.643199 | 0.732862 | 0.650355 | 0.571560 | 0.358000 | 0.552104 |
| L2-7B-PALUG4-R96 | PaLU G-LRD4 | 0.751953 | 6.038581 | 6.883553 | 0.704966 | 0.442833 | 0.741785 | 0.776387 | 0.670087 | 0.771865 | 0.436000 | 0.649132 |
| L2-7B-PALUG4-R80 | PaLU G-LRD4 | 0.626953 | 6.899082 | 7.585134 | 0.653199 | 0.403584 | 0.711014 | 0.765506 | 0.654301 | 0.710703 | 0.392000 | 0.612901 |
| L2-7B-PALUG4-R64 | PaLU G-LRD4 | 0.498047 | 8.333970 | 8.684767 | 0.596801 | 0.366894 | 0.663613 | 0.750272 | 0.643252 | 0.622018 | 0.368000 | 0.572979 |
| L2-7B-C1-R96 | C1 + Two-Sided KL | 0.750000 | 6.052101 | 6.716750 | 0.730219 | 0.446246 | 0.751842 | 0.776387 | 0.691397 | 0.784098 | 0.436000 | 0.659456 |
| L2-7B-C1-R80 | C1 + Two-Sided KL | 0.625000 | 6.401839 | 6.983492 | 0.729377 | 0.449659 | 0.739793 | 0.779652 | 0.690608 | 0.778899 | 0.438000 | 0.657998 |
| L2-7B-C1-R64 | C1 + Two-Sided KL | 0.500000 | 6.987429 | 7.400864 | 0.699495 | 0.427474 | 0.722764 | 0.776931 | 0.692186 | 0.770642 | 0.402000 | 0.641642 |

Llama-2-7B observations:

- C1 has the lowest compressed C4 PPL and highest compressed MCQ average at
  every rank.
- G-LRD4 is only 0.013520 PPL below C1 on WikiText-2 at R96; C1 is best on
  WikiText-2 at R80 and R64.
- C1-R80 is a strong operating point: it retains 62.5% of V while losing only
  0.87 MCQ-average points relative to Dense.
- PaLU does not show a strict M-to-G2-to-G4 improvement at R96, R80, or R64,
  so the corresponding three trend diagnostics fail even though all runs are
  complete.

## Result locations

- Qwen3-8B: `ICLR-results/qwen3-8b/quality-summary.{json,md}` and
  `ICLR-results/qwen3-8b/quality/`
- Llama-3.1-8B: `ICLR-results/llama31-8b/quality-summary.{json,md}` and
  `ICLR-results/llama31-8b/quality/`
- Llama-2-7B: `ICLR-results/llama2-7b/quality-summary.{json,md}` and
  `ICLR-results/llama2-7b/quality/`

All numerical values in this report are copied from the three audited
machine-readable `quality-summary.json` files.
