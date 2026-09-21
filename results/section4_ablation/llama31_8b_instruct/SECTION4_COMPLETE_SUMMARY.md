# Section 4 Ablations — Llama-3.1-8B-Instruct

All results in this directory use `meta-llama/Llama-3.1-8B-Instruct` with Dense
V128 and the original output projection. The frozen model-config SHA-256 is
`29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e`;
the identity-manifest SHA-256 is
`6de42f7a477cfa9bd5a6d1231fb73fe1b66cd6ddc306bd344422125a0c590f49`.

Unless stated otherwise, routing uses 64K sequences, Page32, B2048 with sink32
and recent64 counted inside the budget, 64 fit windows and 16 held-out windows,
Q64/Q32, 40 ALS sweeps, and PCG100. The proxy only selects support; final
attention uses the exact selected post-RoPE keys.

## Base and residual components

| Method | Logical width | Extra state | Routed recall | Page KL ↓ | Pooled post-WO rel-MSE ↓ | RULER-64K |
|---|---:|---:|---:|---:|---:|---:|
| R16-only | 16 | 16 | 0.687520 | 0.436918 | 0.004767 | 76.2500 |
| R20-only | 20 | 20 | 0.726301 | 0.322598 | 0.004132 | 83.8258 |
| B4R16 | 20 | 16 | 0.823286 | 0.146185 | 0.003528 | 85.4735 |
| B16-only | 16 | 0 | 0.711161 | 0.742223 | 0.008254 | 55.2841 |
| B16R16 | 32 | 16 | 0.838779 | 0.120886 | 0.003436 | 85.4735 |
| R32-only | 32 | 32 | 0.801290 | 0.156492 | 0.003452 | 85.6629 |
| Exact-K | 128 | 128 | 1.000000 | 0.000000 | 0.002960 | 85.6629 |

At matched logical width 20, B4R16 improves recall by 9.70 percentage points
and reduces page KL by 54.7% relative to R20-only. At width 32, B16R16 improves
recall by 3.75 points and reduces page KL by 22.8% relative to R32-only while
using half as much additional persistent state.

## Base-only rank sweep

| Base rank | Predictable energy | Routed recall | Page KL ↓ |
|---:|---:|---:|---:|
| 4 | 0.416073 | 0.680362 | 0.819484 |
| 8 | 0.570560 | 0.695947 | 0.781046 |
| 16 | 0.739092 | 0.711161 | 0.742223 |
| 24 | 0.834278 | 0.717965 | 0.726211 |
| 32 | 0.892529 | 0.722319 | 0.716204 |
| 48 | 0.955701 | 0.726161 | 0.707433 |
| 64 | 0.984113 | 0.727969 | 0.703365 |
| 80 | 0.995546 | 0.728818 | 0.701597 |
| 96 | 0.999174 | 0.729177 | 0.701021 |

Nearly perfect key predictability does not imply good routing: B96-only captures
99.9% of predictable energy but remains far behind B16R16 in recall and page KL.

## Residual fitting objective

| Objective | Routed recall | Page KL ↓ | Pooled post-WO rel-MSE ↓ | RULER-64K |
|---|---:|---:|---:|---:|
| Residual-MSE | 0.786112 | 0.427594 | 0.005205 | 84.1477 |
| Score-MSE, Page-Fisher init | **0.847181** | 0.126483 | 0.003471 | **86.0417** |
| Page-Fisher | 0.838779 | **0.120886** | **0.003436** | 85.4735 |

Residual reconstruction alone is insufficient. Score-MSE gives the best recall
and RULER point estimate, while Page-Fisher gives the best KL and post-WO error.

## Fisher-free Score-only fitting

| Method | Routed recall | Page KL ↓ | Pooled post-WO rel-MSE ↓ | Hard RULER-64K | LongBench-32K |
|---|---:|---:|---:|---:|---:|
| Page-Fisher | 0.838779 | **0.120886** | **0.003436** | 59.5111 | 37.8673 |
| Score-MSE, Page-Fisher init | **0.847181** | 0.126483 | 0.003471 | 58.9611 | 38.3949 |
| Fixed Score-only | 0.841876 | 0.136844 | 0.003485 | 59.2333 | 38.0524 |
| Query-Gram Score-only | 0.847167 | 0.126464 | 0.003466 | **59.5944** | **38.4141** |

Query-Gram selection reduces page KL by 7.59% relative to fixed
length-stratified sampling. Its downstream gains are small and not statistically
resolved: +0.3611 Hard RULER (95% paired CI [-0.5556, 1.3722]) and +0.3617
LongBench (95% paired CI [-0.1975, 1.1702]). The clean claim is therefore that
unweighted causal QK-score fitting is sufficient; Query-Gram is an optional
inference-free calibration refinement.

LongBench references are Dense Full 37.5758 and Exact-K 37.8692.

## Pre-RoPE versus post-RoPE base prediction

| Predictor | Post-RoPE rel-MSE ↓ | Routed recall | Page KL ↓ | Attention mass |
|---|---:|---:|---:|---:|
| Pre-RoPE B16 | **0.190069** | **0.710990** | **0.741199** | **0.930427** |
| Post-RoPE B16 | 0.563237 | 0.243576 | 1.979172 | 0.869636 |

| Position | Pre-RoPE recall | Post-RoPE recall | Pre-RoPE KL | Post-RoPE KL |
|---|---:|---:|---:|---:|
| 0–8K | 0.862186 | 0.690489 | 0.273186 | 0.815372 |
| 8–32K | 0.696033 | 0.222018 | 0.506602 | 1.705767 |
| 32–64K | 0.671694 | 0.112247 | 1.051014 | 2.541932 |

A position-independent low-rank map should be fitted in the pre-RoPE coordinate
frame. Direct post-RoPE prediction creates a coordinate-frame mismatch that
worsens sharply with position.

## Main conclusion

The Instruct/Dense-V128 rerun supports three distinct points: a low-rank base and
residual are complementary at matched route width; score-aware residual fitting
is necessary whereas Fisher weighting is optional; and the base predictor must
operate before RoPE when it has no explicit position input.
