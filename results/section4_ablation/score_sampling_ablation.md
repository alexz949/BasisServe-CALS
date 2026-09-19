# Section 4 Fisher-Free Score Routing: Query-Sampling Ablation

Llama-3.1-8B-Instruct with Dense V128 and original W_O. All routed methods use
B16R16, Page32, B2048 with sink32/recent64 inside the budget, and exact selected
post-RoPE keys for final attention. Fitting uses 40 ALS sweeps and PCG100.

## Routing diagnostics

| Method | Routed page recall | Page KL ↓ | Post-W_O rel-MSE ↓ |
|---|---:|---:|---:|
| Exact-K | 1.000000 | 0.000000 | 0.006659 |
| Page-Fisher | 0.838779 | 0.120886 | 0.007407 |
| Score-MSE (Page-Fisher init) | 0.847181 | 0.126483 | 0.007489 |
| Fixed Score-only | 0.841876 | 0.136844 | 0.007513 |
| Query-Gram Score-only | 0.847167 | 0.126464 | 0.007477 |

## Downstream evaluation

| Method | Hard RULER-64K | LongBench-32K |
|---|---:|---:|
| Dense Full | — | 37.5758 |
| Exact-K | — | 37.8692 |
| Page-Fisher | 59.5111 | 37.8673 |
| Score-MSE (Page-Fisher init) | 58.9611 | 38.3949 |
| Fixed Score-only | 59.2333 | 38.0524 |
| Query-Gram Score-only | 59.5944 | 38.4141 |

## Fixed sampling versus Query-Gram selection

Query-Gram selection reduces page KL by **7.59%** relative to fixed
length-stratified positions. Its downstream point estimate is higher by
**0.3611** on Hard RULER and
**0.3617** on LongBench.

- Hard RULER: 6 wins / 3 losses / 291 ties; paired 95% CI [-0.5556, 1.3722].
- LongBench: 27 wins / 27 losses / 138 ties; paired 95% CI [-0.1975, 1.1702].

The confidence intervals include zero. We therefore treat Query-Gram selection as
an inference-free calibration refinement rather than an essential component. The
primary result is that unweighted causal QK-score fitting works without loss
gradients, Fisher weights, or task labels.
