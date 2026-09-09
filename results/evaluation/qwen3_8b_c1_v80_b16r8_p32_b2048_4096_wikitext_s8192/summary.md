# Qwen3-8B C1-V80 Base16+R8 Page32 WikiText-2 PPL

The conditional router reconstructs a rank-16 pre-RoPE Key base from the resident V80 code, adds an independently stored rank-8 residual-Key code, forms Page32 log-mass, and selects one fixed physical page set after a max across the four Query heads sharing each GQA Key head. The first Page32 is pinned within each fixed budget.

| Arm | Repository PPL | Token-weighted PPL | vs Dense | vs C1 exact | Physical selected fraction | Wall time (s) | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 6.25855833 | 6.24799023 | -- | -- | -- | 9.84 | 16.083 |
| c1_exact | 6.75649866 | 6.74337834 | +7.96% | -- | -- | 10.53 | 15.561 |
| ours_b2048 | 6.78599644 | 6.77297310 | +8.43% | +0.44% | 0.436922 | 333.41 | 16.173 |
| ours_b4096 | 6.76157419 | 6.74851886 | +8.04% | +0.08% | 0.749909 | 572.45 | 17.023 |

The two routed arms use strict physical budgets B2048/B4096 per KV head. The deployable same-precision persistent KV scalar ratio is `0.343750` versus dense KV.

`Repository PPL` uses Loki's equal mean over non-overlapping blocks, including the shorter final block. `Token-weighted PPL` weights every predicted token equally. Context resets at each block boundary.
