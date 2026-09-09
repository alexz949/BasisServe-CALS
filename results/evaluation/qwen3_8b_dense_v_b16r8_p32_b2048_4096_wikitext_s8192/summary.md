# Qwen3-8B Dense-V Base16+R8 Page32 WikiText-2 PPL

The original dense rank-128 Values and output projection are used in every arm. The fixed C1-V80 encoder is composed algebraically into the existing Base16 map, so routing is unchanged in real arithmetic and no router is refitted. Residual-R8, Page32, normalized group-max selection, and one pinned prefix page match the C1-V80 experiment.

| Arm | Repository PPL | Token-weighted PPL | vs Dense | Physical selected fraction | Wall time (s) | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| dense | 6.25855833 | 6.24799023 | -- | -- | 35.36 | 16.083 |
| ours_b2048 | 6.28271066 | 6.27226576 | +0.386% | 0.436924 | 1690.68 | 16.928 |
| ours_b4096 | 6.26243246 | 6.25192547 | +0.062% | 0.749910 | 2899.10 | 17.965 |

Budgets are fixed union-after-GQA physical token budgets per KV head. Repository PPL is the equal mean over non-overlapping sequence-block NLLs; token-weighted PPL is also recorded.
