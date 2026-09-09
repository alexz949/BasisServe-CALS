# Qwen3-8B Dense-V Exact Top-K and Loki WikiText-2 PPL

All arms use the original dense rank-128 Value projection and dense output projection. Exact Top-K ranks tokens with full rank-128 QK; Loki ranks tokens with rank-32 projected QK. Both independently select 25% of tokens per Query head, recompute exact rank-128 QK on the selected support, and apply selected-token softmax to dense Values.

| Arm | Repository PPL | Token-weighted PPL | Tokens | Wall time (s) | Peak GiB |
|---|---:|---:|---:|---:|---:|
| dense | 6.25855833 | 6.24799023 | 299041 | 34.89 | 16.083 |
| exact_topk | 6.25832784 | 6.24760602 | 299041 | 1321.54 | 17.910 |
| loki | 6.39377902 | 6.38262786 | 299041 | 1318.00 | 17.854 |

`Repository PPL` reproduces Loki's equal weighting of non-overlapping sequence blocks, including the shorter final block. `Token-weighted PPL` weights every predicted token equally. Context resets at every block boundary.
