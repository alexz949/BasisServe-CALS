# Qwen3-8B C1-V80 + Loki WikiText-2 PPL

Loki uses a pre-RoPE Key-PCA basis, projects post-RoPE Q/K at runtime, selects 25% of tokens independently for every Query head (2,048 at length 8,192), and recomputes exact 128-dimensional QK on the selected support. The exact-TopK control ranks with full 128-dimensional QK. The Value payload is the fixed C1-V80 ALS5 checkpoint in all C1 arms.

| Arm | Repository PPL | Token-weighted PPL | Tokens | Wall time (s) | Peak GiB |
|---|---:|---:|---:|---:|---:|
| c1_exact | 6.75719947 | 6.74406351 | 299041 | 41.35 | 15.561 |
| c1_exact_topk | 6.75987516 | 6.74672690 | 299041 | 1121.88 | 16.961 |

`Repository PPL` reproduces Loki's equal weighting of each non-overlapping sequence block, including the shorter final block. `Token-weighted PPL` weights every predicted token equally. Context resets at every block boundary.

The query-axis tiling and sparse gather avoid the official Python path's full 8192x8192 materialization; they do not change the selected indices or attention equations.
