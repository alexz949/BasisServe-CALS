# Qwen3-8B C1-V80 + Loki WikiText-2 PPL

Loki uses a pre-RoPE Key-PCA basis, projects post-RoPE Q/K at runtime, selects 25% of tokens independently for every Query head (2,048 at length 8,192), and recomputes exact 128-dimensional QK on the selected support. The Value payload is the fixed C1-V80 ALS5 checkpoint in all C1 arms.

| Arm | Repository PPL | Token-weighted PPL | Tokens | Wall time (s) | Peak GiB |
|---|---:|---:|---:|---:|---:|
| dense | 6.25855833 | 6.24799023 | 299041 | 9.83 | 16.083 |
| c1_exact | 6.75649866 | 6.74337834 | 299041 | 11.54 | 15.561 |
| loki | 7.23697864 | 7.21993989 | 299041 | 255.70 | 16.904 |

`Repository PPL` reproduces Loki's equal weighting of each non-overlapping sequence block, including the shorter final block. `Token-weighted PPL` weights every predicted token equally. Context resets at every block boundary.

The query-axis tiling and sparse gather avoid the official Python path's full 8192x8192 materialization; they do not change the selected indices or attention equations.
