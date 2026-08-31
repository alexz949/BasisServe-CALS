# Qwen3-8B C1-V64 + KQ-SVD exact-Key routing full-window PPL

The same fixed evaluation tokens are scored under BF16 dense KV, full C1-V64 with exact QK, and KQ-SVD page routing followed by exact QK over fetched pages.

Windows: `4 x 4096`; scored tokens: `4095` tokens/window; block: `32`; page: `64`.

BF16 dense full-window PPL: `6.58055844`; full C1 exact-QK full-window PPL: `15.09026596` (ratio `2.29315887`).

| R | B | Sparse PPL | Sparse/C1 | Sparse/BF16 | NLL delta vs C1 | Top-1 vs C1 | Physical K fraction | Exact-K MiB/token | GPU KV ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1024 | 15.21520078 | 1.00827917 | 2.31214431 | +8.24508213e-03 | 0.93980464 | 0.55558239 | 82.024 | 0.3125 |
| 32 | 1024 | 15.06425846 | 0.99827654 | 2.28920670 | -1.72494857e-03 | 0.96398046 | 0.57170432 | 84.490 | 0.3750 |

This is full-window teacher-forced PPL, not full-corpus WikiText PPL. Exact K remains physically GPU-resident in this correctness oracle; reported traffic is the logical page-store read volume.
