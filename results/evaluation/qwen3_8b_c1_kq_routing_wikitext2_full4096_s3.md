# Qwen3-8B C1-V64 + KQ-SVD exact-Key routing full-window PPL

The same fixed evaluation tokens are scored under BF16 dense KV, full C1-V64 with exact QK, and KQ-SVD page routing followed by exact QK over fetched pages.

Windows: `18 x 4096`; scored tokens: `4095` tokens/window; block: `32`; page: `64`.

BF16 dense full-window PPL: `6.42898680`; full C1 exact-QK full-window PPL: `7.56807119` (ratio `1.17717946`).

| R | B | Sparse PPL | Sparse/C1 | Sparse/BF16 | NLL delta vs C1 | Top-1 vs C1 | Physical K fraction | Exact-K MiB/token | GPU KV ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1024 | 7.56421979 | 0.99949110 | 1.17658039 | -5.09031195e-04 | 0.97018044 | 0.56915203 | 84.132 | 0.3750 |

This is full-window teacher-forced PPL, not full-corpus WikiText PPL. Exact K remains physically GPU-resident in this correctness oracle; reported traffic is the logical page-store read volume.
