# Qwen3-8B C1-V64 + KQ-SVD exact-Key routing full-window PPL

The same fixed evaluation tokens are scored under BF16 dense KV, full C1-V64 with exact QK, and KQ-SVD page routing followed by exact QK over fetched pages.

Windows: `19 x 4096`; scored tokens: `4095` tokens/window; block: `32`; page: `64`.

BF16 dense full-window PPL: `6.27936686`; full C1 exact-QK full-window PPL: `8.55588712` (ratio `1.36253978`).

| R | B | Sparse PPL | Sparse/C1 | Sparse/BF16 | NLL delta vs C1 | Top-1 vs C1 | Physical K fraction | Exact-K MiB/token | GPU KV ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1024 | 8.55784253 | 1.00022855 | 1.36285118 | +2.28519293e-04 | 0.96896086 | 0.56982503 | 84.221 | 0.3750 |

This is full-window teacher-forced PPL, not full-corpus WikiText PPL. Exact K remains physically GPU-resident in this correctness oracle; reported traffic is the logical page-store read volume.
