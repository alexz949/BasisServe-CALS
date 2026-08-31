# Qwen3-8B C1-V64 + KQ-SVD exact-Key routing full-window PPL

The same fixed suffix tokens are scored under BF16 dense KV, full C1-V64 with exact QK, and KQ-SVD page routing followed by exact QK over fetched pages.

Windows: `1 x 4096`; scored suffix: `4095` tokens/window; block: `32`; page: `64`.

BF16 dense suffix PPL: `4.85684831`; full C1 exact-QK suffix PPL: `5.98040658` (ratio `1.23133485`).

| R | B | Sparse PPL | Sparse/C1 | Sparse/BF16 | NLL delta vs C1 | Top-1 vs C1 | Physical K fraction | Exact-K MiB/token | GPU KV ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1024 | 6.07516976 | 1.01584561 | 1.25084610 | +1.57213783e-02 | 0.94822955 | 0.55807709 | 82.365 | 0.3125 |
| 32 | 1024 | 6.00581012 | 1.00424780 | 1.23656531 | +4.23879890e-03 | 0.96434676 | 0.57375728 | 84.776 | 0.3750 |

This is full-window teacher-forced PPL, not full-corpus WikiText PPL. Exact K remains physically GPU-resident in this correctness oracle; reported traffic is the logical page-store read volume.
