# Qwen3-8B C1-V64 + KQ-SVD exact-Key routing full-window PPL

The same fixed suffix tokens are scored under BF16 dense KV, full C1-V64 with exact QK, and KQ-SVD page routing followed by exact QK over fetched pages.

Windows: `1 x 4096`; scored suffix: `4095` tokens/window; block: `32`; page: `64`.

BF16 dense suffix PPL: `8.32306108`; full C1 exact-QK suffix PPL: `9.83784917` (ratio `1.18199892`).

| R | B | Sparse PPL | Sparse/C1 | Sparse/BF16 | NLL delta vs C1 | Top-1 vs C1 | Physical K fraction | Exact-K MiB/token | GPU KV ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1024 | 9.95503094 | 1.01191132 | 1.19607808 | +1.18409383e-02 | 0.95653236 | 0.57238002 | 84.411 | 0.3125 |
| 32 | 1024 | 9.84087230 | 1.00030730 | 1.18236214 | +3.07248665e-04 | 0.97557998 | 0.58767577 | 86.776 | 0.3750 |

This is full-window teacher-forced PPL, not full-corpus WikiText PPL. Exact K remains physically GPU-resident in this correctness oracle; reported traffic is the logical page-store read volume.
