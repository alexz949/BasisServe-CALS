# Qwen3-8B C1-V64 + KQ-SVD exact-Key routing full-window PPL

The same fixed suffix tokens are scored under BF16 dense KV, full C1-V64 with exact QK, and KQ-SVD page routing followed by exact QK over fetched pages.

Windows: `1 x 4096`; scored suffix: `4095` tokens/window; block: `32`; page: `64`.

BF16 dense suffix PPL: `6.28313470`; full C1 exact-QK suffix PPL: `105.44875832` (ratio `16.78282631`).

| R | B | Sparse PPL | Sparse/C1 | Sparse/BF16 | NLL delta vs C1 | Top-1 vs C1 | Physical K fraction | Exact-K MiB/token | GPU KV ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1024 | 105.75227092 | 1.00287829 | 16.83113224 | +2.87416036e-03 | 0.90134310 | 0.53796028 | 79.538 | 0.3125 |
| 32 | 1024 | 104.25951784 | 0.98872210 | 16.59355129 | -1.13419770e-02 | 0.94676435 | 0.55732194 | 82.431 | 0.3750 |

This is full-window teacher-forced PPL, not full-corpus WikiText PPL. Exact K remains physically GPU-resident in this correctness oracle; reported traffic is the logical page-store read volume.
