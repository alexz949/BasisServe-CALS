# Qwen3-8B C1-V64 + KQ-SVD exact-Key routing full-window PPL

The same fixed suffix tokens are scored under BF16 dense KV, full C1-V64 with exact QK, and KQ-SVD page routing followed by exact QK over fetched pages.

Windows: `1 x 4096`; scored suffix: `4095` tokens/window; block: `32`; page: `64`.

BF16 dense suffix PPL: `7.38307040`; full C1 exact-QK suffix PPL: `8.35824767` (ratio `1.13208289`).

| R | B | Sparse PPL | Sparse/C1 | Sparse/BF16 | NLL delta vs C1 | Top-1 vs C1 | Physical K fraction | Exact-K MiB/token | GPU KV ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1024 | 8.37953688 | 1.00254709 | 1.13496641 | +2.54385155e-03 | 0.95311355 | 0.55391217 | 81.783 | 0.3125 |
| 32 | 1024 | 8.35737959 | 0.99989614 | 1.13196531 | -1.03864824e-04 | 0.96923077 | 0.56806230 | 83.975 | 0.3750 |

This is full-window teacher-forced PPL, not full-corpus WikiText PPL. Exact K remains physically GPU-resident in this correctness oracle; reported traffic is the logical page-store read volume.
