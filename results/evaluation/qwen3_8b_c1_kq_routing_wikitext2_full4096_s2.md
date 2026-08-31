# Qwen3-8B C1-V64 + KQ-SVD exact-Key routing full-window PPL

The same fixed evaluation tokens are scored under BF16 dense KV, full C1-V64 with exact QK, and KQ-SVD page routing followed by exact QK over fetched pages.

Windows: `18 x 4096`; scored tokens: `4095` tokens/window; block: `32`; page: `64`.

BF16 dense full-window PPL: `7.14966173`; full C1 exact-QK full-window PPL: `8.42634064` (ratio `1.17856494`).

| R | B | Sparse PPL | Sparse/C1 | Sparse/BF16 | NLL delta vs C1 | Top-1 vs C1 | Physical K fraction | Exact-K MiB/token | GPU KV ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1024 | 8.42851320 | 1.00025783 | 1.17886881 | +2.57796744e-04 | 0.96792837 | 0.57235967 | 84.588 | 0.3750 |

This is full-window teacher-forced PPL, not full-corpus WikiText PPL. Exact K remains physically GPU-resident in this correctness oracle; reported traffic is the logical page-store read volume.
