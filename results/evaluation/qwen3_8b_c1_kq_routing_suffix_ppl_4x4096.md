# Qwen3-8B C1-V64 + KQ-SVD exact-Key routing suffix PPL

The same fixed suffix tokens are scored under BF16 dense KV, full C1-V64 with exact QK, and KQ-SVD page routing followed by exact QK over fetched pages.

Windows: `4 x 4096`; scored suffix: `128` tokens/window; block: `32`; page: `64`.

BF16 dense suffix PPL: `5.88819237`; full C1 exact-QK suffix PPL: `12.26681085` (ratio `2.08328976`).

| R | B | Sparse PPL | Sparse/C1 | Sparse/BF16 | NLL delta vs C1 | Top-1 vs C1 | Physical K fraction | Exact-K MiB/token | GPU KV ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 512 | 12.32738033 | 1.00493767 | 2.09357636 | +4.92552044e-03 | 0.91015625 | 0.20360678 | 59.697 | 0.3750 |
| 32 | 1024 | 12.23953279 | 0.99777627 | 2.07865709 | -2.22620483e-03 | 0.96484375 | 0.38038783 | 109.938 | 0.3750 |
| 64 | 512 | 12.36183584 | 1.00774651 | 2.09942799 | +7.71666131e-03 | 0.92382812 | 0.20888873 | 61.304 | 0.5000 |
| 64 | 1024 | 12.19432096 | 0.99409057 | 2.07097870 | -5.92696285e-03 | 0.95703125 | 0.39043091 | 112.848 | 0.5000 |

This is suffix-conditioned teacher-forced PPL, not full-corpus WikiText PPL. Exact K remains physically GPU-resident in this correctness oracle; reported traffic is the logical page-store read volume.
