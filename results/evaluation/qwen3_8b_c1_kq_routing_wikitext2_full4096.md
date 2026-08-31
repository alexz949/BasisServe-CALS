# Qwen3-8B C1-V64 + KQ-SVD exact-Key routing full-corpus block PPL

The same fixed evaluation tokens are scored under BF16 dense KV, full C1-V64 with exact QK, and KQ-SVD page routing followed by exact QK over fetched pages.

Windows: `73 x 4096`; scored tokens: `4095` tokens/window; block: `32`; page: `64`.

BF16 dense full-corpus block PPL: `6.51276078`; full C1 exact-QK full-corpus block PPL: `7.97934734` (ratio `1.22518662`).

| R | B | Sparse PPL | Sparse/C1 | Sparse/BF16 | NLL delta vs C1 | Top-1 vs C1 | Physical K fraction | Exact-K MiB/token | GPU KV ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1024 | 7.97839859 | 0.99988110 | 1.22504094 | -1.18908033e-04 | 0.96854166 | 0.57094090 | 84.384 | 0.3750 |

This is non-overlapping block-chunked teacher-forced PPL over every complete 4096-token block in WikiText-2 test: 299,008/299,078 tokenizer tokens are covered, the final 70 tokens are dropped, and context resets at each block boundary. Exact K remains physically GPU-resident in this correctness oracle; reported traffic is the logical page-store read volume.
