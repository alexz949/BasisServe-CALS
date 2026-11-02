# qwen3 GQA V128 joint fit

- K remains dense; each of 8 KV heads retains a mean V rank of 128/128.
- Total KV-cache retention is 100% (0% reduction).
- Fit windows: 256; held-out windows: 64.
- Encoder sweeps: 6; final decoder refitted.
- Mean held-out factor-dtype relative MSE: `0`.

| Layer | Rank | Solver | Held-out relative MSE |
|---:|---:|:---|---:|
| 0 | 128 | exact_full_rank | 0 |
| 1 | 128 | exact_full_rank | 0 |
| 2 | 128 | exact_full_rank | 0 |
| 3 | 128 | exact_full_rank | 0 |
| 4 | 128 | exact_full_rank | 0 |
| 5 | 128 | exact_full_rank | 0 |
| 6 | 128 | exact_full_rank | 0 |
| 7 | 128 | exact_full_rank | 0 |
| 8 | 128 | exact_full_rank | 0 |
| 9 | 128 | exact_full_rank | 0 |
| 10 | 128 | exact_full_rank | 0 |
| 11 | 128 | exact_full_rank | 0 |
| 12 | 128 | exact_full_rank | 0 |
| 13 | 128 | exact_full_rank | 0 |
| 14 | 128 | exact_full_rank | 0 |
| 15 | 128 | exact_full_rank | 0 |
| 16 | 128 | exact_full_rank | 0 |
| 17 | 128 | exact_full_rank | 0 |
| 18 | 128 | exact_full_rank | 0 |
| 19 | 128 | exact_full_rank | 0 |
| 20 | 128 | exact_full_rank | 0 |
| 21 | 128 | exact_full_rank | 0 |
| 22 | 128 | exact_full_rank | 0 |
| 23 | 128 | exact_full_rank | 0 |
| 24 | 128 | exact_full_rank | 0 |
| 25 | 128 | exact_full_rank | 0 |
| 26 | 128 | exact_full_rank | 0 |
| 27 | 128 | exact_full_rank | 0 |
| 28 | 128 | exact_full_rank | 0 |
| 29 | 128 | exact_full_rank | 0 |
| 30 | 128 | exact_full_rank | 0 |
| 31 | 128 | exact_full_rank | 0 |
| 32 | 128 | exact_full_rank | 0 |
| 33 | 128 | exact_full_rank | 0 |
| 34 | 128 | exact_full_rank | 0 |
| 35 | 128 | exact_full_rank | 0 |
