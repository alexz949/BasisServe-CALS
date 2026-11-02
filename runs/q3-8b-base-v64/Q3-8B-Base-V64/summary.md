# qwen3 GQA allocated mean V64 joint fit

- K remains dense; each of 8 KV heads retains a mean V rank of 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Fit windows: 256; held-out windows: 64.
- Encoder sweeps: 6; final decoder refitted.
- Mean held-out factor-dtype relative MSE: `0.142342575`.

| Layer | Rank | Solver | Held-out relative MSE |
|---:|---:|:---|---:|
| 0 | 96 | als | 0.00707059591 |
| 1 | 64 | als | 0.0592042164 |
| 2 | 32 | als | 0.166048834 |
| 3 | 48 | als | 0.106220245 |
| 4 | 48 | als | 0.123767365 |
| 5 | 48 | als | 0.134185633 |
| 6 | 80 | als | 0.0787140873 |
| 7 | 64 | als | 0.145933531 |
| 8 | 80 | als | 0.118494684 |
| 9 | 64 | als | 0.199596485 |
| 10 | 80 | als | 0.126931256 |
| 11 | 64 | als | 0.173350509 |
| 12 | 48 | als | 0.171212085 |
| 13 | 48 | als | 0.1937612 |
| 14 | 64 | als | 0.157683348 |
| 15 | 64 | als | 0.150782236 |
| 16 | 64 | als | 0.156798504 |
| 17 | 64 | als | 0.1107744 |
| 18 | 64 | als | 0.137165239 |
| 19 | 80 | als | 0.0665155907 |
| 20 | 64 | als | 0.143810802 |
| 21 | 64 | als | 0.159061466 |
| 22 | 96 | als | 0.0591955658 |
| 23 | 128 | exact_full_rank | 0 |
| 24 | 96 | als | 0.0394817188 |
| 25 | 48 | als | 0.206238842 |
| 26 | 64 | als | 0.200375431 |
| 27 | 48 | als | 0.247423203 |
| 28 | 32 | als | 0.320633566 |
| 29 | 64 | als | 0.204757177 |
| 30 | 64 | als | 0.150635038 |
| 31 | 48 | als | 0.273818679 |
| 32 | 48 | als | 0.209771088 |
| 33 | 80 | als | 0.132957532 |
| 34 | 64 | als | 0.089743125 |
| 35 | 32 | als | 0.102219406 |
