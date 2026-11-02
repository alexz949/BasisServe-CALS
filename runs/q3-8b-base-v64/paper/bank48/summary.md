# qwen3 GQA V48 joint fit

- K remains dense; each of 8 KV heads retains a mean V rank of 48/128.
- Total KV-cache retention is 68.75% (31.25% reduction).
- Fit windows: 256; held-out windows: 64.
- Encoder sweeps: 6; final decoder refitted.
- Mean held-out factor-dtype relative MSE: `0.19262199`.

| Layer | Rank | Solver | Held-out relative MSE |
|---:|---:|:---|---:|
| 0 | 48 | paper als (published bank) | 0.029472402 |
| 1 | 48 | paper als (published bank) | 0.0861847635 |
| 2 | 48 | paper als (published bank) | 0.104777987 |
| 3 | 48 | paper als (published bank) | 0.101791712 |
| 4 | 48 | paper als (published bank) | 0.139065941 |
| 5 | 48 | paper als (published bank) | 0.130179225 |
| 6 | 48 | paper als (published bank) | 0.179325192 |
| 7 | 48 | paper als (published bank) | 0.203112735 |
| 8 | 48 | paper als (published bank) | 0.238934744 |
| 9 | 48 | paper als (published bank) | 0.269524751 |
| 10 | 48 | paper als (published bank) | 0.249663135 |
| 11 | 48 | paper als (published bank) | 0.23572433 |
| 12 | 48 | paper als (published bank) | 0.168311404 |
| 13 | 48 | paper als (published bank) | 0.188984956 |
| 14 | 48 | paper als (published bank) | 0.221835025 |
| 15 | 48 | paper als (published bank) | 0.209203646 |
| 16 | 48 | paper als (published bank) | 0.222874312 |
| 17 | 48 | paper als (published bank) | 0.158933911 |
| 18 | 48 | paper als (published bank) | 0.196073938 |
| 19 | 48 | paper als (published bank) | 0.146219671 |
| 20 | 48 | paper als (published bank) | 0.210937948 |
| 21 | 48 | paper als (published bank) | 0.224642767 |
| 22 | 48 | paper als (published bank) | 0.200508574 |
| 23 | 48 | paper als (published bank) | 0.219590587 |
| 24 | 48 | paper als (published bank) | 0.161814429 |
| 25 | 48 | paper als (published bank) | 0.214046618 |
| 26 | 48 | paper als (published bank) | 0.277243856 |
| 27 | 48 | paper als (published bank) | 0.247274719 |
| 28 | 48 | paper als (published bank) | 0.235736468 |
| 29 | 48 | paper als (published bank) | 0.282597784 |
| 30 | 48 | paper als (published bank) | 0.205131523 |
| 31 | 48 | paper als (published bank) | 0.276777733 |
| 32 | 48 | paper als (published bank) | 0.211027254 |
| 33 | 48 | paper als (published bank) | 0.276871941 |
| 34 | 48 | paper als (published bank) | 0.126730835 |
| 35 | 48 | paper als (published bank) | 0.0832648372 |
