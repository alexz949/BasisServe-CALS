# qwen3 GQA V64 joint fit

- K remains dense; each of 8 KV heads retains a mean V rank of 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Fit windows: 256; held-out windows: 64.
- Encoder sweeps: 6; final decoder refitted.
- Mean held-out factor-dtype relative MSE: `0.136809567`.

| Layer | Rank | Solver | Held-out relative MSE |
|---:|---:|:---|---:|
| 0 | 64 | paper als (published bank) | 0.0201510203 |
| 1 | 64 | paper als (published bank) | 0.0539074153 |
| 2 | 64 | paper als (published bank) | 0.0691732938 |
| 3 | 64 | paper als (published bank) | 0.0673111597 |
| 4 | 64 | paper als (published bank) | 0.0937240979 |
| 5 | 64 | paper als (published bank) | 0.0868164241 |
| 6 | 64 | paper als (published bank) | 0.120836602 |
| 7 | 64 | paper als (published bank) | 0.140597854 |
| 8 | 64 | paper als (published bank) | 0.169176156 |
| 9 | 64 | paper als (published bank) | 0.193650677 |
| 10 | 64 | paper als (published bank) | 0.178719499 |
| 11 | 64 | paper als (published bank) | 0.168841319 |
| 12 | 64 | paper als (published bank) | 0.118700043 |
| 13 | 64 | paper als (published bank) | 0.134479478 |
| 14 | 64 | paper als (published bank) | 0.158576985 |
| 15 | 64 | paper als (published bank) | 0.149900264 |
| 16 | 64 | paper als (published bank) | 0.161301603 |
| 17 | 64 | paper als (published bank) | 0.113886276 |
| 18 | 64 | paper als (published bank) | 0.141712904 |
| 19 | 64 | paper als (published bank) | 0.101197368 |
| 20 | 64 | paper als (published bank) | 0.152510491 |
| 21 | 64 | paper als (published bank) | 0.161539149 |
| 22 | 64 | paper als (published bank) | 0.143638166 |
| 23 | 64 | paper als (published bank) | 0.154810119 |
| 24 | 64 | paper als (published bank) | 0.108201952 |
| 25 | 64 | paper als (published bank) | 0.153196543 |
| 26 | 64 | paper als (published bank) | 0.201032488 |
| 27 | 64 | paper als (published bank) | 0.178752448 |
| 28 | 64 | paper als (published bank) | 0.169912004 |
| 29 | 64 | paper als (published bank) | 0.202762135 |
| 30 | 64 | paper als (published bank) | 0.14705239 |
| 31 | 64 | paper als (published bank) | 0.200290402 |
| 32 | 64 | paper als (published bank) | 0.151661956 |
| 33 | 64 | paper als (published bank) | 0.201489545 |
| 34 | 64 | paper als (published bank) | 0.0932843847 |
| 35 | 64 | paper als (published bank) | 0.0623498182 |
