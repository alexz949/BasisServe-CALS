# qwen3 GQA allocated mean V64 joint fit

- K remains dense; each of 8 KV heads retains a mean V rank of 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Fit windows: 256; held-out windows: 64.
- Encoder sweeps: 6; final decoder refitted.
- Mean held-out factor-dtype relative MSE: `0.140458521`.

| Layer | Rank | Solver | Held-out relative MSE |
|---:|---:|:---|---:|
| 0 | 80 | paper als (published bank) | 0.0127502289 |
| 1 | 64 | paper als (published bank) | 0.0539074153 |
| 2 | 32 | paper als (published bank) | 0.158701535 |
| 3 | 48 | paper als (published bank) | 0.101791712 |
| 4 | 48 | paper als (published bank) | 0.139065941 |
| 5 | 32 | paper als (published bank) | 0.193494286 |
| 6 | 64 | paper als (published bank) | 0.120836602 |
| 7 | 64 | paper als (published bank) | 0.140597854 |
| 8 | 80 | paper als (published bank) | 0.113484005 |
| 9 | 80 | paper als (published bank) | 0.130713209 |
| 10 | 80 | paper als (published bank) | 0.121140755 |
| 11 | 64 | paper als (published bank) | 0.168841319 |
| 12 | 48 | paper als (published bank) | 0.168311404 |
| 13 | 48 | paper als (published bank) | 0.188984956 |
| 14 | 64 | paper als (published bank) | 0.158576985 |
| 15 | 64 | paper als (published bank) | 0.149900264 |
| 16 | 64 | paper als (published bank) | 0.161301603 |
| 17 | 64 | paper als (published bank) | 0.113886276 |
| 18 | 64 | paper als (published bank) | 0.141712904 |
| 19 | 80 | paper als (published bank) | 0.0673746065 |
| 20 | 64 | paper als (published bank) | 0.152510491 |
| 21 | 64 | paper als (published bank) | 0.161539149 |
| 22 | 96 | paper als (published bank) | 0.0591626636 |
| 23 | 112 | paper als (published bank) | 0.028002994 |
| 24 | 96 | paper als (published bank) | 0.0395282115 |
| 25 | 48 | paper als (published bank) | 0.214046618 |
| 26 | 64 | paper als (published bank) | 0.201032488 |
| 27 | 64 | paper als (published bank) | 0.178752448 |
| 28 | 32 | paper als (published bank) | 0.324304032 |
| 29 | 64 | paper als (published bank) | 0.202762135 |
| 30 | 64 | paper als (published bank) | 0.14705239 |
| 31 | 48 | paper als (published bank) | 0.276777733 |
| 32 | 48 | paper als (published bank) | 0.211027254 |
| 33 | 96 | paper als (published bank) | 0.0824550297 |
| 34 | 80 | paper als (published bank) | 0.064483667 |
| 35 | 32 | paper als (published bank) | 0.107695589 |
