# qwen3 GQA V96 joint fit

- K remains dense; each of 8 KV heads retains a mean V rank of 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Fit windows: 256; held-out windows: 64.
- Encoder sweeps: 6; final decoder refitted.
- Mean held-out factor-dtype relative MSE: `0.0552253438`.

| Layer | Rank | Solver | Held-out relative MSE |
|---:|---:|:---|---:|
| 0 | 96 | paper als (published bank) | 0.00686187803 |
| 1 | 96 | paper als (published bank) | 0.0165881486 |
| 2 | 96 | paper als (published bank) | 0.024090808 |
| 3 | 96 | paper als (published bank) | 0.023324632 |
| 4 | 96 | paper als (published bank) | 0.0337952279 |
| 5 | 96 | paper als (published bank) | 0.0308646422 |
| 6 | 96 | paper als (published bank) | 0.0434744254 |
| 7 | 96 | paper als (published bank) | 0.053892009 |
| 8 | 96 | paper als (published bank) | 0.0673839155 |
| 9 | 96 | paper als (published bank) | 0.0788557139 |
| 10 | 96 | paper als (published bank) | 0.0725978844 |
| 11 | 96 | paper als (published bank) | 0.0692934907 |
| 12 | 96 | paper als (published bank) | 0.0476689868 |
| 13 | 96 | paper als (published bank) | 0.0549537694 |
| 14 | 96 | paper als (published bank) | 0.0655998599 |
| 15 | 96 | paper als (published bank) | 0.0621042703 |
| 16 | 96 | paper als (published bank) | 0.0675316412 |
| 17 | 96 | paper als (published bank) | 0.0473037665 |
| 18 | 96 | paper als (published bank) | 0.059608564 |
| 19 | 96 | paper als (published bank) | 0.0403807638 |
| 20 | 96 | paper als (published bank) | 0.0642275102 |
| 21 | 96 | paper als (published bank) | 0.0670869722 |
| 22 | 96 | paper als (published bank) | 0.0591626636 |
| 23 | 96 | paper als (published bank) | 0.0625006096 |
| 24 | 96 | paper als (published bank) | 0.0395282115 |
| 25 | 96 | paper als (published bank) | 0.0629725895 |
| 26 | 96 | paper als (published bank) | 0.0843553742 |
| 27 | 96 | paper als (published bank) | 0.0745778803 |
| 28 | 96 | paper als (published bank) | 0.0711247246 |
| 29 | 96 | paper als (published bank) | 0.0837914445 |
| 30 | 96 | paper als (published bank) | 0.0611567765 |
| 31 | 96 | paper als (published bank) | 0.0813896655 |
| 32 | 96 | paper als (published bank) | 0.0611205937 |
| 33 | 96 | paper als (published bank) | 0.0824550297 |
| 34 | 96 | paper als (published bank) | 0.0392485407 |
| 35 | 96 | paper als (published bank) | 0.0272393916 |
