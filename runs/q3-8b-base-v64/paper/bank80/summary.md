# qwen3 GQA V80 joint fit

- K remains dense; each of 8 KV heads retains a mean V rank of 80/128.
- Total KV-cache retention is 81.25% (18.75% reduction).
- Fit windows: 256; held-out windows: 64.
- Encoder sweeps: 6; final decoder refitted.
- Mean held-out factor-dtype relative MSE: `0.0921038754`.

| Layer | Rank | Solver | Held-out relative MSE |
|---:|---:|:---|---:|
| 0 | 80 | paper als (published bank) | 0.0127502289 |
| 1 | 80 | paper als (published bank) | 0.03192964 |
| 2 | 80 | paper als (published bank) | 0.0434108043 |
| 3 | 80 | paper als (published bank) | 0.0422081618 |
| 4 | 80 | paper als (published bank) | 0.0597691723 |
| 5 | 80 | paper als (published bank) | 0.0550271853 |
| 6 | 80 | paper als (published bank) | 0.0768684271 |
| 7 | 80 | paper als (published bank) | 0.0924074773 |
| 8 | 80 | paper als (published bank) | 0.113484005 |
| 9 | 80 | paper als (published bank) | 0.130713209 |
| 10 | 80 | paper als (published bank) | 0.121140755 |
| 11 | 80 | paper als (published bank) | 0.11481653 |
| 12 | 80 | paper als (published bank) | 0.0796864768 |
| 13 | 80 | paper als (published bank) | 0.0907238454 |
| 14 | 80 | paper als (published bank) | 0.10812637 |
| 15 | 80 | paper als (published bank) | 0.102065977 |
| 16 | 80 | paper als (published bank) | 0.110606057 |
| 17 | 80 | paper als (published bank) | 0.0776579511 |
| 18 | 80 | paper als (published bank) | 0.0972845243 |
| 19 | 80 | paper als (published bank) | 0.0673746065 |
| 20 | 80 | paper als (published bank) | 0.104643005 |
| 21 | 80 | paper als (published bank) | 0.11019932 |
| 22 | 80 | paper als (published bank) | 0.0976829248 |
| 23 | 80 | paper als (published bank) | 0.103873869 |
| 24 | 80 | paper als (published bank) | 0.06912706 |
| 25 | 80 | paper als (published bank) | 0.103746912 |
| 26 | 80 | paper als (published bank) | 0.137779749 |
| 27 | 80 | paper als (published bank) | 0.122498676 |
| 28 | 80 | paper als (published bank) | 0.116502096 |
| 29 | 80 | paper als (published bank) | 0.137806669 |
| 30 | 80 | paper als (published bank) | 0.100402063 |
| 31 | 80 | paper als (published bank) | 0.13562934 |
| 32 | 80 | paper als (published bank) | 0.102323145 |
| 33 | 80 | paper als (published bank) | 0.137185493 |
| 34 | 80 | paper als (published bank) | 0.064483667 |
| 35 | 80 | paper als (published bank) | 0.0438041193 |
