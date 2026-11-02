# qwen3 GQA V112 joint fit

- K remains dense; each of 8 KV heads retains a mean V rank of 112/128.
- Total KV-cache retention is 93.75% (6.25% reduction).
- Fit windows: 256; held-out windows: 64.
- Encoder sweeps: 6; final decoder refitted.
- Mean held-out factor-dtype relative MSE: `0.0245266682`.

| Layer | Rank | Solver | Held-out relative MSE |
|---:|---:|:---|---:|
| 0 | 112 | paper als (published bank) | 0.00247419511 |
| 1 | 112 | paper als (published bank) | 0.00615735661 |
| 2 | 112 | paper als (published bank) | 0.00964894865 |
| 3 | 112 | paper als (published bank) | 0.00937333802 |
| 4 | 112 | paper als (published bank) | 0.0139628622 |
| 5 | 112 | paper als (published bank) | 0.0125126418 |
| 6 | 112 | paper als (published bank) | 0.0179101943 |
| 7 | 112 | paper als (published bank) | 0.0233028295 |
| 8 | 112 | paper als (published bank) | 0.029506885 |
| 9 | 112 | paper als (published bank) | 0.0347287921 |
| 10 | 112 | paper als (published bank) | 0.0321344186 |
| 11 | 112 | paper als (published bank) | 0.0309998695 |
| 12 | 112 | paper als (published bank) | 0.0212195459 |
| 13 | 112 | paper als (published bank) | 0.0247477693 |
| 14 | 112 | paper als (published bank) | 0.0295936077 |
| 15 | 112 | paper als (published bank) | 0.0282383221 |
| 16 | 112 | paper als (published bank) | 0.0306630688 |
| 17 | 112 | paper als (published bank) | 0.0215393496 |
| 18 | 112 | paper als (published bank) | 0.0273545153 |
| 19 | 112 | paper als (published bank) | 0.0181995598 |
| 20 | 112 | paper als (published bank) | 0.0293770242 |
| 21 | 112 | paper als (published bank) | 0.0305450014 |
| 22 | 112 | paper als (published bank) | 0.0264503441 |
| 23 | 112 | paper als (published bank) | 0.028002994 |
| 24 | 112 | paper als (published bank) | 0.0167663914 |
| 25 | 112 | paper als (published bank) | 0.028534794 |
| 26 | 112 | paper als (published bank) | 0.0384662154 |
| 27 | 112 | paper als (published bank) | 0.0336092874 |
| 28 | 112 | paper als (published bank) | 0.0321989234 |
| 29 | 112 | paper als (published bank) | 0.0374795815 |
| 30 | 112 | paper als (published bank) | 0.0279082371 |
| 31 | 112 | paper als (published bank) | 0.0361280514 |
| 32 | 112 | paper als (published bank) | 0.0272771104 |
| 33 | 112 | paper als (published bank) | 0.0361481922 |
| 34 | 112 | paper als (published bank) | 0.0174077481 |
| 35 | 112 | paper als (published bank) | 0.0123920903 |
