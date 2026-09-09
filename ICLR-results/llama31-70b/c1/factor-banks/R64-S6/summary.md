# Llama-3.1-70B GQA C1 V64 joint fit

- K remains dense; every one of 8 physical V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.218034743`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.00378835272 |
| 1 | after_redecoder | 6 | 0.0536850579 |
| 2 | after_redecoder | 6 | 0.0836976532 |
| 3 | after_redecoder | 6 | 0.0628161736 |
| 4 | after_redecoder | 6 | 0.0375137033 |
| 5 | after_redecoder | 6 | 0.0973467287 |
| 6 | after_redecoder | 6 | 0.113183955 |
| 7 | after_redecoder | 6 | 0.0864746157 |
| 8 | after_redecoder | 6 | 0.0598507289 |
| 9 | after_redecoder | 6 | 0.0891535473 |
| 10 | after_redecoder | 6 | 0.127808876 |
| 11 | after_redecoder | 6 | 0.165991115 |
| 12 | after_redecoder | 6 | 0.0697714851 |
| 13 | after_redecoder | 6 | 0.0938435156 |
| 14 | after_redecoder | 6 | 0.10726513 |
| 15 | after_redecoder | 6 | 0.116364323 |
| 16 | after_redecoder | 6 | 0.105213926 |
| 17 | after_redecoder | 6 | 0.16371565 |
| 18 | after_redecoder | 6 | 0.214989912 |
| 19 | after_redecoder | 6 | 0.184274296 |
| 20 | after_redecoder | 6 | 0.186255566 |
| 21 | after_redecoder | 6 | 0.243060769 |
| 22 | after_redecoder | 6 | 0.302532496 |
| 23 | after_redecoder | 6 | 0.266727917 |
| 24 | after_redecoder | 6 | 0.174186621 |
| 25 | after_redecoder | 6 | 0.226062196 |
| 26 | after_redecoder | 6 | 0.174111654 |
| 27 | after_redecoder | 6 | 0.181523085 |
| 28 | after_redecoder | 6 | 0.167943967 |
| 29 | after_redecoder | 6 | 0.180916778 |
| 30 | after_redecoder | 6 | 0.181874024 |
| 31 | after_redecoder | 6 | 0.190801421 |
| 32 | after_redecoder | 6 | 0.201387149 |
| 33 | after_redecoder | 6 | 0.19805744 |
| 34 | after_redecoder | 6 | 0.18249034 |
| 35 | after_redecoder | 6 | 0.219314007 |
| 36 | after_redecoder | 6 | 0.235793094 |
| 37 | after_redecoder | 6 | 0.235058624 |
| 38 | after_redecoder | 6 | 0.252550262 |
| 39 | after_redecoder | 6 | 0.294668331 |
| 40 | after_redecoder | 6 | 0.26089516 |
| 41 | after_redecoder | 6 | 0.331591674 |
| 42 | after_redecoder | 6 | 0.380176005 |
| 43 | after_redecoder | 6 | 0.275648417 |
| 44 | after_redecoder | 6 | 0.278818807 |
| 45 | after_redecoder | 6 | 0.300872864 |
| 46 | after_redecoder | 6 | 0.439882323 |
| 47 | after_redecoder | 6 | 0.301174315 |
| 48 | after_redecoder | 6 | 0.375981654 |
| 49 | after_redecoder | 6 | 0.296433693 |
| 50 | after_redecoder | 6 | 0.403427664 |
| 51 | after_redecoder | 6 | 0.309535727 |
| 52 | after_redecoder | 6 | 0.230409665 |
| 53 | after_redecoder | 6 | 0.398431329 |
| 54 | after_redecoder | 6 | 0.326489141 |
| 55 | after_redecoder | 6 | 0.201578938 |
| 56 | after_redecoder | 6 | 0.333920733 |
| 57 | after_redecoder | 6 | 0.341707289 |
| 58 | after_redecoder | 6 | 0.345075152 |
| 59 | after_redecoder | 6 | 0.41679641 |
| 60 | after_redecoder | 6 | 0.355880054 |
| 61 | after_redecoder | 6 | 0.433872765 |
| 62 | after_redecoder | 6 | 0.395858428 |
| 63 | after_redecoder | 6 | 0.296874446 |
| 64 | after_redecoder | 6 | 0.297004954 |
| 65 | after_redecoder | 6 | 0.276159879 |
| 66 | after_redecoder | 6 | 0.219401559 |
| 67 | after_redecoder | 6 | 0.299110839 |
| 68 | after_redecoder | 6 | 0.263569752 |
| 69 | after_redecoder | 6 | 0.256962493 |
| 70 | after_redecoder | 6 | 0.192922329 |
| 71 | after_redecoder | 6 | 0.158022338 |
| 72 | after_redecoder | 6 | 0.133400865 |
| 73 | after_redecoder | 6 | 0.114301534 |
| 74 | after_redecoder | 6 | 0.0976066708 |
| 75 | after_redecoder | 6 | 0.146196686 |
| 76 | after_redecoder | 6 | 0.102653784 |
| 77 | after_redecoder | 6 | 0.171363737 |
| 78 | after_redecoder | 6 | 0.227729647 |
| 79 | after_redecoder | 6 | 0.122975252 |
