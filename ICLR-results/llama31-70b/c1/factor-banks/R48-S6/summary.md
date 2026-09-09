# Llama-3.1-70B GQA C1 V48 joint fit

- K remains dense; every one of 8 physical V heads retains rank 48/128.
- Total KV-cache retention is 68.75% (31.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.28916544`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.00624406103 |
| 1 | after_redecoder | 6 | 0.0854289973 |
| 2 | after_redecoder | 6 | 0.129067073 |
| 3 | after_redecoder | 6 | 0.0929717988 |
| 4 | after_redecoder | 6 | 0.0580623095 |
| 5 | after_redecoder | 6 | 0.146706442 |
| 6 | after_redecoder | 6 | 0.166184636 |
| 7 | after_redecoder | 6 | 0.131984831 |
| 8 | after_redecoder | 6 | 0.0893591928 |
| 9 | after_redecoder | 6 | 0.130449308 |
| 10 | after_redecoder | 6 | 0.186718035 |
| 11 | after_redecoder | 6 | 0.229644036 |
| 12 | after_redecoder | 6 | 0.10027697 |
| 13 | after_redecoder | 6 | 0.135455955 |
| 14 | after_redecoder | 6 | 0.153073003 |
| 15 | after_redecoder | 6 | 0.166504614 |
| 16 | after_redecoder | 6 | 0.149045393 |
| 17 | after_redecoder | 6 | 0.226085641 |
| 18 | after_redecoder | 6 | 0.291750087 |
| 19 | after_redecoder | 6 | 0.253493448 |
| 20 | after_redecoder | 6 | 0.251442443 |
| 21 | after_redecoder | 6 | 0.322437093 |
| 22 | after_redecoder | 6 | 0.393121662 |
| 23 | after_redecoder | 6 | 0.345323211 |
| 24 | after_redecoder | 6 | 0.229399059 |
| 25 | after_redecoder | 6 | 0.289520716 |
| 26 | after_redecoder | 6 | 0.226538869 |
| 27 | after_redecoder | 6 | 0.242038554 |
| 28 | after_redecoder | 6 | 0.224119992 |
| 29 | after_redecoder | 6 | 0.24230705 |
| 30 | after_redecoder | 6 | 0.245067795 |
| 31 | after_redecoder | 6 | 0.259504415 |
| 32 | after_redecoder | 6 | 0.261884982 |
| 33 | after_redecoder | 6 | 0.267332164 |
| 34 | after_redecoder | 6 | 0.245248446 |
| 35 | after_redecoder | 6 | 0.297466694 |
| 36 | after_redecoder | 6 | 0.311527236 |
| 37 | after_redecoder | 6 | 0.321867335 |
| 38 | after_redecoder | 6 | 0.335650346 |
| 39 | after_redecoder | 6 | 0.392497893 |
| 40 | after_redecoder | 6 | 0.344706177 |
| 41 | after_redecoder | 6 | 0.434733098 |
| 42 | after_redecoder | 6 | 0.486204736 |
| 43 | after_redecoder | 6 | 0.361100543 |
| 44 | after_redecoder | 6 | 0.368557355 |
| 45 | after_redecoder | 6 | 0.39260681 |
| 46 | after_redecoder | 6 | 0.560454668 |
| 47 | after_redecoder | 6 | 0.396278981 |
| 48 | after_redecoder | 6 | 0.481782781 |
| 49 | after_redecoder | 6 | 0.386986899 |
| 50 | after_redecoder | 6 | 0.52219927 |
| 51 | after_redecoder | 6 | 0.404309387 |
| 52 | after_redecoder | 6 | 0.304909543 |
| 53 | after_redecoder | 6 | 0.517024933 |
| 54 | after_redecoder | 6 | 0.423447202 |
| 55 | after_redecoder | 6 | 0.269424689 |
| 56 | after_redecoder | 6 | 0.437884867 |
| 57 | after_redecoder | 6 | 0.445280783 |
| 58 | after_redecoder | 6 | 0.450410686 |
| 59 | after_redecoder | 6 | 0.527537466 |
| 60 | after_redecoder | 6 | 0.461563128 |
| 61 | after_redecoder | 6 | 0.55697219 |
| 62 | after_redecoder | 6 | 0.510437104 |
| 63 | after_redecoder | 6 | 0.405851786 |
| 64 | after_redecoder | 6 | 0.390652188 |
| 65 | after_redecoder | 6 | 0.352789036 |
| 66 | after_redecoder | 6 | 0.291611867 |
| 67 | after_redecoder | 6 | 0.391860449 |
| 68 | after_redecoder | 6 | 0.351056785 |
| 69 | after_redecoder | 6 | 0.337469795 |
| 70 | after_redecoder | 6 | 0.252182101 |
| 71 | after_redecoder | 6 | 0.207052493 |
| 72 | after_redecoder | 6 | 0.169725086 |
| 73 | after_redecoder | 6 | 0.156894574 |
| 74 | after_redecoder | 6 | 0.130216033 |
| 75 | after_redecoder | 6 | 0.190690328 |
| 76 | after_redecoder | 6 | 0.138889138 |
| 77 | after_redecoder | 6 | 0.225841901 |
| 78 | after_redecoder | 6 | 0.302152776 |
| 79 | after_redecoder | 6 | 0.160683748 |
