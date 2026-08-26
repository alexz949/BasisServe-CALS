# Qwen3-32B GQA C1 V112 joint fit

- K remains dense; every one of 8 physical V heads retains rank 112/128.
- Total KV-cache retention is 93.75% (6.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0242385536`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | decoder_only | 0 | 0.00139650321 |
| 1 | decoder_only | 0 | 0.000882929834 |
| 2 | decoder_only | 0 | 0.00332259779 |
| 3 | decoder_only | 0 | 0.00475833463 |
| 4 | decoder_only | 0 | 0.00754591266 |
| 5 | decoder_only | 0 | 0.0049240217 |
| 6 | decoder_only | 0 | 0.00411863877 |
| 7 | decoder_only | 0 | 0.0191774465 |
| 8 | decoder_only | 0 | 0.0133235472 |
| 9 | decoder_only | 0 | 0.0061477083 |
| 10 | decoder_only | 0 | 0.00935999459 |
| 11 | decoder_only | 0 | 0.0224896569 |
| 12 | decoder_only | 0 | 0.0110981426 |
| 13 | decoder_only | 0 | 0.0164113601 |
| 14 | decoder_only | 0 | 0.0159421863 |
| 15 | decoder_only | 0 | 0.00978188481 |
| 16 | decoder_only | 0 | 0.00796342813 |
| 17 | decoder_only | 0 | 0.00528259749 |
| 18 | decoder_only | 0 | 0.00625631333 |
| 19 | decoder_only | 0 | 0.00938513509 |
| 20 | decoder_only | 0 | 0.0100658307 |
| 21 | decoder_only | 0 | 0.0150185335 |
| 22 | decoder_only | 0 | 0.0171992993 |
| 23 | decoder_only | 0 | 0.0174442034 |
| 24 | decoder_only | 0 | 0.0308754507 |
| 25 | decoder_only | 0 | 0.0234832221 |
| 26 | decoder_only | 0 | 0.0237517266 |
| 27 | decoder_only | 0 | 0.0246774472 |
| 28 | decoder_only | 0 | 0.038590492 |
| 29 | decoder_only | 0 | 0.0287566649 |
| 30 | decoder_only | 0 | 0.0465300768 |
| 31 | decoder_only | 0 | 0.0394542277 |
| 32 | decoder_only | 0 | 0.020125672 |
| 33 | decoder_only | 0 | 0.0161399468 |
| 34 | decoder_only | 0 | 0.0222987316 |
| 35 | decoder_only | 0 | 0.0370365183 |
| 36 | decoder_only | 0 | 0.0279921781 |
| 37 | decoder_only | 0 | 0.0339251399 |
| 38 | decoder_only | 0 | 0.0364842332 |
| 39 | decoder_only | 0 | 0.0296426617 |
| 40 | decoder_only | 0 | 0.0275230419 |
| 41 | decoder_only | 0 | 0.0203647982 |
| 42 | decoder_only | 0 | 0.0298000833 |
| 43 | decoder_only | 0 | 0.0323669555 |
| 44 | decoder_only | 0 | 0.0355517441 |
| 45 | decoder_only | 0 | 0.0316438029 |
| 46 | decoder_only | 0 | 0.0324241697 |
| 47 | decoder_only | 0 | 0.0385880798 |
| 48 | decoder_only | 0 | 0.0310602784 |
| 49 | decoder_only | 0 | 0.0331700475 |
| 50 | decoder_only | 0 | 0.0337156502 |
| 51 | decoder_only | 0 | 0.0279288905 |
| 52 | decoder_only | 0 | 0.035416734 |
| 53 | decoder_only | 0 | 0.0281312196 |
| 54 | decoder_only | 0 | 0.0425603775 |
| 55 | decoder_only | 0 | 0.0371626908 |
| 56 | decoder_only | 0 | 0.0391610126 |
| 57 | decoder_only | 0 | 0.0462426163 |
| 58 | decoder_only | 0 | 0.043140422 |
| 59 | decoder_only | 0 | 0.0398441564 |
| 60 | decoder_only | 0 | 0.0513742991 |
| 61 | decoder_only | 0 | 0.0293570654 |
| 62 | decoder_only | 0 | 0.0527018429 |
| 63 | decoder_only | 0 | 0.0129768531 |
