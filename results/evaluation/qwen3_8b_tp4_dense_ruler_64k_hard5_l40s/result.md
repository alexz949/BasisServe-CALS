# Qwen3-8B TP4 Dense K128/V128 RULER

Dense K128/V128 cache; no sparse page routing.

| Task/sample | Score | Prompt | Output | Prefill tok/s | Decode tok/s |
|:---|---:|---:|---:|---:|---:|
| niah_multikey_2:7 | 100.00% | 65369 | 10 | 10023.32 | 31.850 |
| niah_multivalue:3 | 50.00% | 65405 | 128 | 10013.10 | 32.557 |
| niah_single_2:3 | 100.00% | 65403 | 10 | 10009.96 | 32.058 |
| niah_single_3:3 | 100.00% | 64794 | 32 | 10041.41 | 32.470 |
| fwe:2 | 100.00% | 64756 | 16 | 10043.09 | 31.583 |

Task-balanced hard-5 accuracy: `90.00%`.
Aggregate decode throughput: `32.4068 tok/s`.
Aggregate end-to-end model throughput: `8491.49 tok/s`.
GPU-resident runtime cache: `2.250 GiB/rank`; mapped-host exact-K cache: `0.000 GiB/rank`.
