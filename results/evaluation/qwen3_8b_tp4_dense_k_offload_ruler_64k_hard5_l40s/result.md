# Qwen3-8B TP4 Dense K128 CPU-offload/V128 GPU RULER

Pinned-CPU K128 cache with a full-prefix H2D transfer per decode layer; GPU-resident V128; no sparse page routing.

| Task/sample | Score | Prompt | Output | Prefill tok/s | Decode tok/s |
|:---|---:|---:|---:|---:|---:|
| niah_multikey_2:7 | 100.00% | 65369 | 10 | 9987.41 | 16.532 |
| niah_multivalue:3 | 50.00% | 65405 | 128 | 9971.03 | 16.558 |
| niah_single_2:3 | 100.00% | 65403 | 10 | 9956.10 | 16.534 |
| niah_single_3:3 | 100.00% | 64794 | 32 | 9961.47 | 16.697 |
| fwe:2 | 100.00% | 64756 | 16 | 9976.52 | 16.694 |

Task-balanced hard-5 accuracy: `90.00%`.
Aggregate decode throughput: `16.5888 tok/s`.
Aggregate end-to-end model throughput: `7376.57 tok/s`.
GPU-resident runtime cache: `1.156 GiB/rank`; CPU-resident exact-K cache: `1.125 GiB/rank`.
Full-prefix exact-K H2D traffic: `856.396 GiB`, or `4.484 GiB` per decode step across TP ranks.
