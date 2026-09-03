# Qwen3-8B TP4 C1 Base16+R8 Page32 group-max RULER

Exact K storage: `gpu`; Page32; strict physical B4096 per KV head.

| Task/sample | Score | Prompt | Output | Prefill tok/s | Decode tok/s |
|:---|---:|---:|---:|---:|---:|
| niah_multikey_2:7 | 0.00% | 65369 | 10 | 9338.54 | 3.341 |
| niah_multivalue:3 | 50.00% | 65405 | 128 | 9339.54 | 3.372 |
| niah_single_2:3 | 100.00% | 65403 | 10 | 9347.33 | 3.381 |
| niah_single_3:3 | 100.00% | 64794 | 32 | 9512.21 | 3.379 |
| fwe:2 | 100.00% | 64756 | 16 | 9479.50 | 3.385 |

Task-balanced hard-5 accuracy: `70.00%`.
Aggregate decode throughput: `3.3729 tok/s`.
Aggregate end-to-end model throughput: `3570.87 tok/s`.
Requested physical exact-K reads: `53.719 GiB`, `225312768` K-vector tokens.
Requested physical exact-K read per decode step: `0.28125 GiB`.
GPU-resident runtime cache: `2.461 GiB/rank`; mapped-host exact-K cache: `0.000 GiB/rank`.

Physical fetch is the requested Page32 exact-K payload read by the CUDA kernel; it is not a PCIe hardware-counter measurement.
