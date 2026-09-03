# Qwen3-8B TP4 C1 Base16+R8 Page32 group-max RULER

Exact K storage: `mapped_host`; Page32; strict physical B4096 per KV head.

| Task/sample | Score | Prompt | Output | Prefill tok/s | Decode tok/s |
|:---|---:|---:|---:|---:|---:|
| niah_multikey_2:7 | 0.00% | 65369 | 10 | 9356.31 | 31.087 |
| niah_multivalue:3 | 50.00% | 65405 | 128 | 9328.38 | 31.645 |
| niah_single_2:3 | 100.00% | 65403 | 10 | 9315.74 | 31.045 |
| niah_single_3:3 | 100.00% | 64794 | 32 | 9460.46 | 31.796 |
| fwe:2 | 100.00% | 64756 | 16 | 9433.74 | 31.724 |

Task-balanced hard-5 accuracy: `70.00%`.
Aggregate decode throughput: `31.6201 tok/s`.
Aggregate end-to-end model throughput: `7993.59 tok/s`.
Requested physical exact-K reads: `53.719 GiB`, `225312768` K-vector tokens.
Requested physical exact-K read per decode step: `0.28125 GiB`.
GPU-resident runtime cache: `0.932 GiB/rank`; mapped-host exact-K cache: `1.125 GiB/rank`.
GPU-oracle token-sequence agreement: `5/5`.

Physical fetch is the requested Page32 exact-K payload read by the CUDA kernel; it is not a PCIe hardware-counter measurement.
