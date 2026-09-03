# Qwen3-8B TP4 C1 Base16+R8 Page32 group-max RULER

Exact K storage: `mapped_host`; Page32; strict physical B4096 per KV head.

| Task/sample | Score | Prompt | Output | Prefill tok/s | Decode tok/s |
|:---|---:|---:|---:|---:|---:|
| niah_multikey_2:7 | 0.00% | 65369 | 10 | 9354.57 | 20.748 |
| niah_multivalue:3 | 50.00% | 65405 | 128 | 9354.19 | 18.691 |
| niah_single_2:3 | 100.00% | 65403 | 10 | 9324.77 | 22.477 |
| niah_single_3:3 | 100.00% | 64794 | 32 | 9475.02 | 22.666 |
| fwe:2 | 100.00% | 64756 | 16 | 9446.08 | 22.627 |

Task-balanced hard-5 accuracy: `70.00%`.
Aggregate decode throughput: `19.7734 tok/s`.
Aggregate end-to-end model throughput: `7349.26 tok/s`.
Requested physical exact-K reads: `53.719 GiB`, `225312768` K-vector tokens.
Requested physical exact-K read per decode step: `0.28125 GiB`.
GPU-resident runtime cache: `1.338 GiB/rank`; mapped-host exact-K cache: `1.125 GiB/rank`.
GPU-oracle token-sequence agreement: `5/5`.

Physical fetch is the requested Page32 exact-K payload read by the CUDA kernel; it is not a PCIe hardware-counter measurement.
