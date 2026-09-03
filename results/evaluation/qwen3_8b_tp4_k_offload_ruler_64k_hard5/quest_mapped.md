# Qwen3-8B TP4 QUEST Min/Max Page32 group-max RULER

Exact K storage: `mapped_host`; Page32; strict physical B4096 per KV head.

| Task/sample | Score | Prompt | Output | Prefill tok/s | Decode tok/s |
|:---|---:|---:|---:|---:|---:|
| niah_multikey_2:7 | 0.00% | 65369 | 10 | 9297.69 | 22.329 |
| niah_multivalue:3 | 50.00% | 65405 | 128 | 9285.41 | 26.228 |
| niah_single_2:3 | 100.00% | 65403 | 10 | 9291.53 | 25.820 |
| niah_single_3:3 | 0.00% | 64794 | 31 | 9437.47 | 26.177 |
| fwe:2 | 100.00% | 64756 | 16 | 9424.75 | 26.036 |

Task-balanced hard-5 accuracy: `50.00%`.
Aggregate decode throughput: `25.9710 tok/s`.
Aggregate end-to-end model throughput: `7729.41 tok/s`.
Requested physical exact-K reads: `53.438 GiB`, `224133120` K-vector tokens.
Requested physical exact-K read per decode step: `0.28125 GiB`.
GPU-resident runtime cache: `0.773 GiB/rank`; mapped-host exact-K cache: `1.125 GiB/rank`.

Physical fetch is the requested Page32 exact-K payload read by the CUDA kernel; it is not a PCIe hardware-counter measurement.
