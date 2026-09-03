# Qwen3-8B TP4 ShadowKV-style mean-landmark Page32 group-max RULER

Exact K storage: `mapped_host`; Page32; strict physical B4096 per KV head.

| Task/sample | Score | Prompt | Output | Prefill tok/s | Decode tok/s |
|:---|---:|---:|---:|---:|---:|
| niah_multikey_2:7 | 0.00% | 65369 | 10 | 9292.33 | 24.265 |
| niah_multivalue:3 | 75.00% | 65405 | 128 | 9275.21 | 26.051 |
| niah_single_2:3 | 100.00% | 65403 | 10 | 9273.88 | 25.829 |
| niah_single_3:3 | 0.00% | 64794 | 26 | 9456.88 | 26.077 |
| fwe:2 | 66.67% | 64756 | 15 | 9421.80 | 26.069 |

Task-balanced hard-5 accuracy: `48.33%`.
Aggregate decode throughput: `25.9517 tok/s`.
Aggregate end-to-end model throughput: `7768.42 tok/s`.
Requested physical exact-K reads: `51.750 GiB`, `217055232` K-vector tokens.
Requested physical exact-K read per decode step: `0.28125 GiB`.
GPU-resident runtime cache: `0.738 GiB/rank`; mapped-host exact-K cache: `1.125 GiB/rank`.

Physical fetch is the requested Page32 exact-K payload read by the CUDA kernel; it is not a PCIe hardware-counter measurement.
