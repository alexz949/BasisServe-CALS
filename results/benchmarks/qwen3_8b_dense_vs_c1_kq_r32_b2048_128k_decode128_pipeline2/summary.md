# Qwen3-8B dense versus C1 sparse 128K decode

Prompt: `130929` tokens; generated output: `128` tokens (`127` timed decode forwards); pipeline devices: `2`.

| Arm | Seconds | ms/token | tokens/s |
|:---|---:|---:|---:|
| dense_bf16_full_kv | 15.197 | 119.662 | 8.357 |
| dense_exact_k_c1_v96 | 130.356 | 1026.428 | 0.974 |
| r32_sparse_exact_k_c1_v96 | 84.762 | 667.415 | 1.498 |

Sparse speedup versus dense BF16 full-K/V: `0.179x`.
Sparse speedup versus dense exact-K/C1-V96: `1.538x`.
Dense C1-V96 speedup versus dense BF16 full-K/V: `0.117x`.

Dense BF16 prefill: `107.372` seconds; C1 shared prefill: `951.178` seconds; resident R32 sidecar: `2.248 GiB`.

The sparse arm keeps exact K in GPU HBM. This isolates the model and attention-path latency; it does not include physical PCIe fetches.
