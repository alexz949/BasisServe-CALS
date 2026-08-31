# Qwen3-8B dense versus C1 sparse 32K decode

Prompt: `32628` tokens; generated output: `128` tokens (`127` timed decode forwards); pipeline devices: `2`.

| Arm | Seconds | ms/token | tokens/s |
|:---|---:|---:|---:|
| dense_bf16_full_kv | 6.010 | 47.320 | 21.133 |
| dense_exact_k_c1_v96 | 4.302 | 33.876 | 29.519 |
| r32_sparse_exact_k_c1_v96 | 6.236 | 49.103 | 20.365 |

Sparse speedup versus dense BF16 full-K/V: `0.964x`.
Sparse speedup versus dense exact-K/C1-V96: `0.690x`.
Dense C1-V96 speedup versus dense BF16 full-K/V: `1.397x`.

Dense BF16 prefill: `9.701` seconds; C1 dense prefill: `9.562` seconds; V128-to-V96 cache projection: `0.004` seconds; resident R32 sidecar: `0.560 GiB`.

The sparse arm includes R32 selection, resident exact-K page packing, and CUDA sparse C1-V96 attention. Exact K remains in GPU HBM, so this does not include physical PCIe fetches or any AllGather collective.
