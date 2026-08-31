# Qwen3-8B dense versus C1 sparse 32K decode

Prompt: `32628` tokens; generated output: `8` tokens (`7` timed decode forwards); pipeline devices: `2`.

| Arm | Seconds | ms/token | tokens/s |
|:---|---:|---:|---:|
| dense_bf16_full_kv | 0.334 | 47.779 | 20.929 |
| dense_exact_k_c1_v96 | 1.656 | 236.562 | 4.227 |
| r32_sparse_exact_k_c1_v96 | 1.446 | 206.531 | 4.842 |

Sparse speedup versus dense BF16 full-K/V: `0.231x`.
Sparse speedup versus dense exact-K/C1-V96: `1.145x`.
Dense C1-V96 speedup versus dense BF16 full-K/V: `0.202x`.

Dense BF16 prefill: `9.756` seconds; C1 shared prefill: `56.804` seconds; resident R32 sidecar: `0.560 GiB`.

The sparse arm keeps exact K in GPU HBM. This isolates the model and attention-path latency; it does not include physical PCIe fetches.
