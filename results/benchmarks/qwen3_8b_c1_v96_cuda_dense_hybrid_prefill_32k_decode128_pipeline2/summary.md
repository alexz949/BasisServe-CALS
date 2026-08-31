# Qwen3-8B C1-V96 cuda_dense 32K decode

Prompt: `32628`; generated output: `128` (`127` timed forwards); devices: `2`.

| Arm | Seconds | ms/token | tokens/s |
|:---|---:|---:|---:|
| dense_bf16_full_kv_sdpa | 6.005 | 47.285 | 21.148 |
| dense_exact_k_c1_v96_cuda_shared_gqa | 4.300 | 33.861 | 29.532 |

C1-V96 cuda_dense speedup versus dense BF16 SDPA: `1.396x`.

Dense prefill: `9.740` seconds; C1 prefill: `9.440` seconds; dense-V128 to C1-V96 cache projection: `0.004` seconds; dynamic-to-static cache conversion: `0.020` seconds.

Both arms use exact dense prefill. The C1 arm then projects only the resident V cache to rank 96, preserves exact K, and uses a fixed-capacity decode cache with a device-resident valid length.

One-step C1 SDPA/cuda_dense check: argmax `19` / `19`; relative logits L2 error `0.0058156`; finite `true`.
