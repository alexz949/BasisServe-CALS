# Qwen3-8B C1-V96 cuda_dense 32K decode

Prompt: `32628`; generated output: `128` (`127` timed forwards); devices: `2`.

| Arm | Seconds | ms/token | tokens/s |
|:---|---:|---:|---:|
| dense_bf16_full_kv_sdpa | 13.461 | 105.994 | 9.434 |
| dense_exact_k_c1_v96_cuda_shared_gqa | 4.303 | 33.883 | 29.513 |

C1-V96 cuda_dense speedup versus dense BF16 SDPA: `3.128x`.

Dense prefill: `9.679` seconds; C1 prefill: `56.737` seconds; dynamic-to-static cache conversion: dense `0.023` seconds, C1 `0.044` seconds.

The C1 decode arm uses a fixed-capacity cache with a device-resident valid length.

One-step C1 SDPA/cuda_dense check: argmax `19` / `19`; relative logits L2 error `0.00463355`; finite `true`.
