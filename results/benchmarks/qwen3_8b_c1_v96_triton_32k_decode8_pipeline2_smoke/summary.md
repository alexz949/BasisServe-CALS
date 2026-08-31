# Qwen3-8B C1-V96 Triton 32K decode

Prompt: `32628`; generated output: `8` (`7` timed forwards); devices: `2`.

| Arm | Seconds | ms/token | tokens/s |
|:---|---:|---:|---:|
| dense_bf16_full_kv_sdpa | 0.335 | 47.834 | 20.906 |
| dense_exact_k_c1_v96_triton | 0.791 | 112.945 | 8.854 |

C1-V96 Triton speedup versus dense BF16 SDPA: `0.424x`.

Dense prefill: `9.723` seconds; C1 prefill: `56.812` seconds; dynamic-to-static cache conversion: `0.043` seconds.

The C1 decode arm uses the existing fused Triton online-softmax kernel with a fixed-capacity cache and device-resident valid length.

One-step C1 SDPA/Triton check: argmax `19` / `19`; relative logits L2 error `0.00429404`; finite `true`.
