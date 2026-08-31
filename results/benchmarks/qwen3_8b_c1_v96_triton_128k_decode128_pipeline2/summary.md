# Qwen3-8B C1-V96 Triton 128K decode

Prompt: `130929`; generated output: `128` (`127` timed forwards); devices: `2`.

| Arm | Seconds | ms/token | tokens/s |
|:---|---:|---:|---:|
| dense_bf16_full_kv_sdpa | 15.213 | 119.791 | 8.348 |
| dense_exact_k_c1_v96_triton | 47.104 | 370.901 | 2.696 |

C1-V96 Triton speedup versus dense BF16 SDPA: `0.323x`.

Dense prefill: `107.273` seconds; C1 prefill: `949.055` seconds; dynamic-to-static cache conversion: `0.141` seconds.

The C1 decode arm uses the existing fused Triton online-softmax kernel with a fixed-capacity cache and device-resident valid length.

One-step C1 SDPA/Triton check: argmax `17` / `17`; relative logits L2 error `0.00490548`; finite `true`.
