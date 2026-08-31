# Qwen3-8B C1 KQ sidecar 128K decode benchmark

Prompt: `130929` tokens; generated output: `128` tokens (`127` timed decode forwards); pipeline devices: `2`.

| Arm | Seconds | ms/token | tokens/s |
|:---|---:|---:|---:|
| lazy_full_k_projection | 87.254 | 687.041 | 1.456 |
| incremental_cached_sidecar | 84.703 | 666.956 | 1.499 |

Decode speedup: `1.030x`; generated IDs equal: `True`; logical routing statistics equal: `True`.

Shared prefill: `1014.010` seconds; resident R32 sidecar: `2.248 GiB`.
