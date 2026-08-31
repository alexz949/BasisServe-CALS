# Qwen3-8B C1 KQ sidecar 128K decode benchmark

Prompt: `32628` tokens; forced decode: `4` tokens; pipeline devices: `2`.

| Arm | Seconds | ms/token | tokens/s |
|:---|---:|---:|---:|
| lazy_full_k_projection | 0.643 | 160.654 | 6.225 |
| incremental_cached_sidecar | 0.625 | 156.304 | 6.398 |

Decode speedup: `1.028x`; generated IDs equal: `True`; logical routing statistics equal: `True`.

Shared prefill: `60.059` seconds; resident R32 sidecar: `0.560 GiB`.
