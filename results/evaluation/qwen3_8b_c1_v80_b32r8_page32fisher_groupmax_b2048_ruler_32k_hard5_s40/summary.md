# Qwen3-8B-Base C1 routing RULER-v1 32K

BF16 dense, C1-V80 exact-QK, and C1-V80 with Base32+R8/B2048 V-conditioned predictive-base plus residual routing use identical official base-model prompts and greedy decoding.

| Task | Samples | BF16 dense | C1 exact-QK | Base32+R8/B2048 | Routing-C1 | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 1 | 100.00% | 100.00% | 0.00% | -100.00 pp | 1 | 0 |
| niah_multivalue | 1 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_2 | 1 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_3 | 1 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| fwe | 1 | 100.00% | 100.00% | 33.33% | -66.67 pp | 1 | 0 |
| **Task-balanced mean** | 5 | **100.00%** | **100.00%** | **66.67%** | **-33.33 pp** | **2** | **0** |

Logical exact-K traffic: `144.000 MiB/decode token`; physical selected-K fraction: `0.063677`; persistent GPU KV ratio: `0.3438`.

Prompt prefill uses chunked exact-QK C1-V80 SDPA. The first generated token comes from dense C1 prefill; V-conditioned predictive-base plus residual routing is enabled for subsequent decode tokens. Exact K remains physically GPU-resident, so traffic is a logical CPU page-store read volume rather than measured PCIe latency.
