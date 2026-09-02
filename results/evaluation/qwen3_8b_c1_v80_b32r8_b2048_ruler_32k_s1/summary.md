# Qwen3-8B-Base C1 routing RULER-v1 32K

BF16 dense, C1-V80 exact-QK, and C1-V80 with Base32+R8/B2048 V-conditioned predictive-base plus residual routing use identical official base-model prompts and greedy decoding.

| Task | Samples | BF16 dense | C1 exact-QK | Base32+R8/B2048 | Routing-C1 | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 1 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_multikey_1 | 1 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_multiquery | 1 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| fwe | 1 | 100.00% | 66.67% | 66.67% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 4 | **100.00%** | **91.67%** | **91.67%** | **+0.00 pp** | **0** | **0** |

Logical exact-K traffic: `335.926 MiB/decode token`; physical selected-K fraction: `0.149103`; persistent GPU KV ratio: `0.3438`.

Prompt prefill uses chunked exact-QK C1-V80 SDPA. The first generated token comes from dense C1 prefill; V-conditioned predictive-base plus residual routing is enabled for subsequent decode tokens. Exact K remains physically GPU-resident, so traffic is a logical CPU page-store read volume rather than measured PCIe latency.
