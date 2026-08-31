# Qwen3-8B-Base C1 KQ-routing RULER-v1 32K

BF16 dense, C1-V96 exact-QK, and C1-V96 with R32/B2048 KQ routing use identical official base-model prompts and greedy decoding.

| Task | Samples | BF16 dense | C1 exact-QK | R32/B2048 | Routing-C1 | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 1 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 1 | **100.00%** | **100.00%** | **100.00%** | **+0.00 pp** | **0** | **0** |

Logical exact-K traffic: `270.977 MiB/decode token`; physical selected-K fraction: `0.117884`; persistent GPU KV ratio: `0.5000`.

Prompt prefill uses chunked exact-QK C1-V96 SDPA. The first generated token comes from dense C1 prefill; KQ routing is enabled for subsequent decode tokens. Exact K remains physically GPU-resident, so traffic is a logical CPU page-store read volume rather than measured PCIe latency.
