# Qwen3-8B-Base C1 KQ-routing RULER-v1 32K

BF16 dense, C1-V64 exact-QK, and C1-V64 with R32/B1024 KQ routing use identical official base-model prompts and greedy decoding.

| Task | Samples | BF16 dense | C1 exact-QK | R32/B1024 | Routing-C1 | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 1 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 1 | **100.00%** | **100.00%** | **100.00%** | **+0.00 pp** | **0** | **0** |

Logical exact-K traffic: `139.979 MiB/decode token`; physical selected-K fraction: `0.060792`; persistent GPU KV ratio: `0.3750`.

Prompt prefill uses chunked exact-QK C1-V64 SDPA. The first generated token comes from dense C1 prefill; KQ routing is enabled for subsequent decode tokens. Exact K remains physically GPU-resident, so traffic is a logical CPU page-store read volume rather than measured PCIe latency.
