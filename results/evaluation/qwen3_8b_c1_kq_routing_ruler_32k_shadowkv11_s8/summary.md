# Qwen3-8B-Base C1 KQ-routing RULER-v1 32K

BF16 dense, C1-V64 exact-QK, and C1-V64 with R32/B1024 KQ routing use identical official base-model prompts and greedy decoding.

| Task | Samples | BF16 dense | C1 exact-QK | R32/B1024 | Routing-C1 | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_2 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_3 | 8 | 100.00% | 87.50% | 75.00% | -12.50 pp | 1 | 0 |
| niah_multikey_1 | 8 | 87.50% | 75.00% | 75.00% | +0.00 pp | 0 | 0 |
| niah_multikey_2 | 8 | 100.00% | 25.00% | 12.50% | -12.50 pp | 1 | 0 |
| niah_multiquery | 8 | 93.75% | 87.50% | 90.62% | +3.12 pp | 0 | 1 |
| niah_multivalue | 8 | 100.00% | 78.12% | 78.12% | +0.00 pp | 2 | 3 |
| vt | 8 | 95.00% | 45.00% | 55.00% | +10.00 pp | 1 | 2 |
| fwe | 8 | 87.50% | 66.67% | 66.67% | +0.00 pp | 1 | 1 |
| qa_1 | 8 | 50.00% | 37.50% | 50.00% | +12.50 pp | 0 | 1 |
| qa_2 | 8 | 37.50% | 25.00% | 25.00% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 88 | **86.48%** | **66.12%** | **66.17%** | **+0.06 pp** | **6** | **8** |

Logical exact-K traffic: `130.784 MiB/decode token`; physical selected-K fraction: `0.056773`; persistent GPU KV ratio: `0.3750`.

Prompt prefill uses chunked exact-QK C1-V64 SDPA. The first generated token comes from dense C1 prefill; KQ routing is enabled for subsequent decode tokens. Exact K remains physically GPU-resident, so traffic is a logical CPU page-store read volume rather than measured PCIe latency.
