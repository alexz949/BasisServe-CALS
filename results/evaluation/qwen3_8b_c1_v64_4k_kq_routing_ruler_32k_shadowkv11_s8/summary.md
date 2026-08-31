# Qwen3-8B-Base C1 KQ-routing RULER-v1 32K

BF16 dense, C1-V64 exact-QK, and C1-V64 with R32/B1024 KQ routing use identical official base-model prompts and greedy decoding.

| Task | Samples | BF16 dense | C1 exact-QK | R32/B1024 | Routing-C1 | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_2 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_3 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_multikey_1 | 8 | 87.50% | 87.50% | 87.50% | +0.00 pp | 0 | 0 |
| niah_multikey_2 | 8 | 100.00% | 50.00% | 25.00% | -25.00 pp | 2 | 0 |
| niah_multiquery | 8 | 93.75% | 84.38% | 90.62% | +6.25 pp | 0 | 2 |
| niah_multivalue | 8 | 100.00% | 87.50% | 84.38% | -3.12 pp | 2 | 1 |
| vt | 8 | 95.00% | 87.50% | 90.00% | +2.50 pp | 1 | 1 |
| fwe | 8 | 87.50% | 79.17% | 58.33% | -20.83 pp | 4 | 0 |
| qa_1 | 8 | 50.00% | 50.00% | 50.00% | +0.00 pp | 0 | 0 |
| qa_2 | 8 | 37.50% | 12.50% | 25.00% | +12.50 pp | 0 | 1 |
| **Task-balanced mean** | 88 | **86.48%** | **76.23%** | **73.71%** | **-2.52 pp** | **9** | **5** |

Logical exact-K traffic: `132.350 MiB/decode token`; physical selected-K fraction: `0.057331`; persistent GPU KV ratio: `0.3750`.

Prompt prefill uses chunked exact-QK C1-V64 SDPA. The first generated token comes from dense C1 prefill; KQ routing is enabled for subsequent decode tokens. Exact K remains physically GPU-resident, so traffic is a logical CPU page-store read volume rather than measured PCIe latency.
