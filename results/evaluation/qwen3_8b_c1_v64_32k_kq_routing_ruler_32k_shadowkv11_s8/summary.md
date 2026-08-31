# Qwen3-8B-Base C1 KQ-routing RULER-v1 32K

BF16 dense, C1-V64 exact-QK, and C1-V64 with R32/B1024 KQ routing use identical official base-model prompts and greedy decoding.

| Task | Samples | BF16 dense | C1 exact-QK | R32/B1024 | Routing-C1 | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_2 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_3 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_multikey_1 | 8 | 87.50% | 87.50% | 87.50% | +0.00 pp | 0 | 0 |
| niah_multikey_2 | 8 | 100.00% | 87.50% | 62.50% | -25.00 pp | 2 | 0 |
| niah_multiquery | 8 | 93.75% | 93.75% | 90.62% | -3.12 pp | 1 | 0 |
| niah_multivalue | 8 | 100.00% | 90.62% | 90.62% | +0.00 pp | 1 | 1 |
| vt | 8 | 95.00% | 82.50% | 77.50% | -5.00 pp | 2 | 0 |
| fwe | 8 | 87.50% | 83.33% | 58.33% | -25.00 pp | 3 | 0 |
| qa_1 | 8 | 50.00% | 50.00% | 62.50% | +12.50 pp | 0 | 1 |
| qa_2 | 8 | 37.50% | 25.00% | 25.00% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 88 | **86.48%** | **81.84%** | **77.69%** | **-4.15 pp** | **9** | **2** |

Logical exact-K traffic: `131.580 MiB/decode token`; physical selected-K fraction: `0.057087`; persistent GPU KV ratio: `0.3750`.

Prompt prefill uses chunked exact-QK C1-V64 SDPA. The first generated token comes from dense C1 prefill; KQ routing is enabled for subsequent decode tokens. Exact K remains physically GPU-resident, so traffic is a logical CPU page-store read volume rather than measured PCIe latency.
