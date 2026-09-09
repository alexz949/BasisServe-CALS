# Qwen3-8B-Base C1 routing RULER-v1 32K

BF16 dense, C1-V80 exact-QK, and C1-V80 with R32/B8192 KQ routing use identical official base-model prompts and greedy decoding.

| Task | Samples | BF16 dense | C1 exact-QK | R32/B8192 | Routing-C1 | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_2 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_3 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_multikey_1 | 8 | 87.50% | 75.00% | 75.00% | +0.00 pp | 0 | 0 |
| niah_multikey_2 | 8 | 100.00% | 50.00% | 37.50% | -12.50 pp | 1 | 0 |
| niah_multiquery | 8 | 93.75% | 93.75% | 78.12% | -15.62 pp | 4 | 1 |
| niah_multivalue | 8 | 100.00% | 90.62% | 90.62% | +0.00 pp | 0 | 0 |
| vt | 8 | 95.00% | 80.00% | 75.00% | -5.00 pp | 2 | 0 |
| fwe | 8 | 87.50% | 79.17% | 70.83% | -8.33 pp | 2 | 0 |
| qa_1 | 8 | 50.00% | 37.50% | 37.50% | +0.00 pp | 1 | 1 |
| qa_2 | 8 | 37.50% | 25.00% | 12.50% | -12.50 pp | 1 | 0 |
| **Task-balanced mean** | 88 | **86.48%** | **75.55%** | **70.64%** | **-4.91 pp** | **11** | **2** |

Logical exact-K traffic: `905.449 MiB/decode token`; physical selected-K fraction: `0.399367`; persistent GPU KV ratio: `0.4375`.

Prompt prefill uses chunked exact-QK C1-V80 SDPA. The first generated token comes from dense C1 prefill; KQ routing is enabled for subsequent decode tokens. Exact K remains physically GPU-resident, so traffic is a logical CPU page-store read volume rather than measured PCIe latency.
