# Qwen3-8B-Base C1 KQ-routing RULER-v1 32K

BF16 dense, C1-V96 exact-QK, and C1-V96 with R32/B1024 KQ routing use identical official base-model prompts and greedy decoding.

| Task | Samples | BF16 dense | C1 exact-QK | R32/B1024 | Routing-C1 | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_2 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_3 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_multikey_1 | 8 | 87.50% | 87.50% | 87.50% | +0.00 pp | 0 | 0 |
| niah_multikey_2 | 8 | 100.00% | 75.00% | 62.50% | -12.50 pp | 1 | 0 |
| niah_multiquery | 8 | 93.75% | 87.50% | 90.62% | +3.12 pp | 1 | 1 |
| niah_multivalue | 8 | 100.00% | 93.75% | 100.00% | +6.25 pp | 0 | 1 |
| vt | 8 | 95.00% | 92.50% | 92.50% | +0.00 pp | 0 | 0 |
| fwe | 8 | 87.50% | 83.33% | 70.83% | -12.50 pp | 2 | 0 |
| qa_1 | 8 | 50.00% | 62.50% | 50.00% | -12.50 pp | 1 | 0 |
| qa_2 | 8 | 37.50% | 37.50% | 37.50% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 88 | **86.48%** | **83.60%** | **81.04%** | **-2.56 pp** | **5** | **2** |

Logical exact-K traffic: `131.571 MiB/decode token`; physical selected-K fraction: `0.057104`; persistent GPU KV ratio: `0.5000`.

Prompt prefill uses chunked exact-QK C1-V96 SDPA. The first generated token comes from dense C1 prefill; KQ routing is enabled for subsequent decode tokens. Exact K remains physically GPU-resident, so traffic is a logical CPU page-store read volume rather than measured PCIe latency.
