# Qwen3-8B-Base Store80/Route32 RULER-v1 32K

Store80 exact-K and Store80 Route32-selected exact-K use the same joint 80-dimensional cache and identical greedy-decoding prompts. Route32 reads the first 32 orthogonal Store80 coordinates; it does not allocate a second routing sidecar.

| Task | Samples | BF16 dense | Store80 exact-K | Route32/B2048 | Routing delta | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 8 | 100.00% | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_2 | 8 | 100.00% | 100.00% | 62.50% | -37.50 pp | 3 | 0 |
| niah_single_3 | 8 | 100.00% | 100.00% | 62.50% | -37.50 pp | 3 | 0 |
| niah_multikey_1 | 8 | 87.50% | 87.50% | 62.50% | -25.00 pp | 2 | 0 |
| niah_multikey_2 | 8 | 100.00% | 100.00% | 12.50% | -87.50 pp | 7 | 0 |
| niah_multiquery | 8 | 93.75% | 93.75% | 68.75% | -25.00 pp | 6 | 0 |
| niah_multivalue | 8 | 100.00% | 96.88% | 59.38% | -37.50 pp | 7 | 1 |
| vt | 8 | 95.00% | 95.00% | 92.50% | -2.50 pp | 1 | 0 |
| fwe | 8 | 87.50% | 100.00% | 87.50% | -12.50 pp | 3 | 0 |
| qa_1 | 8 | 50.00% | 37.50% | 50.00% | +12.50 pp | 1 | 2 |
| qa_2 | 8 | 37.50% | 25.00% | 25.00% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 88 | **86.48%** | **85.06%** | **62.10%** | **-22.95 pp** | **33** | **3** |

Logical exact-K traffic: `271.097 MiB/decode token`; physical selected-K fraction: `0.118885`; deployable joint-cache scalar ratio: `0.3125`.

Prompt prefill and the first generated token use Store80 exact-K. Route32 is enabled for subsequent decode tokens. Exact K remains GPU-resident in this correctness oracle, so exact-K traffic is logical.
