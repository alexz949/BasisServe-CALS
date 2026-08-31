# Qwen3-8B-Base Store80/Route32 RULER-v1 32K

Store80 exact-K and Store80 Route32-selected exact-K use the same joint 80-dimensional cache and identical greedy-decoding prompts. Route32 reads the first 32 orthogonal Store80 coordinates; it does not allocate a second routing sidecar.

| Task | Samples | BF16 dense | Store80 exact-K | Route32/B1024 | Routing delta | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 8 | 100.00% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| niah_single_2 | 8 | 100.00% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| niah_single_3 | 8 | 100.00% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| niah_multikey_1 | 8 | 87.50% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| niah_multikey_2 | 8 | 100.00% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| niah_multiquery | 8 | 93.75% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| niah_multivalue | 8 | 100.00% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| vt | 8 | 95.00% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| fwe | 8 | 87.50% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| qa_1 | 8 | 50.00% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| qa_2 | 8 | 37.50% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 88 | **86.48%** | **0.00%** | **0.00%** | **+0.00 pp** | **0** | **0** |

Logical exact-K traffic: `151.673 MiB/decode token`; physical selected-K fraction: `0.066482`; deployable joint-cache scalar ratio: `0.3125`.

Prompt prefill and the first generated token use Store80 exact-K. Route32 is enabled for subsequent decode tokens. Exact K remains GPU-resident in this correctness oracle, so exact-K traffic is logical.
