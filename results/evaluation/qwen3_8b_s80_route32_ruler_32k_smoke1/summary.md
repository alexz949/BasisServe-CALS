# Qwen3-8B-Base Store80/Route32 RULER-v1 32K

Store80 exact-K and Store80 Route32-selected exact-K use the same joint 80-dimensional cache and identical greedy-decoding prompts. Route32 reads the first 32 orthogonal Store80 coordinates; it does not allocate a second routing sidecar.

| Task | Samples | BF16 dense | Store80 exact-K | Route32/B1024 | Routing delta | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 1 | 100.00% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 1 | **100.00%** | **0.00%** | **0.00%** | **+0.00 pp** | **0** | **0** |

Logical exact-K traffic: `143.189 MiB/decode token`; physical selected-K fraction: `0.062171`; deployable joint-cache scalar ratio: `0.3125`.

Prompt prefill and the first generated token use Store80 exact-K. Route32 is enabled for subsequent decode tokens. Exact K remains GPU-resident in this correctness oracle, so exact-K traffic is logical.
