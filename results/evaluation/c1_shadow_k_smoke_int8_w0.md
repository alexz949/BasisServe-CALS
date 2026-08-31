# C1 shadow-Key speculative verification oracle

This is a correctness and acceptance oracle, not a production throughput or CPU-offload benchmark.

Model: `Qwen/Qwen3-32B`; prompts: `4`; shadow bits: `8`; recent exact window: `0`.

| Draft | Exact match | Mean accepted | Mean shadow accepted | Full block | Shadow top-1 | Block/seq top-1 | Exact calls/token | Corrections |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | True | 1.9766 | 0.9766 | 0.9766 | 0.9766 | 0.9922 | 1.500000 | 3 |
| 4 | True | 3.9844 | 2.9844 | 0.9844 | 0.9948 | 0.9922 | 1.250000 | 1 |
| 8 | True | 7.2000 | 6.2000 | 0.8857 | 0.9819 | 0.9844 | 1.136719 | 4 |

The first proposal in every round is seeded by exact-target logits. `Mean shadow accepted` removes that forced seed. On rejection, the oracle strictly replays every emitted token through one-token exact target forwards so committed cache state follows ordinary sequential C1 greedy execution. Block/seq top-1 reports finite-precision divergence in block verification.
