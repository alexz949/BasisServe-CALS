# C1 shadow-Key speculative verification oracle

This is a correctness and acceptance oracle, not a production throughput or CPU-offload benchmark.

Model: `Qwen/Qwen3-32B`; prompts: `4`; shadow bits: `16`; recent exact window: `0`.

| Draft | Exact match | Mean accepted | Mean shadow accepted | Full block | Shadow top-1 | Block/seq top-1 | Exact calls/token | Corrections |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | True | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.500000 | 0 |
| 4 | True | 4.0000 | 3.0000 | 1.0000 | 1.0000 | 0.9922 | 1.250000 | 0 |

The first proposal in every round is seeded by exact-target logits. `Mean shadow accepted` removes that forced seed. On rejection, the oracle strictly replays every emitted token through one-token exact target forwards so committed cache state follows ordinary sequential C1 greedy execution. Block/seq top-1 reports finite-precision divergence in block verification.
