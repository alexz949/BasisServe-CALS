# C1 shadow-Key speculative verification oracle

This is a correctness and acceptance oracle, not a production throughput or CPU-offload benchmark.

Model: `Qwen/Qwen3-32B`; prompts: `4`; shadow bits: `16`; recent exact window: `0`.

| Draft | Exact match | Mean accepted | Mean shadow accepted | Full block | Shadow top-1 | Verify calls/token | Corrections |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 2 | True | 1.9844 | 0.9844 | 0.9844 | 0.9844 | 0.507812 | 1 |
| 4 | False | 3.9375 | 2.9375 | 0.9375 | 0.9792 | 0.265625 | 2 |

The first proposal in every round is seeded by exact-target logits. `Mean shadow accepted` removes that forced seed. On rejection, the oracle runs one exact correction-token synchronization forward so committed cache length always equals emitted sequence length.
