# C1 shadow-Key speculative verification oracle

This is a correctness and acceptance oracle, not a production throughput or CPU-offload benchmark.

Model: `Qwen/Qwen3-32B`; prompts: `4`; shadow bits: `8`; recent exact window: `0`.

| Draft | Exact match | Mean accepted | Mean shadow accepted | Full block | Shadow top-1 | Block/seq top-1 | Exact calls/token | Corrections |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | False | 1.9922 | 0.9922 | 0.9922 | 0.9922 | n/a | 0.503906 | 1 |
| 4 | False | 3.8182 | 2.8182 | 0.9394 | 0.9789 | n/a | 0.273438 | 4 |
| 8 | False | 6.7568 | 5.7568 | 0.8378 | 0.9726 | n/a | 0.167969 | 6 |

The first proposal in every round is seeded by exact-target logits. `Mean shadow accepted` removes that forced seed. The selected commit policy is `direct_block`. Direct block commit uses pending exact target KV and only performs an additional one-token target forward for a correction. Exact sequence match is always measured against ordinary sequential C1 greedy decoding.
