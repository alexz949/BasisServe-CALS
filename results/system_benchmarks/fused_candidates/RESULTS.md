# Fused candidate selector results

Llama-3.1-8B base, single L40S, TP1, batch1, 64K window64, Dense V128, original Wo, CPU K offload, B16R16 two-stage routing, 512 candidate pages, hard2048 including sink32/recent64. Fused append is enabled in BOTH arms; only the candidate selector differs.

| ABBA run | CUDA median ms/step, after first10 |
|---|---:|
| Original selector 20 | 30.3022 |
| Fused selector 20 | 29.1005 |
| Fused selector 21 | 29.0801 |
| Original selector 21 | 30.2817 |

Mean of run medians: **30.2920 -> 29.0903 ms/step**, saving **1.2017 ms**, reduction **3.97%**. Each run uses100 continuous decode steps; this is decode timing, excluding prefill. Four recorded101-token sequences match exactly and all logits are finite.

A separate instrumented real-query check compared **3232/3232 identical candidate ID sets**. Its timing is excluded from speed results. The8K model smoke verified selected-support attention in all32 layers (at8K all pages fit within512, so the64K check is essential for validating the new selector).

Synthetic validation covers24 cases: batch2/8KVheads, random/tied/large-magnitude scores, 8K-128K and non-power-of-two page boundaries. All selected score multisets match the original exactly. Threshold ties use ascending IDs; PyTorch topk tie order is unspecified, so this does not guarantee identical tie resolution on all future inputs.

| Pages (batch2) | Original selector us | Fused selector us |
|---|---:|---:|
| 1024 | 37.12 | 8.04 |
| 2048 | 40.60 | 9.98 |
| 2049 | 42.14 | 9.73 |
| 2049 | 42.04 | 9.88 |
| 4096 | 50.38 | 13.82 |

The microbenchmark uses CUDA graphs and synthetic inputs; use full decode timings for serving conclusions. The previous append-only experiment measured31.31 ->30.31ms; the present29.09ms is consistent with roughly7.1% cumulative reduction from that older unfused reference, but that cumulative comparison is across runs, not this experiment's paired contrast.

Environment: basis. Commands:

```bash
python -m benchmarks.system.validate_fused_candidates
python -m benchmarks.system.run_fused_candidates
```

Completed model job8327476. The initial full-sort prototype hit the installed Triton's infinity handling issue; finite sentinels fixed it. Full sorting was then replaced with exact radix selection after poor128K timing. Current radix validation and model jobs passed. Attempts are retained in validation.log.

Code is opt-in via --fused-select; this has not changed the default RULER runtime. The previously completed330-question evaluation predates these fusions. No commit or push.
