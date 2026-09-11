# Qwen3-8B Base: frozen eight-sample FWE diagnosis

Same FWE64..71 from local RULER seed42/32K. BF16,50-token cap, greedy,
lowrank environment, direct GPUs3/6. No old results overwritten.

| Configuration | Score |
|---|---:|
| C1 KL96 + Full K (existing reference) |66.6667|
| C1 KL96 + ShadowKV rank160 (repeat) |37.5000|
| C1 KL96 + ShadowKV selection, exact historical K |70.8333|
| Original exact V + Full K |83.3333|
| Original exact V + ShadowKV rank160 |79.1667|

All four diagnostic arms completed eight samples. Repeated compressed-V ShadowKV
generated token IDs match existing records exactly. Exact-K substitution preserves
the selection algorithm, not necessarily selected IDs on later divergent queries.
This implicates reconstruction in the compressed-V degradation; exact-V ShadowKV
shows a much smaller gap. It does not establish an implementation bug or generalize
beyond these eight examples. K error and exact attention-mass coverage were measured
only at the first decode step per layer, stored in per-sample routing statistics.

Command: python -u evaluation/diagnose_shadowkv_fwe.py --arm ARM
(repeat, exact_k, dense_full, dense_shadow).
Logs: results/logs/shadowkv_fwe. Later Llama experiments are separate controls.
