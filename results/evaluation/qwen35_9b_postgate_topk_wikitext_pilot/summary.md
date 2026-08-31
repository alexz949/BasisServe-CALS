# Qwen3.5 post-gate Top-K cross-domain quality

Exact checkpoint gates and states are retained. Top-K is applied to the post-gate wire immediately before the dense output projection. The hook is a quality oracle and is not a sparse-kernel timing result. Packet bytes assume fixed-cardinality BF16 values plus one source-local bitmask.

## WikiText-2

| Variant | K/source | Packet B/source/layer | PPL | Delta NLL | Paired SE | Top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| dense | 512 | 1024 | 12.3135 | 0 | 0 | 1 |
| full_topk_50_source_local | F256 | 576 | 12.40386 | +0.007311583 | 0.002131162 | 0.959516 |
| gdn_topk_50_source_local | G256 | 576 | 12.37271 | +0.004797697 | 0.001608798 | 0.966365 |
| both_topk_50_source_local | F256/G256 | 576 | 12.47278 | +0.01285267 | 0.00263537 | 0.950465 |
| full_topk_75_gdn_topk_50_source_local | F384/G256 | 640 | 12.387 | +0.005951881 | 0.001635059 | 0.962696 |
| full_topk_50_gdn_topk_75_source_local | F256/G384 | 768 | 12.43628 | +0.009922266 | 0.003299148 | 0.958293 |
