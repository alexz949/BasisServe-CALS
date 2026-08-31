# Qwen3-8B coordinate-selected MLP C1 WikiText-2

Static Gram/SRRQR coordinates use a free least-squares decoder. Runtime is a single-GPU quality equivalent; communication reduction is analytical.

| Variant | K/source | PPL | PPL delta | Delta NLL | Paired SE | Top-1 agreement | Mean calibration output MSE | Ideal communication reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1536 | 7.002508 | 0 | 0 | 0 | 1 | 0 | 0% |
| coordinate C1 | 576 | 30.90577 | +341.353% | +1.484674 | 0.03195274 | 0.478114 | 0.193786 | 43.750% |
