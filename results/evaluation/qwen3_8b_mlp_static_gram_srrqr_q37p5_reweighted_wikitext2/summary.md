# Qwen3-8B MLP static Gram/SRRQR WikiText-2

Fixed TP-source-local masks are calibrated on C4 with the full activation/output contribution Gram. Evaluation uses dense masked execution; reported sparse-exchange reductions are analytical.

| Variant | K/source | PPL | PPL delta | Delta NLL | Paired SE | Top-1 agreement | Retained input energy | Transformed input energy | Calibration output residual | Ideal sparse exchange reduction vs AllReduce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1536 | 7.002508 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0% |
| mlp_static_gram_srrqr_37p5_source_local_reweighted | 576 | 55.96937 | +699.276% | +2.078536 | 0.07994176 | 0.399539 | 0.672638 | 0.686090 | 0.308920 | 43.750% |
