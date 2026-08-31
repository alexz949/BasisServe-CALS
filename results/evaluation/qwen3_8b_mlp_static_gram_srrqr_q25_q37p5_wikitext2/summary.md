# Qwen3-8B MLP static Gram/SRRQR WikiText-2

Fixed TP-source-local masks are calibrated on C4 with the full activation/output contribution Gram. Evaluation is a dense zero-fill quality oracle; reported sparse-exchange reductions are analytical.

| Variant | K/source | PPL | PPL delta | Delta NLL | Paired SE | Top-1 agreement | Retained activation energy | Ideal sparse exchange reduction vs AllReduce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1536 | 7.002508 | 0 | 0 | 0 | 1 | 1 | 0% |
| mlp_static_gram_srrqr_25_source_local | 384 | 4809.226 | +68578.623% | +6.532023 | 0.09796035 | 0.040129 | 0.459085 | 62.500% |
| mlp_static_gram_srrqr_37p5_source_local | 576 | 55.92717 | +698.673% | +2.077782 | 0.08285246 | 0.398609 | 0.670824 | 43.750% |
