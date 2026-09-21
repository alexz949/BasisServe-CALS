# Section 4 Routing Ablation: Mechanism Results

Model: Llama-3.1-8B-Instruct. All arms use Dense V128, original W_O, Page32,
sink32/recent64 inside B2048, and exact selected post-RoPE K for final attention.

## Matched logical width

| Method | Width | Extra state | Mass | Routed page recall | Page KL | Post-W_O rel-MSE |
|---|---:|---:|---:|---:|---:|---:|
| B4R16 | 20 | 16 | 0.945889 | 0.823286 | 0.146185 | 0.003528 |
| R20-only | 20 | 20 | 0.942864 | 0.726301 | 0.322598 | 0.004132 |

## Residual objective

| Objective | Mass | Routed page recall | Page KL | Post-W_O rel-MSE |
|---|---:|---:|---:|---:|
| residual_mse | 0.938064 | 0.786112 | 0.427594 | 0.005205 |
| score_mse | 0.946463 | 0.847181 | 0.126483 | 0.003471 |
| page_fisher | 0.946392 | 0.838779 | 0.120886 | 0.003436 |

## Base-rank sweep

Best Base-only retained mass: rank 96.
See `plots/base_rank_sweep.pdf` and the layer-level CSV for the complete trajectory.

RULER columns remain intentionally empty until the mechanism results have been inspected.
