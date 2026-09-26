# Section 4 / Experiment 1: V -> pre-RoPE K affine predictability

Model `models--Qwen--Qwen3-8B` (config sha `f7c4eadfbbf5`), compressed V `ckpt_v96` (manifest sha `ee184aa35a06`, C1 uniform V96 encoder and decoder). Calibration `c4_retrieval_50_50` (sha `b3e7142cbd60`): fit windows [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], held-out windows [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], 131072 tokens each (4194304 fit / 2097152 held-out tokens per layer).

Target: pre-RoPE K = W_k h (after k_norm for Qwen3), per KV group, all tokens of every window. Metric: centered explained energy 1 - ||K_c - Khat_c||_F^2 / ||K_c||_F^2 on the held-out windows (K_c centered with the held-out mean); affine predictors fitted on the fit windows only; fit-split values reported alongside.

## Held-out explained K energy (mean over KV groups)

| layers | local compressed V | all-group compressed V | local raw V | all-group raw V |
|---|---:|---:|---:|---:|
| first third (0-11) | 48.42% | 80.42% | 52.66% | 84.05% |
| middle third (12-23) | 54.46% | 83.76% | 57.83% | 86.50% |
| last third (24-35) | 32.96% | 71.63% | 36.83% | 76.31% |
| all (0-35) | 45.28% | 78.61% | 49.11% | 82.29% |

Fit-split values (same predictors, fit windows): local compressed V 46.07%, all-group compressed V 79.25%, local raw V 49.89%, all-group raw V 82.86%.

## Per layer (held-out, mean over groups; min-max over groups for local compressed V)

| layer | local compressed V | all-group compressed V | local raw V | all-group raw V | local compressed V min-max |
|---:|---:|---:|---:|---:|---:|
| 0 | 77.27% | 89.77% | 79.86% | 91.49% | 66.1-88.5% |
| 1 | 56.92% | 89.81% | 62.45% | 92.51% | 43.9-68.2% |
| 2 | 45.07% | 79.31% | 49.19% | 83.85% | 26.6-59.7% |
| 3 | 47.32% | 84.06% | 51.86% | 87.48% | 32.9-59.6% |
| 4 | 45.46% | 77.11% | 49.65% | 81.40% | 27.6-59.1% |
| 5 | 49.86% | 80.66% | 54.21% | 84.08% | 38.0-62.2% |
| 6 | 46.84% | 80.27% | 52.49% | 84.33% | 38.3-62.5% |
| 7 | 43.13% | 79.93% | 47.85% | 83.75% | 20.2-59.9% |
| 8 | 42.95% | 77.95% | 46.93% | 81.84% | 29.6-63.7% |
| 9 | 36.53% | 71.46% | 40.34% | 75.92% | 15.1-59.8% |
| 10 | 43.65% | 76.53% | 47.63% | 80.42% | 25.8-60.7% |
| 11 | 46.00% | 78.24% | 49.45% | 81.49% | 36.1-56.7% |
| 12 | 52.71% | 83.49% | 56.32% | 86.23% | 35.4-76.3% |
| 13 | 53.56% | 77.84% | 56.91% | 81.18% | 35.3-70.5% |
| 14 | 62.20% | 86.15% | 65.85% | 88.80% | 40.3-78.7% |
| 15 | 57.66% | 82.35% | 60.67% | 85.01% | 42.9-75.2% |
| 16 | 44.86% | 81.50% | 48.59% | 84.90% | 25.4-66.9% |
| 17 | 60.79% | 85.00% | 63.84% | 87.50% | 35.7-82.6% |
| 18 | 56.68% | 89.78% | 59.54% | 91.62% | 27.1-88.0% |
| 19 | 51.85% | 82.47% | 55.11% | 85.13% | 23.1-82.9% |
| 20 | 58.76% | 87.91% | 61.77% | 90.10% | 29.7-82.9% |
| 21 | 59.16% | 89.08% | 61.94% | 91.04% | 20.3-76.8% |
| 22 | 47.95% | 79.46% | 51.64% | 82.70% | 25.9-76.1% |
| 23 | 47.42% | 80.07% | 51.74% | 83.81% | 28.0-69.3% |
| 24 | 38.96% | 79.60% | 43.19% | 83.52% | 17.5-65.7% |
| 25 | 45.38% | 76.59% | 48.54% | 79.99% | 24.1-78.2% |
| 26 | 33.49% | 72.12% | 37.06% | 76.55% | 10.9-46.2% |
| 27 | 40.14% | 73.82% | 43.68% | 77.79% | 23.5-65.2% |
| 28 | 48.39% | 83.19% | 51.73% | 85.76% | 27.0-89.0% |
| 29 | 23.86% | 69.56% | 29.09% | 75.74% | 10.4-37.2% |
| 30 | 27.56% | 63.93% | 30.95% | 69.12% | 14.1-43.5% |
| 31 | 30.86% | 74.66% | 35.75% | 80.05% | 13.6-67.0% |
| 32 | 29.49% | 65.66% | 33.54% | 70.67% | 16.9-45.8% |
| 33 | 23.09% | 66.86% | 27.37% | 72.91% | 15.6-34.0% |
| 34 | 26.00% | 65.78% | 29.48% | 71.35% | 14.6-47.7% |
| 35 | 28.30% | 67.86% | 31.61% | 72.27% | 9.3-54.2% |

Plots: `v_to_k_by_layer.pdf` (held-out explained energy vs layer, four predictors), `v_to_k_by_layer_group.pdf` (layer x KV-group heatmaps, compressed V, local and all-group panels). Exact values in `result.json`.
