# Section 4 / Experiment 1: V -> pre-RoPE K affine predictability

Model `models--meta-llama--Llama-3.1-8B-Instruct` (config sha `29e4c210b0d6`), compressed V `ckpt_B` (manifest sha `4fa8237e16ff`, C1 V96 encoder and decoder). Calibration `c4_retrieval_50_50` (sha `252df671b860`): fit windows [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], held-out windows [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], 131072 tokens each (4194304 fit / 2097152 held-out tokens per layer).

Target: pre-RoPE K = W_k h (after k_norm for Qwen3), per KV group, all tokens of every window. Metric: centered explained energy 1 - ||K_c - Khat_c||_F^2 / ||K_c||_F^2 on the held-out windows (K_c centered with the held-out mean); affine predictors fitted on the fit windows only; fit-split values reported alongside.

## Held-out explained K energy (mean over KV groups)

| layers | local compressed V | all-group compressed V | local raw V | all-group raw V |
|---|---:|---:|---:|---:|
| first third (0-9) | 56.99% | 85.37% | 60.96% | 88.10% |
| middle third (10-20) | 50.95% | 83.53% | 54.47% | 86.38% |
| last third (21-31) | 28.47% | 67.52% | 31.41% | 72.09% |
| all (0-31) | 45.11% | 78.60% | 48.57% | 82.00% |

Fit-split values (same predictors, fit windows): local compressed V 46.22%, all-group compressed V 79.71%, local raw V 49.68%, all-group raw V 82.99%.

## Per layer (held-out, mean over groups; min-max over groups for local compressed V)

| layer | local compressed V | all-group compressed V | local raw V | all-group raw V | local compressed V min-max |
|---:|---:|---:|---:|---:|---:|
| 0 | 68.83% | 93.28% | 72.02% | 95.29% | 43.4-94.5% |
| 1 | 71.96% | 95.92% | 76.62% | 97.00% | 52.6-88.1% |
| 2 | 47.60% | 77.44% | 51.55% | 81.41% | 24.8-59.9% |
| 3 | 45.34% | 78.28% | 50.06% | 82.14% | 30.9-62.7% |
| 4 | 51.72% | 86.20% | 56.08% | 88.86% | 35.9-67.8% |
| 5 | 53.33% | 80.52% | 56.78% | 83.50% | 29.5-70.1% |
| 6 | 51.58% | 83.28% | 55.30% | 86.27% | 38.1-61.1% |
| 7 | 64.61% | 87.94% | 68.49% | 90.23% | 50.5-73.6% |
| 8 | 58.46% | 85.73% | 62.17% | 88.23% | 33.3-78.1% |
| 9 | 56.53% | 85.12% | 60.56% | 88.04% | 27.1-77.7% |
| 10 | 60.03% | 86.05% | 63.52% | 88.38% | 41.0-70.6% |
| 11 | 69.45% | 88.22% | 72.58% | 90.49% | 41.7-80.3% |
| 12 | 57.04% | 86.24% | 60.78% | 89.15% | 41.3-67.3% |
| 13 | 56.36% | 90.70% | 59.50% | 92.51% | 23.8-90.4% |
| 14 | 62.46% | 90.37% | 65.51% | 92.23% | 39.4-86.9% |
| 15 | 39.52% | 75.90% | 43.67% | 79.89% | 20.6-70.2% |
| 16 | 50.62% | 82.76% | 54.74% | 85.89% | 21.2-79.9% |
| 17 | 39.71% | 77.51% | 43.57% | 81.05% | 22.9-63.6% |
| 18 | 42.06% | 81.99% | 45.98% | 85.03% | 17.8-64.1% |
| 19 | 44.90% | 78.53% | 48.04% | 81.86% | 15.8-73.4% |
| 20 | 38.34% | 80.62% | 41.26% | 83.67% | 23.5-81.6% |
| 21 | 35.75% | 78.83% | 39.22% | 82.37% | 16.4-72.2% |
| 22 | 29.45% | 69.64% | 32.69% | 74.47% | 15.6-41.7% |
| 23 | 25.24% | 60.57% | 28.12% | 65.82% | 15.4-42.8% |
| 24 | 27.17% | 63.88% | 29.91% | 69.07% | 13.2-50.1% |
| 25 | 28.74% | 66.54% | 31.19% | 70.65% | 14.6-51.9% |
| 26 | 24.18% | 61.57% | 26.86% | 66.93% | 12.6-32.4% |
| 27 | 29.27% | 65.92% | 31.87% | 70.56% | 18.0-44.5% |
| 28 | 27.11% | 64.42% | 29.57% | 68.87% | 10.0-41.2% |
| 29 | 24.96% | 66.13% | 27.52% | 70.80% | 8.6-45.9% |
| 30 | 23.65% | 61.06% | 26.23% | 66.13% | 7.9-38.8% |
| 31 | 37.68% | 84.10% | 42.32% | 87.30% | 21.8-59.6% |

Plots: `v_to_k_by_layer.pdf` (held-out explained energy vs layer, four predictors), `v_to_k_by_layer_group.pdf` (layer x KV-group heatmaps, compressed V, local and all-group panels). Exact values in `result.json`.
