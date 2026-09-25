# Section 4 / Experiment 3: page granularity (Llama-3.1-8B-Instruct, V96, B16R16, no sink, recent 64)

3A: exact attention mass captured by sink 0 + recent 64 + B routed tokens in pages of P on the 16 held-out calibration windows (131072 tokens, 32 queries per window, 32 layers). Oracle = exact-QK page selection; router = the page-4 B16R16 factors scored at every page size (fixed, no refit); refit = the B16R16 bank fitted at that page size. Page recall = fraction of the oracle's routed pages the router selects; dispersion = fraction of 256-token segments of the routable region touched by the routed support.

## Table A: retained attention mass

| Page | Router mass B256 | Oracle mass B256 | Router mass B2048 | Oracle mass B2048 | Refit router mass B256 | Refit router mass B2048 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 86.59% | 87.75% | 94.65% | 95.23% | 86.63% | 94.67% |
| 4 | 84.19% | 85.27% | 93.14% | 93.71% | 84.19% | 93.14% |
| 8 | 82.93% | 83.95% | 92.45% | 93.01% | 82.91% | 92.43% |
| 32 | 79.57% | 80.48% | 91.01% | 91.57% | 79.29% | 90.88% |

| Page | Router page recall B256 | Router dispersion B256 | Oracle dispersion B256 | Router page recall B2048 | Router dispersion B2048 | Oracle dispersion B2048 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 68.0% | 22.7% | 21.9% | 78.1% | 51.8% | 51.8% |
| 4 | 70.3% | 12.7% | 12.4% | 79.7% | 33.8% | 33.5% |
| 8 | 71.9% | 9.0% | 8.9% | 80.7% | 26.6% | 26.2% |
| 32 | 74.6% | 4.1% | 4.0% | 82.8% | 16.4% | 16.1% |

## Table B: RULER-128K on niah_multikey_2, fwe (200 prompts, identical across arms)

| Page | niah_multikey_2 | fwe | mean | protocol |
|---|---:|---:|---:|---|
| p1 | 80.0 | 46.3 | 63.17 | Page-Fisher refit, no sink, 2048 + recent 64 |
| p4 | 78.0 | 46.7 | 62.33 | Page-Fisher refit, no sink, 2048 + recent 64 |
| p8 | 75.0 | 50.7 | 62.83 | Page-Fisher refit, no sink, 2048 + recent 64 |
| p32 | 66.0 | 60.3 | 63.17 | released bank: pinned sink page (32) + 1952 routed + recent 64 = 2048 total |
| full | 83.0 | 52.7 | 67.83 | exact attention |

Paired per-task differences (bootstrap 95% CI, 100 prompts per task):

- p1 - p4 | niah_multikey_2: +2.0 [+0.0, +5.0], win/loss 2/0
- p1 - p4 | fwe: -0.3 [-3.7, +3.0], win/loss 9/10
- p1 - p8 | niah_multikey_2: +5.0 [+1.0, +9.0], win/loss 5/0
- p1 - p8 | fwe: -4.3 [-7.3, -1.3], win/loss 4/16
- p1 - p32 | niah_multikey_2: +14.0 [+7.0, +21.0], win/loss 15/1
- p1 - p32 | fwe: -14.0 [-18.0, -10.0], win/loss 2/42
- p4 - p8 | niah_multikey_2: +3.0 [-1.0, +8.0], win/loss 4/1
- p4 - p8 | fwe: -4.0 [-7.3, -1.0], win/loss 5/16
- p32 - p4 | niah_multikey_2: -12.0 [-20.0, -5.0], win/loss 2/14
- p32 - p4 | fwe: +13.7 [+9.0, +18.0], win/loss 47/8
- p32 - p8 | niah_multikey_2: -9.0 [-16.0, -2.0], win/loss 2/11
- p32 - p8 | fwe: +9.7 [+5.7, +13.7], win/loss 34/5

Plots: `page_granularity_mass.pdf` (3A), `page_granularity_ruler.pdf` (3B). Exact values in `diagnostic.json` and `downstream.json`.
