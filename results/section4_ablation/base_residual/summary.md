# Section 4 / Experiment 2: Base / residual components (Llama-3.1-8B-Instruct, V96, page 4, 2048 routed + recent 64, no sink)

RULER-128K on the first 30 prompts of each of the 11 tasks (330 prompts, identical across arms; matched by index and input hash). Retained mass = exact attention mass captured by the selected support (sink 0 + recent 64 + 2048 routed tokens in pages of 4) on the 16 held-out calibration windows (131072 tokens, 32 queries per window, all layers); oracle (exact-QK page selection) = 93.71%.

| Method | V-derived dims | K-derived dims | Retained mass | Page recall | RULER Avg. |
|---|---:|---:|---:|---:|---:|
| B16 | 16 | 0 | 41.99% | 60.4% | 4.20 |
| B16 + pinned first page (4-token sink) | 16 | 0 | 84.43% (sink 4, oracle 93.61%) | 62.5% | 46.78 |
| R16 | 0 | 16 | 92.02% | 64.7% | 76.48 |
| B16R16 | 16 | 16 | 93.14% | 79.7% | 79.48 |
| R32 | 0 | 32 | 93.02% | 75.5% | 79.52 |
| Full-K (exact routing) | - | - | 93.71% (oracle) | 100% | 82.57 |

Paired differences vs B16R16 (task-balanced mean, bootstrap 95% CI on the common prompts):

- b16 - b16r16: -75.29 [-78.68, -71.76], win/loss/tie 1/272/57
- b16sink - b16r16: -32.71 [-36.80, -28.62], win/loss/tie 10/150/170
- r16 - b16r16: -3.00 [-5.17, -0.80], win/loss/tie 11/23/296
- r32 - b16r16: +0.03 [-1.12, +1.01], win/loss/tie 10/6/314
- full - b16r16: +3.09 [+1.25, +4.96], win/loss/tie 34/15/281

## Per task

| task | B16 | B16 + pinned first page (4-token sink) | R16 | B16R16 | R32 | Full-K (exact routing) |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 16.7 | 70.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| niah_single_2 | 0.0 | 86.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| niah_single_3 | 0.0 | 16.7 | 96.7 | 96.7 | 96.7 | 100.0 |
| niah_multikey_1 | 3.3 | 86.7 | 96.7 | 96.7 | 96.7 | 96.7 |
| niah_multikey_2 | 0.0 | 23.3 | 50.0 | 76.7 | 73.3 | 86.7 |
| niah_multiquery | 1.7 | 60.0 | 97.5 | 99.2 | 99.2 | 98.3 |
| niah_multivalue | 2.5 | 25.0 | 95.8 | 95.8 | 94.2 | 94.2 |
| vt | 2.0 | 27.3 | 64.7 | 66.0 | 68.0 | 78.0 |
| fwe | 10.0 | 32.2 | 43.3 | 43.3 | 50.0 | 57.8 |
| qa_1 | 3.3 | 53.3 | 60.0 | 63.3 | 60.0 | 60.0 |
| qa_2 | 6.7 | 33.3 | 36.7 | 36.7 | 36.7 | 36.7 |

Exact values, protocols and prompt indices in `result.json`.
