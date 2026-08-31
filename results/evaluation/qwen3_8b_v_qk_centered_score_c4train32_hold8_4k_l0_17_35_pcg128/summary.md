# Qwen3-8B Value-to-QK centered-score proxy

Each query head has its own linear map `M_h`. The fit directly minimizes causal per-query centered QK score error; it does not fit or materialize a full approximate Key as its objective.

## Aggregate

| source | split | centered-score rel-RMSE | KL(P||proxy) | R@512 | mass@512 | oracle-mass@512 |
|:---|:---|---:|---:|---:|---:|---:|
| c1_v64 | fit | 0.797941 | 4.206144 | 0.540844 | 0.565769 | 0.963019 |
| c1_v64 | heldout | 0.831023 | 4.349091 | 0.510146 | 0.526317 | 0.964391 |
| dense_v128 | fit | 0.771419 | 4.087245 | 0.567082 | 0.609534 | 0.963019 |
| dense_v128 | heldout | 0.820699 | 4.301333 | 0.522622 | 0.553063 | 0.964391 |

## Per layer held-out

| layer/source | split | centered-score rel-RMSE | KL(P||proxy) | R@512 | mass@512 | oracle-mass@512 |
|:---|:---|---:|---:|---:|---:|---:|
| L0 dense_v128 | heldout | 0.848098 | 4.327400 | 0.516387 | 0.492720 | 0.961730 |
| L0 c1_v64 | heldout | 0.850033 | 4.337548 | 0.508212 | 0.482581 | 0.961730 |
| L17 dense_v128 | heldout | 0.702427 | 3.564231 | 0.626857 | 0.686778 | 0.971896 |
| L17 c1_v64 | heldout | 0.732577 | 3.645217 | 0.607251 | 0.661283 | 0.971896 |
| L35 dense_v128 | heldout | 0.977109 | 5.012369 | 0.424622 | 0.479692 | 0.959547 |
| L35 c1_v64 | heldout | 0.976958 | 5.064507 | 0.414975 | 0.435087 | 0.959547 |

Exact KL and Top-k metrics are evaluation metrics. The fitted objective is centered score MSE solved by batched matrix-free CG.
