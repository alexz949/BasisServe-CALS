# Qwen3-8B Value-to-QK centered-score proxy

Each query head has its own linear map `M_h`. The fit directly minimizes causal per-query centered QK score error; it does not fit or materialize a full approximate Key as its objective.

## Aggregate

| source | split | centered-score rel-RMSE | KL(P||proxy) | R@512 | mass@512 | oracle-mass@512 |
|:---|:---|---:|---:|---:|---:|---:|
| c1_v64 | fit | 0.798702 | 4.210205 | 0.539844 | 0.564054 | 0.963019 |
| c1_v64 | heldout | 0.829921 | 4.345769 | 0.510836 | 0.526175 | 0.964391 |
| dense_v128 | fit | 0.773081 | 4.096601 | 0.565050 | 0.607274 | 0.963019 |
| dense_v128 | heldout | 0.818053 | 4.289546 | 0.524386 | 0.556778 | 0.964391 |

## Per layer held-out

| layer/source | split | centered-score rel-RMSE | KL(P||proxy) | R@512 | mass@512 | oracle-mass@512 |
|:---|:---|---:|---:|---:|---:|---:|
| L0 dense_v128 | heldout | 0.846865 | 4.321648 | 0.517148 | 0.495674 | 0.961730 |
| L0 c1_v64 | heldout | 0.849616 | 4.335457 | 0.508581 | 0.483734 | 0.961730 |
| L17 dense_v128 | heldout | 0.698748 | 3.552827 | 0.629105 | 0.687491 | 0.971896 |
| L17 c1_v64 | heldout | 0.730893 | 3.640522 | 0.608195 | 0.659174 | 0.971896 |
| L35 dense_v128 | heldout | 0.972075 | 4.994163 | 0.426906 | 0.487170 | 0.959547 |
| L35 c1_v64 | heldout | 0.974896 | 5.061329 | 0.415732 | 0.435617 | 0.959547 |

Exact KL and Top-k metrics are evaluation metrics. The fitted objective is centered score MSE solved by batched matrix-free CG.
