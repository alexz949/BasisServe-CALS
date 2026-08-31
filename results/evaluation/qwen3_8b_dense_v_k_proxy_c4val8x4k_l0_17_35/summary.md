# Qwen3-8B dense-V to K linear probe

The probe is fit on C4 windows that are disjoint from every reported held-out metric. `pre_rope` uses exact token positions only after the linear V-to-K prediction; `direct_post` is a single position-independent map. C1-V64 is evaluated on the same examples as a controlled source ablation.

## Aggregate

| layer | proxy | pre-K centered rel-MSE | post-K centered rel-MSE | post-K cosine | centered-score rel-RMSE | KL(P||P_proxy) | R@512 | mass@512 | oracle-mass@512 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | c1_v64_direct_post | - | 0.567793 | 0.871922 | 0.864609 | 4.353163 | 0.471621 | 0.475700 | 0.962250 |
| all | c1_v64_pre_rope | 0.356207 | 0.262325 | 0.933590 | 0.497142 | 2.948103 | 0.754396 | 0.703878 | 0.962250 |
| all | dense_v_direct_post | - | 0.540705 | 0.878629 | 0.853295 | 4.317110 | 0.484428 | 0.498499 | 0.962250 |
| all | dense_v_pre_rope | 0.309111 | 0.227649 | 0.942140 | 0.461487 | 2.864469 | 0.774648 | 0.716571 | 0.962250 |

## Per layer

| layer | proxy | pre-K centered rel-MSE | post-K centered rel-MSE | post-K cosine | centered-score rel-RMSE | KL(P||P_proxy) | R@512 | mass@512 | oracle-mass@512 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | dense_v_pre_rope | 0.189027 | 0.144962 | 0.998434 | 0.313366 | 0.281609 | 0.870908 | 0.954414 | 0.963179 |
| 0 | dense_v_direct_post | - | 0.425812 | 0.991290 | 0.902496 | 4.250822 | 0.441052 | 0.451558 | 0.963179 |
| 0 | c1_v64_pre_rope | 0.228892 | 0.175518 | 0.998290 | 0.336049 | 0.329333 | 0.853729 | 0.951376 | 0.963179 |
| 0 | c1_v64_direct_post | - | 0.452060 | 0.991256 | 0.903449 | 4.238753 | 0.436345 | 0.440789 | 0.963179 |
| 17 | dense_v_pre_rope | 0.399744 | 0.323161 | 0.891966 | 0.547197 | 3.341869 | 0.732605 | 0.703453 | 0.969013 |
| 17 | dense_v_direct_post | - | 0.582098 | 0.791135 | 0.715306 | 3.734109 | 0.603913 | 0.640292 | 0.969013 |
| 17 | c1_v64_pre_rope | 0.474876 | 0.383897 | 0.869790 | 0.602113 | 3.449211 | 0.702496 | 0.686030 | 0.969013 |
| 17 | c1_v64_direct_post | - | 0.627381 | 0.772657 | 0.753218 | 3.845659 | 0.578757 | 0.613862 | 0.969013 |
| 35 | dense_v_pre_rope | 0.827377 | 0.443103 | 0.936018 | 0.618188 | 4.969931 | 0.720431 | 0.491848 | 0.954557 |
| 35 | dense_v_direct_post | - | 0.944218 | 0.853462 | 0.987791 | 4.966398 | 0.408320 | 0.403649 | 0.954557 |
| 35 | c1_v64_pre_rope | 0.866826 | 0.464230 | 0.932689 | 0.640017 | 5.065766 | 0.706964 | 0.474229 | 0.954557 |
| 35 | c1_v64_direct_post | - | 0.953855 | 0.851853 | 0.982127 | 4.975076 | 0.399759 | 0.372450 | 0.954557 |

All Top-k metrics use proxy logits for selection. The oracle mass uses the true exact-QK Top-k on the identical query rows. This is a quality diagnostic, not a PCIe/runtime measurement.
