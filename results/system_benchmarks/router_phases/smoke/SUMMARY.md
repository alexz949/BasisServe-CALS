# Router phase experiment

First real decode query in each of 32 layers. Graph-replayed kernels exclude Python overhead. Phase proportions are clock64 block cycles from a split, instrumented layout, including barriers/stalls; scaled wall proxies are NOT independent measured phase latencies. Transformed query omits original intermediate BF16 K/RoPE rounding and is a diagnostic prototype, not a serving replacement. V80 is a dimension-only control; actual native values are Dense V128.

Command: `python -m benchmarks.system.profile_router_phases --smoke`; environment: `basis`.

| Timed component | Sum over 32 layers (ms) |
|---|---:|
| original_page_kernel | 1.3439 |
| instrumented_split_kernel | 1.4958 |
| transformed_query_total | 3.3194 |
| residual_query_projection | 0.1441 |
| append_V128_to_B16 | 0.0702 |
| append_V80_to_B16_dimension_control | 0.0736 |

| Phase | Approximate wall proxy (ms) |
|---|---:|
| load_base_right | 0.2064 |
| B16_to_K128 | 0.1683 |
| bias_BF16_round | 0.1994 |
| RoPE_BF16_round | 0.2295 |
| load_query_codes | 0.1586 |
| QK | 0.1659 |
| residual_dot_score_round | 0.1320 |
| Page_LSE | 0.0839 |

Mean selected-page overlap with original: 99.741679%.
Maximum page-score difference: 0.253555.
