# Router phase experiment

First real decode query in each of 32 layers. Graph-replayed kernels exclude Python overhead. Phase proportions are clock64 block cycles from a split, instrumented layout, including barriers/stalls; scaled wall proxies are NOT independent measured phase latencies. Transformed query omits original intermediate BF16 K/RoPE rounding and is a diagnostic prototype, not a serving replacement. V80 is a dimension-only control; actual native values are Dense V128.

Command: `python -m benchmarks.system.profile_router_phases`; environment: `basis`.

| Timed component | Sum over 32 layers (ms) |
|---|---:|
| original_page_kernel | 9.7085 |
| original_wrapper_eager | 10.4817 |
| instrumented_split_kernel | 10.6498 |
| transformed_query_total | 24.9690 |
| transformed_coefficients_once | 0.0570 |
| residual_query_projection | 0.1302 |
| append_V128_to_B16 | 0.0546 |
| append_V80_to_B16_dimension_control | 0.0537 |

| Phase | Approximate wall proxy (ms) |
|---|---:|
| load_base_right | 1.3848 |
| B16_to_K128 | 1.3216 |
| bias_BF16_round | 1.2977 |
| RoPE_BF16_round | 1.5841 |
| load_query_codes | 1.0751 |
| QK | 1.3638 |
| residual_dot_score_round | 1.0551 |
| Page_LSE | 0.6263 |

Mean selected-page overlap with original: 99.401456%.
Maximum page-score difference: 0.497707.
