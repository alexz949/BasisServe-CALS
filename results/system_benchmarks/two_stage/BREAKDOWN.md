# Two-stage decode breakdown

CUDA events in actual continuous decode steps 10..19, summed across all 32 layers. Stage spans include launch gaps and instrumentation. Residual categories are subtractions of nested intervals from this same run; no mixed-run subtraction or zero-router lower-bound claim.

Instrumented mean: 34.9245 ms; uninstrumented reference mean steady median: 31.2932 ms. Different sampling statistics; their difference is a diagnostic estimate of perturbation, not a rigorous correction.

Generated sequence unchanged: True.

| Phase | ms/step | Share of instrumented total |
|---|---|---|
| qkv_linear | 2.3290 | 6.67% |
| query_key_rope | 0.1516 | 0.43% |
| wo_linear | 1.8193 | 5.21% |
| mlp_gate_up_linear | 10.2768 | 29.43% |
| mlp_down_linear | 5.2396 | 15.00% |
| append_cpu_key | 0.6114 | 1.75% |
| metadata_update | 0.1929 | 0.55% |
| coarse_scoring | 1.6471 | 4.72% |
| select_512 | 1.6585 | 4.75% |
| fine_router_including_query_projection | 2.3984 | 6.87% |
| final_selection | 0.5252 | 1.50% |
| slot_planner | 0.3969 | 1.14% |
| fetch_missing_key | 1.4897 | 4.27% |
| sparse_attention | 0.9694 | 2.78% |
| pre_norm_and_dispatch | 0.2277 | 0.65% |
| post_norm_silu_residual_dispatch | 0.6302 | 1.80% |
| cache_append_codes_support_and_dispatch | 2.5902 | 7.42% |
| outside_layers_and_dispatch | 1.7707 | 5.07% |

Timing does not include prefill. Configuration: basis environment, single L40S, TP1, Llama-3.1-8B, 64K, B16R16, 512 candidates, final hard2048 including sink32 and recent64.

`python -m benchmarks.system.profile_two_stage`

The earlier 4.21 ms complete-router measurement is a repeated CUDA-graph operator microbenchmark. This event profile measures spans inside normal eager decode, including launch gaps and instrumentation under its actual cache state; its router spans sum to 6.23 ms. These scopes must not be combined or substituted for each other.
