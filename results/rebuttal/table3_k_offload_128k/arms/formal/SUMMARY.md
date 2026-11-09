# Table 3 matched Key-offload arms

Llama-3.1-8B-Instruct BF16, TP1/B1, dense V128, B16R16 full scan, 1984 routed plus 64 recent tokens, NVIDIA L40S, torch 2.13.0+cu126, conda env `basis213`. 16 conditioning and 128 measured decode steps after a full prefill; prefill is excluded from decode timing.

Arms: `dense_resident` and `basis_resident` are the archived Table 3 runtimes. `basis_resident_nobase` reads Base from the first 16 Value coordinates instead of a separate Base cache. `basis_k_offload_nobase` additionally keeps exact Keys in mapped pinned host memory and fetches selected Keys into persistent GPU slots. Arms without a suffix use the archived Page32 router; `_pN` arms use the page-size-general build of the same router with N-token pages.

## 130048 tokens: median over prompt cohorts

| Arm | Cohorts | Attention block ms | Full-model wall ms | Full-model CUDA ms | Decode resident GiB | Decode peak allocated GiB | Decode peak reserved GiB | Host pinned-Key GiB |
|---|---|---|---|---|---|---|---|---|
| `dense_resident` | 3 | 23.821 | 47.347 | 47.324 | 30.86 | 30.86 | 31.12 | 0.00 |
| `basis_resident` | 3 | 14.177 | 37.400 | 37.378 | 32.98 | 32.98 | 33.23 | 0.00 |
| `basis_resident_nobase` | 3 | 14.411 | 37.830 | 37.801 | 31.98 | 31.98 | 32.22 | 0.00 |
| `basis_k_offload_nobase` | 3 | 16.301 | 39.481 | 39.458 | 24.29 | 24.29 | 24.35 | 7.95 |

Speedup over `dense_resident` with the same timing definition:

| Arm | Attention block | Full-model wall | Full-model CUDA |
|---|---|---|---|
| `basis_resident` | 1.68 | 1.27 | 1.27 |
| `basis_resident_nobase` | 1.65 | 1.25 | 1.25 |
| `basis_k_offload_nobase` | 1.46 | 1.20 | 1.20 |

`basis_k_offload_nobase` against its matched resident control (`basis_resident_nobase`):

- Full-model wall latency: 1.044x (37.830 to 39.481 ms).
- Attention block: 1.131x.
- GPU decode resident memory: 31.98 to 24.29 GiB, 7.69 GiB less (24.0%).
- The same 7.95 GiB of exact Keys now sits in pinned host memory; total system memory is not reduced.

- Key slots: 521589 selected Keys per step over all layers, 131705 fetched from host (32.155 MiB per step).

Memory by phase, GiB (PyTorch allocated unless stated):

| Phase | `dense_resident` | `basis_resident` | `basis_resident_nobase` | `basis_k_offload_nobase` |
|---|---|---|---|---|
| Model and preallocated caches | 30.85 | 32.97 | 31.97 | 24.29 |
| Prefill peak allocated | 38.36 | 40.48 | 39.48 | 31.79 |
| After prefill and placement | 30.86 | 32.98 | 31.98 | 24.29 |
| Steady decode start | 30.86 | 32.98 | 31.98 | 24.29 |
| Steady decode peak allocated | 30.86 | 32.98 | 31.98 | 24.29 |
| Steady decode end | 30.86 | 32.98 | 31.98 | 24.29 |
| Overall peak allocated | 38.36 | 40.48 | 39.48 | 31.79 |
| Overall peak reserved | 39.31 | 41.42 | 41.47 | 33.60 |
| Decode peak reserved, allocator cache released after prefill | 31.12 | 33.23 | 32.22 | 24.35 |
| Device used in decode, allocator cache released | 31.62 | 33.74 | 32.76 | 24.90 |

Attention-block components, ms per step summed over 32 layers (separate profiled run):

| Arm | attention_block | cache_append | dense_attention | key_refresh | page_selection | router_scan | sparse_attention |
|---|---|---|---|---|---|---|---|
| `dense_resident` | 23.821 | 0.243 | 23.377 | - | - | - | - |
| `basis_resident` | 14.177 | 0.387 | - | - | 0.922 | 11.750 | 0.805 |
| `basis_resident_nobase` | 14.411 | 0.393 | - | - | 1.013 | 11.896 | 0.807 |
| `basis_k_offload_nobase` | 16.301 | 0.338 | - | 1.877 | 1.011 | 11.905 | 0.786 |

Per cohort:

| Arm | Cohort | Status | Attention block ms | Full wall ms | Full CUDA ms | Decode resident GiB |
|---|---|---|---|---|---|---|
| `dense_resident` | 0 | complete | 23.859 | 47.696 | 47.668 | 30.86 |
| `dense_resident` | 1 | complete | 23.805 | 47.287 | 47.264 | 30.86 |
| `dense_resident` | 2 | complete | 23.821 | 47.347 | 47.324 | 30.86 |
| `basis_resident` | 0 | complete | 14.192 | 37.739 | 37.706 | 32.98 |
| `basis_resident` | 1 | complete | 14.163 | 37.400 | 37.378 | 32.98 |
| `basis_resident` | 2 | complete | 14.177 | 37.390 | 37.367 | 32.98 |
| `basis_resident_nobase` | 0 | complete | 14.368 | 37.830 | 37.801 | 31.98 |
| `basis_resident_nobase` | 1 | complete | 14.411 | 37.620 | 37.597 | 31.98 |
| `basis_resident_nobase` | 2 | complete | 14.470 | 37.919 | 37.891 | 31.98 |
| `basis_k_offload_nobase` | 0 | complete | 16.301 | 39.386 | 39.364 | 24.29 |
| `basis_k_offload_nobase` | 1 | complete | 16.279 | 39.481 | 39.458 | 24.29 |
| `basis_k_offload_nobase` | 2 | complete | 16.457 | 39.962 | 39.931 | 24.29 |

## State-level bytes

Tensor inventory after the measured decode, lowest cohort, counted once per backing storage. Caches are preallocated for prompt plus generated tokens; active bytes scale the token-indexed states by the tokens actually stored.

### `dense_resident`, 130048 tokens (130192 stored of 130193 capacity)

| Device | State | Tensors | Allocated GiB | Active GiB |
|---|---|---|---|---|
| gpu | model parameters | parameters | 14.9575 | 14.9575 |
| gpu | exact historical Keys | key_cache | 7.9464 | 7.9463 |
| gpu | Values | value_cache | 7.9464 | 7.9463 |

GPU tensors in this inventory: 30.850 GiB; other live CUDA tensors: 0.001 GiB; PyTorch allocated at the same moment: 30.859 GiB (difference 8.1 MiB, allocator block rounding).

### `basis_resident`, 130048 tokens (130192 stored of 130193 capacity)

| Device | State | Tensors | Allocated GiB | Active GiB |
|---|---|---|---|---|
| gpu | model parameters | parameters | 14.9575 | 14.9575 |
| gpu | Values | value_cache | 7.9464 | 7.9463 |
| gpu | exact historical Keys | key_cache | 7.9464 | 7.9463 |
| gpu | Value-derived routing state | base_cache | 0.9933 | 0.9933 |
| gpu | Key residual routing state | residual_cache | 0.9933 | 0.9933 |
| gpu | index tables | page_offsets, token_range | 0.0310 | 0.0310 |
| gpu | RoPE tables | shared_rope | 0.0310 | 0.0310 |
| gpu | attention and router workspaces | query_code, attention_workspace, attention_output, slot_workspace (view), single_rope | 0.0240 | 0.0240 |
| gpu | page scores | router_output, fine_router_output | 0.0175 | 0.0175 |
| gpu | router factors | base_left, base_right, base_bias, residual_encoder, residual_query | 0.0069 | 0.0069 |
| gpu | selected indices | candidate_ids, selected_pages, support_ids | 0.0050 | 0.0050 |

GPU tensors in this inventory: 32.952 GiB; other live CUDA tensors: 0.001 GiB; PyTorch allocated at the same moment: 32.975 GiB (difference 22.1 MiB, allocator block rounding).

### `basis_resident_nobase`, 130048 tokens (130192 stored of 130193 capacity)

| Device | State | Tensors | Allocated GiB | Active GiB |
|---|---|---|---|---|
| gpu | model parameters | parameters | 14.9575 | 14.9575 |
| gpu | Values | value_cache | 7.9464 | 7.9463 |
| gpu | exact historical Keys | key_cache | 7.9464 | 7.9463 |
| gpu | Key residual routing state | residual_cache | 0.9933 | 0.9933 |
| gpu | index tables | page_offsets, token_range | 0.0310 | 0.0310 |
| gpu | RoPE tables | shared_rope | 0.0310 | 0.0310 |
| gpu | attention and router workspaces | query_code, attention_workspace, attention_output, slot_workspace (view), single_rope | 0.0240 | 0.0240 |
| gpu | page scores | router_output, fine_router_output | 0.0175 | 0.0175 |
| gpu | router factors | base_left, base_right, base_bias, residual_encoder, residual_query | 0.0069 | 0.0069 |
| gpu | selected indices | candidate_ids, selected_pages, support_ids | 0.0050 | 0.0050 |
| gpu | Value-derived routing state | base_cache (view) | 0.0000 | 0.0000 |

GPU tensors in this inventory: 31.959 GiB; other live CUDA tensors: 0.001 GiB; PyTorch allocated at the same moment: 31.975 GiB (difference 15.3 MiB, allocator block rounding).

### `basis_k_offload_nobase`, 130048 tokens (130192 stored of 130193 capacity)

| Device | State | Tensors | Allocated GiB | Active GiB |
|---|---|---|---|---|
| gpu | model parameters | parameters | 14.9575 | 14.9575 |
| gpu | Values | value_cache | 7.9464 | 7.9463 |
| gpu | Key residual routing state | residual_cache | 0.9933 | 0.9933 |
| gpu | slot maps | slot_resident, slot_lookup, selected_slots, slot_missing, slot_counts | 0.1339 | 0.1339 |
| gpu | selected-Key slots | slot_key_cache | 0.1250 | 0.1250 |
| gpu | index tables | page_offsets, token_range | 0.0310 | 0.0310 |
| gpu | RoPE tables | shared_rope | 0.0310 | 0.0310 |
| gpu | attention and router workspaces | query_code, attention_workspace, attention_output, slot_workspace (view), single_rope | 0.0240 | 0.0240 |
| gpu | page scores | router_output, fine_router_output | 0.0175 | 0.0175 |
| gpu | router factors | base_left, base_right, base_bias, residual_encoder, residual_query | 0.0069 | 0.0069 |
| gpu | selected indices | candidate_ids, selected_pages, support_ids | 0.0050 | 0.0050 |
| gpu | Value-derived routing state | base_cache (view) | 0.0000 | 0.0000 |
| host | exact historical Keys | host_key | 7.9464 | 7.9463 |

GPU tensors in this inventory: 24.272 GiB; other live CUDA tensors: 0.001 GiB; PyTorch allocated at the same moment: 24.288 GiB (difference 16.1 MiB, allocator block rounding).

