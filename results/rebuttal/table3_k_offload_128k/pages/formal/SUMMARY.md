# Table 3 matched Key-offload arms

Llama-3.1-8B-Instruct BF16, TP1/B1, dense V128, B16R16 full scan, 1984 routed plus 64 recent tokens, NVIDIA L40S, torch 2.13.0+cu126, conda env `basis213`. 16 conditioning and 128 measured decode steps after a full prefill; prefill is excluded from decode timing.

Arms: `dense_resident` and `basis_resident` are the archived Table 3 runtimes. `basis_resident_nobase` reads Base from the first 16 Value coordinates instead of a separate Base cache. `basis_k_offload_nobase` additionally keeps exact Keys in mapped pinned host memory and fetches selected Keys into persistent GPU slots. Arms without a suffix use the archived Page32 router; `_pN` arms use the page-size-general build of the same router with N-token pages.

## 130048 tokens: median over prompt cohorts

| Arm | Cohorts | Attention block ms | Full-model wall ms | Full-model CUDA ms | Decode resident GiB | Decode peak allocated GiB | Decode peak reserved GiB | Host pinned-Key GiB |
|---|---|---|---|---|---|---|---|---|
| `basis_resident_nobase_p32` | 3 | 14.000 | 37.227 | 37.205 | 31.98 | 31.98 | 32.22 | 0.00 |
| `basis_k_offload_nobase_p32` | 3 | 15.906 | 39.100 | 39.078 | 24.29 | 24.29 | 24.35 | 7.95 |
| `basis_resident_nobase_p4` | 3 | 15.987 | 39.234 | 39.211 | 32.09 | 32.09 | 32.21 | 0.00 |
| `basis_k_offload_nobase_p4` | 3 | 18.048 | 41.200 | 41.177 | 24.40 | 24.40 | 24.47 | 7.95 |

`basis_k_offload_nobase_p32` against its matched resident control (`basis_resident_nobase_p32`):

- Full-model wall latency: 1.050x (37.227 to 39.100 ms).
- Attention block: 1.136x.
- GPU decode resident memory: 31.98 to 24.29 GiB, 7.69 GiB less (24.0%).
- The same 7.95 GiB of exact Keys now sits in pinned host memory; total system memory is not reduced.

- Key slots: 521589 selected Keys per step over all layers, 131692 fetched from host (32.151 MiB per step).

`basis_k_offload_nobase_p4` against its matched resident control (`basis_resident_nobase_p4`):

- Full-model wall latency: 1.050x (39.234 to 41.200 ms).
- Attention block: 1.129x.
- GPU decode resident memory: 32.09 to 24.40 GiB, 7.69 GiB less (24.0%).
- The same 7.95 GiB of exact Keys now sits in pinned host memory; total system memory is not reduced.

- Key slots: 524025 selected Keys per step over all layers, 153413 fetched from host (37.454 MiB per step).

Memory by phase, GiB (PyTorch allocated unless stated):

| Phase | `basis_resident_nobase_p32` | `basis_k_offload_nobase_p32` | `basis_resident_nobase_p4` | `basis_k_offload_nobase_p4` |
|---|---|---|---|---|
| Model and preallocated caches | 31.97 | 24.29 | 32.09 | 24.40 |
| Prefill peak allocated | 39.48 | 31.79 | 39.59 | 31.90 |
| After prefill and placement | 31.98 | 24.29 | 32.09 | 24.40 |
| Steady decode start | 31.98 | 24.29 | 32.09 | 24.40 |
| Steady decode peak allocated | 31.98 | 24.29 | 32.09 | 24.40 |
| Steady decode end | 31.98 | 24.29 | 32.09 | 24.40 |
| Overall peak allocated | 39.48 | 31.79 | 39.59 | 31.90 |
| Overall peak reserved | 41.47 | 33.60 | 41.46 | 33.72 |
| Decode peak reserved, allocator cache released after prefill | 32.22 | 24.35 | 32.21 | 24.47 |
| Device used in decode, allocator cache released | 32.76 | 24.90 | 32.74 | 25.02 |

Attention-block components, ms per step summed over 32 layers (separate profiled run):

| Arm | attention_block | cache_append | key_refresh | page_selection | router_scan | sparse_attention |
|---|---|---|---|---|---|---|
| `basis_resident_nobase_p32` | 14.000 | 0.386 | - | 0.614 | 11.889 | 0.800 |
| `basis_k_offload_nobase_p32` | 15.906 | 0.338 | 1.874 | 0.616 | 11.875 | 0.779 |
| `basis_resident_nobase_p4` | 15.987 | 0.388 | - | 1.122 | 13.357 | 0.808 |
| `basis_k_offload_nobase_p4` | 18.048 | 0.338 | 2.126 | 1.120 | 13.317 | 0.783 |

Per cohort:

| Arm | Cohort | Status | Attention block ms | Full wall ms | Full CUDA ms | Decode resident GiB |
|---|---|---|---|---|---|---|
| `basis_resident_nobase_p32` | 0 | complete | 13.992 | 37.214 | 37.192 | 31.98 |
| `basis_resident_nobase_p32` | 1 | complete | 14.000 | 37.256 | 37.234 | 31.98 |
| `basis_resident_nobase_p32` | 2 | complete | 14.029 | 37.227 | 37.205 | 31.98 |
| `basis_k_offload_nobase_p32` | 0 | complete | 15.849 | 38.975 | 38.947 | 24.29 |
| `basis_k_offload_nobase_p32` | 1 | complete | 15.906 | 39.100 | 39.078 | 24.29 |
| `basis_k_offload_nobase_p32` | 2 | complete | 16.067 | 39.224 | 39.201 | 24.29 |
| `basis_resident_nobase_p4` | 0 | complete | 15.989 | 39.234 | 39.211 | 32.09 |
| `basis_resident_nobase_p4` | 1 | complete | 15.971 | 39.228 | 39.206 | 32.09 |
| `basis_resident_nobase_p4` | 2 | complete | 15.987 | 39.266 | 39.244 | 32.09 |
| `basis_k_offload_nobase_p4` | 0 | complete | 18.048 | 41.171 | 41.148 | 24.40 |
| `basis_k_offload_nobase_p4` | 1 | complete | 18.015 | 41.200 | 41.177 | 24.40 |
| `basis_k_offload_nobase_p4` | 2 | complete | 18.239 | 41.382 | 41.358 | 24.40 |

## State-level bytes

Tensor inventory after the measured decode, lowest cohort, counted once per backing storage. Caches are preallocated for prompt plus generated tokens; active bytes scale the token-indexed states by the tokens actually stored.

### `basis_resident_nobase_p32`, 130048 tokens (130192 stored of 130193 capacity)

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

### `basis_k_offload_nobase_p32`, 130048 tokens (130192 stored of 130193 capacity)

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

### `basis_resident_nobase_p4`, 130048 tokens (130192 stored of 130193 capacity)

| Device | State | Tensors | Allocated GiB | Active GiB |
|---|---|---|---|---|
| gpu | model parameters | parameters | 14.9575 | 14.9575 |
| gpu | Values | value_cache | 7.9464 | 7.9463 |
| gpu | exact historical Keys | key_cache | 7.9464 | 7.9463 |
| gpu | Key residual routing state | residual_cache | 0.9933 | 0.9933 |
| gpu | page scores | router_output, fine_router_output | 0.1261 | 0.1261 |
| gpu | index tables | page_offsets, token_range | 0.0310 | 0.0310 |
| gpu | RoPE tables | shared_rope | 0.0310 | 0.0310 |
| gpu | attention and router workspaces | query_code, attention_workspace, attention_output, slot_workspace (view), single_rope | 0.0240 | 0.0240 |
| gpu | router factors | base_left, base_right, base_bias, residual_encoder, residual_query | 0.0069 | 0.0069 |
| gpu | selected indices | candidate_ids, selected_pages, support_ids | 0.0058 | 0.0058 |
| gpu | Value-derived routing state | base_cache (view) | 0.0000 | 0.0000 |

GPU tensors in this inventory: 32.068 GiB; other live CUDA tensors: 0.001 GiB; PyTorch allocated at the same moment: 32.085 GiB (difference 16.1 MiB, allocator block rounding).

### `basis_k_offload_nobase_p4`, 130048 tokens (130192 stored of 130193 capacity)

| Device | State | Tensors | Allocated GiB | Active GiB |
|---|---|---|---|---|
| gpu | model parameters | parameters | 14.9575 | 14.9575 |
| gpu | Values | value_cache | 7.9464 | 7.9463 |
| gpu | Key residual routing state | residual_cache | 0.9933 | 0.9933 |
| gpu | slot maps | slot_resident, slot_lookup, selected_slots, slot_missing, slot_counts | 0.1339 | 0.1339 |
| gpu | page scores | router_output, fine_router_output | 0.1261 | 0.1261 |
| gpu | selected-Key slots | slot_key_cache | 0.1250 | 0.1250 |
| gpu | index tables | page_offsets, token_range | 0.0310 | 0.0310 |
| gpu | RoPE tables | shared_rope | 0.0310 | 0.0310 |
| gpu | attention and router workspaces | query_code, attention_workspace, attention_output, slot_workspace (view), single_rope | 0.0240 | 0.0240 |
| gpu | router factors | base_left, base_right, base_bias, residual_encoder, residual_query | 0.0069 | 0.0069 |
| gpu | selected indices | candidate_ids, selected_pages, support_ids | 0.0058 | 0.0058 |
| gpu | Value-derived routing state | base_cache (view) | 0.0000 | 0.0000 |
| host | exact historical Keys | host_key | 7.9464 | 7.9463 |

GPU tensors in this inventory: 24.381 GiB; other live CUDA tensors: 0.001 GiB; PyTorch allocated at the same moment: 24.399 GiB (difference 16.9 MiB, allocator block rounding).

