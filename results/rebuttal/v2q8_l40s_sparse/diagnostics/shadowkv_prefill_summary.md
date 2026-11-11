# Why ShadowKV's prefill is slower than BasisKV's in V2Q8 round 2

2026-11-11, 4x L40S node 2, Llama-3.1-8B-Instruct BF16, V128. Numbers are from `formal_v2/` unless a row says
"trial". Trials are single runs on cohort 0 under `smoke_v3/exp2/` and are not formal results.

## Summary

- At TP4, 64K prompt, 16 concurrent requests, all 16 requests are prefilled in 103.5 s by BasisKV and in 231.5 s by
  upstream ShadowKV (median of 3 cohorts): a factor of 2.2. At TP1, 130K, one request, it is 36.5 s against
  53.3 s: a factor of 1.46.
- The model computation the two methods share takes the same time in both. The whole gap is work that only
  ShadowKV does during prefill: a per-layer SVD of the Keys, and landmark selection with the Value offload.
- That extra work does not shrink under tensor parallelism, while the shared attention cost shrinks a lot. This is
  why the factor grows from 1.46 at TP1 to 2.2 at TP4.
- The per-layer, in-line SVD is upstream ShadowKV's own code. Our TP port adds only the transfer of Keys and
  factors between GPUs around it.
- The prefill group size is not the cause. We tried 1, 4 (upstream) and 16 requests per forward; ShadowKV's best
  total is 215.0 s, still twice BasisKV's.

## Terms

- **Request**: one prompt plus its generated tokens.
- **Batch** (B16): the number of requests served at once. Their caches are resident together and each decode step
  produces one token for every request.
- **Prefill group**: how many of those requests go through the model in one prefill forward. It is a scheduling
  choice inside a fixed batch.

| Method | Batch in Exp2 | Requests per prefill forward | Where it comes from |
|---|---|---|---|
| BasisKV | 16 | 1 | our implementation |
| ShadowKV | 16 | 4 | upstream `batch_prefill`, hard-coded (8 for 120K-200K prompts) |
| LRQK | 16 | 16 | upstream runs the whole batch in one forward |

## Where upstream ShadowKV runs the SVD

`external/ShadowKV` is at `e51904cd` with a clean working tree. `get_svd` has one call site in the repository,
inside `layer_compute` (`models/base.py`, lines 129-133), which runs once per layer:

```python
if q_len > 4*1024: # prefill
    # svd unrope key and save
    self.kv_cache.get_svd(key_states, layer_idx=layer_idx)
    query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
    self.kv_cache.prefill_kv_cache(value_states, layer_idx, key_states, query_states[:, :, -1:])
```

So the SVD is not one step after the whole prefill. Each of the 32 layers decomposes its own pre-RoPE Keys before
that layer's attention runs, and the layer does not continue until the SVD has finished. "One-time SVD" means once
per prompt per layer, with nothing recomputed during decode. Inside `get_svd` (`models/kv_cache.py`, line 480)
upstream synchronizes the GPU, calls `gc.collect()` and `torch.cuda.empty_cache()`, converts the Keys to FP32,
runs `torch.svd` on the 8 KV heads joined into 1024 dimensions, keeps 160 components, and copies U and SV to CPU.

| Behaviour | Source |
|---|---|
| SVD inside each layer's forward, before attention | upstream |
| GPU synchronize, `gc.collect`, `empty_cache` before the SVD | upstream |
| FP32, 8 heads joined (1024 dims), rank 160 | upstream |
| Request groups of 4 in `batch_prefill` | upstream |
| Gather the Keys of all GPUs on one GPU, then send U and SV back | our TP port (upstream has no TP) |

Under TP4 each GPU holds 2 KV heads (256 dims). To keep upstream's joint decomposition of all 8 heads, the port
(`basisserve/core/shadowkv_upstream_tp.py`, `tp_get_svd`) sends every GPU's Keys to the layer's owner GPU
(`layer % 4`), runs upstream's SVD there, sends the full U to every GPU (U is shared by all heads) and each GPU's
slice of SV. The other three GPUs wait meanwhile. A per-GPU SVD of the local 256 dimensions would be a different
compression, so the port does not do that. At TP1 there is no port: upstream runs as is.

## Prefill time per request, by component

From the breakdown runs (cohort 0). These use synchronized timers, so totals are a little higher than in the
uninstrumented runs (6.5 s and 14.0 s per request in Exp2).

**Exp2: TP4, 64K, B16**

| Component | BasisKV | ShadowKV upstream |
|---|---|---|
| Attention | 1.28 s | 1.42 s |
| Other model work (projections, MLP, norms, AllReduce) | 5.14 s | 5.18 s |
| **Shared work** | **6.42 s** | **6.60 s** |
| BasisKV routing codes and Key offload | 0.27 s | n/a |
| ShadowKV `get_svd` | n/a | 7.24 s |
| ShadowKV landmarks, outlier chunks and Value offload | n/a | 1.70 s |
| **Method-specific work** | **0.27 s** | **8.94 s** |
| One forward, per request | 6.70 s | 15.55 s |

ShadowKV also spends 0.44 s per request after prefill moving its factors back to the GPU (`H2D`).

**Exp1: TP1, 130K, B1**

| Component | BasisKV | ShadowKV upstream |
|---|---|---|
| Attention | 24.67 s | 23.54 s |
| Other model work | 11.38 s | 10.59 s |
| **Shared work** | **36.05 s** | **34.13 s** |
| BasisKV routing codes and Key offload | 0.53 s | n/a |
| ShadowKV `get_svd` | n/a | 10.29 s |
| ShadowKV landmarks, outlier chunks and Value offload | n/a | 5.36 s |
| **Method-specific work** | **0.53 s** | **15.65 s** |
| One forward | 36.59 s | 49.78 s |

`H2D` adds 1.08 s for ShadowKV.

## Why the factor is larger at TP4

- Attention is quadratic in the prompt length and its heads are split over the GPUs. From TP1 130K to TP4 64K it
  falls from about 24 s to about 1.4 s per request.
- ShadowKV's extra work is linear in the prompt length and TP does not divide it: the SVD of a layer runs on one
  GPU. It falls only from 15.65 s to 8.94 s.
- Its share of ShadowKV's forward therefore rises from 31% to 57%.

## What is inside `get_svd` at TP4 (7.24 s per request)

| Part | Per request |
|---|---|
| SVD of the 8 layers this GPU owns | 0.94 s |
| Waiting for the other GPUs' 24 layers, then receiving the factors | 2.94 s |
| Sending this GPU's Keys to the owner | 0.28 s |
| Rest: upstream's synchronize, `gc.collect`, `empty_cache`, and copying U to CPU | 3.08 s |

Of these, the Key transfer (0.28 s) is added by the port. The factor transfer is inside the 2.94 s waiting row and
was not timed on its own. At TP1 the same function costs 10.29 s: 6.13 s in the SVD and 4.16 s in the rest.

## Prefill group size trials for ShadowKV (TP4, 64K, B16, cohort 0, 128 output tokens)

Arms `shadowkv_upstream_g1_v128` and `shadowkv_upstream_g16_v128` repeat the loop of upstream `batch_prefill` in
`bench_v2q8_tp.py` with its group constant changed. Upstream's cache already accepts any group size, so no
upstream file is edited. These are not the upstream schedule.

| | Groups of 1 (trial) | Groups of 4 (upstream) | All 16 (trial) | BasisKV |
|---|---|---|---|---|
| All 16 requests prefilled | 332.6 s | 231.1 s | 215.0 s | 103.5 s |
| Per request | 20.3 s | 14.0 s | 12.9 s | 6.5 s |
| First request's TTFT | 20.4 s | 57.4 s | 207.2 s | 6.5 s |
| Median TTFT | 171.4 s | 140.8 s | 207.2 s | 55.0 s |
| Last request's TTFT | 325.6 s | 223.9 s | 207.2 s | 103.5 s |
| End to end | 337.9 s | 236.2 s | 219.9 s | 108.6 s |
| Mean decode step | 41.7 ms | 40.0 ms | 38.7 ms | 40.1 ms |
| Peak PyTorch allocation during prefill | at most 18.6 GiB | at most 18.6 GiB | 32.2 GiB | 26.1 GiB |
| Peak PyTorch reservation during prefill | at most 20.0 GiB | at most 20.0 GiB | 43.0 GiB | 27.9 GiB |
| Resident GPU memory per GPU | 18.7 GiB | 18.7 GiB | 18.7 GiB | 24.2 GiB |

- Larger groups are faster in total for ShadowKV, with little gained from 4 to 16.
- With all 16 in one forward the reservation peaks at 43.0 GiB of the L40S's 44.4 GiB.
- Larger groups delay the first token: with all 16 together every request waits 207 s.
- Groups of 1 and 4 generate identical tokens for all 16 requests. With all 16 together, 15 requests are
  identical to groups of 4 and one differs from token 19 on. The run is not teacher-forced, so everything after
  the first difference differs too. The cause was not examined.
- Whether BasisKV would gain from more than one request per forward was not tested.

The formal ShadowKV result keeps upstream's groups of 4.

## Limits of these statements

- This is the speed of upstream's implementation, not a lower bound for the method. The factors are not used
  until decode, so an implementation could run the SVD in the background instead of in line. Upstream does not,
  and we did not add it. If `get_svd` cost nothing, ShadowKV's forward would be 8.31 s per request against 6.70 s
  for BasisKV; a background SVD would still compete with the forward for the GPU.
- The trials for groups of 1 and 16 are one run each. No breakdown was run for them, so the reason groups of 1
  are slower per request is not measured.
- The TP1 result agrees with the earlier independent `tp1_offline` run of upstream ShadowKV: 53.5 s prefill then,
  53.3 s now.

## Files

- Formal runs: `formal_v2/exp1/`, `formal_v2/exp2/` (`*_request_*`, `*_steady_*`, `*_breakdown_*`).
- Trials: `smoke_v3/exp2/shadowkv_upstream_g1_v128_request_b16_t65536/`,
  `smoke_v3/exp2/shadowkv_upstream_g16_v128_request_b16_t65536/`.
- Runner log: `logs/round2_runner.log`.
- Deviations from upstream: `audit/COMPATIBILITY.md` sections 12 to 12.3.
