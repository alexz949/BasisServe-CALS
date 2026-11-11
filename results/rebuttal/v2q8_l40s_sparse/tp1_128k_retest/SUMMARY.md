# TP1 request-mode retest, 130048 prompt tokens

Formal runs complete: 6 of 6. Request mode only (same-shape warmup request, fresh state, one measured request of 128 tokens); no steady runs, so there are no per-step percentiles and no host-memory inventory. The three cohorts are three different prompts.

- Model revision `0e9e39f249a16976918f6564b8830bc894c89659`, TP1, B1, BF16, original V128/Wo; inputs `tp1_offline/inputs/prompts.safetensors` (sha256 `7c6485069e8d5802`).
- Host `node6.athena`, NVIDIA L40S, driver 550.54.15; formal runs on physical GPU(s) at PCI 00000000:4F:00.0; Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz, 24 threads on 12 physical cores of the GPU's NUMA node.
- conda env `basis`, PyTorch 2.13.0+cu129, Transformers 5.17.0; repository `be7c8a2` plus uncommitted changes.

## Arms

- `basis_koff_v128_reuse_opt`: BasisKV, exact K offloaded (Base read from V, lean forward, fast router). CUDA allocator setting: expandable_segments:True.
- `shadowkv_upstream_nohk_v128`: ShadowKV with per-layer pre-SVD GC and empty-cache calls disabled. CUDA allocator setting: PyTorch default.

## Results

Median of the cohorts, range in brackets.

| | `basis_koff_v128_reuse_opt` | `shadowkv_upstream_nohk_v128` |
|---|---|---|
| Prefill end (TTFT) | 35.12 s (34.86 to 35.29; n=3) | 44.98 s (44.95 to 45.03; n=3) |
| Placement after prefill (ShadowKV H2D) | 0.00 s (0.00 to 0.00; n=3) | 1.12 s (1.10 to 1.16; n=3) |
| Decode-ready | 35.12 s (34.86 to 35.29; n=3) | 46.11 s (46.10 to 46.13; n=3) |
| Request end to end, 128 tokens | 39.32 s (39.08 to 39.52; n=3) | 50.13 s (50.12 to 50.19; n=3) |
| Request: mean decode step (127 steps) | 33.20 ms (33.06 to 33.35; n=3) | 31.66 ms (31.61 to 31.98; n=3) |
| GPU resident, allocated | 24.23 GiB (24.23 to 24.23; n=3) | 17.60 GiB (17.60 to 17.60; n=3) |
| Prefill peak allocated | 30.75 GiB (30.75 to 30.75; n=3) | 20.61 GiB (20.61 to 20.61; n=3) |
| Prefill peak reserved | 31.44 GiB (31.44 to 31.44; n=3) | 21.72 GiB (21.72 to 21.72; n=3) |
| Decode peak allocated | 24.23 GiB (24.23 to 24.23; n=3) | 17.60 GiB (17.60 to 17.60; n=3) |
| Decode peak reserved | 31.44 GiB (31.44 to 31.44; n=3) | 18.42 GiB (18.42 to 18.42; n=3) |
| Process RSS | 9.71 GiB (9.71 to 9.75; n=3) | 10.18 GiB (10.18 to 10.18; n=3) |

Per run:

| Arm | Cohort | Status | Physical GPU | TTFT s | Placement s | Decode-ready s | E2E s | Step ms | Others on pinned CPUs, max |
|---|---|---|---|---|---|---|---|---|---|
| `basis_koff_v128_reuse_opt` | 0 | complete | 00000000:4F:00.0 | 34.86 | 0.00 | 34.86 | 39.08 | 33.20 | 0.6 |
| `basis_koff_v128_reuse_opt` | 1 | complete | 00000000:4F:00.0 | 35.12 | 0.00 | 35.12 | 39.32 | 33.06 | 0.6 |
| `basis_koff_v128_reuse_opt` | 2 | complete | 00000000:4F:00.0 | 35.29 | 0.00 | 35.29 | 39.52 | 33.35 | 0.5 |
| `shadowkv_upstream_nohk_v128` | 0 | complete | 00000000:4F:00.0 | 44.95 | 1.16 | 46.11 | 50.13 | 31.61 | 0.6 |
| `shadowkv_upstream_nohk_v128` | 1 | complete | 00000000:4F:00.0 | 44.98 | 1.12 | 46.10 | 50.12 | 31.66 | 0.8 |
| `shadowkv_upstream_nohk_v128` | 2 | complete | 00000000:4F:00.0 | 45.03 | 1.10 | 46.13 | 50.19 | 31.98 | 0.9 |

## Checks, probes and failures

- Cleanup scope (8192 tokens, call counts, not timed), `shadowkv_upstream_v128`: inside `get_svd` {'collect': 32, 'empty_cache': 32, 'synchronize': 64, 'svd': 32}; inside `H2D` {'collect': 2, 'empty_cache': 2, 'synchronize': 3}; bindings restored after an exception inside `get_svd`: True; passed: True.
- Cleanup scope (8192 tokens, call counts, not timed), `shadowkv_upstream_nohk_v128`: inside `get_svd` {'collect': 0, 'empty_cache': 0, 'synchronize': 64, 'svd': 32}; inside `H2D` {'collect': 2, 'empty_cache': 2, 'synchronize': 3}; bindings restored after an exception inside `get_svd`: True; passed: True.
- LRQK stream-order check: ordered copy wrong elements {'K': 0, 'V': 0, 'A_K': 0}; passed: True.
- probe_130048, `shadowkv_upstream_v128`: complete; TTFT 51.03 s, decode-ready 52.23 s.
- probe_130048, `shadowkv_upstream_nohk_v128`: complete; TTFT 46.00 s, decode-ready 47.28 s.
- probe_130048, `shadowkv_upstream_nohk_v128` fed the tokens of `shadowkv_upstream_v128`: argmax equal at 8 of 8 positions, max abs log-probability difference 0.0.
- probe_130048, `lrqk_upstream_koff_v128`: failed. **Failed, kind `out_of_memory`, in phase `same_shape_warmup_request`**: [rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 508.00 MiB. GPU 0 has a total capacity of 44.52 GiB of which 46.25 MiB is free. Including non-PyTorch memory, this process has 44.46 GiB memory in use. Of the allocated memory 42.23 GiB is allocated by PyTorch, and 1.30 GiB is reserved by PyTorch but unalloca (innermost upstream frame: [rank0]:   File "/home/yl1139/BasisServe-CALS/external/LRQK/lrqk_attention.py", line 399, in _lrqk_prefill_inv_w1; log `results/rebuttal/v2q8_l40s_sparse/tp1_128k_retest/probe/lrqk_upstream_koff_v128_request_b1_t130048_smoke/run.log`)
- probe_130048, `lrqk_upstream_kvoff_v128`: failed. **Failed, kind `out_of_memory`, in phase `same_shape_warmup_request`**: [rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 508.00 MiB. GPU 0 has a total capacity of 44.52 GiB of which 506.25 MiB is free. Including non-PyTorch memory, this process has 43.98 GiB memory in use. Of the allocated memory 42.24 GiB is allocated by PyTorch, and 820.88 MiB is reserved by PyTorch but unall (innermost upstream frame: [rank0]:   File "/home/yl1139/BasisServe-CALS/external/LRQK/lrqk_attention.py", line 399, in _lrqk_prefill_inv_w1; log `results/rebuttal/v2q8_l40s_sparse/tp1_128k_retest/probe/lrqk_upstream_kvoff_v128_request_b1_t130048_smoke/run.log`)
- probe_130048, `basis_koff_v128_reuse_opt`: complete; TTFT 35.00 s, decode-ready 35.00 s.
- smoke_8k, `shadowkv_upstream_v128`: complete; TTFT 7.74 s, decode-ready 8.09 s.
- smoke_8k, `shadowkv_upstream_nohk_v128`: complete; TTFT 2.72 s, decode-ready 3.09 s.
- smoke_8k, `shadowkv_upstream_nohk_v128` fed the tokens of `shadowkv_upstream_v128`: argmax equal at 8 of 8 positions, max abs log-probability difference 0.0.
- smoke_8k, `basis_koff_v128_reuse_opt`: complete; TTFT 0.70 s, decode-ready 0.70 s.

A probe is one short run (8 generated tokens) on cohort 0; probe timings are not results.
