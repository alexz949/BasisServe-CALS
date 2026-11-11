# TP1 request-mode retest, 65536 prompt tokens

Formal runs complete: 6 of 6. Request mode only (same-shape warmup request, fresh state, one measured request of 128 tokens); no steady runs, so there are no per-step percentiles and no host-memory inventory. The three cohorts are three different prompts.

- Model revision `0e9e39f249a16976918f6564b8830bc894c89659`, TP1, B1, BF16, original V128/Wo; inputs `tp4_64k/inputs/prompts.safetensors` (sha256 `88e50b1c337df818`).
- Host `node6.athena`, NVIDIA L40S, driver 550.54.15; formal runs on physical GPU(s) at PCI 00000000:4F:00.0, 00000000:D5:00.0; Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz, 24 threads on 12 physical cores of the GPU's NUMA node.
- conda env `basis`, PyTorch 2.13.0+cu129, Transformers 5.17.0; repository `be7c8a2` plus uncommitted changes.

## Arms

- `basis_koff_v128_reuse_opt`: BasisKV, exact K offloaded (Base read from V, lean forward, fast router). CUDA allocator setting: expandable_segments:True.
- `shadowkv_upstream_nohk_v128`: ShadowKV with per-layer pre-SVD GC and empty-cache calls disabled. CUDA allocator setting: PyTorch default.

## Results

Median of the cohorts, range in brackets.

| | `basis_koff_v128_reuse_opt` | `shadowkv_upstream_nohk_v128` |
|---|---|---|
| Prefill end (TTFT) | 11.47 s (11.34 to 11.61; n=3) | 17.42 s (17.35 to 17.44; n=3) |
| Placement after prefill (ShadowKV H2D) | 0.00 s (0.00 to 0.00; n=3) | 0.82 s (0.78 to 0.87; n=3) |
| Decode-ready | 11.47 s (11.34 to 11.61; n=3) | 18.25 s (18.13 to 18.31; n=3) |
| Request end to end, 128 tokens | 15.22 s (15.08 to 15.35; n=3) | 22.05 s (21.92 to 22.09; n=3) |
| Request: mean decode step (127 steps) | 29.44 ms (29.43 to 29.58; n=3) | 29.86 ms (29.78 to 29.94; n=3) |
| GPU resident, allocated | 19.72 GiB (19.72 to 19.72; n=3) | 16.46 GiB (16.46 to 16.46; n=3) |
| Prefill peak allocated | 23.00 GiB (23.00 to 23.00; n=3) | 23.30 GiB (23.30 to 23.30; n=3) |
| Prefill peak reserved | 23.35 GiB (23.35 to 23.35; n=3) | 27.79 GiB (27.79 to 27.79; n=3) |
| Decode peak allocated | 19.72 GiB (19.72 to 19.72; n=3) | 16.46 GiB (16.46 to 16.46; n=3) |
| Decode peak reserved | 23.35 GiB (23.35 to 23.35; n=3) | 16.78 GiB (16.78 to 16.78; n=3) |
| Process RSS | 5.77 GiB (5.77 to 5.77; n=3) | 10.18 GiB (10.18 to 10.18; n=3) |

Per run:

| Arm | Cohort | Status | Physical GPU | TTFT s | Placement s | Decode-ready s | E2E s | Step ms | Others on pinned CPUs, max |
|---|---|---|---|---|---|---|---|---|---|
| `basis_koff_v128_reuse_opt` | 0 | complete | 00000000:4F:00.0 | 11.34 | 0.00 | 11.34 | 15.08 | 29.44 | 0.6 |
| `basis_koff_v128_reuse_opt` | 1 | complete | 00000000:D5:00.0 | 11.47 | 0.00 | 11.47 | 15.22 | 29.58 | 2.4 |
| `basis_koff_v128_reuse_opt` | 2 | complete | 00000000:D5:00.0 | 11.61 | 0.00 | 11.61 | 15.35 | 29.43 | 1.4 |
| `shadowkv_upstream_nohk_v128` | 0 | complete | 00000000:4F:00.0 | 17.35 | 0.78 | 18.13 | 21.92 | 29.86 | 0.4 |
| `shadowkv_upstream_nohk_v128` | 1 | complete | 00000000:D5:00.0 | 17.44 | 0.87 | 18.31 | 22.09 | 29.78 | 0.8 |
| `shadowkv_upstream_nohk_v128` | 2 | complete | 00000000:D5:00.0 | 17.42 | 0.82 | 18.25 | 22.05 | 29.94 | 0.9 |

## Checks, probes and failures

- probe_65536, `basis_koff_v128_reuse_opt`: complete; TTFT 11.03 s, decode-ready 11.03 s.
- probe_65536, `shadowkv_upstream_v128`: complete; TTFT 22.05 s, decode-ready 22.75 s.
- probe_65536, `shadowkv_upstream_nohk_v128`: complete; TTFT 17.14 s, decode-ready 18.05 s.
- probe_65536, `shadowkv_upstream_nohk_v128` fed the tokens of `shadowkv_upstream_v128`: argmax equal at 8 of 8 positions, max abs log-probability difference 0.0.

A probe is one short run (8 generated tokens) on cohort 0; probe timings are not results.
