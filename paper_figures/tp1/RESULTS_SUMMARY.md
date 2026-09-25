# TP1 Efficiency Figure Results

Generated: 2026-09-25T10:38:40.542199+00:00.

Environment: `basis`; Llama-3.1-8B-Instruct BF16; TP1/B1 for Figures 1/3, fixed active batch for Figure 2; one NVIDIA L40S.

## Verification and Run Status

All supplied historical main-figure values match archived raw results at published precision. The 24K Dense/Basis smoke passed. Figure 1 is a new same-version seven-context sweep with three fixed cohorts in independent processes per method/context. Figure 2 reuses the original same-source 64K grid; Figure 3 reuses archived complete requests. No quality benchmarks or LRQK reruns were launched.

New formal trials complete: **42/42**. Failures: 0. Fully aggregated context pairs: 7/7.

## Figure 1: GPU-Local Sparse Decode

| Prompt | Dense attention ms | Basis attention ms | Attention speedup | Dense model ms | Basis model ms | Model speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 16384 | 3.809 | 4.395 | 0.867x | 27.085 | 27.619 | 0.981x |
| 24576 | 5.267 | 5.167 | 1.019x | 28.633 | 28.331 | 1.011x |
| 32768 | 6.704 | 5.845 | 1.147x | 30.106 | 29.000 | 1.038x |
| 49152 | 9.611 | 7.315 | 1.314x | 33.021 | 30.432 | 1.085x |
| 65536 | 12.584 | 8.700 | 1.446x | 36.008 | 31.929 | 1.128x |
| 98304 | 18.334 | 11.769 | 1.558x | 41.777 | 34.979 | 1.194x |
| 130048 | 23.849 | 14.674 | 1.625x | 47.313 | 37.866 | 1.250x |

Exact new/archived latencies and percentage changes are in `fig1_archive_comparison.csv`. The largest absolute change among completed historical overlaps is 0.24%.

## Figure 2: 64K K-Offload Throughput

| Method | Batch | tok/s | Status / failure phase |
|---|---:|---:|---|
| dense_local | 1 | 27.657 | complete |
| dense_local | 2 | 41.115 | complete |
| dense_local | 4 | - | cache_allocation |
| dense_k_offload | 1 | 5.079 | complete |
| dense_k_offload | 2 | 5.418 | complete |
| dense_k_offload | 4 | 5.618 | complete |
| dense_k_offload | 8 | - | cache_allocation |
| basis_k_offload | 1 | 29.362 | complete |
| basis_k_offload | 2 | 46.099 | complete |
| basis_k_offload | 4 | 68.182 | complete |
| basis_k_offload | 8 | - | cache_allocation |

Largest successful tested batches: Dense-local 2, Dense K-offload 4, BasisKV K-offload 4. All plotted capacity failures are cache-allocation OOMs, not decode-kernel OOMs.

## Figure 3: Complete 128-Output-Token Requests

| Prompt | Dense s | BasisKV s | ShadowKV s | Shadow/Basis |
|---:|---:|---:|---:|---:|
| 32768 | 8.433 | 8.306 | 19.131 | 2.303x |
| 65536 | 16.709 | 16.399 | 29.508 | 1.799x |
| 130048 | 43.418 | 42.898 | 58.805 | 1.371x |

All three cohorts completed for every main request value. ShadowKV's longest-context tail decode is faster than BasisKV's; complete requests are slower. Separate build profiles are non-additive. LRQK runtime failures and compiler warnings remain appendix-only.

## Artifacts and Provenance

Figures and raw plotting CSV: `paper_figures/tp1/fig{1,2,3}_*.{pdf,png,csv}`. Captions: `captions.md`; point-level source fields: `provenance.json`; LRQK: `lrqk_appendix.md` / `lrqk_appendix_data.csv`. New raw trial JSON/logs: `results/system_benchmarks/tp1_paper_scaling/formal/`. Exact launch/reproduction commands: `README.md`, per-trial `command.json` and run manifests.

Historical source HEAD: `bab891d3e108d4280eeee65e7bb6c07f86dea83a`. The archived source bytes, not a dirty checkout or commit alone, define the runtime. Offline HF revision: `86c03092c663dc6654584143130b5e5abf2fedaa`; capacity HF revision: `0d0b7a569eb3adcfbf5d63d78a8beb767980fcad`. No SHA256 checks. Historical raw-data archive links and the publication file scope are in `README.md`.
