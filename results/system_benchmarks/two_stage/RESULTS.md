# Exact-K coarse screening plus B16R16 refinement

Conda environment: basis. Slurm on lovelace, one L40S, TP1, Llama-3.1-8B base, 64K context, batch 1, Dense V128, original Wo. Full reference uses the new 8-warp register-consumed B16R16 router, not the older slower implementation. Final support is the existing 62 historical Page32 pages (including sink page 0), masked at the historical boundary, plus exactly recent64; no budget expansion.

## Implementation and interpretation

Exact BF16 post-RoPE K supplies coordinatewise page minima/maxima. FP32 coarse dot products add log(valid historical token count). Four heads normalize independently over non-forced eligible pages, then merge by max in log space. Coarse top-k is 512 total per physical KV head, sorted by original page ID. Page0 and every page intersecting recent64 are forced into these 512. Recent-only pages have fine score -infinity. Candidate-only MMA maps page IDs to original positions; it does not gather full routing codes into a new sequence. Fine normalization is over candidates and is not equivalent to full normalization.

Metadata construction occurs before CPU K offload. A 64-token GPU ring delays historical-summary updates until each token exits recent64. The coarse scoring and normalization kernels are Triton; selection uses torch.topk and sort, and final selection reuses the existing CUDA selector. This is a first implementation, not a claim of fully optimized selection. No CPU historical K is read during routing.

Min/max page scoring is borrowed from [Quest](https://arxiv.org/html/2406.10774v2); the normalized GQA candidate generator and B16R16 cascade here are the tested adaptation, not an official Quest reproduction.

## Validation

Synthetic checks cover 30 incremental metadata checkpoints, page-boundary lengths 127/128/129 and 8K/64K, multiple batches/heads, forced recent pages, unique sorted candidates, and candidate/full score equivalence. 8K two-stage model smoke checks actual attention and finite logits. Real trace additionally asserts bitwise-equal candidate scores versus corresponding full-scan scores (before renormalization).

Completed diagnostic windows: [64, 65, 66]; total layer/query samples: 1056. Each 64K trace uses full-router generation, collecting step 0 and steps 10–19 across all 32 layers. All three methods receive identical queries, K/V and positions. Dense output uses FP32 QK/softmax/PV with TF32 disabled. These are local fixed-input diagnostics, not RULER accuracy.

## Candidate quality

| Window | Adaptive-page candidate recall mean / p1 / minimum | Lost reference head mass mean / p95 / maximum | Final two-stage page overlap |
|---|---|---|---|
| 64 | 98.442012% / 66.748635% / 32.786885% | 7.213938% / 21.406885% / 81.802690% | 97.649948% |
| 65 | 97.832681% / 78.333336% / 16.393442% | 8.849635% / 23.884481% / 74.668270% | 97.123577% |
| 66 | 96.227480% / 63.583332% / 27.868852% | 8.528385% / 26.698341% / 71.906310% | 95.653865% |
| all | 97.500724% / 68.333334% / 16.393442% | 8.197319% / 24.183443% / 81.802690% | 96.809130% |

Candidate recall excludes forced sink/recent-intersecting pages. Lost reference mass uses full B16R16 page probabilities over that same non-forced set. Final page overlap includes sink among 62 historical pages. A low mean does not rule out bad individual heads/queries.

## Attention output

| Window | Method | Dense-relative MSE (sum squared error / sum squared dense output) | Mean retained mass | Mean non-sink/non-recent retained mass | Mean output delta vs full router |
|---|---|---|---|---|---|
| 64 | full | 0.01913796 | 85.6743% | 67.3755% | 0.00000000 |
| 64 | coarse | 0.03429223 | 81.4173% | 57.6503% | 0.02354660 |
| 64 | two | 0.01912174 | 85.6367% | 67.2797% | 0.00088421 |
| 65 | full | 0.00785572 | 90.2411% | 70.0651% | 0.00000000 |
| 65 | coarse | 0.01633922 | 86.8052% | 58.7501% | 0.01060561 |
| 65 | two | 0.00786232 | 90.2145% | 69.9907% | 0.00063355 |
| 66 | full | 0.00835878 | 92.7642% | 72.4393% | 0.00000000 |
| 66 | coarse | 0.01985612 | 89.9787% | 61.4286% | 0.01072619 |
| 66 | two | 0.00825455 | 92.7218% | 72.3080% | 0.00021419 |
| all | full | 0.01162756 | 89.5598% | 69.9600% | 0.00000000 |
| all | coarse | 0.02320109 | 86.0671% | 59.2764% | 0.01495947 |
| all | two | 0.01159178 | 89.5243% | 69.8595% | 0.00057732 |

## Complete routing latency

CUDA-graph microbench of full routing calls, including output allocations within capture, coarse scoring, candidate normalization/top512/sort, fine projection/scan, final selection and original-ID mapping. Summary construction/update, attention, fetch and other model computation are excluded here; normal decode below includes incremental updates. Timings are sums of layer medians averaged over steps 10 and 19. Diagnostic trace wall time is not serving performance.

| Window | Full B16R16 ms/step | Coarse-only ms/step | Two-stage ms/step |
|---|---|---|---|
| 64 | 6.9025 | 0.7571 | 4.2100 |
| 65 | 6.9056 | 0.7569 | 4.2085 |
| 66 | 6.9051 | 0.7573 | 4.2080 |

## Normal continuous decode

One GPU allocation, order full/coarse/two/two/coarse/full, fresh processes, 100 generated tokens; steady CUDA median discards the first ten steps. No full-scan reference is evaluated in coarse/two serving. Prefill is excluded. Different algorithms can change generation and subsequent cache reuse; paired microbench above isolates routing cost on matched queries.

| Method | Mean of two steady CUDA medians ms/step | Native wall ms/step | Matching recorded decode-input positions vs full (first run) |
|---|---|---|---|
| full | 33.2523 | 36.1515 | 101/101 |
| coarse | 27.1775 | 33.1648 | 7/101 |
| two | 31.2932 | 36.4663 | 101/101 |

Native whole-loop wall measurements can include cold Triton compilation at newly encountered page counts; they are retained for transparency and are not used as steady speedup estimates. The coarse-only control still retains the common Base/Residual code cache and append computation, while skipping its scan; it is not a fully optimized standalone Quest runtime. The inherited `w4` directory label is the old harness slot-attention parameter; the full and candidate-only router kernels both use 8 warps.

Per-run steady decode medians:

- full: 33.236992 ms (finite logits=True, slot hit fraction=0.725358), 33.267694 ms (finite logits=True, slot hit fraction=0.725368)
- coarse: 27.173888 ms (finite logits=True, slot hit fraction=0.783883), 27.181055 ms (finite logits=True, slot hit fraction=0.783891)
- two: 31.261696 ms (finite logits=True, slot hit fraction=0.726813), 31.324672 ms (finite logits=True, slot hit fraction=0.726793)

Metadata allocation (min/max plus recent ring): 260.500 MiB. Nominal 64K min/max alone is 256 MiB; capacity slack and ring account for the remainder. Prefill-summary event totals include cold compilation if present and are retained in diagnostic JSON, not interpreted as steady kernel cost.

## Commands

```bash
python -m benchmarks.system.validate_two_stage
python -m benchmarks.system.run_two_stage --phase trace
python -m benchmarks.system.run_two_stage --phase decode
python -m benchmarks.system.summarize_two_stage
```

Implementation: benchmarks/system/two_stage_router.py; real benchmark: bench_two_stage.py; runner: run_two_stage.py. The reusable native benchmark now accepts a diagnostic window index. Production router/cache defaults remain unchanged. No GitHub commit or push.
