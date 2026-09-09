# Experiment Summary: Query-Gram Page Overlap to C1-V96 LongBench

Snapshot: 2026-09-07 UTC (2026-09-06 US Eastern). This document consolidates existing completed experiments and separately records the in-progress V96 sparse pipeline. No new benchmark was launched to prepare this summary.

## 1. Scope and terminology

All experiments below use **Qwen3-8B-Base**, with 36 layers, 32 query heads, 8 physical KV heads, four query heads per GQA group, and original K/V head dimension 128. GPU experiments use the `basis` environment and NVIDIA L40S GPUs; some selection and aggregation stages run on CPUs. Model inference and deployed factors use BF16 unless explicitly stated otherwise.

| Term | Meaning in these experiments |
|---|---|
| C1-V80 / C1-V96 | Uniform C1 Value latent width 80 / 96 per physical KV head; C1 also supplies the output decoder. |
| Full exact K | All uncompressed cached keys participate in attention. Values may still be C1-compressed. This does **not** mean original dense K/V inference. |
| Base16 | Affine, rank-16, closed-form MSE reduced-rank regression from resident C1 Value coordinates to pre-RoPE K. |
| Residual R8 | Eight stored residual coordinates per token/KV group, fitted relative to the Base with a causal non-sink Page-Fisher objective. |
| Q32 / Q8 | Number of calibration query positions per window used for residual fitting, not a query-compression rank or retrieval budget. |
| Page32 / B2048 | 32 tokens per page, at most 64 selected pages including pinned page 0; 63 non-pinned slots when enough pages are valid. |
| C1 prefill | C1 Value and output projections participate throughout the prompt forward; the first generated token is its argmax. |
| Original dense prefill | Original Qwen3 attention modules, dense V128 and original output projections process the prompt; C1 is introduced afterward. |

Sparse experiments discussed here use all 36 layers, with no adaptive budget and no forced-current-page rule. Exact K is GPU-resident in the evaluation implementation. The conditional routing reference path materializes a post-RoPE Base128 plus residual-R8 sidecar, width 136. These are quality experiments, **not measurements of CPU offload traffic, compact production storage, or optimized decode speed**.

Exact keys are uncompressed keys generated on the relevant model trajectory. C1-prefill and dense-prefill trajectories do not necessarily produce identical hidden states or keys in later layers.

## 2. Query-Gram sampling and residual fitting

The original Query-Gram experiment kept C1-V80 and the existing closed-form MSE-RRR Base16 frozen, and refitted only uniform R8 residual factors.

| Component | Setting |
|---|---|
| Router fit set | Existing C4 windows 0–63: 64 × 32,768 tokens |
| Router diagnostic set | Windows 64–79: 16 × 32,768 tokens |
| Window construction | Existing packed C4 windows, not newly collected native contiguous 32K documents |
| Candidate queries | 512 positions/window: 63, 127, …, 32,767, at stride 64 |
| Position selection | Fit-only per-head uncentered whitening, then deterministic Query-Gram pivots |
| Stratification | Four 8K bins; eight positions selected per bin |
| Selected positions | 32 per layer, shared across windows within that layer |
| Selection precision | FP64 CPU; whitening epsilon 1e-6 |
| Residual objective | Original causal non-sink Page-Fisher; page 0 excluded |
| Residual solver | 40 BCD sweeps; PCG damping/tolerance 1e-5, maximum 100 iterations |
| Endpoint selection | Fixed final factors; diagnostics do not select the endpoint |

Each head has 64 × 32 = 2,048 fit query/window observations and 16 × 32 = 512 diagnostic observations. Diagnostic queries do not participate in whitening or position selection. No Adam, model-weight training, or benchmark-answer fitting is used.

The sampler changes **which Q positions are used to fit residual routing**. It does not introduce a new online query sampler, change the Base objective, or increase the online retrieval budget. Fitting losses evaluated at different sampler-specific positions are not directly interchangeable; the page audit below uses common evaluation queries.

Sources: [full fitting protocol](qgram32_full_protocol.md), [frozen Query-Gram position manifest](../results/evaluation/qgram32/positions.json).

## 3. Common-query exact/proxy page overlap

### 3.1 Measurement definition

Both old terminal-Q32 and new stratified Query-Gram Q32 routers were evaluated on **the same dense-teacher C4 captures**:

- All 36 layers and all 8 GQA groups.
- 16 diagnostic windows, indices 64–79.
- 32 common terminal-8K Q positions: 24,831, 25,087, …, 32,767.
- Page32/B2048, pinned page 0, identical GQA-max selection policy.

This gives 36 × 8 × 16 × 32 = **147,456 conditions per router**. Layers 0 and 1 also use the sparse rule in this audit.

The exact reference computes FP32 QK from captured BF16 Q/K. The proxy uses the native BF16 routing arithmetic. Each head's non-sink page mass is normalized before taking the maximum over the four GQA query heads. This group routing score is not the head-average teacher mass.

For exact-selected and proxy-selected page sets, each of size 64:

\[
\mathrm{overlap}=\frac{|S_{\mathrm{exact}}\cap S_{\mathrm{proxy}}|}{64},
\qquad
\mathrm{nonpinned\ overlap}=\frac{|S_{\mathrm{exact}}\cap S_{\mathrm{proxy}}|-1}{63}.
\]

Overlap is **not IoU and not attention-mass coverage**. Means equally weight queries, windows, groups, and layers.

### 3.2 Aggregate results

| Residual-fitting query policy | Mean shared pages / 64 | Page overlap | Excluding pinned page 0 |
|---|---:|---:|---:|
| Old terminal-Q32 | 52.3170 | 81.7453% | 81.4556% |
| Stratified Query-Gram Q32 | 51.2775 | 80.1211% | 79.8056% |

Query-Gram Q32 is **1.6242 percentage points lower** on this common terminal-query overlap audit. Every layer's mean is lower in this comparison; the full 36-layer table is included in Appendix A.

### 3.3 Saved ranking detail

For every layer/group/window/query, complete valid-page tables retain:

- Exact and proxy scores, selected IDs and masks.
- Intersection, missed pages, and extra pages.
- Inclusive `rank_min`/`rank_max` intervals for ties.
- Score cutoffs, score-minus-cutoff margins, and the owning GQA head.
- Per-head and head-average teacher mass, with and without sink normalization.

Pinned page 0 has rank 0; the routed cutoff is rank 63. Actual selected masks determine membership when scores tie. Each page can be traced to token positions `[32p, 32p+31]`.

The human-readable missed-page CSV contains the 20 highest head-average teacher-mass misses per layer for the designated new router. It is a diagnostic subset, not the full miss distribution. Complete page tables are retained separately. Historical exact and old-Q32 tables were checked bitwise against the prior audit.

Sources: [page-overlap results](../results/evaluation/qgram_pages/summary.md), [all-layer CSV](../results/evaluation/qgram_pages/layer_overlap.csv), [ranking protocol and artifact definitions](qgram_page_rankings_protocol.md).

## 4. RULER results and the matched terminal-Q8 comparison

### 4.1 RULER protocol

The repeated RULER 32K pilot contains **11 tasks × 8 prompts = 88 prompts**, not the complete 13-task suite. All arms use the same frozen C1-V80 payload and a shared **full-attention C1 Triton prefill**, including the first generated token. Subsequent full-K and sparse decoding use independent immutable-prefix forks.

This is important: these RULER results were **not obtained with original dense prefill**. The full-K C1 reference isolates the added routing effect relative to that shared C1 trajectory; it is not a dense-V128 baseline.

Scoring is the existing case-insensitive substring metric: fraction of reference answers recovered for `match_type=all`, or any-reference hit for `match_type=part`. Consequently, an eight-prompt task can have scores such as 93.75%; it is not necessarily binary exact match per prompt. The headline mean equally weights the 11 tasks.

### 4.2 Terminal-Q32 versus Query-Gram Q32

| Task | Full exact K + C1-V80 | Terminal-Q32 R8 | Query-Gram Q32 R8 |
|---|---:|---:|---:|
| niah_single_1 | 100.0000% | 100.0000% | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 100.0000% |
| niah_single_3 | 100.0000% | 100.0000% | 100.0000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 87.5000% |
| niah_multikey_2 | 87.5000% | 50.0000% | 75.0000% |
| niah_multiquery | 96.8750% | 96.8750% | 93.7500% |
| niah_multivalue | 93.7500% | 96.8750% | 96.8750% |
| vt | 92.5000% | 92.5000% | 92.5000% |
| fwe | 91.6667% | 83.3333% | 79.1667% |
| qa_1 | 50.0000% | 50.0000% | 50.0000% |
| qa_2 | 37.5000% | 37.5000% | 37.5000% |
| Task-balanced mean | **85.2083%** | **81.3258%** | **82.9356%** |

Query-Gram Q32 increases the pilot mean by **1.6098 pp** versus terminal-Q32, despite its lower common-query page overlap. Its gap to full-K C1 is 2.2727 pp. In `niah_multikey_2`, saved sample indices 33 and 37 change from incorrect to correct; sample 38 remains incorrect. These are three identified prompts, not an exhaustive explanation of all score changes.

The overlap audit and RULER evaluation measure different objects on different data; their numerical directions are reported separately. Both benchmark sets have been reused during development, and eight prompts per task provide limited evidence for small differences.

Sources: [terminal-Q32 result](../results/evaluation/mse_base_q32_ruler32k/result.json), [Query-Gram RULER summary](../results/evaluation/qgram32_ruler32k/summary.md).

### 4.3 Matched terminal-Q8 test

Both Q8 arms freeze the same C1-V80 and Base16 and **newly fit R8 with eight terminal-8K Q positions**. They are not obtained by truncating an already-fitted Q32 checkpoint.

Uniform positions are 25,599, 26,623, 27,647, 28,671, 29,695, 30,719, 31,743, and 32,767. Query-Gram Q8 reuses each layer's eight pivots in the final bin of the original Q32 manifest. Its whitening still comes from full-window candidate queries; this is not terminal-only whitening.

| Policy | Common-Q page overlap | Non-pinned overlap | RULER mean | multikey_2 | fwe |
|---|---:|---:|---:|---:|---:|
| Terminal uniform Q8 | 78.0621% | 77.7139% | 77.4053% | 37.5000% | 70.8333% |
| Inherited terminal Query-Gram Q8 | 76.7247% | 76.3552% | 80.8902% | 50.0000% | 79.1667% |

The Q8 overlap audit still uses the same 32 common terminal evaluation queries, independently of the eight fitting queries. Query-Gram Q8 raises RULER mean by 3.4848 pp while lowering page overlap by 1.3374 pp versus uniform Q8.

Sources: [Q8 comparison protocol](terminal_q8_comparison_protocol.md), [Q8 overlap](../results/evaluation/terminal8_pages/summary.md), [uniform-Q8 RULER](../results/evaluation/terminal8_uniform_ruler32k/summary.md), [Query-Gram-Q8 RULER](../results/evaluation/terminal8_qgram_ruler32k/summary.md).

## 5. LongBench dataset and initial C1-V80 comparison

### 5.1 Fixed inputs and scoring

All LongBench results below reuse exactly the same **192 saved prompts: six tasks × 32 prompts**. Tasks are `qasper`, `multifieldqa_en`, `hotpotqa`, `2wikimqa`, `gov_report`, and `qmsum`.

- Official base completion prompts; no chat template.
- Identical saved input token IDs and reference answers.
- Greedy generation; unchanged tokenizer/model EOS.
- Per-task output caps: 128 / 64 / 32 / 32 / 512 / 512 tokens.
- Input plus reserved-output cap: 32,768 tokens.
- Actual prompt lengths: 1,192–30,431; mean 9,244.5885; no prompts truncated.
- QA uses official F1; summarization uses official ROUGE-L; best alternative reference.
- Scores are on a 0–100 scale; the headline mean is the arithmetic mean of six tasks.

This is not full LongBench or LongBench-E, and the task scores should not all be called accuracy. Four independent L40S workers process sample shards; this is not tensor-parallel execution.

### 5.2 Initial C1-prefill four-arm result

The four C1-V80 arms share full causal **C1-V80 Triton prefill** and its first token. Only decode differs. Full-K decode uses SDPA; sparse arms use the native BF16 selected-page attention path.

| Task | Original dense K/V | C1 full K | C1 exact-QK sparse | C1 Query-Gram Q32 | C1 terminal-Q32 |
|---|---:|---:|---:|---:|---:|
| qasper | 39.3018 | 19.6094 | 18.8967 | 19.5324 | 19.7245 |
| multifieldqa_en | 52.8498 | 30.7387 | 29.1173 | 29.8343 | 29.3452 |
| hotpotqa | 60.8872 | 29.7403 | 32.8628 | 36.3228 | 30.6941 |
| 2wikimqa | 50.1190 | 31.2642 | 37.0228 | 31.5359 | 33.3671 |
| gov_report | 29.1896 | 27.2987 | 30.0715 | 29.4190 | 27.8504 |
| qmsum | 26.1460 | 26.3270 | 26.8727 | 26.1977 | 25.4020 |
| Mean | **43.0822** | **27.4964** | **29.1406** | **28.8070** | **27.7305** |

The exact-QK sparse arm uses full exact QK for page selection under the same Page32/B2048 rule, then attends only to selected pages. It is not full attention and is not a mathematical upper bound on downstream task score. All sparse arms retain the C1 payload.

Query-Gram minus terminal-Q32 is +1.0765 score points, with 50 improved, 31 regressed, and 111 tied prompt scores. Query-Gram minus exact-QK sparse is −0.3336 points.

Sources: [four-arm result](../results/evaluation/longbench_c1_32k/summary.md), [dense result](../results/evaluation/longbench_dense_32k/summary.md).

## 6. Matched-calibration PaLU controls

Two V-only PaLU Fisher checkpoints were evaluated on the same LongBench inputs. Both use exact full K, with their approximate V participating during prefill and decode.

| Checkpoint | Realized average per-KV-head-equivalent rank | Six-task mean |
|---|---:|---:|
| PaLU M | 81.7778 | 25.2234 |
| PaLU G-LRD4 | 80.0000 | 24.5116 |

Both whitening/factor fitting and newly measured Fisher importance use exactly the C1 **32 × 32K C4 fit token IDs**. M factorizes eight KV heads independently; G4 uses two groups of four adjacent KV heads with nominal group rank 320 and Fisher-allocated layer/group ranks. The two realized budgets are not exactly equal.

The evaluation executes a BF16 latent writer plus Value reconstruction, then a V128 cache with SDPA. It is not a compact-cache or speed measurement. PaLU/dense prefill uses SDPA, while the original C1 four-arm prefill uses Triton; these comparisons do not isolate only the factor-fitting method.

All 384 new predictions passed scoring audits. M produced immediate EOS/empty output on 5/192 prompts; G4 on 52/192. These are actual outcomes under the fixed EOS policy, not missing records.

Source: [matched-calibration PaLU results](../results/evaluation/longbench_palu_32k/summary.md), [protocol and per-task table](longbench_palu_protocol.md).

## 7. Original dense prefill, C1 decode, and two-sided KL allocation

### 7.1 What changed at the prefill boundary

For the original-dense-prefill arms, original Qwen3 attention modules process every complete prompt with K128/V128 and the original output projection. The first generated token is the dense argmax. Each cached V is then projected into its C1 coordinates; exact K tensors are preserved across this conversion. C1 participates in computing the second and subsequent generated tokens.

Original modules are restored before each new prompt, so this is not merely dense prefill for the first sample in a worker. All 192 first tokens matched the saved dense baseline in these arms.

### 7.2 Results under original dense prefill

| Task | Dense decode | Uniform V80 full K | KL avg80 full K | KL avg80 + Base16/R8 sparse |
|---|---:|---:|---:|---:|
| qasper | 39.3018 | 34.9041 | 35.2802 | 34.9063 |
| multifieldqa_en | 52.8498 | 47.9694 | 48.6936 | 49.2924 |
| hotpotqa | 60.8872 | 59.3363 | 62.1941 | 62.6062 |
| 2wikimqa | 50.1190 | 47.2545 | 46.9940 | 46.9940 |
| gov_report | 29.1896 | 29.3136 | 28.6770 | 28.8272 |
| qmsum | 26.1460 | 27.3302 | 26.9735 | 26.2185 |
| Mean | **43.0822** | **41.0180** | **41.4687** | **41.4741** |

For uniform V80, changing the prefill path changes mean score from 27.4964 to 41.0180, a +13.5216-point difference. The C1 checkpoint, full-K decode support, and SDPA decode backend remain matched. The prefill change also changes the first token, hidden-state trajectory, and subsequent generated keys; it is not a cache-only intervention.

### 7.3 Allocation and matched-router details

The existing two-sided factorized-KL allocation uses alpha = 1, anchor rank 64, actual probes at 32 and 96, and bank ranks 32/48/64/80/96/112/128. The final average is exactly 80, with rank counts 3/3/11/6/5/3/5 layers.

Payload factors use C4 32 × 32K fit and 4 × 32K diagnostic local-MSE data, with six ALS sweeps. Allocation profiling uses a separate 32 × 32K C4 set and 12 × 32K confirmation set, with 1,024 sampled terminal positions per window. No LongBench data enters allocation.

The exported allocation also canonicalizes encoder coordinates and closes/refits decoders for non-anchor, non-full-rank candidates. Its six rank80 layers are not bitwise identical to uniform V80 factors. This is a comparison of exported checkpoints, **not a strict rank-index-only ablation**.

A new Base16/R8 bank was fitted for the allocated coordinates; the old uniform-V80 router was not attached unchanged. The router uses the same 64/16 C4 windows, Query-Gram Q32, and non-sink Page-Fisher settings. Sparse decode remains Page32/B2048 on all 36 layers.

| Paired comparison | Mean change | Improved / regressed / tied prompts |
|---|---:|---:|
| KL full K versus uniform V80 full K | +0.4507 points | 46 / 37 / 109 |
| KL sparse versus KL full K | +0.005358 points | 34 / 36 / 122 |

The near-equal KL sparse/full means do not imply identical selected pages, outputs, or generation sequences.

Sources: [dense-prefill V80](../results/evaluation/longbench_c1_denseprefill_32k/summary.md), [KL full K](../results/evaluation/longbench_c1_kl_denseprefill_32k/summary.md), [KL matched sparse router](longbench_c1_kl_base_residual_protocol.md).

## 8. C1-prefill kernel numerical diagnosis

Eight fixed LongBench prompts were checked across all 36 layers: **288 layer/input pairs**, with lengths 1,192–30,431 including non-aligned lengths. The local comparison used exactly the same real Q/K/C1-V80 tensors for:

1. Existing Triton compressed-Value prefill.
2. Forced Flash-SDPA with zero-padded Value features and unchanged Q/K scale and causal support.
3. Explicit FP32 QK/softmax/Value accumulation at selected causal query rows, with TF32 disabled.

Zero padding in the reference changes neither the C1 factors nor payload capacity. C1 output-projection error was compared using the same decoder in FP32.

| Maximum relative L2 over layer/input pairs | Triton versus FP32 | Flash versus FP32 |
|---|---:|---:|
| Sampled attention latent | 0.1954% | 0.1851% |
| Sampled C1-decoded output | 0.2532% | 0.2507% |

The maximum full-tensor Triton-versus-Flash relative L2 was 0.1841%. Across 6,624 sampled layer/query positions, the maximum row error for Triton versus FP32 was 0.3004%; each row metric aggregates query heads. All recorded comparisons were finite.

Replacing all 36 C1 prefill kernels with Flash changed **0/8 first tokens**. All three C1-versus-dense first-token disagreements remained. Mean final-position full-vocabulary KL was:

| Comparison | Mean KL |
|---|---:|
| Dense to Triton-C1 | 0.502051 |
| Dense to Flash-C1 | 0.496512 |
| Triton-C1 to Flash-C1 | 0.003239 |

No large Triton-specific numerical anomaly was identified on the checked inputs or sampled boundaries. This is limited numerical evidence, not proof that all possible inputs are safe or complete generations are identical. No full Flash-reference generation benchmark was run, and no production kernel was changed.

Source: [numerical diagnosis](../results/evaluation/c1_prefill_kernel/summary.md).

## 9. Latest completed result: uniform C1-V96 prefill and full-K decode

### 9.1 Matched payload calibration

The V80 and V96 checkpoints use the same calibration capture manifest, **32 × 32K C4 fit plus 4 × 32K diagnostic windows**, full-layer attention-output MSE with cross-head covariance, activation-weighted SVD initialization, and six encoder ALS sweeps followed by decoder refitting. Non-rank fitting settings match.

| Local diagnostic | V80 | V96 |
|---|---:|---:|
| Mean held-out relative attention-output MSE | 0.0944713 | 0.0564016 |
| Layer33 relative attention-output MSE | 0.2272885 | 0.1294652 |

The mean local MSE decreases by approximately 40.3%. These are local factor diagnostics, not task scores or a terminal-distribution guarantee.

### 9.2 Full-K LongBench result

V96 uses **C1-V96 Triton prefill followed by full exact-K/C1-V96 SDPA decode**, all 36 layers, without routing. The first token comes from C1-V96, not original dense prefill.

| Task | C1-V80 prefill/decode | C1-V96 prefill/decode | Original dense |
|---|---:|---:|---:|
| qasper | 19.6094 | 32.4074 | 39.3018 |
| multifieldqa_en | 30.7387 | 42.8418 | 52.8498 |
| hotpotqa | 29.7403 | 57.9676 | 60.8872 |
| 2wikimqa | 31.2642 | 40.2530 | 50.1190 |
| gov_report | 27.2987 | 30.5532 | 29.1896 |
| qmsum | 26.3270 | 27.6797 | 26.1460 |
| Mean | **27.4964** | **38.6171** | **43.0822** |

V96 minus V80: **+11.1207 points**. V96 minus dense: **−4.4651 points**. Paired against V80: 91 improved, 44 regressed, 57 tied prompt scores. Paired against dense: 59 improved, 65 regressed, 68 tied scores.

V96 first-token agreement with dense is 156/192. Output-cap exits without EOS are 39/192, compared with 100/192 for C1-prefill V80 and 16/192 for dense. These cap exits are completed generations under the fixed policy, not missing samples.

All 192 V96 outputs passed token/text, EOS/cap, official-score, provenance, and shard-coverage checks. Smoke took 32 seconds. Four evaluation workers took 8:36, 7:18, 9:12, and 11:23; CPU summary took 18 seconds. Peak allocated GPU memory was 21.613 GiB. All jobs exited successfully without retries, non-finite logits, or failed assertions. Optional FuzzyWuzzy acceleration warnings do not affect the selected F1/ROUGE-L metrics.

This comparison increases capacity in **both prefill and decode**; it does not isolate the prefill contribution. No original-dense-prefill V96 arm was run.

Sources: [V96 result and paired scores](../results/evaluation/longbench_c1_v96/summary.md), [V96 protocol](longbench_c1_v96_protocol.md), [V96 factor diagnostics](../results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6/summary.md).

## 10. In progress: matched V96 Base16/R8 sparse decode

This pipeline is separate from the completed V96 result above. It freezes C1-V96 and newly fits Base16/R8 in its 96-dimensional coordinates. The old V80 Base-left factor has shape `(8,80,16)` and is not directly reused for V96; the new shape is `(8,96,16)`. Residual is refitted relative to the new Base.

The fitting and deployment settings remain 64 × 32K fit, 16 × 32K diagnostic, closed-form MSE Base16, Query-Gram Q32 Page-Fisher R8, 40 BCD sweeps, Page32/B2048, pinned page 0, all layers. The evaluation preserves C1-V96 prefill and checks its first token against the saved full-K V96 baseline. Only subsequent decode becomes sparse.

Status checked at **2026-09-07 02:11 UTC**:

| Stage | Job | Status |
|---|---|---|
| Full-protocol layer0/35 fit checks | 8301037_0–1 | Completed successfully |
| All-layer fit | 8301039_0–3 | Running on four L40S; 10 layer records present at the snapshot |
| Sparse decode smoke | 8301040 | Dependency-pending |
| LongBench evaluation | 8301041_0–3 | Dependency-pending |
| CPU summary and audit | 8301042 | Dependency-pending |

The layer0 fit took 88.2 seconds, with residual Page-Fisher fit/diagnostic NMSE 0.292923/0.557809. Layer35 took 101.6 seconds, with 0.220877/0.327183. These are fitting diagnostics, **not sparse LongBench scores**.

Some PCG query solves reached the existing 100-iteration cap: final maximum relative residuals were 0.001165 and 0.005217, above the requested 1e-5 tolerance. Factors and losses are finite, but numerical convergence is not claimed. The original solver budget is retained for the comparison.

**No completed V96 sparse LongBench score is available at this snapshot.** In particular, 38.6171 is the full-K V96 result, not a sparse score. The existing pipeline continues independently of this summary.

Source: [V96 matched-router protocol and jobs](longbench_c1_v96_router_protocol.md).

## Appendix A. All 36 layers: terminal-Q32 versus Query-Gram-Q32 page overlap

Values include pinned page 0 and use the common terminal-Q32 diagnostic queries described in Section 3. These percentages are page-set overlap, not teacher-mass recall or task accuracy.

| Layer | Terminal-Q32 | Query-Gram Q32 |
|---|---:|---:|
| 0 | 86.4819% | 84.7504% |
| 1 | 85.4237% | 82.7591% |
| 2 | 81.2420% | 79.6558% |
| 3 | 77.1835% | 73.7259% |
| 4 | 84.4742% | 82.4848% |
| 5 | 84.6043% | 81.0974% |
| 6 | 81.2626% | 79.3518% |
| 7 | 72.1046% | 69.6537% |
| 8 | 84.3639% | 83.0391% |
| 9 | 75.2068% | 73.5325% |
| 10 | 88.4651% | 87.5710% |
| 11 | 83.5949% | 82.5153% |
| 12 | 82.9590% | 81.7768% |
| 13 | 70.6638% | 68.9777% |
| 14 | 85.9169% | 84.5249% |
| 15 | 79.1462% | 77.5928% |
| 16 | 79.9736% | 78.3325% |
| 17 | 83.2737% | 82.0686% |
| 18 | 82.9498% | 81.6933% |
| 19 | 82.9884% | 81.4285% |
| 20 | 81.5186% | 80.3825% |
| 21 | 82.3448% | 81.4716% |
| 22 | 82.8568% | 81.0966% |
| 23 | 84.7992% | 83.4091% |
| 24 | 80.4523% | 79.3343% |
| 25 | 84.7523% | 83.8234% |
| 26 | 82.1526% | 81.1939% |
| 27 | 82.4348% | 81.3660% |
| 28 | 84.8701% | 83.6960% |
| 29 | 77.6123% | 75.8572% |
| 30 | 81.2359% | 79.5341% |
| 31 | 80.6976% | 78.6030% |
| 32 | 80.3524% | 78.3188% |
| 33 | 77.7615% | 76.2817% |
| 34 | 81.9633% | 80.4157% |
| 35 | 84.7481% | 83.0452% |
| Mean | **81.7453%** | **80.1211%** |

The matched terminal-Q8 all-layer table is retained in the [Q8 page-audit summary](../results/evaluation/terminal8_pages/summary.md) and [CSV](../results/evaluation/terminal8_pages/layer_overlap.csv).

## Appendix B. Result provenance

This summary uses final result JSONs and generated result summaries where available, rather than treating older submission-time protocol text as completed results. Protocol documents contain the detailed execution commands; this summary does not reproduce shell commands.

Selected result fingerprints checked when preparing this summary:

| Result | SHA256 of result.json |
|---|---|
| Original C1-V80 four-arm LongBench | `8aad51f62e5bd33e120ee2ef02efd18c290e63a544e6edf70d87a9ec68fa566b` |
| Dense-prefill uniform V80 | `2f31de5a2b0184e876d44374088ae91494209a0bf8267158ac0ffceb6e925c0d` |
| Dense-prefill KL avg80 full K | `58492193ff089862c625a4ad128f7aebb67a5aef4f5f00550b8c3529fdcdc58a` |
| Dense-prefill KL avg80 Base16/R8 sparse | `96ee7f31315e7159eecb440ccf9fa9941f70e5ac665752fa908e955499cdac42` |
| C1-V80 prefill kernel diagnosis | `25cd6a21356b355d6379f0335886dddf6ed3b1001dffa546d638c05f013d348f` |
| C1-V96 prefill/full-K decode | `1efcd87d4d887e4f47080a301089af59c14a328c6b827d52dd981c2adab6b01e` |

The page diagnostics, RULER pilot, and LongBench pilot are separate measurements. No significance test, unseen-benchmark generalization claim, theoretical task-score ceiling, or measured PCIe speedup is implied by their juxtaposition.
