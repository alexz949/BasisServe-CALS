# C1-V80 Key Routing: From MSE-RRR Base to Q128 and Page-Boundary Repair

## 1. Scope and current status

This report consolidates the completed closed-form Base experiments, residual query-sampling sweeps, the statistics-construction optimization, and the subsequent direct page-ranking diagnostics. It describes Qwen3-8B-Base, not Qwen3-32B or a different model variant.

Two different kinds of results must be kept separate:

| Experiment family | Data and coverage | Reported metric |
| --- | --- | --- |
| Frozen MSE-RRR Base16 + Page-Fisher residual Q1/Q16/Q32/Q64/Q128 | 64 fit + 16 diagnostic C4 windows, 32K each; all 36 layers | RULER32K task-balanced accuracy on 88 reused prompts |
| Direct page-boundary repair | 2 fit + 2 diagnostic windows, four sampled query positions/window; initially one group, then all groups of layers 15/33 | Teacher attention-mass coverage, not RULER accuracy |

The terminal-Q32 Fisher configuration achieved the highest observed RULER mean within the closed-form Base query sweep: **81.3258%**. Terminal Q64 and Q128 achieved **80.5303%** and **79.6212%**. The direct-ranking implementation is functional, but its small diagnostic results are mixed and no repaired checkpoint has been evaluated on RULER.

## 2. Fixed architecture and what is approximated

The model has 36 layers, eight KV/GQA groups, 32 query heads, and a 128-dimensional Key per group. Four query heads share each KV group. The frozen payload is C1-V80.

For one layer/group, use row-vector notation:

\[
c_t=v_tF,\qquad
\widehat k_t^{\mathrm{pre}}=c_tAB+b,
\qquad
\bar k_t=\operatorname{RoPE}_t(\widehat k_t^{\mathrm{pre}}).
\]

Dimensions are raw V128, C1 code C80, Base factors A80×16 and B16×128, and a 128-dimensional predicted Key. Base16 is a rank constraint on the predictive map; it does not mean that Q is compressed to 16 dimensions.

The post-RoPE residual and residual code are

\[
r_t=k_t^{\mathrm{post}}-\bar k_t,\qquad z_t=r_tE_g,
\quad E_g\in\mathbb R^{128\times8}.
\]

Each query head has its own U128×8. The routing score is

\[
\hat s_{h,t}=\frac{q_h\bar k_t^\top+(q_hU_h)(r_tE_g)^\top}{\sqrt{128}}.
\]

Base uses resident Value information. Residual uses the exact Key when creating the token's sidecar and retains information that Base cannot recover from Value. The fitted E/U are fixed at inference; there is no online refitting.

These approximate scores select pages. Attention on the selected support subsequently uses **exact K and resident C1-V80**, not the proxy scores as final attention logits. The experiments keep exact K on GPU and materialize a Base128+R8 sidecar. They are accuracy oracles, not measured CPU offload or PCIe speedups.

## 3. Closed-form MSE-RRR Base

The Base map solves an affine, rank-constrained pre-RoPE Key regression:

\[
\min_{\operatorname{rank}(W)\le16,b}
\sum_{t\in\mathcal T_{\mathrm{fit}}}
\|k_t^{\mathrm{pre}}-c_tW-b\|_2^2.
\]

The implementation centers the captured input/target statistics, whitens the input covariance using its numerical pseudoinverse, takes a truncated SVD, and restores the affine bias. This is unweighted Key-space MSE-RRR, not per-query QK fitting, Page-Fisher fitting, exact softmax KL, or page-recall optimization.

The existing Base tensors were reused from `results/checkpoints/q8_residual_kl_bank`, whose original Base artifacts reside in `results/checkpoints/qwen3_8b_v80_32k_base16_p32`. Base and C1 payload remained frozen throughout the residual query sweeps.

### Historical Q-aware Base branch

Before returning to closed-form Base, another branch fitted Base with sampled, causal, RoPE-aware raw-QK loss using Adam on captured activations. It is a different Base, and does not satisfy the project's requested definition excluding Adam from training-free fitting.

| Historical configuration | RULER mean |
| --- | ---: |
| Adam Q-aware Base16 + terminal-Q1 Fisher R8 | 69.5076% |
| Adam Q-aware Base16 + terminal-Q16 Fisher R8 | 80.4356% |
| Closed-form MSE-RRR Base16 + terminal-Q1 Fisher R8 | 75.3598% |
| Closed-form MSE-RRR Base16 + terminal-Q16 Fisher R8 | 80.5682% |

These numbers are RULER scores, not Key-reconstruction accuracy. In particular, 69.51% and 80.44% refer to the historical Adam-Base branch with different residual query coverage. Later Q32/Q64/Q128 experiments used the closed-form Base instead.

## 4. Page-Fisher residual objective and fitting

Residual fitting uses 64 fit windows and 16 diagnostic windows from C4. Each window has 32768 tokens, giving 2,097,152 fit tokens and 524,288 diagnostic tokens. These windows pack eight 4096-token source windows without separators; they are not native contiguous 32K documents.

For every sampled query and head, fitting uses the entire causal Key prefix outside the pinned first page. It does not subsample the corresponding K prefix down to Q tokens. Increasing Q means adding sampled query positions, with all 32 query heads retained at each position.

Exact teacher logits define a softmax over causal non-sink tokens. For each page p, compute its probability mass m_p and teacher-weighted residual representative rho_p. The compact statistic is

\[
\mu=\sum_pm_p\rho_p,\qquad
G=\sum_pm_p(\rho_p-\mu)^\top(\rho_p-\mu).
\]

With \(v_h=q_h(U_hE_g^\top-I)\), the residual objective is

\[
\mathcal L_{\mathrm{Fisher}}=
\frac{1}{2\cdot128}\sum_{\mathrm{window,query},h}v_hG_hv_h^\top.
\]

The centering includes softmax normalization. Nevertheless, this is a local Page-Fisher approximation: it linearizes page log-sum-exp around teacher scores. It does not directly optimize hard selected-page recall, exact finite-error KL, C1-decoded output, or task accuracy.

The solver uses eigenvector initialization, 40 BCD sweeps, a final query-factor solve, and PCG within block updates. Damping and tolerance are 1e-5; maximum PCG iterations are 100. Residual rank is uniformly eight. Each query retains its own Gram and causal prefix; queries are not averaged. The final sweep is retained, and diagnostic loss does not select residual checkpoints. No Adam or full-model backward pass is used in this residual stage.

## 5. Query-sampling configurations

Positions below are zero-based. The notation start / stride / end includes the endpoint.

| Residual setting | Q positions | Fit observations/head | Diagnostic observations/head |
| --- | --- | ---: | ---: |
| Q1 | 32767 | 64 | 16 |
| Terminal Q16 | 25087 / 512 / 32767 | 1024 | 256 |
| Terminal Q32 | 24831 / 256 / 32767 | 2048 | 512 |
| Full-window Q32 | 1023 / 1024 / 32767 | 2048 | 512 |
| Terminal Q64 | 24703 / 128 / 32767 | 4096 | 1024 |
| Terminal Q128 | 24639 / 64 / 32767 | 8192 | 2048 |

The terminal configurations use the final 8192-token interval and are nested. Q64 and Q128 contain all terminal-Q32 positions. Q1 is only the final position: the Q1-to-Q16 change adds both query count and positional coverage. Full-window Q32 changes the position/prefix-length distribution while keeping Q count fixed; its first two prefixes fit within B2048 without needing sparse selection.

Capture/audit stages checked immutable input provenance and bitwise equality of shared teacher queries. Q64/Q128 smoke additionally checked all Q32 overlaps in the smoke window. Base equality, factor hashes and finite values were audited for the new banks.

## 6. Shared RULER protocol

All main query-sweep evaluations use the same 11 tasks × eight prompts, or 88 paired prompts, at the RULER32K configuration. This is a reused pilot, not an untouched final test set or the complete 13-task RULER suite.

Page selection uses per-head non-sink page-LSE normalization, followed by max across the four GQA query heads, then one fixed physical page set per group. It is not a union of four independently budgeted Top-k sets. Page32/B2048 selects page0 plus 63 routed pages when enough history is available.

All 36 layers use sparse decode. Prefill is full-support C1 Triton attention; the first generated token is shared. Each arm gets an independent immutable-prefix cache fork. Decode uses the native BF16 path, greedy generation, official base prompts and task-specific generation caps/EOS. A full exact-K + C1-V80 reference is rerun in each evaluation.

Scores are case-insensitive reference substring matches: fractional reference hits for `all`, any-reference hits for `part`. The reported aggregate is the mean of task scores. Full exact-K + C1-V80 is not a dense-V128 baseline.

## 7. RULER results

### Aggregate scores

| Frozen closed-form Base16 configuration | RULER mean | Difference from terminal Q32 |
| --- | ---: | ---: |
| Q1 Fisher R8 | 75.3598% | -5.9659 pp |
| Terminal Q16 Fisher R8 | 80.5682% | -0.7576 pp |
| Terminal Q32 Fisher R8 | **81.3258%** | 0 |
| Full-window Q32 Fisher R8 | 80.4167% | -0.9091 pp |
| Terminal Q64 Fisher R8 | 80.5303% | -0.7955 pp |
| Terminal Q128 Fisher R8 | 79.6212% | -1.7045 pp |
| Full exact K + C1-V80 | 85.2083% | +3.8826 pp |

### Per-task scores

All numbers are percentages. Every column has eight prompts/task.

| Task | Q1 | Q16 | Q32 | Full-window Q32 | Q64 | Q128 | Exact K + C1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| niah_single_1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_single_2 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_single_3 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multikey_1 | 87.5 | 87.5 | 87.5 | 87.5 | 87.5 | 87.5 | 87.5 |
| niah_multikey_2 | 25 | 50 | 50 | 50 | 50 | 50 | 87.5 |
| niah_multiquery | 87.5 | 100 | 96.875 | 96.875 | 93.75 | 93.75 | 96.875 |
| niah_multivalue | 65.625 | 93.75 | 96.875 | 96.875 | 93.75 | 93.75 | 93.75 |
| vt | 92.5 | 92.5 | 92.5 | 95 | 90 | 92.5 | 92.5 |
| fwe | 83.3333 | 75 | 83.3333 | 70.8333 | 83.3333 | 70.8333 | 91.6667 |
| qa_1 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| qa_2 | 37.5 | 37.5 | 37.5 | 37.5 | 37.5 | 37.5 | 37.5 |

### Paired sample changes

| New configuration versus reference | Improved | Regressed | Tied |
| --- | ---: | ---: | ---: |
| Terminal Q32 versus terminal Q16 | 3 | 1 | 84 |
| Full-window Q32 versus terminal Q32 | 4 | 6 | 78 |
| Terminal Q64 versus terminal Q32 | 2 | 5 | 81 |
| Terminal Q128 versus terminal Q32 | 1 | 6 | 81 |
| Terminal Q128 versus terminal Q64 | 2 | 4 | 82 |

Independent rescoring reproduced the reported results. Cross-run checks confirmed unchanged exact-K reference generations for the comparisons recorded in the experiment reports, including all 88 sequences across Q32/Q64/Q128. Base tensors remained unchanged; new residual banks contain refitted E/U.

The Q1-to-Q16 gain is 5.2083 pp over a 16× query-count increase; Q16-to-Q32 gains 0.7576 pp over 2×. Comparing these absolute gains alone does not establish diminishing returns. The subsequent Q64/Q128 runs, however, did not improve this pilot. They do not establish that more Q must generally hurt, that more independent windows would be ineffective, or that R8 capacity is the identified cause.

## 8. Statistics-construction optimization and runtime

The original multi-query builder repeated C1 encoding, Base prediction, RoPE and residual construction for every query. The revised builder is window-major: those token features are computed once/window, while exact QK, causal teacher probabilities and Page-Fisher statistics remain query-specific.

The output remains query-major/document-minor, preserving Q/Gram alignment. Final CPU Grams are preallocated instead of concatenating a list into a second full buffer. This is not a fused all-query Fisher kernel.

On a real-cache A100 smoke using two windows, terminal Q32 and layers 0/33, the statistics-only median speedups were 7.2812× and 7.2852×. Gram differences were zero on these examples and Fisher losses matched. This does not imply a 7× full-fitting speedup or universal bitwise equality on all hardware/shapes.

| Completed fit | Four-worker elapsed range | Statistics path |
| --- | --- | --- |
| Terminal Q32 | 40:24–41:13 | Previous query-major builder |
| Full-window Q32 | 13:56–14:52 | Window-major builder |
| Terminal Q64 | 20:44–22:04 | Window-major builder |
| Terminal Q128 | 34:12–36:14 | Window-major builder |

Fit runs used L40S and `basis`. Full-window Q32 differs from terminal Q32 in both query placement and implementation, so their elapsed-time difference cannot be attributed solely to either change. RULER workers generally took about six to seven minutes. All these times include startup and are not isolated kernel timings.

FP32 fit-plus-diagnostic Gram storage is approximately 5 GiB/layer for Q32, 10 GiB for Q64, and 20 GiB for Q128, excluding other tensors and solver temporaries. Q64 and Q128 completed without OOM. An optional Q128 `nvidia-smi` check was denied execution permission; it did not affect fitting and did not yield a GPU-memory measurement.

## 9. Direct page-boundary repair: implemented method

This is a local repair of the completed terminal-Q32 Fisher router, not a new Base fit or a fresh residual fit from scratch.

For each sampled query, use the actual native BF16 selector to identify its chosen physical pages. Teacher full-attention page mass is averaged across the four heads in the group. Pair high-mass omitted pages with low-mass selected non-sink pages, retaining only positive teacher-mass gains. At most eight pairs are used per query. Pinned page0 is never removed.

The deployment-equivalent log ranking score is

\[
a_p=\max_h\left[
\log\sum_{i\in p}e^{\hat s_{h,i}}
-\log\sum_{i\notin\mathrm{sink}}e^{\hat s_{h,i}}
\right].
\]

The local weighted squared-hinge surrogate is

\[
\sum_{p^+,p^-}w_{+,-}
\left[0.05-(a_{p^+}-a_{p^-})\right]_+^2,
\quad w_{+,-}=\bar m_{p^+}-\bar m_{p^-}>0.
\]

E and U are alternated for two sweeps. Each block uses analytic derivatives, with the current max-head owner held fixed locally and both page-LSE and non-sink normalization differentiated. The active linearized constraints are solved by damped least squares in observation space:

\[
\Delta=A^\top(AA^\top+\lambda I)^{-1}b,
\quad\lambda=0.01\,\mathrm{mean}\,\mathrm{diag}(AA^\top),
\]

with a numerical damping floor and square-root normalized pair weights in A/b. This implementation does not use autograd or Adam. It is not an exact global optimizer of discontinuous Top-B recall.

Step sizes 1, 0.5, 0.25 and 0.125 are tried. Every trial recomputes BF16 sidecar scores and real selected pages. An update is accepted only if aggregate fit teacher-mass coverage does not decrease and the fixed-pair native hinge loss decreases. Diagnostic examples do not enter fitting or acceptance.

This gate guarantees only non-decreasing aggregate fit coverage on that fixed set. It does not guarantee per-query improvement, non-sink-mass improvement, diagnostic improvement, or RULER accuracy.

## 10. Direct-ranking data scale and completed results

### Actual scale

The repair experiments use fit windows 0/1 and diagnostic windows 64/65, with queries at 26623, 28671, 30719 and 32767. That is only eight query positions per split per group, each with four heads. Each query still reads its full causal K prefix, but many K rows do not replace independent Q/document diversity.

The first run covered layer 33/group 0 on A100. The expansion covered layers 15 and 33, all eight groups each, on four L40S workers. **Only layer/group coverage increased; the window and query counts did not.** It is therefore a broader implementation smoke, not a full-sized test of the new method.

### Initial single-group smoke

| Metric, layer 33/group 0 | Before | After |
| --- | ---: | ---: |
| Fit full teacher mass | 91.6388% | 91.8595% |
| Diagnostic full teacher mass | 96.2683% | 96.5162% |
| Diagnostic non-sink mass | 95.9837% | 96.2486% |

Three of four E/U block updates were accepted. The fit increase is partly enforced by the gate; the diagnostic increase is not.

### All-group expansion

| Layer | Fit full mass before → after | Diagnostic full mass before → after | Diagnostic change | Groups improved / regressed |
| --- | --- | --- | ---: | ---: |
| 15 | 93.3861% → 93.7735% | 93.2512% → 93.2088% | -0.0424 pp | 2 / 6 |
| 33 | 89.5770% → 90.3516% | 92.7950% → 92.8958% | +0.1008 pp | 5 / 3 |

Diagnostic non-sink mass changed from 89.4487% to 89.3617% in layer 15 and from 92.5084% to 92.6142% in layer 33. Across sixteen groups, diagnostic full mass improved in seven and regressed in nine. Layer 33's mean gain is strongly influenced by group 7 (+1.2912 pp).

All sixteen L40S tasks completed successfully in 10–12 seconds each including startup. Actual per-task program time was about 1.0–1.8 seconds with less than 0.184 GiB peak allocated GPU memory. Such short times reflect the tiny cached-data scope, not the cost of 64/16-window fitting or end-to-end evaluation.

The initial implementation failed once before optimization due to an incorrect legacy-manifest field lookup. It was corrected to check the direct-manifest hash against audited Q32 provenance, and the same smoke was rerun successfully without relaxing numerical checks. The first A100 expansion was cancelled while pending at the user's request because no A100 was free; the later L40S expansion completed. No original checkpoint was overwritten.

## 11. Verification and limits of the evidence

- New ranking tests cover finite-difference checks of both analytic factor derivatives, pinned-page exclusion, positive swap weights, frozen inputs, fit acceptance, full-budget no-op behavior and native BF16 proxy arithmetic. The combined regression run passed 26 tests.
- Real ranking runs compare the assembled score against the production sidecar at sampled causal positions. This is a cached single-group check, not a live full-model C1 trajectory test.
- The ranking results store source hashes, per-block accepted steps, factor hashes and metrics. The output files are single-group E/U tensors, not installed full-model checkpoints.
- Fit improvement does not demonstrate generalization because acceptance explicitly enforces it. Two diagnostic windows are insufficient for a reliable method-level conclusion.
- No repaired router has been evaluated on RULER or PPL. No full 64-fit/16-diagnostic ranking experiment has run.
- The ranking solver currently forms a matrix in constraint space; simply increasing the number of sampled wrong-page pairs would incur quadratic storage growth. A scalable replacement was discussed but has not been implemented.
- The exact reason that Q64/Q128 underperform Q32 remains unresolved. Common-diagnostic-Q Fisher-loss comparisons and convergence diagnostics were discussed, not completed in this sequence. Objective mismatch, finite-sample effects and solver behavior are possible explanations, not established findings.

## 12. Checkpoints, code and detailed records

### Main checkpoints

| Configuration | Directory under `results/checkpoints/` |
| --- | --- |
| Closed-form Base + Q1 residual | `q8_residual_kl_bank` |
| Closed-form Base + Q16 residual | `mse_base_q16_r8` |
| Closed-form Base + terminal Q32 residual | `mse_base_q32_r8` |
| Closed-form Base + full-window Q32 residual | `mse_base_uniform_q32_r8` |
| Closed-form Base + Q64 residual | `mse_base_q64_r8` |
| Closed-form Base + Q128 residual | `mse_base_q128_r8` |

### Implementation entries

- [Query capture](../scripts/capture_qwen3_8b_q16.py): nested Q16–Q128 and teacher-overlap checks.
- [Residual Fisher fitting](../evaluation/fit_qwen3_8b_q8_fisher_residual.py): frozen Base, window-major statistics and uniform R8 fitting.
- [RULER evaluator](../evaluation/eval_qwen3_8b_residual_rank_ruler.py): paired native sparse versus exact-K/C1 evaluation.
- [Page-boundary repair](../basisserve/core/residual_page_ranking.py): analytic local alternating updates and actual-selection acceptance.
- [Ranking diagnostic driver](../evaluation/smoke_residual_page_ranking.py): per-layer/group cached-data runs.

### Detailed reports and execution commands

The following reports retain the exact program commands, environments, Slurm stages, artifact paths and detailed checks. This consolidation did not submit a new GPU experiment.

- [Closed-form Base/Q1](mse_base_q1_ruler_protocol.md).
- [Closed-form Base/Q16](mse_base_q16_ruler_protocol.md).
- [Closed-form Base/terminal Q32](mse_base_q32_ruler_protocol.md).
- [Closed-form Base/full-window Q32](mse_base_uniform_q32_ruler_protocol.md).
- [Q64 and Q128](mse_base_q64_q128_ruler_protocol.md).
- [Window-major optimization](fisher_window_major_summary.md).
- [Initial page-ranking smoke](residual_page_ranking_smoke.md).
- [Layers 15/33 all-group ranking results](residual_page_ranking_l15_l33.md).

The main RULER means and per-task scores in this report were rechecked from their saved `result.json` records in `basis`. Historical reports describe state at the time they were written; this report distinguishes subsequently completed experiments from proposals that remain unimplemented. No GitHub commit or push was performed while preparing this summary.
