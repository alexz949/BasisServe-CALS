# Qwen3-8B C1 Two-Sided Factorized-KL Summary

## 1. Scope

This experiment implemented and evaluated the scalable, forward-only Two-Sided Factorized-KL rank allocator, identified in code as `two_sided_factorized_kl`.

Checkpoints produced before this naming cleanup retain the historical identifier `factorized_two_sided_dp`; the recorded algorithm and schedule are unchanged.

The completed scope was:

- skip the restricted candidate-rank-bank diagnostic;
- evaluate more terminal-position sample counts;
- sample causal prediction positions while retaining the complete 2048-token context;
- replace stored teacher vocabulary logits with FP32 teacher sufficient statistics;
- compute the student partition function by exact full-vocabulary streaming;
- replace complete probe forwards with batch-local anchor caching and suffix replay;
- run the complete 128-window C4 profile;
- export the selected schedule and evaluate full WikiText-2 test PPL.

The candidate-rank bank remained:

\[
\mathcal R=\{32,48,64,80,96,112,128\}.
\]

The anchor rank was 64. The measured two-sided intervention ranks were 32 and 96.

---

## 2. Position-sampled terminal objective

A length-2048 window contains 2047 valid next-token prediction positions:

\[
t\in\{0,\ldots,2046\}.
\]

For every window, a random permutation of these positions was generated without replacement. The measured position sets were nested:

\[
\mathcal T_{64}
\subset
\mathcal T_{128}
\subset
\mathcal T_{256}
\subset
\mathcal T_{512}
\subset
\mathcal T_{1024}.
\]

All teacher, anchor, layer, and rank interventions used exactly the same sampled positions. For a model configuration \(M\), the sampled terminal objective was:

\[
\widehat K(M)
=
\frac{1}{Nm}
\sum_{i=1}^{N}
\sum_{t\in\mathcal T_{i,m}}
\operatorname{KL}\!\left(
p_T(\cdot\mid x_i,t)
\Vert
p_M(\cdot\mid x_i,t)
\right).
\]

The full 2048-token attention computation was retained. Only the terminal positions sent through the output head were sampled. The vocabulary was never sampled or truncated.

---

## 3. Teacher sufficient statistics

Let the bias-free output head be:

\[
z=W h,
\qquad
p=\operatorname{softmax}(z).
\]

For each selected teacher position, the implementation retained:

\[
\mu_T=W^\top p_T,
\]

\[
c_T=\sum_v p_T(v)\log p_T(v),
\]

and the teacher log normalizer:

\[
\log Z_T=\log\sum_v\exp(z_{T,v}).
\]

The statistics were accumulated and stored in FP32. Full teacher vocabulary logits were not retained.

For student hidden state \(h_S\), the terminal KL is:

\[
\operatorname{KL}(T\Vert S)
=
c_T-\mu_T^\top h_S+\log Z_S.
\]

For an anchor \(A\) and probe \(P\), the paired KL difference becomes:

\[
\boxed{
\Delta\operatorname{KL}_{P-A}
=
\mu_T^\top(h_A-h_P)
+
\log Z_P-
\log Z_A
}.
\]

This removes all repeated teacher-softmax work from the individual layer probes.

---

## 4. Exact streaming vocabulary normalization

The student normalizer remained an exact full-vocabulary operation:

\[
\log Z_S
=
\log\sum_{v=1}^{151936}
\exp(w_v^\top h_S).
\]

The vocabulary was processed in chunks of 8192 entries. Each chunk produced a local FP32 log-sum-exp, and the chunks were combined with FP32 `logaddexp` accumulation.

The implementation therefore materialized only:

\[
[\text{selected positions},\ 8192]
\]

logits at one time instead of full sequence-by-vocabulary logits.

The output-head weights and output-head arithmetic for the new profiler were FP32. The transformer backbone and deployed C1 factors remained BF16.

---

## 5. Batch-local suffix replay

For the uniform rank-64 anchor, the profiler retained the input to each decoder layer:

\[
H_0^A,H_1^A,\ldots,H_{35}^A.
\]

For an intervention at layer \(\ell\), the probe began at the saved anchor state:

\[
H_\ell^A
\rightarrow
\text{modified layer }\ell
\rightarrow
\ell+1
\rightarrow
\cdots
\rightarrow
35.
\]

Layers before \(\ell\) were not recomputed. The attention mask, position IDs, RoPE embeddings, and final RMS normalization from the anchor forward were reused exactly.

For 128 windows and batch size 16, each shard processed eight data batches. Across four shards, execution consisted of:

- 64 complete teacher/anchor backbone forwards;
- 576 suffix probe replays;
- 360 complete-forward equivalents after weighting each suffix by its actual number of decoder layers.

The previous complete-forward implementation required 640 complete-forward equivalents under the same four-shard accounting.

---

## 6. Implementation

The implementation added:

- `basisserve/core/sampled_terminal_kl.py`
  - deterministic nested position sampling;
  - per-window hidden-state gathering;
  - FP32 teacher sufficient statistics;
  - exact streaming output-head log-sum-exp;
  - exact KL reconstruction and paired KL deltas;
  - selected-position NLL.
- `basisserve/core/qwen_suffix_replay.py`
  - batch-local Qwen anchor-state capture;
  - exact replay from an arbitrary intervention layer.
- `tests/test_sampled_terminal_kl.py`
  - nested-sampling tests;
  - dense-versus-streaming KL equivalence;
  - dense-versus-streaming NLL equivalence;
  - suffix replay equivalence.

The existing layer Global-KL profiler was extended in:

- `evaluation/run_qwen3_32b_c1_layer_global_kl_sharded.py`
- `evaluation/run_qwen3_8b_c1_two_sided_factorized_kl_sharded.py`
- `evaluation/build_qwen3_8b_c1_two_sided_factorized_kl_schedule.py`
  - `sampled_suffix` profiling backend;
  - batch-granular resumable checkpoints;
  - nested position-count metric records;
  - per-position-count Factorized-KL allocation;
  - peak GPU memory accounting.

The existing Qwen3-8B shim uses the same implementation after activating the audited Qwen3-8B geometry.

---

## 7. Numerical validation

The CPU and tiny-Qwen regression suite completed with:

\[
\boxed{27\text{ tests passed}}.
\]

The real-Qwen suffix replay validation reproduced a complete forward element by element after a single-layer intervention.

A one-layer L40S validation used two C4 windows and all 2047 prediction positions. The measured layer-0 deltas were:

| Intervention | New FP32 sampled-suffix backend | Previous BF16 full-forward backend | Mean difference |
|---|---:|---:|---:|
| rank 32 | 0.017957762 | 0.018065522 | -0.000107760 |
| rank 96 | -0.001999792 | -0.001941800 | -0.000057992 |

The signs and magnitudes agreed. The small numerical difference includes the change from BF16 output-head logits to FP32 output-head arithmetic.

---

## 8. Complete C4 profiling configuration

| Setting | Value |
|---|---|
| Model | Qwen3-8B-Base |
| Calibration domain | C4 train, fresh document-disjoint windows |
| Window start | 320 |
| Profile windows | 128 |
| Sequence length | 2048 |
| Batch size | 16 |
| Anchor rank | 64 |
| Compression probe | 32 |
| Expansion probe | 96 |
| Candidate ranks | 32, 48, 64, 80, 96, 112, 128 |
| Local error | held-out post-ALS BF16 relative MSE |
| Error exponent | 1.25 |
| Position counts | 64, 128, 256, 512, 1024 |
| Selected position count | 1024 |
| Position seed | 20260903 |
| Vocabulary chunk | 8192 |
| Model/factor dtype | BF16 |
| Teacher statistics | FP32 |
| Output-head arithmetic | FP32 |
| Attention backend | SDPA |
| Hardware | 4 × NVIDIA L40S |
| Shards | 4, with 9 layers per shard |

The four shard core elapsed times were:

| Shard | Seconds |
|---:|---:|
| 0 | 447.764 |
| 1 | 429.837 |
| 2 | 416.640 |
| 3 | 401.977 |

The slowest shard completed in 7 minutes 28 seconds of profiler time and 7 minutes 41 seconds of Slurm wall time.

All four shards reported the same peak CUDA allocation:

\[
32{,}479{,}114{,}240\text{ bytes}
=
30.25\text{ GiB}.
\]

Compared with the previous two-shard full-forward profile:

- previous total GPU time: 3608.66 GPU-seconds;
- new total GPU time: 1696.22 GPU-seconds;
- normalized profiling speedup: approximately \(2.13\times\);
- previous wall time: approximately 30.1 minutes;
- new four-GPU wall time: approximately 7.7 minutes.

---

## 9. Position-count convergence

Each position count independently generated a two-sided Factorized-KL schedule under the exact average-rank-64 budget.

| Positions per window | Exact layer matches to 1024 | Rank MAE to 1024 | Sensitivity Spearman, compression | Sensitivity Spearman, expansion |
|---:|---:|---:|---:|---:|
| 64 | 30/36 | 2.6667 | 0.9931 | 0.9719 |
| 128 | 34/36 | 0.8889 | 0.9882 | 0.9810 |
| 256 | 34/36 | 0.8889 | 0.9931 | 0.9941 |
| 512 | 34/36 | 0.8889 | 0.9959 | 0.9979 |
| 1024 | 36/36 | 0.0000 | 1.0000 | 1.0000 |

The rank histograms were:

| Positions | r32 | r48 | r64 | r80 | r96 | r112 | r128 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4 | 8 | 13 | 7 | 3 | 1 | 0 |
| 128 | 3 | 10 | 12 | 7 | 3 | 1 | 0 |
| 256 | 3 | 10 | 11 | 8 | 4 | 0 | 0 |
| 512 | 3 | 10 | 12 | 7 | 3 | 1 | 0 |
| 1024 | 3 | 10 | 11 | 8 | 4 | 0 | 0 |

The 1024-position probe deltas were strongly aligned with the previous 2047-position profile:

| Probe | Spearman versus old 2047-position profile | Mean absolute delta difference | Maximum absolute delta difference |
|---|---:|---:|---:|
| rank 32 | 0.998713 | 0.00007666 | 0.00065604 |
| rank 96 | 0.998970 | 0.00004204 | 0.00013176 |

The 1024-position schedule matched 32 of 36 layers in the previous 2047-position schedule, with rank MAE 1.7778.

---

## 10. Selected 1024-position schedule

The selected per-layer ranks were:

```text
[80, 64, 32, 48, 48, 48, 64, 48, 80, 80, 80, 80,
 48, 48, 64, 64, 64, 64, 64, 80, 64, 64, 96, 96,
 96, 48, 64, 48, 32, 64, 80, 48, 48, 96, 80, 32]
```

Its histogram was:

| Rank | Layers |
|---:|---:|
| 32 | 3 |
| 48 | 10 |
| 64 | 11 |
| 80 | 8 |
| 96 | 4 |
| 112 | 0 |
| 128 | 0 |

The exact layer-rank sum was:

\[
\sum_{\ell=0}^{35}r_\ell
=
36\times64
=
2304.
\]

The corresponding physical-source rank sum was:

\[
8\times2304=18432,
\]

which retains exactly 50% of dense Value-cache dimensions.

---

## 11. C4 confirmation

The exported schedule was evaluated on 16 C4 confirmation windows disjoint from the 128 profile windows.

| Schedule | Terminal KL | NLL | Exponentiated NLL |
|---|---:|---:|---:|
| Uniform C1-V64 | 0.300654555 | 2.265406869 | 9.635044008 |
| Old 2047-position allocation | 0.249093473 | 2.214503236 | — |
| New 1024-position allocation | **0.247002658** | **2.213140666** | **9.144390823** |

For the new schedule versus uniform V64:

\[
\operatorname{mean}(\Delta\mathrm{KL})=-0.053651896,
\]

\[
\operatorname{median}(\Delta\mathrm{KL})=-0.013680134.
\]

The new schedule improved terminal KL on all 16 confirmation windows. One window contributed a much larger improvement than the other 15. Removing only that largest-magnitude window gave:

\[
\operatorname{trimmed\ mean}(\Delta\mathrm{KL})=-0.014107282.
\]

For NLL:

\[
\operatorname{mean}(\Delta\mathrm{NLL})=-0.052266203,
\]

\[
\operatorname{median}(\Delta\mathrm{NLL})=-0.012480974,
\]

and the 15-window mean after removing the largest-magnitude window was:

\[
-0.012448653.
\]

The exponentiated confirmation NLL is a C4 confirmation statistic and is not the WikiText-2 test PPL.

---

## 12. Full WikiText-2 test PPL

The final evaluation used:

- WikiText-2 test split;
- sequence length 2048;
- batch size 1;
- 146 chunks;
- 298,862 prediction tokens;
- FP32 loss accumulation;
- BF16 Qwen3-8B and C1 factors;
- one NVIDIA L40S.

| Schedule | Full WikiText-2 test PPL |
|---|---:|
| Uniform C1-V64 | 8.416738065 |
| C4 128-window, old 2047-position allocation | 8.245895888 |
| C4 128-window, new 1024-position allocation | **8.229340360** |

Relative to uniform C1-V64, the new schedule reduced PPL by:

\[
8.229340360-8.416738065
=
-0.187397705,
\]

or 2.2265% relative.

Relative to the old 2047-position allocation, it reduced PPL by:

\[
8.229340360-8.245895888
=
-0.016555528,
\]

or 0.2008% relative.

The new PPL run accumulated:

\[
\mathrm{NLL\ sum}=629913.188964844.
\]

The old 2047-position schedule accumulated:

\[
\mathrm{NLL\ sum}=630513.826171875.
\]

The new schedule therefore reduced the total NLL by 600.637207031 over the same 298,862 tokens.

Evaluation time was 33.24 seconds. Peak CUDA allocation was 19,501,001,216 bytes, or 18.16 GiB.

---

## 13. Interpretation

The measured layer sensitivities converge substantially before all 2047 positions are evaluated. With 128 C4 windows:

- 64 positions preserve the broad ranking but still alter six schedule layers relative to 1024;
- 128, 256, and 512 positions each match 34 of 36 layers in the 1024-position schedule;
- 512 positions reach sensitivity Spearman correlations of 0.9959 and 0.9979 against 1024;
- 1024 positions reproduce the old full-position layer ordering with Spearman above 0.9987 for both probes.

The exact DP remains sensitive near rank-allocation boundaries. Very similar sensitivity rankings can exchange rank between two or four layers while preserving the same total budget. Consequently, exact schedule match is stricter than the underlying sensitivity agreement.

The new 1024-position schedule did not sacrifice downstream language-model quality. Its full WikiText-2 PPL was lower than both uniform V64 and the previous full-position allocation.

The profiling speedup came from the combination of:

- fewer terminal positions sent through the large vocabulary head;
- no retained full teacher vocabulary logits;
- no repeated teacher-softmax computation for probes;
- exact streaming normalization;
- suffix replay instead of restarting every intervention at layer 0;
- four independent layer shards.

The one-layer oracle comparison also shows that changing the output-head arithmetic from BF16 to FP32 introduces small numerical differences. The measured layer rankings and final conclusions were stable to this difference.

---

## 14. Linear alpha cleanup

The completed 128-window, 1024-position two-sided profile was reused without any additional model profiling. Only the offline allocation exponent was changed from

\[
\alpha=1.25
\]

to

\[
\alpha=1.
\]

With the linear form, the two measured one-layer slopes are

\[
s_\ell^-
=
\frac{
K(M_{\ell\leftarrow32})-K(M_{64})
}{
e_{\ell,32}-e_{\ell,64}
},
\qquad
s_\ell^+
=
\frac{
K(M_{\ell\leftarrow96})-K(M_{64})
}{
e_{\ell,96}-e_{\ell,64}
}.
\]

The offline layer cost is therefore

\[
\widehat C_\ell(r)=
\begin{cases}
s_\ell^-(e_{\ell,r}-e_{\ell,64}), & r<64,\\
0, & r=64,\\
s_\ell^+(e_{\ell,r}-e_{\ell,64}), & r>64.
\end{cases}
\]

The rank allocation is the exact dynamic-programming solution to

\[
\min_{\sum_\ell r_\ell=64L}
\sum_\ell \widehat C_\ell(r_\ell).
\]

### Allocated schedules

The linear schedule was:

```text
[80, 64, 32, 48, 48, 32, 64, 64, 80, 80, 80, 64,
 48, 48, 64, 64, 64, 64, 64, 80, 64, 64, 96, 112,
 96, 48, 64, 48, 32, 64, 64, 48, 64, 96, 80, 32]
```

Its rank histogram was:

| Rank | Layer count |
|---:|---:|
| 32 | 4 |
| 48 | 7 |
| 64 | 15 |
| 80 | 6 |
| 96 | 3 |
| 112 | 1 |

The rank sum remained exactly 2304. Relative to the \(\alpha=1.25\) schedule, 30 of 36 layer ranks matched and the rank MAE was 2.6667. The six changed layers were:

| Layer | \(\alpha=1\) | \(\alpha=1.25\) |
|---:|---:|---:|
| 5 | 32 | 48 |
| 7 | 64 | 48 |
| 11 | 64 | 80 |
| 23 | 112 | 96 |
| 30 | 64 | 80 |
| 32 | 64 | 48 |

### C4 confirmation

| Allocation exponent | Mean terminal KL | Mean NLL |
|---:|---:|---:|
| 1 | 0.244414151 | 2.210243862 |
| 1.25 | 0.247002658 | 2.213140666 |

For \(\alpha=1\) minus \(\alpha=1.25\), the paired 16-window differences were:

| Metric | Mean difference | Median difference | Standard error | Better / worse windows for \(\alpha=1\) |
|---|---:|---:|---:|---:|
| Terminal KL | -0.002588507 | +0.000111301 | 0.003063670 | 7 / 9 |
| NLL | -0.002896804 | -0.000448167 | 0.002889300 | 9 / 7 |

### Full WikiText-2 test PPL

All PPL rows used the same 298,862 prediction tokens, 146 independent 2048-token chunks, and FP32 loss accumulation.

| Allocation | PPL | NLL sum |
|---|---:|---:|
| Sampled two-sided, \(\alpha=1.25\) | **8.229340360** | 629,913.188965 |
| Sampled two-sided, \(\alpha=1\) | 8.241004649 | 630,336.496826 |
| Previous 2047-position two-sided allocation | 8.245895888 | 630,513.826172 |
| Uniform V64 | 8.416738065 | 636,642.511963 |

The linear schedule was 0.011664289 PPL, or 0.1417%, above the \(\alpha=1.25\) schedule. It was 0.004891239 PPL, or 0.0593%, below the previous 2047-position allocation and 0.175733416 PPL, or 2.0879%, below uniform V64.

The \(\alpha=1.25\) schedule remains the numerically best WikiText result. The linear \(\alpha=1\) schedule gives nearly the same PPL while removing the nonlinear exponent from the allocator, so it satisfies the stated simplification criterion for the formal method; \(\alpha=1.25\) can be retained as the tuned comparison.

The \(\alpha=1\) WikiText evaluation completed in 33.20 seconds with a peak CUDA allocation of 19,501,001,216 bytes, or 18.16 GiB.

---

## 15. Limited MCQ diagnostic

The linear \(\alpha=1\) allocation and uniform V64 were evaluated with lm-eval on seven zero-shot tasks. Each task was limited to 500 examples, giving 3,500 evaluated examples per arm. The evaluation used BF16 model weights, SDPA, a maximum length of 4096, and batch size 8. The reported metric is normalized accuracy when the task provides it and raw accuracy otherwise.

| Task | Metric | Uniform V64 | Factorized \(\alpha=1\) | Change |
|---|---|---:|---:|---:|
| ARC-Easy | acc_norm | 72.6% | **74.8%** | +2.2 pp |
| ARC-Challenge | acc_norm | 49.0% | **50.2%** | +1.2 pp |
| HellaSwag | acc_norm | 61.0% | **62.6%** | +1.6 pp |
| PIQA | acc_norm | **81.4%** | 80.8% | -0.6 pp |
| WinoGrande | acc | 70.8% | **71.2%** | +0.4 pp |
| BoolQ | acc | 78.0% | 78.0% | 0.0 pp |
| OpenBookQA | acc_norm | 39.4% | **40.0%** | +0.6 pp |
| **Seven-task macro** | equal-task mean | 64.6000% | **65.3714%** | **+0.7714 pp** |

The factorized allocation improved five tasks, tied one, and reduced one. Because every task used exactly 500 examples, the macro improvement also corresponds to 27 additional selected-metric correct predictions over 3,500 examples. The relative macro-accuracy increase was 1.1942%.

This limited evaluation supports the same direction as the WikiText result: redistributing the fixed V64-equivalent rank budget with the linear two-sided allocator improves over uniform rank allocation. The individual task changes remain small relative to their approximately two-percentage-point single-task standard errors. Per-example predictions were not logged, so this run does not provide a paired significance test.

The four task shards ran concurrently on four L40S GPUs and completed in 80–82 seconds without evaluation warnings or failures.

---

## 16. Artifacts

### Profiling implementation

- `basisserve/core/sampled_terminal_kl.py`
- `basisserve/core/qwen_suffix_replay.py`
- `evaluation/run_qwen3_32b_c1_layer_global_kl_sharded.py`
- `tests/test_sampled_terminal_kl.py`
- `tests/test_qwen3_32b_layer_global_kl.py`

### One-layer validation

- `results/evaluation/qwen3_8b_c1_sampled_suffix_validation_l0/shard_00.json`

### Complete 128-window profile

- `results/evaluation/qwen3_8b_c1_factorized_c4_128measure_twosided_sampled_suffix_profile/shard_00.json`
- `results/evaluation/qwen3_8b_c1_factorized_c4_128measure_twosided_sampled_suffix_profile/shard_01.json`
- `results/evaluation/qwen3_8b_c1_factorized_c4_128measure_twosided_sampled_suffix_profile/shard_02.json`
- `results/evaluation/qwen3_8b_c1_factorized_c4_128measure_twosided_sampled_suffix_profile/shard_03.json`

### Exported allocation checkpoint

- `results/checkpoints/qwen3_8b_c1_factorized_c4_128measure_twosided_sampled1024_avg64_a1p25/result.json`
- `results/checkpoints/qwen3_8b_c1_factorized_c4_128measure_twosided_sampled1024_avg64_a1p25/summary.md`
- `results/checkpoints/qwen3_8b_c1_factorized_c4_128measure_twosided_sampled1024_avg64_a1p25/selected_factors/`
- `results/checkpoints/qwen3_8b_c1_factorized_c4_128measure_twosided_sampled1024_avg64_a1p0/result.json`
- `results/checkpoints/qwen3_8b_c1_factorized_c4_128measure_twosided_sampled1024_avg64_a1p0/summary.md`
- `results/checkpoints/qwen3_8b_c1_factorized_c4_128measure_twosided_sampled1024_avg64_a1p0/selected_factors/`

### Full WikiText-2 PPL

- `results/evaluation/qwen3_8b_c1_factorized_c4_128measure_twosided_sampled1024_full_wikitext_ppl.json`
- `results/evaluation/qwen3_8b_c1_factorized_c4_128measure_twosided_sampled1024_a1p0_full_wikitext_ppl.json`

### Limited MCQ comparison

- `results/evaluation/qwen3_8b_c1_factorized_alpha1_vs_uniform_v64_mcq7_limit500_summary.json`
- `results/evaluation/qwen3_8b_c1_uniform_v64_mcq7_limit500_shard0.json`
- `results/evaluation/qwen3_8b_c1_uniform_v64_mcq7_limit500_shard1.json`
- `results/evaluation/qwen3_8b_c1_factorized_alpha1_mcq7_limit500_shard0.json`
- `results/evaluation/qwen3_8b_c1_factorized_alpha1_mcq7_limit500_shard1.json`

### Slurm jobs

| Purpose | Job ID | Status |
|---|---:|---|
| One-layer full-position validation | 8298741 | completed |
| Four-shard 128-window profile | 8298745 | completed |
| Confirmation and factor export | 8298750 | completed |
| Full WikiText-2 PPL | 8298751 | completed |
| Linear-alpha confirmation and factor export | 8298753 | completed |
| Linear-alpha full WikiText-2 PPL | 8298754 | completed |
| Four-way 500-example MCQ comparison | 8298759 | completed |

The WikiText evaluator emitted a tokenizer warning because the concatenated WikiText token stream exceeded the model's configured maximum sequence length. Evaluation subsequently split that stream into independent 2048-token chunks, so the warning did not change the evaluated context length or the reported PPL.
