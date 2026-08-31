# C1 K-only Reverse ShadowKV

This design keeps the complete C1-compressed Value cache on GPU and offloads
exact post-RoPE Keys.  Small post-RoPE Key metadata select physical Key pages;
the selected exact Keys are fetched and used for sparse exact attention against
the corresponding resident C1-Value slices.

It reverses the data placement used by
[ShadowKV](https://bytedance-seed.github.io/ShadowKV/): ShadowKV retains a
compact/reconstructable Key representation on GPU and fetches selected Values,
whereas this design selects and fetches exact Keys while C1-Values stay local.

## Decode step

For physical KV head `g`, page `p`, consuming GQA query heads `H_g`, and `M`
contiguous mean landmarks per page, the deployable selector uses

\[
\hat s_{g,p} = \max_{h\in H_g}\max_{m\le M}
\frac{q_h^\top \bar k_{g,p,m}}{\sqrt d}.
\]

Page selection happens once per physical KV head. Thus pages preferred by any
query head in a GQA group are unioned without fetching duplicate Key pages.
Recent pages and an external hot/outlier page mask are forced into the support;
the remaining page budget is filled by landmark score.

This `physical_shared` policy is a system-oriented GQA adaptation, not the
selection granularity used by the original QUEST algorithm.  The
`per_query_head` policy instead assigns an independent page budget to every
Query head, exactly as in QUEST.  Exact K pages are fetched once per physical
KV head by taking the union of its consuming Query-head supports, but sparse QK
and C1-V attention retain the distinct support of each Query head.  Results
therefore report both logical Query-head sparsity and the larger physical K-page
union.  Paper-faithful decode also forces the current final page into each
Query-head support and charges it against the page budget.

The selected pages then follow one common path:

1. Read each selected exact post-RoPE Key page through `ExactKeyPageStore`.
2. Compute exact QK scores only on those pages.
3. Apply an FP32 online softmax over the sparse support.
4. Read only the matching slices of the resident C1-Value cache.
5. Return the C1 latent output for the existing C1 decoder.

The implementation never expands Values across GQA query heads and never scans
the full C1-Value cache after page selection.

## Selectors

- `quest_minmax` stores the coordinate-wise minimum and maximum post-RoPE Key
  for every physical page. For query head `h`, it computes the QUEST bound

  \[
  U_{h,p}=\frac{1}{\sqrt d}\sum_j
  \max(q_{h,j}k^{\min}_{p,j},q_{h,j}k^{\max}_{p,j}),
  \]

  then takes the maximum over the query heads sharing a physical GQA KV head.
  The score upper-bounds every exact QK logit in the page when the metadata is
  exact. It can still rank pages poorly because independently optimal channels
  need not come from the same token. This is a GQA adaptation of
  [QUEST](https://proceedings.mlr.press/v235/tang24l.html).
  `physical_shared` takes this group maximum before Top-K. `per_query_head`
  performs Top-K before any GQA union and is the paper-faithful quality policy.

- `mean_landmark` is an ablation. `landmarks_per_page=1` is the
  direct ShadowKV-style mean; `4` tests whether small deterministic subchunks
  improve page recall without an activation-aware low-rank fit.
- `centroid_radius` stores the same centroid plus an FP32 residual radius

  \[
  r_{g,p,m}=\max_{t\in\mathcal P_{p,m}}\|k_{g,t}-\hat\mu_{g,p,m}\|_2.
  \]

  The radius is computed around the centroid after its BF16/FP16 storage
  quantization. Cauchy--Schwarz then gives the calibration-free page bound

  \[
  U_{h,p}=\max_m\frac{q_h^\top\hat\mu_{g,p,m}
  +\|q_h\|_2r_{g,p,m}}{\sqrt d}.
  \]

  This is a maximum-QK upper bound, although its ranking can be loose.
- `teacher_exact` ranks pages with the maximum exact QK score. It deliberately
  consults full exact K and is only an oracle for separating landmark-selection
  error from sparse-budget error. It is not a runtime policy and is not
  necessarily the page set that maximizes aggregate softmax mass across all GQA
  query heads.
- `teacher_mass` computes the full exact attention distribution and ranks each
  physical page by its probability mass summed over all consuming GQA query
  heads. For a fixed physical-page count, this is the exact optimum for the
  aggregate selected attention-mass objective. It is also not deployable.

All selectors use the same exact-page attention path after selection.

## Measurements

The replay oracle reports:

- exact full-attention mass contained in selected pages;
- exact top-page and top-token recall;
- finite `KL(sparse || exact)` and attention-probability L1;
- C1 latent relative L2;
- decoded C1 output relative L2, maximum absolute error, and cosine similarity;
- logical resident landmark/C1-Value bytes, exact-Key page bytes read, and QK
  FLOP counts.

Exact-to-sparse KL is intentionally omitted because it is infinite whenever the
exact distribution assigns positive mass outside the sparse support.

## Memory and transfer model

For sequence length `S`, page size `P`, head dimension `D`, C1 Value rank `rV`,
and `M` landmarks per page, a mean-landmark configuration stores approximately

\[
B H_{kv} S r_V b_V
\; + \;
B H_{kv}\lceil S/P\rceil M D b_L,
\]

plus recent/hot exact-Key pages and a staging buffer. Per decode step, selecting
`N` physical pages transfers approximately

\[
N P D b_K
\]

bytes of exact Key data. The page budget therefore controls both approximation
quality and host-to-device traffic directly.

The QUEST selector replaces the mean-landmark term with two vectors per page:

\[
2 B H_{kv}\lceil S/P\rceil D b_L.
\]

Thus its Key metadata is approximately `2/P` of a full Key cache at the same
dtype: 12.5% for page 16, 6.25% for page 32, and 3.125% for page 64.

## Current scope

`basisserve/core/c1_k_reverse_shadow.py` is an algorithm correctness oracle.
The existing page-store boundary can read a CPU tensor, but the reference call
is synchronous and is not a pinned-memory, asynchronous, double-buffered
offload implementation. Latency claims require a later CPU-backed runtime with
overlapped page selection, transfer, and sparse attention.

The capture replay entry point is
`evaluation/eval_qwen3_c1_k_reverse_shadow_oracle.py`.

## Frozen 16 x 4096 held-out oracle

The first all-layer protocol was frozen before looking at these examples:

- Qwen3-8B-Base with the existing rank-64-per-KV-head C1 Value factors;
- 16 document-disjoint windows from the C4 `validation` split, each 4096
  tokens, with the decode query at position 4095;
- page size 16 and BF16 QUEST Min/Max metadata;
- layers 0--1 use full exact K resident on GPU;
- layers 2--35 use CPU exact K selected by `quest_minmax`;
- sparse-layer exact-token budgets 256, 512, and 1024, with no recent-window
  addition.

The C1 factors were fitted on a separate C4 `train` bank. The held-out capture
contains zero K-proxy fit pairs, so none of these 16 windows calibrate or train
the selector. Metrics are emitted separately for every layer and window: each
budget therefore has 36 x 16 = 576 observations.

| sparse-layer budget | selected mass mean | selected mass minimum | decoded rel-L2 mean | decoded rel-L2 P95 | decoded rel-L2 maximum (layer/window) | window layer-mean maximum |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.754409 | 0.313374 | 0.353957 | 0.749470 | 2.300912 (33/8) | 0.517865 |
| 512 | 0.826419 | 0.460887 | 0.237918 | 0.536753 | 1.574445 (33/8) | 0.332951 |
| 1024 | 0.895180 | 0.595039 | 0.138653 | 0.327469 | 0.729362 (15/7) | 0.213562 |

All 1728 records are finite, and the full-exact layers have exactly unit
selected mass and zero decoded-output error. Increasing the sparse budget
improves the aggregate metrics substantially, but budget 512 is not
near-lossless: its P95 decoded relative L2 is 0.537 and the worst observation is
1.574. Even budget 1024 retains a material tail. The worst layer-mean decoded
errors at budget 512 are layers 33 (0.498), 15 (0.497), and 9 (0.339).

These are independent layer-local attention-output oracles, not end-to-end NLL,
PPL, task accuracy, or measured CPU-offload latency. They show that QUEST page
selection plus the tested sparse budgets remains the dominant algorithmic risk;
they do not by themselves establish the application-level quality loss.

Artifacts:

- `results/evaluation/qwen3_8b_c1_k_reverse_shadow_quest_c4val16_s4096.json`
  (SHA256 `80212a03df3513e4a06f9a4b75099436ab990506d2bd56069fa1ea2db8c5db5d`)
- `results/evaluation/qwen3_8b_c1_k_reverse_shadow_quest_c4val16_s4096.md`
  (SHA256 `6e202b4d2e2315fd5c893106251b84fabda2c8a8fe45c503c1db99ef02d44d4e`)

## Paper-faithful QUEST versus physical GQA selection

The end-to-end comparison uses the same 16 held-out C4 validation documents,
an exact 3968-token prefill, and scores the final 128 tokens of each 4096-token
window. Layers 0--1 remain exact. Every sparse Query forces its current final
page into the budget. `physical_shared` selects one support per physical KV
head; `per_query_head` reproduces QUEST's independent Query-head Top-K and then
unions pages only for physical K reads.

| support | logical budget | Dense-K + C1 PPL | sparse PPL | PPL change | physical K fraction | logical Query fraction |
|:---|---:|---:|---:|---:|---:|---:|
| physical shared | 512 | 6.028448 | 6.163613 | +2.2421% | 12.5108% | 12.5108% |
| physical shared | 1024 | 6.028448 | 6.047548 | +0.3168% | 25.2077% | 25.2077% |
| physical shared | 2048 | 6.028448 | 6.023979 | -0.0741% | 50.6014% | 50.6014% |
| per Query head | 512 | 6.028448 | 6.141478 | +1.8749% | 21.5000% | 12.5108% |
| per Query head | 1024 | 6.028448 | 6.051222 | +0.3778% | 39.2143% | 25.2077% |
| per Query head | 2048 | 6.028448 | 6.031436 | +0.0496% | 67.2681% | 50.6014% |

At budget 1024, physical-shared selection is the best measured quality/traffic
point: its mean NLL increase is `0.003163 +/- 0.004019` paired sample SE while
reading 25.21% of physical K. Paper-faithful per-Query selection does not
improve mean PPL at this budget and expands physical K reads to 39.21% because
the four Query heads sharing each Qwen3 KV head request different pages. The
per-Query policy sometimes reduces individual shared-support tail errors, but
it is not a better aggregate GQA deployment policy in this protocol.

Artifacts:

- `results/evaluation/qwen3_8b_c1_quest_faithful_c4val16_s4096_suffix128.json`
  (SHA256 `8474d1d4626cbab03c123cc428f5f3d84a71ad3f533ae3aabacc40c01a7a884c`)
- `results/evaluation/qwen3_8b_c1_quest_faithful_c4val16_s4096_suffix128.md`
  (SHA256 `7175a887e1753f3182934871000ce2d1a079feb92510cbbf3a7f18f1c1cfc3ff`)
