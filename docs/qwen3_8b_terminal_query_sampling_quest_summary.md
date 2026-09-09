# Qwen3-8B C1 conditional K routing: query sampling and QUEST summary

## 1. Scope

This document summarizes the current Qwen3-8B conditional Key-routing
experiments, the mathematical difference between the implemented selector and
original QUEST, and the definition of the proposed terminal-Q128 diagnostic.

No terminal-Q128/Q256 fitting or evaluation result is reported here.

## 2. Current routing configuration

The fixed model and cache configuration is:

| Component | Setting |
|---|---|
| Model | Qwen3-8B-Base, 36 layers |
| Attention | 8 KV heads, 4 Query heads per KV head, head dimension 128 |
| Resident Value payload | Frozen C1-V80 |
| Key-derived Base | Per-GQA-group affine Base16 predicted from C1-V80 |
| Key innovation | Post-RoPE residual sidecar R8 |
| Page size | 32 tokens |
| Sparse budget | B2048 = 64 physical pages per KV group |
| Prefix handling | Page 0 excluded from residual fitting and pinned at selection |
| Sparse payload | Selected exact post-RoPE K with resident C1-V80 |
| Evaluation | 32K RULER pilot, 11 tasks x 8 examples = 88 prompts |

For token position \(i\), the approximate post-RoPE Key is

\[
\widetilde k_i
=
R_i\widehat k_i^{\mathrm{pre}}
+
\widehat\delta_i,
\]

where

\[
\widehat k_i^{\mathrm{pre}}
=
\operatorname{Base16}(C_{V,i})
\]

is predicted from the resident C1-V80 code, and \(\widehat\delta_i\) is the
rank-8 post-RoPE innovation sidecar.

The selected pages are only a routing decision. Attention over selected pages
uses exact K; therefore, exact attention cannot recover a page omitted by the
router.

## 3. Current Page-LSE selector

For Query head \(h\), the proxy token scores are

\[
\widetilde s_{h,i}
=
\frac{q_h^\top\widetilde k_i}{\sqrt{128}}.
\]

The score of page \(p\) is its proxy log-sum-exp:

\[
\widetilde\ell_{h,p}
=
\log\sum_{i\in p}\exp(\widetilde s_{h,i}).
\]

It is normalized over pages:

\[
\widetilde m_{h,p}
=
\frac{\exp(\widetilde\ell_{h,p})}
{\sum_{p'}\exp(\widetilde\ell_{h,p'})}.
\]

The four Query heads that share KV group \(g\) are combined using

\[
G_{g,p}=\max_{h\in g}\widetilde m_{h,p}.
\]

The router pins the prefix page and selects the remaining highest-ranked pages
under a total budget of 64 pages per KV group.

This is an approximate-Key Page-LSE selector. It is not the original QUEST
min/max selector.

## 4. Original QUEST

Original QUEST stores elementwise extrema of the exact cached post-RoPE Keys
for every page:

\[
k^{\min}_{p,j}=\min_{i\in p}k_{i,j},
\qquad
k^{\max}_{p,j}=\max_{i\in p}k_{i,j}.
\]

Given Query \(q\), it assigns page criticality

\[
U_p(q)
=
\frac{1}{\sqrt d}
\sum_j
\max\left(q_jk^{\min}_{p,j},q_jk^{\max}_{p,j}\right).
\]

This is an upper bound because

\[
\max_{i\in p}q^\top k_i
\le
\sum_j\max_{i\in p}q_jk_{i,j}
=
\sqrt d\,U_p(q).
\]

QUEST ranks pages by \(U_p(q)\), loads exact K/V from selected pages, and
computes sparse attention. It does not fit a learned Base or residual and does
not require C4 calibration.

References:

- [QUEST paper](https://arxiv.org/abs/2406.10774)
- [Official QUEST implementation](https://github.com/mit-han-lab/Quest)

The two selectors therefore have different approximations:

| Property | Original QUEST | Current C1 router |
|---|---|---|
| Resident metadata | Exact-Key per-page min/max | Per-token Base16 + R8 proxy |
| Calibration | None | C4 activation calibration |
| Page objective | Upper bound on maximum token logit | Approximate page attention mass |
| Main selection error | Loose coordinatewise upper bound | Learned proxy error and distribution shift |
| Selected attention | Exact K/V | Exact K with resident C1-V80 |

## 5. Query-sampling experiments

The C4 calibration population contains 64 fit and 16 validation windows of
32768 tokens. Each window packs eight 4096-token C4 chunks and is not a native
32K document.

### 5.1 Terminal residual sampling

The terminal experiments keep the same fitted Q-aware Base16 and change only
the number of Query positions used to fit residual R8.

| Residual fitting | Query positions/window | Position region | RULER mean |
|---|---:|---|---:|
| Q1 | 1 | terminal position | 69.5076% |
| Q8 | 8 | last 8K, spacing 1024 | 80.0947% |
| Q16 | 16 | last 8K, spacing 512 | 80.4356% |
| C1 exact-K reference | Exact K | full routing reference | 85.2083% |

Q1 to Q8 produced a large improvement of 10.5871 percentage points. Q8 to Q16
produced only 0.3409 percentage points, with four samples improving, five
regressing, and 79 tying. This indicates a large benefit from moving beyond a
single terminal Query, followed by substantial saturation or task-level
variance between Q8 and Q16.

### 5.2 Uniform-32K Q16

The uniform experiment used positions

\[
2047,4095,\ldots,32767
\]

and refitted both Base16 and residual R8. It therefore differs from the
terminal residual-only comparison in two ways: position distribution and Base
factors.

| Configuration | RULER mean |
|---|---:|
| Terminal-Q16 residual with the existing terminal-Q8 Base | 80.4356% |
| Uniform-32K-Q16 jointly refitted Base16 + residual R8 | 79.1477% |
| C1 exact-K reference | 85.2083% |

Uniform-32K Q16 was 1.2879 percentage points below terminal-Q16. All 88 exact-K
generated token sequences were identical across the matched evaluations.

The new uniform Base improved its own validation raw-QK NMSE in all 36 layers;
the median relative reduction from initialization was 0.62485%. Thus a small
improvement in the smooth local fitting objective did not produce better
downstream discrete page selection.

Because the uniform experiment refitted both Base and residual, its accuracy
difference cannot be assigned uniquely to one component.

## 6. Interpretation of the observed behavior

### 6.1 Smooth fitting loss versus discrete retrieval

Base raw-QK regression and residual Page-Fisher are smooth average objectives.
Deployment instead applies a discrete Top-page operator:

\[
\operatorname{TopK}_p\widetilde\ell_p.
\]

A small proxy-score error around the selection cutoff can remove a critical
page even when average score error decreases. Exact-K attention is applied only
after selection and cannot repair such a false negative.

### 6.2 C4 teacher mass versus RULER retrieval

For teacher probability \(a_i\), Page-Fisher assigns page mass

\[
m_p=\sum_{i\in p}a_i.
\]

A page that receives little mass under ordinary C4 Queries contributes little
to the fitted metric. The same page can become the unique answer-bearing page
under a RULER needle Query. Increasing the number of C4 Query positions reduces
sampling noise but does not remove this objective/domain mismatch.

### 6.3 Position distribution

RULER answering and autoregressive decode use Queries at the end of the full
prompt. Terminal sampling therefore matches this position regime more closely.
Uniform-32K Q16 spends most sampled positions on earlier-prefix regimes while
the rank-limited Base16/R8 representation must accommodate their different
causal contexts, attention patterns, and RoPE phases.

### 6.4 What the result says about QUEST

The result does not show that original QUEST fails. Original QUEST has no
learned Key proxy and therefore avoids Base/residual calibration error.

QUEST can nevertheless produce false positives because the minimum or maximum
for different Key dimensions may come from different tokens. Its summed bound
can describe a coordinatewise combination that no real token realizes. It also
bounds the maximum token logit rather than exact page softmax mass. These are
different failure modes from those measured in the current C1 router.

## 7. Defined terminal-Q128 diagnostic

The clean next sample-count diagnostic is to freeze the existing terminal-Q8
Base16 and refit only residual R8 with 128 Query positions uniformly distributed
over the final 8192 tokens:

\[
t_j=24576+64(j+1)-1,
\qquad j=0,\ldots,127.
\]

Thus the positions begin at 24639 and end at 32767. All 32 Query heads
participate at every position.

Per Query head, the number of fit document-position observations changes from

\[
64\times16=1024
\]

for Q16 to

\[
64\times128=8192
\]

for Q128.

This residual-only comparison extends the controlled curve

\[
Q1\rightarrow Q8\rightarrow Q16\rightarrow Q128
\]

without introducing a simultaneous Base change.

Terminal-Q256 would sample every 32 tokens and provide 16384 fit
document-position observations per head. Its adjacent Queries are more
correlated, so its nominal doubling over Q128 would not imply twice as many
independent observations. It remains undefined as a completed experiment.

The terminal-Q128 result has four direct interpretations:

| Observation | Meaning |
|---|---|
| Page-Fisher validation and RULER both improve | Q16 residual fitting was sample-limited |
| Page-Fisher improves but RULER does not | Local objective or calibration-domain mismatch dominates |
| Both remain approximately unchanged | Residual R8 or the predictable routing subspace is saturated |
| Per-task/sample changes remain unstable | Rare-page retrieval is not adequately represented by average C4 Page-Fisher statistics |

