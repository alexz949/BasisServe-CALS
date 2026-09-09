# C1-V80 Conditional Pre-K Routing: Algorithm and 32K Results

## Objective

This experiment asks whether the resident C1-V80 latent can serve two roles at decode time:

1. reconstruct the Value payload used by C1 attention; and
2. generate a low-cost routing proxy that identifies which exact Key pages should be loaded.

The target system keeps C1-V80 resident on GPU, stores exact post-RoPE K outside the active GPU cache, predicts routing scores from C1-V80 plus a small residual sidecar, loads the selected exact K pages, and then evaluates sparse exact-QK attention against the corresponding resident C1 Value payload.

The experiments here isolate the routing component. They do not measure end-to-end language-model PPL or RULER accuracy.

## Fixed C1 representation

- Model: BF16 Qwen3-8B-Base.
- Transformer layers: 36.
- Physical KV heads/groups per layer: 8.
- Query heads: 32.
- Head dimension: 128.
- Resident Value representation: uniform C1-V80.
- C1 checkpoint: ALS6 fit using 32 C4 documents of length 32,768.
- The C1-V80 factors are frozen throughout both the predictive-spectrum analysis and the residual-rank experiment.

Therefore, these experiments do not refit the Value latent. They measure how much Key-routing information can be extracted from the already fitted C1-V80 coordinates.

## Base16 conditional Key predictor

For a token and KV group, let

\[
c_i\in\mathbb R^{80}
\]

be its resident C1 Value code, and let

\[
k_i^{\mathrm{pre}}\in\mathbb R^{128}
\]

be its exact pre-RoPE Key. The affine rank-constrained predictor is

\[
\widehat k_i^{\mathrm{pre}}
=
c_iL_rR_r+b,
\qquad
\operatorname{rank}(L_rR_r)\le r.
\]

Base16 uses \(r=16\). It is trained offline by exact reduced-rank regression, not by BCD. After centering the C1 and Key rows, define

\[
G_{CC}=C^\top C,
\qquad
G_{CK}=C^\top K.
\]

The input covariance is whitened and the SVD

\[
G_{CC}^{-1/2}G_{CK}=U\Sigma V^\top
\]

is truncated to rank \(r\). This gives the minimum-MSE affine rank-\(r\) map. The token's exact RoPE transformation is then applied:

\[
\widehat k_i^{(0)}
=
\operatorname{RoPE}_{p_i}\!\left(\widehat k_i^{\mathrm{pre}}\right).
\]

Predicting pre-RoPE K and then applying exact position-dependent RoPE avoids forcing a single static linear map to represent all rotary positions.

## Predictive-spectrum experiment

### Protocol

- Fit split: 64 independent C4 documents of length 32,768.
- Held-out split: 16 independent C4 documents of length 32,768.
- Layers: all 36.
- Predictor ranks: 4, 8, 16, 24, 32, 48, 64, and 80.
- Targets: pre-RoPE K and direct post-RoPE K.
- Score metric: raw-score NMSE using the paired final-token Query from each held-out document.
- This analysis streams previously captured activations and performs no model forward.

If the singular values of the whitened cross-moment are \(\sigma_j\), the exact rank-\(r\) captured target energy is

\[
E_r=\sum_{j=1}^{r}\sigma_j^2,
\]

and the optimal centered reconstruction loss is

\[
\mathcal L_r
=
\operatorname{tr}(G_{KK})-E_r.
\]

### All-layer pre-RoPE results

| Rank | Captured total K energy | Captured V-predictable K energy | Held-out centered K MSE | Paired-query score NMSE |
|---:|---:|---:|---:|---:|
| 4 | 20.47% | 49.93% | 0.797411 | 0.155089 |
| 8 | 26.89% | 65.50% | 0.734022 | 0.143095 |
| 16 | 33.53% | 81.10% | 0.668312 | 0.132624 |
| 24 | 36.96% | 89.04% | 0.634376 | 0.128241 |
| 32 | 38.92% | 93.58% | 0.614999 | 0.126026 |
| 48 | 40.87% | 98.04% | 0.595744 | 0.124123 |
| 64 | 41.58% | 99.63% | 0.588718 | 0.123448 |
| 80 | 41.74% | 100.00% | 0.587072 | 0.123289 |

An unrestricted affine V80 predictor explains only 41.74% of centered pre-RoPE K energy. Thus, the strong claim that V80 linearly reconstructs most of K is false. However, rank 16 captures 81.10% of everything that is linearly predictable from V80, and increasing the rank from 16 to 80 changes paired-query score NMSE only from 0.132624 to 0.123289. Base16 is therefore close to the useful low-rank conditional-mean knee.

### Pre-RoPE versus direct post-RoPE prediction

| Rank | Target | Captured total K energy | Paired-query score NMSE |
|---:|:---|---:|---:|
| 16 | pre-RoPE, followed by exact RoPE | 33.53% | 0.132624 |
| 16 | direct post-RoPE | 13.02% | 0.417762 |
| 80 | pre-RoPE, followed by exact RoPE | 41.74% | 0.123289 |
| 80 | direct post-RoPE | 15.11% | 0.409934 |

At rank 16, the direct post-RoPE score error is about 3.15 times the pre-RoPE-plus-exact-RoPE error. The pre-RoPE rank-16 captured-energy fraction is also stable across position buckets, remaining between 33.81% and 34.08% from positions 0 through 32K. This supports treating RoPE analytically instead of asking the learned predictor to absorb it.

### Layer variation

- Unrestricted pre-RoPE predictable fraction: mean 41.74%, ranging from 77.60% at layer 0 to 19.72% at layer 33.
- Rank-16 captured predictable fraction: mean 81.10%; the lowest observed value is 74.96% at layer 13.
- Rank-16 paired-query score NMSE: mean 0.132624, ranging from 0.010820 at layer 0 to 0.350340 at layer 33.

The all-layer average hides a large structural difference: early layers can be routed almost entirely by the Base16 conditional mean, while some late layers require information outside that conditional mean.

## Query-visible residual router

For each token, define the exact post-RoPE residual

\[
e_i
=
k_i^{\mathrm{post}}-\widehat k_i^{(0)}.
\]

For KV group \(g\), a rank-\(r\) token encoder stores

\[
z_{g,i}=e_{g,i}A_g,
\qquad
A_g\in\mathbb R^{128\times r}.
\]

For query head \(h\), the query-side coordinates are

\[
u_h(q)=qU_h,
\qquad
U_h\in\mathbb R^{128\times r}.
\]

The complete routing score is

\[
\widehat s_{h,i}
=
\frac{1}{\sqrt{128}}
\left[
q_h^\top\widehat k_{g,i}^{(0)}
+
(q_hU_h)z_{g,i}^\top
\right].
\]

Only the rank-\(r\) token code \(z_{g,i}\) is additional persistent per-token routing state. Base16 is generated from the already resident C1-V80 code, while the small query factors are shared model parameters.

The residual factors \(A_g\) and \(U_h\) are optimized by alternating BCD. The objective is a Page32 softmax-Fisher loss constructed from exact teacher attention after excluding the pinned sink page. Base16 and C1-V80 remain frozen.

## Residual-rank sweep

### Protocol

- Diagnostic layers: 0, 13, and 33.
- Base predictor: frozen Base16.
- Residual ranks: 0, 2, 4, 8, 12, 16, 24, and 32.
- Residual fit: 64 independent C4 documents of length 32,768.
- Residual validation: 16 independent C4 documents of length 32,768.
- Fresh evaluation: 4 additional C4 documents of length 32,768.
- Query sampling: final-token decode Query; 32 query heads per document.
- Page size: 32 tokens.
- Page0 is always pinned.
- Routed pages per KV group: 63 additional pages.
- Total selected tokens: 2,048 of 32,768, or 6.25%.
- BCD sweeps: 40.
- Hardware: three L40S GPUs.
- Slurm job: 8299968; all tasks completed successfully without non-finite results.

### Frozen Base16 reconstruction error

| Layer | Base16 post-RoPE relative K MSE |
|---:|---:|
| 0 | 0.007413 |
| 13 | 0.325821 |
| 33 | 0.457325 |

### Fresh routing results

#### Layer 0

| Residual rank | Validation Fisher NMSE | Selected attention mass | Non-sink page recall | Exact-refined output rel-MSE |
|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 0.860093 | 0.840278 | 0.001316 |
| 2 | 0.689526 | 0.860588 | 0.812996 | 0.001176 |
| 4 | 0.461405 | 0.861803 | 0.834325 | 0.001147 |
| 8 | 0.446679 | 0.864083 | 0.867063 | 0.001129 |
| 12 | 0.389740 | 0.864332 | 0.885913 | 0.001125 |
| 16 | 0.395713 | 0.864695 | 0.894841 | 0.001114 |
| 24 | 0.331940 | 0.864966 | 0.901290 | 0.001107 |
| 32 | 0.299701 | 0.865526 | 0.913690 | 0.001109 |

Layer 0 is already well represented by Base16. R24 changes selected attention mass by only 0.49 percentage points and reduces output relative MSE by 15.9%. Most of the increased exact page overlap corresponds to pages carrying little attention mass.

#### Layer 13

| Residual rank | Validation Fisher NMSE | Selected attention mass | Non-sink page recall | Exact-refined output rel-MSE |
|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 0.930427 | 0.648810 | 0.016654 |
| 2 | 1.116161 | 0.931660 | 0.650794 | 0.012554 |
| 4 | 0.967088 | 0.933212 | 0.687500 | 0.015262 |
| 8 | 0.831584 | 0.939671 | 0.710317 | 0.011848 |
| 12 | 0.692324 | 0.940347 | 0.735119 | 0.011436 |
| 16 | 0.587253 | 0.942773 | 0.746528 | 0.011740 |
| 24 | 0.426214 | 0.943630 | 0.775298 | 0.011486 |
| 32 | 0.352343 | 0.945612 | 0.790675 | 0.010696 |

Layer 13 is intermediate. R8 reduces output relative MSE by 28.9% relative to R0, while R32 reaches a 35.8% reduction. The small-rank validation Fisher curve is noisy: R2 is worse than the normalized R0 baseline and R4 provides little held-out Fisher gain, even though some fresh downstream metrics improve. Fit or validation Fisher loss alone does not completely determine sparse routing quality.

#### Layer 33

| Residual rank | Validation Fisher NMSE | Selected attention mass | Non-sink page recall | Exact-refined output rel-MSE |
|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 0.419835 | 0.615079 | 0.457695 |
| 2 | 0.092866 | 0.885259 | 0.709325 | 0.047690 |
| 4 | 0.105970 | 0.886517 | 0.709821 | 0.046109 |
| 8 | 0.103890 | 0.898177 | 0.748016 | 0.024357 |
| 12 | 0.083404 | 0.913232 | 0.798115 | 0.015584 |
| 16 | 0.069198 | 0.917404 | 0.814484 | 0.014017 |
| 24 | 0.056640 | 0.923966 | 0.846726 | 0.012567 |
| 32 | 0.050542 | 0.926379 | 0.857143 | 0.011787 |

Layer 33 cannot use Base16 alone as a viable router. R2 raises selected attention mass from 0.420 to 0.885 and cuts output relative MSE by 89.6%. R12 reaches 0.913 selected mass and a 96.6% error reduction. R32 reaches 0.926 selected mass and a 97.4% error reduction.

## Combined interpretation

The results support a two-part statement:

\[
\boxed{
\text{C1-V80 contains a useful low-rank conditional mean of pre-K,}
\quad
\text{but it does not contain all query-visible routing information.}
}
\]

Base16 efficiently extracts most of the K component that is linearly predictable from C1-V80. Its role is not full K reconstruction. The separate score residual represents the remaining directions that matter under the deployment Query and attention distribution.

The residual is especially effective in layer 33: a two-dimensional correction recovers most of the lost attention mass despite the large raw Base16 K reconstruction error. This shows why raw K MSE and routing quality are not equivalent. A residual direction can contain little total K energy while producing a large correction to \(q^\top k\) on the relevant Queries.

The need for residual capacity is strongly layer-dependent:

\[
\boxed{
\text{layer 0: R0--R2 is nearly sufficient,}
\quad
\text{layer 13: R8 captures most practical gain,}
\quad
\text{layer 33: R12 or more is beneficial.}
}
\]

The three-layer sweep therefore rejects a universal assumption that every layer should use the same residual rank. It supports adaptive per-layer residual capacity, but it does not by itself define an all-layer rank schedule.

## Scope of the evidence

- Predictive spectra cover all 36 layers, but the residual-rank sweep covers only layers 0, 13, and 33.
- The residual experiment uses C4 for fit, validation, and fresh evaluation.
- Each document contributes a final-token decode Query rather than all Queries.
- The fresh split contains four 32K documents, producing 128 query-head observations per layer.
- The reported output error uses exact selected K pages and the corresponding resident C1 Value payload; it is an attention-output oracle metric, not end-to-end PPL or task accuracy.
- The experiment validates the mathematical decomposition and its routing ceiling. It does not measure PCIe overlap, runtime kernel efficiency, or full-model generation latency.
