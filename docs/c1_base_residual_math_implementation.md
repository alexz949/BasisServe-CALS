# C1 Base + Residual Routing: Mathematics and Implementation

This document describes the implemented Qwen3-8B-Base routing path, its offline fitting procedure, and its online computation. The concrete reference is **C1-V80 + Q8-fitted Base16 + Q16 Page-Fisher residual R8**, with Page32 and a 2048-token selection budget. Other query-sampling experiments do not change the mathematical structure below.

The tested Base uses **Adam on captured activations**. The residual uses **block coordinate descent (BCD), with preconditioned conjugate gradient (PCG) inside each block update**. The shared-metric closed-form Base discussed more recently is a distinct alternative; it must not be confused with the Base used in the reported routing experiments.

## 1. What the two components do

The decomposition is

\[
\boxed{
\text{resident C1-V}
\longrightarrow \text{predicted pre-RoPE K}
\longrightarrow \text{predicted post-RoPE K}
\quad + \quad \text{low-rank post-RoPE residual score}.
}
\]

Base exploits the part of Key that can be predicted from the resident Value latent. Residual stores additional information derived from the actual Key, rather than trying to obtain all Key information from Value.

Together they predict routing scores. After page selection, the attention computation uses **selected exact K and the resident C1 Value codes**. The approximate routing scores are not reused as the final attention logits.

The fitted parameters are fixed during inference. “Q-aware” means that calibration queries enter the fitting objective, not that Base is refitted for every decoding query.

## 2. Notation and dimensions

The layer index is omitted below. Each layer has separate parameters. Token and feature vectors use row-vector notation unless explicitly stated otherwise.

| Symbol | Meaning | Reference dimensions |
| --- | --- | --- |
| \(g\) | KV/GQA group | 8 groups |
| \(h\) | Query head; \(g(h)\) is its KV group | 32 heads; 4 per group |
| \(t\) | Key token position | Up to the current context length |
| \(j\) | Query position | \(t\le j\) for causal attention |
| \(v_{t,g}\) | Raw Value | \(1\times128\) |
| \(c_{t,g}\) | Frozen C1 Value code | \(1\times80\) |
| \(k^{pre}_{t,g},k^{post}_{t,g}\) | Key before/after RoPE | \(1\times128\) |
| \(q_{j,h}\) | Actual post-RoPE query used by attention | \(1\times128\) |
| \(A_g,B_g,b_g\) | Base factors and bias | \(80\times16,16\times128,1\times128\) |
| \(E_g\) | Residual token encoder | \(128\times8\) |
| \(U_h\) | Residual query projector | \(128\times8\) |
| \(P\) | Page size | 32 tokens |
| \(B_{tok}\) | Total selected-token budget | 2048 tokens |

There are 36 transformer layers. Base and the residual token encoder are local to a KV group. The four query heads in that group have separate residual query projectors.

The payload encoder is frozen:

\[
c_{t,g}=v_{t,g}F_g,\qquad F_g\in\mathbb R^{128\times80}.
\]

Neither the C1 encoder/decoder nor the original model weights are updated by the Base/residual fitting stages described here.

## 3. Calibration data and actual query coverage

| Item | Base stage | Reference residual stage |
| --- | --- | --- |
| Fit windows | C4 windows 0–63 | Same 64 windows |
| Validation windows | C4 windows 64–79 | Same 16 windows |
| Window length | 32768 tokens | 32768 tokens |
| Query observations/window | 8 positions, all 32 heads | 16 positions, all 32 heads |
| Key positions for each query | Every causal token from position 32 through \(j\) | Same causal/non-sink rule |
| Purpose of validation | Select Base epoch | Diagnostic loss only |
| Rank | Base16 | Uniform residual R8 |

These C4 windows pack eight 4096-token source windows without separators; they are not native contiguous 32K documents. There are 2,097,152 fit tokens and 524,288 validation tokens. Query sampling limits the observed query positions, not the number of captured Key rows.

Base query positions, zero-based:

\[
J_{base}=\{25599,26623,27647,28671,29695,30719,31743,32767\}.
\]

Residual Q16 positions, zero-based:

\[
\begin{aligned}
J_{res}=\{&25087,25599,26111,26623,27135,27647,28159,28671,\\
&29183,29695,30207,30719,31231,31743,32255,32767\}.
\end{aligned}
\]

Both sets are in the terminal portion of the window, not uniform across the entire 32K range. Q16 contains the Q8 positions. All four Q heads in each group remain separate observations; they are not replaced by their mean.

The captures come from dense-teacher forward passes. Raw V and exact post-RoPE K are retained, and current C1-V80 codes are computed offline using the frozen encoder. Repeated fitting epochs operate on these captures, not on repeated full-model forward/backward passes.

The original Q8 Base bank used a terminal-Q1 residual. The reference Q16 checkpoint freezes that same Q8 Base and **refits the residual** with 16 queries/window. A residual-stage manifest saying `base_optimizer=None` means “Base was not updated in this stage,” not “Base was never fitted with Adam.”

## 4. Base: prediction and objective

### 4.1 Per-token affine low-rank prediction

For every token in a group,

\[
\widehat k^{pre}_{t,g}=c_{t,g}A_gB_g+b_g,
\qquad \operatorname{rank}(A_gB_g)\le16.
\]

The predicted pre-RoPE Key is then rotated at its actual Key position:

\[
\bar k^{post}_{t,g}=\operatorname{RoPE}_t(\widehat k^{pre}_{t,g}).
\]

Rank16 is the bottleneck of the map from C80 to K128. Its output is still a 128-dimensional predicted Key, not a 16-dimensional final attention Key. Base does not mix different tokens or different KV groups.

### 4.2 Actual causal, position-aware raw-QK loss

Define exact and Base-only scores:

\[
s_{w,j,h,t}=\frac{q_{w,j,h}(k^{post}_{w,t,g(h)})^\top}{\sqrt{128}},
\qquad
\widehat s^{base}_{w,j,h,t}=\frac{q_{w,j,h}(\bar k^{post}_{w,t,g(h)})^\top}{\sqrt{128}}.
\]

The implemented objective is

\[
\boxed{
\mathcal L_{base}=\frac12
\sum_w\sum_{j\in J_{base}}\sum_h\sum_{t=32}^{j}
\left(\widehat s^{base}_{w,j,h,t}-s_{w,j,h,t}\right)^2.
}
\]

It preserves each sampled query, its position, and its causal prefix. It is neither raw unweighted K reconstruction nor a softmax/Page-Fisher objective. It does not directly optimize page recall, selected-support attention output, or terminal language-model KL.

The first 32 tokens are excluded because their page is pinned during routing. All other causally eligible K rows contribute for each sampled Q; there is no corresponding 8-token Key subsample.

### 4.3 Why pre-RoPE prediction can still be position-aware

For this paragraph use column vectors, and let \(R_t\) denote the RoPE matrix. Write the pre-RoPE prediction error as \(e_{w,t,g}\). The same loss can be expressed as

\[
\mathcal L_{base}=\frac12\sum_{w,t,g}
e_{w,t,g}^{\top}H_{w,t,g}e_{w,t,g},
\]

\[
\boxed{
H_{w,t,g}=\frac1{128}R_t^\top
\left[
\sum_{\substack{j\in J_{base}\\j\ge t}}
\sum_{h:g(h)=g}q_{w,j,h}q_{w,j,h}^\top
\right]R_t.
}
\]

Thus the **output coordinate system** is pre-RoPE, while the **error metric** depends on position and observed post-RoPE queries. Different tokens generally have different metrics because both RoPE and causal query eligibility vary.

The code evaluates explicit score errors; it does not materialize one \(128\times128\) metric for every token.

### 4.4 Actual numerical solver

The tested Base starts from the MSE-Base16 factors in `q8_residual_kl_bank`, balances the two factor matrices, and optimizes \(A_g,B_g,b_g\) using Adam.

| Setting | Value |
| --- | --- |
| Factor arithmetic/parameters | FP32 |
| Maximum epochs | 12 |
| Documents accumulated per optimizer step | 4 |
| Factor learning rate | 0.002 |
| Bias learning rate | 0.0005 |
| Gradient-norm clipping | 1 |
| Early-stopping patience | 4 epochs |
| Shuffle seed | 73 |
| Checkpoint selection | Lowest validation raw-QK NMSE, including epoch 0 |

Each document's loss is backpropagated only through the small Base factor computation. Four documents contribute gradients before an optimizer step. The implementation applies a fixed normalization derived from the initial fit exact-score energy.

The reported local metric is

\[
\operatorname{NMSE}_{QK}=
\frac{\sum(\widehat s^{base}-s)^2}{\sum s^2}.
\]

This is uncentered score-energy normalization, not centered regression \(R^2\). Validation chooses the epoch, so this validation set is not an untouched final test.

All 36 layers in the reference Base bank selected an epoch greater than zero. Consequently, this checkpoint is genuinely Adam-fitted, not simply the closed-form initializer saved under a different name. Under the requested definition excluding Adam, **this tested Base does not qualify as the desired training-free Base**. It nevertheless requires no full-model backward pass.

## 5. Residual: what is encoded

Freeze Base, then form the token innovation in **post-RoPE coordinates**:

\[
\boxed{\epsilon_{t,g}=k^{post}_{t,g}-\bar k^{post}_{t,g}.}
\]

The residual code and query code are

\[
z_{t,g}=\epsilon_{t,g}E_g\in\mathbb R^{1\times8},
\qquad a_{j,h}=q_{j,h}U_h\in\mathbb R^{1\times8}.
\]

The complete routing-score approximation is

\[
\boxed{
\widehat s_{j,h,t}=
\frac{q_{j,h}(\bar k^{post}_{t,g(h)})^\top
 +(q_{j,h}U_h)(\epsilon_{t,g(h)}E_{g(h)})^\top}
{\sqrt{128}}.
}
\]

Only the second term is rank8. The scale remains \(1/\sqrt{128}\), not \(1/\sqrt8\).

The residual is not predicted solely from C1-V: it uses the exact Key when that token is appended. It preserves information unavailable to Base. The same residual code is shared by four Q heads, but \(U_h\) differs by head. Therefore the implementation primarily represents a low-rank **score correction**, not a single universally shared reconstructed K vector.

The identity \(K=\bar K+\epsilon\) is exact by construction. However, the fitted rank-constrained, Q-weighted Base is not an unrestricted Euclidean least-squares projector. Its residual must not automatically be identified with an orthogonal projection residual or a Schur-complement conditional covariance.

## 6. Page-Fisher: from exact teacher attention to compact statistics

The following construction is repeated separately for every fit window, sampled query position, and Q head. Let \(e\) index a window/query-position pair; the head index remains explicit.

### 6.1 Exact teacher distribution over non-sink causal tokens

Restrict to token positions \(32\le t\le j\), then compute

\[
p_{e,h,t}=\frac{\exp(s_{e,h,t})}
{\sum_{u=32}^{j}\exp(s_{e,h,u})}.
\]

The pinned prefix is removed **before** softmax normalization. These probabilities are based on exact K, not on Base or the current residual approximation.

For each Page32 block \(p\), define its mass and teacher-weighted residual representative:

\[
m_{e,h,p}=\sum_{t\in p}p_{e,h,t},
\qquad
\rho_{e,h,p}=\frac{\sum_{t\in p}p_{e,h,t}\epsilon_{t,g(h)}}{m_{e,h,p}}.
\]

This is not an unweighted page mean. A partial final page contains only valid causal tokens; padding receives zero probability. The implementation clamps a vanishing mass denominator for numerical safety.

### 6.2 Centered residual-feature Gram

Define

\[
\mu_{e,h}=\sum_p m_{e,h,p}\rho_{e,h,p},
\]

\[
\boxed{
G_{e,h}=\sum_p m_{e,h,p}
(\rho_{e,h,p}-\mu_{e,h})^\top
(\rho_{e,h,p}-\mu_{e,h})
\in\mathbb R^{128\times128}.
}
\]

This covariance is the compact Page-Fisher statistic used by the solver. The code symmetrizes it numerically. Page masses, within-page conditional weights, and feature centering are all included.

### 6.3 Why this is normalization-aware

For page logits, softmax has Fisher matrix

\[
F_{page}=\operatorname{diag}(m)-mm^\top.
\]

For a small token-score perturbation \(\Delta s_t\), the first-order page-log-sum-exp perturbation is

\[
\Delta\ell_p\approx
\sum_{t\in p}\frac{p_t}{m_p}\Delta s_t.
\]

The local teacher-to-proxy page-distribution KL is consequently approximated by

\[
\frac12\Delta\ell^\top F_{page}\Delta\ell
=\frac12\sum_p m_p
\left(\Delta\ell_p-\sum_{p'}m_{p'}\Delta\ell_{p'}\right)^2.
\]

Centering removes a common additive logit shift, which cannot change a normalized distribution. This is the normalization term absent from uncentered numerator-only objectives.

### 6.4 Implemented residual objective

With row queries define

\[
v_{e,h}=q_{e,h}(U_hE_{g(h)}^\top-I_{128}).
\]

Then the fitting objective is

\[
\boxed{
\mathcal L_{res}=\frac1{2\cdot128}
\sum_{e,h}v_{e,h}G_{e,h}v_{e,h}^\top.
}
\]

This is an exact quadratic in either factor block when the other block and teacher statistics are fixed. Its interpretation as page-distribution KL is a **local approximation**: it linearizes page log-sum-exp and uses the teacher Fisher curvature. It does not optimize exact finite-error softmax KL, hard Top-B recall, or the final C1-decoded output directly.

The normalization used for residual Fisher NMSE is

\[
\mathcal E_{res}=\frac1{2\cdot128}\sum_{e,h}q_{e,h}G_{e,h}q_{e,h}^\top,
\qquad
\operatorname{NMSE}_{Fisher}=\mathcal L_{res}/\mathcal E_{res}.
\]

For nonzero teacher energy, a zero residual correction has NMSE 1. This metric is neither raw-K reconstruction NMSE nor attention-mass recall. Its denominator depends on Base, so normalized residual losses from different Bases do not necessarily have a common absolute scale.

## 7. Residual fitting: BCD with matrix-free linear solves

### 7.1 Statistics and initialization

Q16 gives \(64\times16=1024\) fit examples per head and \(16\times16=256\) validation examples per head. The builder preserves every Q/Gram pairing; it does not average Q vectors.

After teacher statistics are built, the BCD loop needs queries and \(128\times128\) Grams, not the full token sequences. Raw K/V captures are not duplicated 16 times. The fit Grams alone occupy 2 GiB per layer in FP32; validation Grams occupy 0.5 GiB. Additional captures, factors, and working buffers are separate.

Initialization takes the leading eight eigenvectors of the sum of the fit Grams over examples and the four heads of each group as \(E_g\). Initially \(U_h\) is copied from its group's encoder. This residual initialization is not the earlier pairwise KQ-SVD checkpoint.

### 7.2 Query-projector update

Fix \(E_g\). Define \(D_{e,h}=E_g^\top G_{e,h}E_g\). Omitting the common positive attention-scale factor, the undamped normal equation for each \(U_h\) is

\[
\boxed{
\sum_e(q_{e,h}^\top q_{e,h})U_hD_{e,h}
=\sum_e q_{e,h}^\top(q_{e,h}G_{e,h}E_g).
}
\]

There are 32 independent query-projector blocks per layer. Each unknown is a \(128\times8\) matrix.

### 7.3 Residual-encoder update

Fix \(U_h\), and let \(a_{e,h}=q_{e,h}U_h\). For each group, the undamped normal equation is

\[
\boxed{
\sum_{h:g(h)=g}\sum_e
G_{e,h}E_g(a_{e,h}^\top a_{e,h})
=\sum_{h:g(h)=g}\sum_e
G_{e,h}q_{e,h}^\top a_{e,h}.
}
\]

There are eight encoder blocks per layer. Each block combines its group's four Q heads and again has \(128\times8\) unknowns.

### 7.4 Numerical implementation

The solver does not explicitly construct the \(1024\times1024\) matrix obtained by vectorizing a block. Instead, it implements the left-hand-side matrix operator and applies PCG.

| Setting | Value |
| --- | --- |
| Alternating sweeps | 40 |
| Sweep order | Refit all query projectors, then all residual encoders |
| After final sweep | Refit query projectors once more |
| Relative damping | \(10^{-5}\), scaled by the block operator's diagonal magnitude |
| PCG relative tolerance | \(10^{-5}\) |
| PCG maximum iterations/block | 100 |
| Preconditioner | Diagonal |
| Current statistics/factor/PCG arithmetic | FP32 |
| Residual checkpoint choice | Fixed final sweep; validation is diagnostic |

The practical system is \((\mathcal A+\lambda I)X=\mathrm{RHS}\), with block-specific damping. PCG starts from zero and records convergence diagnostics. A tolerance and iteration cap are not an exact symbolic solve, and this document does not assert that every block reached its tolerance.

The procedure uses no Adam and no model backward pass. It is nevertheless iterative: an outer BCD loop contains inner PCG solves. It should be described as **BCD with iterative linear-system solves**, not “one closed-form SVD.” Numerical damping and inexact solves also mean that ideal undamped block-minimization guarantees should not be silently attributed to every recorded update.

Base is frozen throughout this stage. Changing Base changes \(\epsilon\), hence the residual Grams and optimal \(E,U\); the old residual factors cannot be assumed fitted for the new Base.

## 8. Online implementation: append, route, then exact selected attention

### 8.1 When a token enters the cache

The actual Key is available when the model computes the new token. The implementation computes

\[
c_t\rightarrow c_tA_gB_g+b_g
\rightarrow\bar k_t^{post},
\qquad
z_t=(k_t^{post}-\bar k_t^{post})E_g.
\]

The native conditional-routing implementation materializes the sidecar

\[
\boxed{d_{t,g}=[\bar k_{t,g}^{post},z_{t,g}]\in\mathbb R^{136}.}
\]

The corresponding query projector represents

\[
\widetilde q_{j,h}=[q_{j,h},q_{j,h}U_h],
\qquad
\widehat s_{j,h,t}=\widetilde q_{j,h}d_{t,g(h)}^\top/\sqrt{128}.
\]

The prefix sidecar is built once, then appended incrementally. Computing metadata once does not mean the routing scores need to be computed only once: each new Q changes those scores.

### 8.2 Page selection

For each head, compute proxy page log-sum-exp over valid causal tokens:

\[
\widehat\ell_{j,h,p}=\log\sum_{t\in p}\exp(\widehat s_{j,h,t}).
\]

Exclude pinned pages and normalize separately per Q head:

\[
\widehat m_{j,h,p}=\operatorname{softmax}_{p\notin\mathrm{pinned}}
(\widehat\ell_{j,h,p}).
\]

The group score is

\[
u_{j,g,p}=\max_{h:g(h)=g}\widehat m_{j,h,p}.
\]

For a sufficiently long prefix, Page32/B2048 selects one pinned page plus the 63 highest-scoring non-pinned pages: 64 pages and at most 2048 valid tokens per group. Padded/future tokens are masked. The budget includes the pinned page.

This is a fixed shared page set per GQA group, **not** a union of four independently selected 2048-token head budgets. It is page-level selection, not exact token Top-2048. Page metadata are not merely page means: the reference path first computes per-token proxy scores and then aggregates them.

### 8.3 Final selected attention

Let \(S_{j,g}\) be the selected token set. Recompute scores using exact selected K:

\[
\alpha_{j,h,t}=
\frac{\exp(q_{j,h}(k^{post}_{t,g(h)})^\top/\sqrt{128})}
{\sum_{u\in S_{j,g(h)}}\exp(q_{j,h}(k^{post}_{u,g(h)})^\top/\sqrt{128})},
\quad t\in S_{j,g(h)}.
\]

Accumulate the resident C1 payload and apply the existing C1 output decoding path:

\[
o^{latent}_{j,h}=\sum_{t\in S_{j,g(h)}}\alpha_{j,h,t}c_{t,g(h)}.
\]

Attention is exact on this selected support with respect to K and its softmax. It is not equal to full-support attention, and its Value payload is C1-compressed rather than dense V128.

The referenced accuracy runs use full-support C1 prefill and native BF16 selected decode. Exact K remains GPU-resident in those runs; they are not measurements of CPU-to-GPU Key offload. First-two-full-layer variants bypass routing in those two layers without changing the Base/residual fitting definition.

## 9. Computation and storage accounting

### 9.1 Arithmetic structure

Counts below are algebraic multiply-accumulates (MACs), not measured latency; one MAC is conventionally two FLOPs. Actual contraction ordering, fusion, GQA reuse, and memory traffic depend on the backend.

| Operation | Per-token or per-query cost |
| --- | --- |
| Factorized Base append, per KV group | \(80\cdot16+16\cdot128=3328\) MACs |
| Residual encoding at append, per KV group | \(128\cdot8=1024\) MACs |
| Residual query projection, per Q head | \(128\cdot8=1024\) MACs |
| Materialized-sidecar routing scan, per Q head | \(L(128+8)=136L\) MACs |
| Proxy page-LSE reductions | Linear in the scanned tokens per head |
| Page selection | Operates on approximately \(L/32\) page scores/group |
| Selected exact-QK plus latent-V accumulation, per Q head | Approximately \(B_{tok}(128+80)\) MACs |

Bias, RoPE, subtraction, softmax, indexing, and C1 output decoding are additional operations. The append counts assume the mathematical two-factor Base contraction; they are not a claim about the optimized order chosen for every tensor expression.

The reference routing scan remains **\(O(L)\)**. In particular, the materialized Base128+R8 implementation scans 136 features per token, not just eight residual features. Sparse final attention does not by itself make routing an \(O(B_{tok})\) operation.

### 9.2 Actual cached tensors versus the intended small sidecar

For BF16 storage, per token and KV group:

| Tensor | Scalars | Bytes |
| --- | ---: | ---: |
| Exact K128 | 128 | 256 |
| Resident C1-V80 | 80 | 160 |
| Materialized predicted post-RoPE Base K128 | 128 | 256 |
| Residual code R8 | 8 | 16 |
| Reference cache tensor subtotal | 344 | 688 |

This subtotal excludes temporary buffers, allocator overhead, model weights, and routing/factor metadata. In the reference accuracy path all four listed tensors are GPU-resident.

The architectural motivation “derive Base from C1-V and add only an R8 code” is different from this materialized implementation. C80+R8 alone would be 176 bytes/token/group, but that is **not** the total GPU cache measured by these accuracy runs; exact-K placement, Base materialization or recomputation, and selected-Key staging must also be specified.

### 9.3 Fitted parameter storage

The five stored FP32 tensors per layer are:

| Checkpoint key | Shape | Scalars |
| --- | --- | ---: |
| `base_left_b16` | `[8,80,16]` | 10240 |
| `base_right_b16` | `[8,16,128]` | 16384 |
| `base_bias_b16` | `[8,128]` | 1024 |
| `residual_encoder_b16_r8` | `[8,128,8]` | 8192 |
| `residual_query_b16_r8` | `[32,128,8]` | 32768 |
| Total | | 68608 |

These routing parameters occupy 268 KiB/layer, approximately 9.42 MiB over 36 layers, excluding C1 payload factors. They do not grow with context length. The native BF16 cache path casts the factors to the payload's runtime dtype when applying them.

## 10. Shared-metric closed-form Base: a separate alternative

The repository contains `fit_affine_metric_reduced_rank_map`, which solves a different objective with **one shared PSD Key-space metric** \(H\):

\[
\min_{\operatorname{rank}(W)\le16,b}
\left\|(CW+\mathbf1b-K^{pre})H^{1/2}\right\|_F^2.
\]

Center \(C,K^{pre}\), and write

\[
G_C=C_c^\top C_c,\qquad J=C_c^\top K_c^{pre},\qquad
T=G_C^{-1/2}JH^{1/2}.
\]

For nonsingular metrics, if \([T]_{16}\) is the rank16 truncated SVD,

\[
\boxed{
W=G_C^{-1/2}[T]_{16}H^{-1/2},\qquad
b=\overline K^{pre}-\overline C W.
}
\]

Singular cases use numerical pseudoinverses and retained eigenspaces. Directions in the nullspace of \(H\) are not determined by the objective; the helper returns a particular solution on the retained metric support. It uses whitening, eigendecomposition/SVD, and mean statistics rather than Adam.

This alternative still predicts **every token** through \(\widehat k_t^{pre}=c_tW+b\), followed by that token's RoPE. What is no longer per-token is the **fitting metric**: \(H\) replaces the distinct \(H_{w,t,g}\). It does not imply page-averaging Value tokens.

For example, averaging the correctly rotated position-specific metrics produces a shared-metric approximation, not the original exact causal/RoPE objective. Directly applying a post-RoPE query covariance to pre-RoPE Key errors also cannot be called the exact position-aware objective without the appropriate coordinate transformation.

The presence of this closed-form helper does **not** mean that the tested Q8 Base was solved this way. No replacement of the reference Adam Base by this alternative is established by the cited checkpoints.

## 11. Position-aware Base using only BCD: discussed, not the tested solver

The exact sampled-query Base objective can instead be written as a sum of small quadratic blocks. In column notation,

\[
\widehat k_t=b+\sum_{m=1}^{16}(a_m^\top c_t)v_m.
\]

Let \(r_t^{(-m)}=k_t-b-\sum_{n\ne m}(a_n^\top c_t)v_n\), with \(H_t\) denoting the actual window/position-specific metric. Holding \(v_m\) fixed gives an 80-dimensional normal equation:

\[
\left[\sum_t(v_m^\top H_t v_m)c_tc_t^\top\right]a_m
=\sum_t c_t\,v_m^\top H_t r_t^{(-m)}.
\]

Holding \(a_m\) fixed and writing \(z_t=a_m^\top c_t\) gives a 128-dimensional equation:

\[
\left[\sum_t z_t^2H_t\right]v_m
=\sum_t z_tH_t r_t^{(-m)}.
\]

Bias has a similar 128-dimensional solve. Sums include all relevant windows and positions. Direct factorizations or pseudoinverses could solve these blocks without Adam or an inner PCG loop. Exact block minimization concerns the fixed sampled-query objective, not a guarantee of the globally optimal rank16 map. Accumulating these statistics is still computational work.

This describes the mathematical BCD-only alternative discussed with the user. It is not the optimizer that generated `q8_qbase_fisher_bank`.

## 12. Precise status distinctions

| Question | Answer for the reference checkpoint/path |
| --- | --- |
| Is Base per-token? | Yes: each token's C80 predicts its own pre-K128. |
| Does Base preserve Q positions during fitting? | Yes, for eight sampled positions/window, not all Q positions. |
| Is Base currently closed-form? | No; the tested Q8 Base uses Adam after an MSE-Base initializer. |
| Is residual also pre-RoPE? | No; it encodes exact post-K minus rotated Base prediction. |
| Is residual fitted with all Q? | No; this reference uses 16 sampled positions/window, with all heads. |
| Is Page-Fisher exact terminal KL? | No; it is a fixed-teacher local page-distribution quadratic. |
| Is residual fit one SVD? | No; eigenvector initialization followed by BCD with PCG. |
| Is Base jointly updated with residual? | No; Base is frozen during residual fitting. |
| Is residual rank adaptively allocated here? | No; uniform R8. |
| Is only R8 metadata materialized online? | No; the reference sidecar contains Base128+R8. |
| Does final attention use approximate K? | It recomputes QK using selected exact K. |
| Does the accuracy path measure CPU offload? | No; its exact K remains GPU-resident. |
| Is Page16 residual already represented here? | No; the referenced fitted objective and checkpoints use Page32. |

Changing page size changes the teacher page representatives and their Fisher Grams. Keeping a 32-token pinned prefix with Page16 would mean two pinned/excluded pages, not one. Base's raw-QK objective depends on this excluded token range but does not otherwise aggregate tokens into pages; the residual objective explicitly does.

## 13. Source and artifact map

All paths below are relative to the repository and identify the inspected implementation, not execution commands.

| Component | Source |
| --- | --- |
| Q8 Base fitting entry and configuration | [fit_qwen3_8b_qaware_base_fisher_bank.py](../evaluation/fit_qwen3_8b_qaware_base_fisher_bank.py) |
| Explicit causal/RoPE raw-QK Base loss | [eval_qwen3_8b_v80_exact_query_weighted_rrr.py](../evaluation/eval_qwen3_8b_v80_exact_query_weighted_rrr.py), `_query_weighted_document_loss` |
| Actual Adam Base loop | [eval_qwen3_8b_v80_pre_rope_fisher_base.py](../evaluation/eval_qwen3_8b_v80_pre_rope_fisher_base.py), `_train_base` |
| Multi-query residual-only refit | [fit_qwen3_8b_q8_fisher_residual.py](../evaluation/fit_qwen3_8b_q8_fisher_residual.py) |
| Residual formation/statistics and initialization | [eval_qwen3_8b_v80_conditional_residual_router.py](../evaluation/eval_qwen3_8b_v80_conditional_residual_router.py) |
| Page-Fisher Gram, sidecar, closed-form RRR helper | [c1_v_conditional_k_router.py](../basisserve/core/c1_v_conditional_k_router.py) |
| BCD query/encoder updates | [gqa_joint_routing_payload_s80_ablation.py](../basisserve/core/gqa_joint_routing_payload_s80_ablation.py) |
| Compact Fisher objective | [gqa_joint_routing_payload_s80_fisher.py](../basisserve/core/gqa_joint_routing_payload_s80_fisher.py) |
| Matrix-free PCG | [gqa_routed_ov_joint.py](../basisserve/core/gqa_routed_ov_joint.py), `conjugate_gradient_matrix` |
| Shared-group page selection and selected attention | [c1_conditional_page_attention.py](../basisserve/core/c1_conditional_page_attention.py) |
| Q16 capture positions | [capture_qwen3_8b_q16.py](../scripts/capture_qwen3_8b_q16.py) |

Reference checkpoints:

- Frozen C1 payload: `results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6`.
- Q8 Adam-fitted Base source: `results/checkpoints/q8_qbase_fisher_bank`.
- Frozen Q8 Base + Q16-fitted R8 residual: `results/checkpoints/q8_qbase_fisher16_r8`.

Related records:

- [Q8 Base fitting protocol and audit](q8_qaware_base_fisher_protocol.md).
- [Q16 residual-only fitting protocol and results](q8_residual_fisher16_protocol.md).
- [Loki, QUEST, and C1 routing comparison summary](loki_quest_c1_routing_summary.md).

This document records mathematics and implementation. Its preparation does not refit a checkpoint, change a routing kernel, or launch an evaluation.
