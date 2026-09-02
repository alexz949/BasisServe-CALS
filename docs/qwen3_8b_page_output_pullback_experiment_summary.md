# Qwen3-8B Page-Output Pullback Routing Experiment

## Scope

This experiment evaluated whether the resident C1 value payload could improve a
separate low-rank Key router without placing Value coordinates in the routing
cache. The evaluated design was:

- Qwen3-8B Base.
- C1-V80 as the resident value payload.
- A post-RoPE, K-only Route32 cache.
- Page64 candidate selection.
- A local output-pullback quadratic derived from the C1 value decoder.
- Page-Fisher as an optional stabilizing component.

The experiment did not reconstruct pre-RoPE Keys. Value information affected
the routing metric only; the Route32 cache coordinates remained entirely
K-derived.

## Mathematical Objective

For one query head, let

\[
s = \frac{Kq}{\sqrt{d_K}},
\qquad
p = \operatorname{softmax}(s),
\qquad
C = V E_V,
\]

where \(K\) is the exact post-RoPE Key matrix and \(C\) is the resident C1-V80
latent. With the head-specific C1 decoder \(D_h\), the decoded attention output
is

\[
y = D_h^\top C^\top p.
\]

The K-only router uses

\[
\widehat s
=
\frac{K E_K U_h^\top q}{\sqrt{d_K}},
\qquad
\Delta_h = E_K U_h^\top-I.
\]

For Page64, the teacher attention distribution defines page masses

\[
\rho_j = \sum_{i\in\mathcal P_j}p_i
\]

and teacher-conditional page representatives \(\bar K\) and \(\bar C\). With

\[
J_\rho = \operatorname{Diag}(\rho)-\rho\rho^\top,
\qquad
G_h^D = D_hD_h^\top,
\]

the K-side Page-output metric is

\[
H_{K,h}^{\mathrm{out}}
=
\bar K^\top J_\rho \bar C
G_h^D
\bar C^\top J_\rho \bar K.
\]

The corresponding local quadratic is

\[
\mathcal L_{\mathrm{out}}
=
\frac{1}{2d_K}
q^\top\Delta_h^\top
H_{K,h}^{\mathrm{out}}
\Delta_h q.
\]

The Page-Fisher component is

\[
H_{K,h}^{\mathrm{pageF}}
=
\bar K^\top J_\rho\bar K.
\]

Both components were collected as separate packed symmetric \(128\times128\)
Grams. Consequently, their weights could be changed during fitting without
repeating the 32K teacher forward pass.

The output metric is local and first-order around the teacher attention
distribution. It is block-diagonal across query heads: it measures each
head-specific decoded output but does not include cross-head output terms,
downstream MLP effects, or later Transformer layers.

## Calibration Protocol

The compact statistics used the following protocol:

| Setting | Value |
|---|---:|
| Fit documents | 64 independent C4 documents |
| Validation documents | 16 independent C4 documents |
| Sequence length | 32,768 tokens |
| Queries per document | 8 |
| Fit observations per query head | 512 |
| Validation observations per query head | 128 |
| Query span | Final 8,192 tokens |
| Query positions | 25,599; 26,623; 27,647; 28,671; 29,695; 30,719; 31,743; 32,767 |
| Page size | 64 tokens |
| Key convention | Exact post-RoPE K |
| Payload | Fixed C1-V80 encoder and decoder |

Each query used its own causal prefix. The teacher softmax distribution, page
masses, page Key representative, and page C1-payload representative were
therefore recomputed for every sampled query.

The teacher forward and Gram construction used BF16 model activations and FP32
statistics. The resulting fit and validation artifacts covered all 36 layers
and occupied approximately 48 GiB. They contained compact sufficient
statistics rather than token-level Jacobians or residual-stream output tables.

## Numerical Verification

The implementation was checked against explicit small-matrix constructions.
The verification covered:

- Equality between the compact K-side quadratic and the explicitly constructed
  decoded-output perturbation.
- Equality between the compact Page-Fisher Gram and the explicit
  \(J_\rho\) quadratic.
- Positive semidefiniteness of both metrics.
- Zero output-pullback loss when all decoded page payloads are identical.
- Correct use of a separate causal prefix for every sampled query.
- Monotonic conditional BCD updates on a synthetic K-only routing problem.

All nine relevant tests passed.

## Attempt 1: Pure Output Pullback

The first fit used only \(H^{\mathrm{out}}\), with no Page-Fisher component.
It used KQ-SVD Route32 initialization, deterministic BCD, FP64 PCG, relative
damping \(10^{-5}\), at most 70 PCG iterations, and 40 BCD sweeps.

The run was stopped after 12 representative layers because the validation gap
was already systematic.

| Aggregate over 12 completed layers | Relative loss |
|---|---:|
| Fit mean | 0.001679 |
| Validation mean | 0.283634 |
| Validation / fit | 168.9× |

Representative layers showed the same pattern:

| Layer | Fit | Validation |
|---:|---:|---:|
| 0 | 0.000432 | 0.058448 |
| 9 | 0.001444 | 0.246195 |
| 18 | 0.002245 | 0.286111 |
| 27 | 0.002665 | 0.424326 |

The pure objective could nearly eliminate calibration loss while retaining a
large independent-document loss. This matched the expected large nullspace of
the output metric: score perturbations that do not change the sampled decoded
payload output are weakly constrained.

## Metric-Scale Measurement

Before constructing a hybrid, the raw teacher energies of the two components
were compared. The ratio

\[
E_{\mathrm{out}}/E_{\mathrm{pageF}}
\]

varied from approximately 0.0077 in the earliest layer to a maximum of 242.2
in the latest layers. A single unnormalized global Page-Fisher weight would
therefore have represented very different effective regularization strengths
at different depths.

## Attempt 2: Teacher-Energy-Normalized Hybrid

The second fit normalized each component independently within every layer and
split:

\[
\widetilde H_{\mathrm{out}}
=
\frac{H_{\mathrm{out}}}{E_{\mathrm{out}}},
\qquad
\widetilde H_{\mathrm{pageF}}
=
\frac{H_{\mathrm{pageF}}}{E_{\mathrm{pageF}}}.
\]

The fitted metric was

\[
H_{\mathrm{hybrid}}
=
\widetilde H_{\mathrm{out}}
+
0.3\widetilde H_{\mathrm{pageF}}.
\]

The solver configuration otherwise remained unchanged: KQ-SVD Route32
initialization, FP64 deterministic BCD/PCG, 40 sweeps, relative damping
\(10^{-5}\), and at most 70 PCG iterations. Final routing factors were exported
in BF16.

All 36 layers completed. Every layer improved the combined validation metric
relative to its KQ-SVD initialization.

| Combined normalized metric | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| Initial validation | 6.183460 | 0.283467 | 98.502297 |
| Final fit | 0.004514 | 0.001714 | 0.007599 |
| Final validation | 0.219997 | 0.031539 | 0.513338 |

The final validation mean was substantially below the initialization, but the
validation-to-fit ratio remained approximately 48.7×.

## Component Decomposition

The normalized combined value hid very different generalization behavior in
its two components. Evaluating the final factors against each component
separately gave:

| Component | Initial fit | Final fit | Initial validation | Final validation | Validation / fit |
|---|---:|---:|---:|---:|---:|
| Output pullback | 6.659230 | 0.000971 | 7.081429 | 0.257288 | 264.8× |
| Page-Fisher | 3.159520 | 0.016335 | 3.190229 | 0.095674 | 5.86× |

Both components improved over initialization on all 36 layers. However, almost
all of the remaining fit/validation discrepancy came from the output-pullback
component. The Page-Fisher component generalized much more consistently across
independent documents.

## Independent 32K Routing Evaluation

The fitted router was evaluated on four independent 32K C4 documents that were
not used by either fit or validation. Each query head selected 32 Page64 pages,
corresponding to a nominal 2,048-token budget. Pages were then unioned across
the four query heads in each physical GQA group. The resulting physical token
count was therefore measured rather than assumed to be exactly 2,048.

The comparison used the same fixed C1-V80 payload and the same fresh direct
capture. The reference Page-Fisher K32 router used the same post-RoPE K-only
coordinates and KQ-SVD initialization.

| Fresh4×32K metric | Page-Fisher K32 | Normalized output-pullback hybrid | Difference |
|---|---:|---:|---:|
| Physical-page recall | **86.57%** | 79.20% | −7.37 pp |
| Attention-mass recall | **93.63%** | 92.97% | −0.66 pp |
| Exact-refined output relative MSE ↓ | **0.005890** | 0.007332 | +24.49% |
| Softmax-Fisher NMSE ↓ | **0.116030** | 0.200108 | +72.46% |
| Raw-score NMSE ↓ | **0.109315** | 0.371326 | +239.68% |
| Mean selected physical tokens | **3,325** | 3,449 | +3.73% |

The output-pullback hybrid used slightly more physical tokens after the GQA
union but produced lower page recall, lower attention-mass recall, and higher
exact-refined output error. Under the measured routing metrics, the Page-Fisher
K32 reference dominated the normalized output-pullback hybrid.

## Runtime Record

All large stages used four L40S GPUs in layer-sharded parallel execution.

| Stage | Wall time | Status |
|---|---:|---|
| 32K compact-statistics collection | 8 min 53 s | Complete, 36/36 layers |
| Normalized hybrid fitting | 13 min 23 s | Complete, 36/36 layers |
| Component and fresh-routing evaluation | 29 s | Complete, 36/36 layers |

## Final Experimental Finding

The experiment successfully implemented a deterministic, forward-only,
C1-aware Page-output pullback for a post-RoPE K-only Route32 router. Compact
sufficient statistics and deterministic BCD/PCG produced very low calibration
loss, and teacher-energy normalization made a single hybrid weight consistent
across depth.

The independent results nevertheless showed that the output-pullback component
had a severe calibration-to-validation gap. The normalized hybrid improved over
its KQ-SVD initialization but did not improve over the existing Page-Fisher K32
router. It was worse in physical-page recall, attention-mass recall,
exact-refined output error, Fisher NMSE, and raw-score NMSE while selecting more
physical tokens.
