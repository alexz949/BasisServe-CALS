# Qwen3-8B Raw-V128 versus C1-V80 Pre-K Predictability

Follow-up scope correction: the results below measure same-group Value-to-Key prediction. The completed [four-ceiling experiment](../qwen3_8b_four_v_pre_k_ceilings_64f16h_32k/summary.md) finds layer-33 fit explained fractions of 25.78% for local raw V and 66.63% for all raw V groups (held-out: 25.05% and 64.66%). Therefore the earlier interpretation below that most missing Key variance is intrinsic to Values as a whole is too broad. A substantial part is recoverable across groups; approximately 33.37%, rather than 74.22%, remains unexplained by all raw V under the fit-split affine hypothesis. The numerical local results are unchanged.

## Configuration

- Model: BF16 Qwen3-8B-Base.
- Target: pre-RoPE K128 from the same layer and KV group.
- Inputs: raw V128 versus the fixed C1-V80 latent derived from that V.
- Fit split: 64 independent C4 windows of length 32768.
- Held-out split: 16 independent C4 windows of length 32768.
- Layers: all 36 transformer layers; eight KV groups per layer.
- Predictor: unrestricted affine regression, rank128 for raw V and rank80 for C1-V80.
- Environment: `basis` Python executable, PyTorch 2.6.0+cu124.
- Hardware: four NVIDIA L40S GPUs on `lovelace`.

The comparison reuses the previously completed C1-V80 spectra and computes the raw-V128 control from the exact same activation windows. It streams existing captures and performs no model forward.

## Metric

For centered input (X\in\{V^{128},C^{80}\}) and target pre-RoPE K, unrestricted linear predictability is

\[
R^2_{\mathrm{linear}}
=
\frac{\left\|G_{XX}^{-1/2}G_{XK}\right\|_F^2}
{\operatorname{tr}(G_{KK})}.
\]

The fit predictable fraction reports this quantity on the 64-window fit split. Held-out centered MSE applies the fitted affine map to the separate 16-window split; (1-\mathrm{MSE}) is the corresponding held-out explained fraction.

## All-Layer Results

| Layer | Raw-V fit predictable K | C1-V80 fit predictable K | Raw-V held-out MSE | C1-V80 held-out MSE |
|---:|---:|---:|---:|---:|
| 0 | 0.808375 | 0.775993 | 0.193978 | 0.227909 |
| 1 | 0.611507 | 0.526461 | 0.394677 | 0.479694 |
| 2 | 0.485918 | 0.421230 | 0.519845 | 0.584524 |
| 3 | 0.518127 | 0.448116 | 0.487012 | 0.556797 |
| 4 | 0.494996 | 0.431524 | 0.511436 | 0.575288 |
| 5 | 0.533688 | 0.464698 | 0.469945 | 0.538977 |
| 6 | 0.514301 | 0.433634 | 0.488589 | 0.569580 |
| 7 | 0.480261 | 0.408559 | 0.522597 | 0.594206 |
| 8 | 0.474169 | 0.412491 | 0.528963 | 0.590515 |
| 9 | 0.410426 | 0.352455 | 0.591889 | 0.650351 |
| 10 | 0.476644 | 0.410114 | 0.527647 | 0.594827 |
| 11 | 0.504386 | 0.444603 | 0.499924 | 0.559736 |
| 12 | 0.561455 | 0.499277 | 0.442396 | 0.505389 |
| 13 | 0.562411 | 0.501020 | 0.439621 | 0.500910 |
| 14 | 0.640009 | 0.580137 | 0.361971 | 0.422295 |
| 15 | 0.603966 | 0.551231 | 0.397804 | 0.450459 |
| 16 | 0.476712 | 0.417172 | 0.525871 | 0.585896 |
| 17 | 0.625641 | 0.573445 | 0.376910 | 0.429594 |
| 18 | 0.582391 | 0.532519 | 0.418495 | 0.468597 |
| 19 | 0.530702 | 0.474105 | 0.472276 | 0.528752 |
| 20 | 0.602313 | 0.547307 | 0.399360 | 0.454484 |
| 21 | 0.597894 | 0.546674 | 0.404980 | 0.456034 |
| 22 | 0.480493 | 0.419190 | 0.523921 | 0.585316 |
| 23 | 0.492923 | 0.423333 | 0.511670 | 0.581207 |
| 24 | 0.407901 | 0.337049 | 0.597146 | 0.668455 |
| 25 | 0.450724 | 0.399170 | 0.554271 | 0.605497 |
| 26 | 0.325653 | 0.272487 | 0.679993 | 0.732927 |
| 27 | 0.389890 | 0.335031 | 0.615911 | 0.670534 |
| 28 | 0.483947 | 0.427358 | 0.520515 | 0.576757 |
| 29 | 0.273118 | 0.205951 | 0.733398 | 0.799985 |
| 30 | 0.297386 | 0.247405 | 0.711136 | 0.759831 |
| 31 | 0.332360 | 0.260695 | 0.675189 | 0.746293 |
| 32 | 0.323083 | 0.265792 | 0.684296 | 0.740728 |
| 33 | 0.257791 | 0.197232 | 0.749540 | 0.808995 |
| 34 | 0.280067 | 0.233095 | 0.729235 | 0.774689 |
| 35 | 0.297073 | 0.251425 | 0.715109 | 0.758571 |

## Depth Summary

| Layers | Raw-V fit predictable K | C1-V80 fit predictable K | Raw-V held-out MSE | C1-V80 held-out MSE | C1 absolute fit loss |
|:---|---:|---:|---:|---:|---:|
| 0--11 | 0.526067 | 0.460823 | 0.478042 | 0.543534 | 0.065244 |
| 12--23 | 0.563076 | 0.505451 | 0.439606 | 0.497411 | 0.057625 |
| 24--35 | 0.343249 | 0.286057 | 0.663812 | 0.720272 | 0.057192 |
| All | 0.477464 | 0.417444 | 0.527153 | 0.587072 | 0.060020 |

- Spearman correlation between layer depth and raw-V fit predictability: -0.6471.
- Spearman correlation between layer depth and C1-V80 fit predictability: -0.6131.
- Spearman correlation between the raw-V and C1-V80 all-layer predictability curves: 0.9933.
- Mean C1 loss of fit predictable K energy: 6.00 percentage points.
- The C1 absolute fit loss is 6.52 points in layers 0--11, 5.76 points in layers 12--23, and 5.72 points in layers 24--35.

## Representative Layers

| Layer | Raw-V fit explained K | C1 fit explained K | Raw-V held-out explained K | C1 held-out explained K |
|---:|---:|---:|---:|---:|
| 0 | 80.84% | 77.60% | 80.60% | 77.21% |
| 13 | 56.24% | 50.10% | 56.04% | 49.91% |
| 33 | 25.78% | 19.72% | 25.05% | 19.10% |

## Dimension-Matched Rank-80 Control

To separate the 80-dimensional bottleneck from the orientation chosen by C1, the stored raw-V singular spectra also provide the best possible rank-80 raw-V-to-K regression.

| Layers | Raw-V unrestricted | Best raw-V rank80 | C1-V80 unrestricted | Rank-80 capacity loss | C1 subspace-orientation loss |
|:---|---:|---:|---:|---:|---:|
| 0--11 | 0.526067 | 0.523482 | 0.460823 | 0.002585 | 0.062659 |
| 12--23 | 0.563076 | 0.560948 | 0.505451 | 0.002128 | 0.055497 |
| 24--35 | 0.343249 | 0.341927 | 0.286057 | 0.001322 | 0.055869 |
| All | 0.477464 | 0.475452 | 0.417444 | 0.002012 | 0.058008 |

| Layer | Raw-V unrestricted | Best raw-V rank80 | C1-V80 unrestricted | Rank-80 capacity loss | C1 subspace-orientation loss |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.808375 | 0.807959 | 0.775993 | 0.000416 | 0.031966 |
| 13 | 0.562411 | 0.559790 | 0.501020 | 0.002621 | 0.058771 |
| 33 | 0.257791 | 0.257280 | 0.197232 | 0.000512 | 0.060048 |

Reducing the optimal raw-V predictor from rank128 to rank80 costs only 0.20 percentage points on average. The fixed C1-V80 subspace loses a further 5.80 points relative to the K-aware optimal rank80 subspace. At layer 33, the corresponding losses are 0.05 and 6.00 points. Thus the additional C1 loss is caused almost entirely by subspace orientation rather than insufficient dimensional capacity.

## Canonical Correlations

| Layer | Raw-V mean rho 1--16 | C1-V80 mean rho 1--16 | Raw-V rho16 | C1-V80 rho16 |
|---:|---:|---:|---:|---:|
| 0 | 0.792672 | 0.755580 | 0.634744 | 0.585446 |
| 13 | 0.910252 | 0.895465 | 0.858952 | 0.836166 |
| 33 | 0.822319 | 0.785292 | 0.756329 | 0.702651 |

The leading canonical correlations do not collapse monotonically with depth: layer 13 has stronger whitened top-16 correlations than layer 0, and layer 33 remains above layer 0 on these averages. CCA weights whitened K directions equally, whereas the Schur-trace explained fraction weights directions by their actual K variance. The observed late-layer failure is therefore better described as growth of variance-weighted conditional Key covariance than as a universal collapse of all canonical correlations.

## Interpretation

Within the unrestricted affine hypothesis, the dominant cause of the late-layer failure is model-intrinsic K/V decoupling. Raw V128 itself loses most of its ability to predict pre-RoPE K: held-out explained K falls from 80.60% at layer 0 to 56.04% at layer 13 and 25.05% at layer 33. The late-layer collapse therefore exists before applying C1 compression.

C1-V80 does discard additional K-predictive directions, but this is a secondary and approximately depth-independent absolute loss. Across all layers it removes about 6.00 percentage points of linearly predictable K energy. The dimension-matched control shows that nearly all of this gap comes from the payload-oriented C1 subspace rather than the rank-80 capacity limit. The loss does not grow in the late third of the model, and the raw-V and C1 curves have almost identical layer ordering.

At layer 33, C1 retains 19.72/25.78 = 76.5% of the K energy that was linearly predictable from raw V, but raw V already leaves roughly three quarters of centered pre-K energy unexplained. Consequently, changing the V compression alone cannot recover most of the missing late-layer routing information. A separate routing residual is justified primarily because the required K information is absent from V itself, not merely because C1-V80 projected it away.

This conclusion is specifically about affine predictability. It does not exclude a nonlinear relationship between raw V and pre-RoPE K.

## Execution

- Slurm array job: 8300099.
- Runtime: 4:40--5:26 per shard.
- Peak reported host memory: 39.27--40.00 GiB per shard, dominated by streamed activation mappings and page residency accounting.
- Status: all four shards completed with exit code 0.
- Warning: Transformers reported that the `device` argument to `Qwen3RotaryEmbedding` is deprecated. No NaN or runtime failure occurred.
