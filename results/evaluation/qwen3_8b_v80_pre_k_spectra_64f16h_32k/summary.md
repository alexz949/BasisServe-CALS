# Qwen3-8B C1-V80 to Key Predictive Spectra

## Protocol

- Model: BF16 Qwen3-8B-Base.
- Value representation: uniform C1-V80 ALS6 calibrated on 32 C4 documents of length 32,768.
- Predictive-spectrum fit split: 64 independent C4 documents of length 32,768.
- Held-out split: 16 independent C4 documents of length 32,768.
- Layers: all 36 transformer layers; eight physical KV groups per layer.
- Target conventions: pre-RoPE K and direct post-RoPE K.
- Position buckets: 0--2,048, 2,048--8,192, 8,192--16,384, and 16,384--32,768.
- Ranks: 0, 4, 8, 16, 24, 32, 48, 64, and 80.
- Score metric: raw-score NMSE from each held-out document's paired final-token Query.
- Environment: `basis`, PyTorch 2.6.0+cu124.
- Hardware: four L40S GPUs on `lovelace`.
- Slurm array job: `8299870`; all four shards completed successfully in 5:43--5:51 with exit code 0.

The analysis streams existing activation captures and does not perform a model forward. Each layer also stores its global pre-RoPE and post-RoPE fit moments for later offline reuse.

## Mathematical quantities

For centered C1 codes \(C\) and Key targets \(K\), the predictive matrix is

\[
T=G_{CC}^{-1/2}G_{CK}.
\]

If its singular values are \(\sigma_j\), the optimal affine rank-\(r\) Key reconstruction loss is

\[
\mathcal L_r
=
\operatorname{tr}(G_{KK})-
\sum_{j=1}^{r}\sigma_j^2.
\]

`Captured K energy` is \(\sum_{j\le r}\sigma_j^2/\operatorname{tr}(G_{KK})\). `Captured predictable energy` normalizes the same numerator by the unrestricted rank-80 predictable energy.

## All-layer rank curves

### Pre-RoPE target

| Rank | Captured K energy | Captured predictable energy | Held-out centered K MSE | Held-out paired-query score NMSE |
|---:|---:|---:|---:|---:|
| 4 | 20.47% | 49.93% | 0.797411 | 0.155089 |
| 8 | 26.89% | 65.50% | 0.734022 | 0.143095 |
| 16 | 33.53% | 81.10% | 0.668312 | 0.132624 |
| 24 | 36.96% | 89.04% | 0.634376 | 0.128241 |
| 32 | 38.92% | 93.58% | 0.614999 | 0.126026 |
| 48 | 40.87% | 98.04% | 0.595744 | 0.124123 |
| 64 | 41.58% | 99.63% | 0.588718 | 0.123448 |
| 80 | 41.74% | 100.00% | 0.587072 | 0.123289 |

### Direct post-RoPE target

| Rank | Captured K energy | Captured predictable energy | Held-out centered K MSE | Held-out paired-query score NMSE |
|---:|---:|---:|---:|---:|
| 4 | 7.70% | 53.34% | 0.924249 | 0.439042 |
| 8 | 10.30% | 70.00% | 0.898974 | 0.427464 |
| 16 | 13.02% | 86.59% | 0.872618 | 0.417762 |
| 24 | 14.26% | 94.29% | 0.860521 | 0.412720 |
| 32 | 14.82% | 97.90% | 0.855262 | 0.410935 |
| 48 | 15.10% | 99.88% | 0.852890 | 0.409959 |
| 64 | 15.11% | 99.99% | 0.852868 | 0.409934 |
| 80 | 15.11% | 100.00% | 0.852872 | 0.409934 |

## Rank-16 position buckets

| Position bucket | Target | Captured K energy | Captured predictable energy | Held-out centered K MSE | Held-out score NMSE |
|:---|:---|---:|---:|---:|---:|
| 0--2,048 | pre-RoPE | 33.98% | 80.98% | 0.722490 | 0.185419 |
| 0--2,048 | post-RoPE | 19.06% | 81.52% | 1.252349 | 0.394834 |
| 2,048--8,192 | pre-RoPE | 33.81% | 81.08% | 0.677274 | 0.171997 |
| 2,048--8,192 | post-RoPE | 16.79% | 83.05% | 1.089035 | 0.407697 |
| 8,192--16,384 | pre-RoPE | 34.08% | 81.05% | 0.672759 | 0.163405 |
| 8,192--16,384 | post-RoPE | 15.68% | 83.43% | 0.971154 | 0.364585 |
| 16,384--32,768 | pre-RoPE | 33.88% | 81.15% | 0.666790 | 0.130701 |
| 16,384--32,768 | post-RoPE | 14.19% | 84.89% | 0.961018 | 0.652943 |

The pre-RoPE captured-energy curve is position-stable: rank-16 captured K energy stays between 33.81% and 34.08% across all four buckets. Direct post-RoPE predictability decreases with position, from 19.06% in the first 2K positions to 14.19% in the final 16K positions.

## Layer variation

For pre-RoPE K, the unrestricted V80-predictable fraction has mean 41.74% and median 42.23%. It ranges from 77.60% at layer 0 to 19.72% at layer 33.

Rank16 captures a mean 33.53% of total centered K energy and 81.10% of the unrestricted predictable energy. The captured-predictable fraction ranges from 98.73% at layer 0 to 74.96% at layer 13.

The held-out pre-RoPE rank-16 paired-query score NMSE has mean 0.132624 and median 0.127189. It ranges from 0.010820 at layer 0 to 0.350340 at layer 33. Direct post-RoPE rank-16 score NMSE has mean 0.417762.

## Interpretation

The strong hypothesis that C1-V80 can reconstruct most of pre-RoPE K is not supported: even the unrestricted rank-80 affine predictor explains only 41.74% of centered K energy on average.

The weaker low-rank conditional-mean hypothesis is supported. Rank16 captures 81.10% of all K energy that is linearly predictable from V80. Increasing the rank from 16 to 80 improves held-out paired-query score NMSE only from 0.132624 to 0.123289.

Pre-RoPE prediction followed by exact token RoPE is strongly supported. At rank16 it reduces paired-query score NMSE from 0.417762 to 0.132624 relative to a static direct post-RoPE predictor. Its predictive-energy fraction is also stable across absolute-position buckets, whereas direct post-RoPE predictability declines with position.

Layer 33 is the clearest difficult case for both total V-to-K predictability and paired-query score preservation. The result is consistent with the existing observation that a separately stored query-visible residual is essential, especially in late layers.

The paired-query metric uses one final-token Query per calibration document. It measures decode-style routing for those Queries and is not an all-query prefill metric.
