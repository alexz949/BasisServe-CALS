# Qwen3-8B Pre-RoPE V-to-K Retrieval and Store80 Long-Position Failure

## Scope

This document summarizes two related investigations on Qwen3-8B-Base:

1. whether dense or C1-compressed Value activations contain enough information to
   predict Key activations for sparse-page routing; and
2. why the Store80 exact-K and Route32 RULER-32K evaluation produced zero accuracy.

The experiments show that a per-head linear map from Value to **pre-RoPE Key** is a
useful routing proxy, while a position-independent direct map to post-RoPE Key is
not. They also show that the failed Store80 RULER run was not caused by Route32,
chunked prefill, or a causal-mask error. The primary failure was extrapolating a
post-RoPE Store80 representation calibrated only at positions 0--2047 to absolute
positions near 30K--32K.

The 32K experiment used the model's native context configuration:
`max_position_embeddings=32768`, `rope_theta=1,000,000`, and
`rope_scaling=None`. Therefore, the failure discussed below is an
**absolute-position/RoPE-phase generalization failure**, not an incorrectly enabled
YaRN or other RoPE-scaling configuration.

## 1. Pre-RoPE V-to-K retrieval

### 1.1 Method

For every physical KV head, an affine ridge-regression probe is fit from either
dense V or the C1-V latent to pre-RoPE K:

\[
\widehat{k}^{\mathrm{pre}}_{g,i} = A_g v_{g,i} + b_g.
\]

The exact token RoPE transform is then applied at evaluation time:

\[
\widehat{k}^{\mathrm{post}}_{g,i}
= R(i)\widehat{k}^{\mathrm{pre}}_{g,i}.
\]

The query uses its exact post-RoPE representation. Proxy attention logits are
therefore

\[
\widehat{s}_{h,i}
= \frac{\langle q^{\mathrm{post}}_{h},
\widehat{k}^{\mathrm{post}}_{g(h),i}\rangle}{\sqrt{128}}.
\]

A control probe directly predicts post-RoPE K with one position-independent map.
That control performs poorly because a fixed linear map cannot represent the
token-dependent RoPE transformation.

### 1.2 Matched evaluation protocol

| Item | Setting |
|:---|:---|
| Model | Qwen3-8B-Base |
| Layers | 0, 17, 35 |
| Sequence length | 32,768 |
| Fit windows | C4 indices 0--3 |
| Held-out windows | C4 indices 32--35 |
| Query | Final token of each held-out window |
| Token budgets | 512, 1,024, 2,048 |
| Page size | 64 tokens |
| Retained page budget | 32 pages = 2,048 tokens/query head |
| GQA geometry | 4 query heads per physical KV head |
| Probe | Per-physical-KV-head affine ridge regression |

The V64 and V96 results below use the same prompts, query rows, layers, budgets,
and exact-QK teacher, so their differences are directly comparable.

### 1.3 Token-level Top2048 results

| Source | Post-K cosine | R@2048 | Selected teacher mass | Exact token oracle mass |
|:---|---:|---:|---:|---:|
| Dense-V128, pre-RoPE probe | 0.952277 | 73.16% | 68.81% | 96.58% |
| C1-V96, pre-RoPE probe | 0.948971 | 71.93% | 68.44% | 96.58% |
| C1-V64, pre-RoPE probe | 0.944597 | 70.90% | 68.22% | 96.58% |
| C1-V96, direct post-RoPE control | 0.836403 | 17.53% | 20.85% | 96.58% |
| C1-V64, direct post-RoPE control | 0.833908 | 17.05% | 19.87% | 96.58% |

The main conclusion is that C1-V retains substantial information about pre-RoPE
K. Applying exact RoPE after the prediction is essential. V96 is slightly better
than V64, but the practical selected-mass difference is small:

| V96 minus V64 | Gain |
|:---|---:|
| R@2048 | +1.03 pp |
| Selected teacher mass | +0.22 pp |
| Page recall | +0.42 pp |
| Mass-page recall | +0.58 pp |
| GQA-union mass-page recall | +0.43 pp |
| GQA-union selected mass | +0.08 pp |

This suggests that V96's larger system-level benefit is more likely to come from
payload reconstruction quality than from substantially better V-to-K routing.

## 2. Page64 routing at a 2,048-token budget

Proxy token logits are reduced to page scores using page log-sum-exp. Each query
head selects 32 Page64 pages. The four query heads sharing one physical KV head
then form a GQA union.

### 2.1 V64 and V96 page quality

| Source | Page recall | Mass-page recall | Selected mass | GQA-union recall | GQA-union mass | Mean physical union pages |
|:---|---:|---:|---:|---:|---:|---:|
| C1-V64 pre-RoPE | 65.63% | 75.89% | 66.31% | 82.60% | 68.72% | 54.32 |
| C1-V96 pre-RoPE | 66.06% | 76.46% | 66.34% | 83.03% | 68.81% | 54.51 |
| Dense-V128 pre-RoPE | 66.91% | 77.47% | 66.47% | 83.72% | 68.90% | 54.39 |

The two GQA metrics measure different things:

- **GQA-union mass-page recall** is the fraction of each query head's exact
  Top32 attention-mass pages that appear in the shared physical-page union. Each
  oracle page counts equally.
- **GQA-union selected mass** is the total exact attention probability carried by
  all tokens in the selected union pages. It weights pages by their actual
  importance over the full attention distribution.

V96's +0.43 pp union-recall gain but only +0.08 pp union-mass gain means that the
additional recovered pages tend to have low attention weight.

### 2.2 Exact oracles

| Oracle | Retained exact attention mass |
|:---|---:|
| Exact token Top2048 | 96.58% |
| Exact Page64 Top32 | 92.43% |
| Exact Page64 Top32 with GQA union | 94.37% |

The Page64 constraint itself loses about 4.15 pp of mass relative to exact token
Top2048. These oracle values require a full exact-QK scan and are diagnostic upper
bounds, not deployable selectors.

In the normal sparse path, proxy-selected pages are loaded and then scored with
their exact K. False-positive pages can be rejected or receive negligible exact
attention weight, but a false-negative page cannot be recovered if it was never
loaded.

## 3. Overfetch 64 pages, exact-rerank, retain 32

To recover proxy false negatives without doubling the final attention payload,
the selector was evaluated as a two-stage pipeline:

1. select 64 Page64 candidates per query head with the C1-V96 proxy;
2. load exact K for those candidates;
3. rank candidate pages with exact-QK page log-sum-exp; and
4. retain 32 pages per query head for final V loading and attention.

### 3.1 Aggregate result

| Metric | Direct proxy Top32 | Proxy Top64 -> exact rerank -> Top32 | Gain |
|:---|---:|---:|---:|
| Mass-page recall | 76.46% | 89.05% | +12.59 pp |
| Selected attention mass | 66.34% | 68.01% | +1.67 pp |
| GQA-union mass-page recall | 83.03% | 90.72% | +7.69 pp |
| GQA-union selected mass | 68.81% | 70.92% | +2.12 pp |

The Top64 candidate set contains 89.05% of the exact Top32 mass pages. Exact
reranking retains all recoverable oracle Top32 pages in this diagnostic, so final
mass-page recall equals candidate recall.

### 3.2 Physical-page cost

| Stage | Mean unique pages/physical KV head | Approximate tokens |
|:---|---:|---:|
| Direct Top32 GQA union | 54.51 | 3,489 |
| Top64 candidate GQA union: exact K required | 101.31 | 6,484 |
| Reranked Top32 GQA union: final V/attention | 55.19 | 3,532 |

The candidate exact-K footprint increases by about 85.9%, while the final
V/attention footprint increases by only about 1.2%. This design is attractive only
when K and V can be fetched independently or candidate K is substantially cheaper
than fetching the complete K/V page. If K and V are physically bundled, the system
effectively fetches about 101 pages and loses most of the intended bandwidth
advantage.

Layer 17 benefits most from reranking: its GQA-union selected mass rises from
69.69% to 75.10%, a gain of 5.41 pp.

## 4. Store80 Route32 RULER-32K failure

### 4.1 Completed but invalid full run

The Store80/Route32 RULER-v1 run completed 88 samples at 32K:

| Arm | Task-balanced accuracy |
|:---|---:|
| BF16 dense baseline | 86.48% |
| Store80 exact-K | 0.00% |
| Store80 Route32/B1024 | 0.00% |

The zero result is not evidence that Route32 failed:

- all 88 Store80 exact-K and Route32 samples already differed from dense on the
  first generated token;
- the first token is produced by their shared Store80 exact prefill, before
  Route32 is enabled;
- Store80 exact-K and Route32 shared the same first token in all 88 samples; and
- generated continuations were repetitive and nonsensical.

Therefore, sparse routing quality cannot be inferred from this full run.

### 4.2 Controlled same-token diagnostic

One RULER prompt's final 2,048 tokens were evaluated twice with identical token
content:

1. position IDs reset to 0--2047; and
2. their actual prompt position IDs near 30K--32K.

Each case compared dense and Store80 full prefill, chunk512 prefill, and an
explicit bottom-right causal mask.

| Same tail tokens | Store80 vs dense at positions 0--2047 | Store80 vs dense at actual 30K--32K positions |
|:---|---:|---:|
| Top1 match | Yes | No |
| Top10 overlap | 60% | 10% |
| Centered logit relative RMSE | 0.2072 | 0.6596 |
| Centered logit cosine | 0.9786 | 0.7632 |
| KL(dense || Store80) | 0.2111 | 6.7412 |
| Dense next token | 220 (`" "`) | 220 (`" "`) |
| Store80 next token | 220 (`" "`) | 1182 (`" back"`) |

For both dense and Store80, full prefill and chunk512 retained the same Top1.
Store80 with an explicit causal mask was identical to Store80 with the mask
generated by Transformers. This rules out chunking, cache rollback, and causal-mask
alignment as the primary failure.

### 4.3 Evidence-supported diagnosis

The Store80 bank records `key_convention="post_rope"`. Its statistics and direct
residual captures were collected from 2,048-token C4 windows at positions
0--2047. The post-RoPE KQ initialization was also fit only at sequence length
2,048.

The controlled diagnostic directly localizes the failure to absolute-position
generalization: identical token content behaves much worse when moved from the
calibration range to positions near 32K. The following mechanism is the strongest
current explanation, but has not yet been isolated by an intervention.

Store80 places routing-related post-RoPE K information and Value payload
information in one 80-dimensional latent. The payload decoder must reconstruct the
attention output while suppressing the injected K component. The 2K calibration
fits that interaction only over the RoPE phases observed at positions 0--2047.
At positions near 32K, unseen post-RoPE orientations likely allow the K component
to leak into the decoded payload and corrupt the model output.

This also explains two otherwise surprising observations:

- WikiText-2K perplexity remains reasonable at 8.31 because it stays inside the
  calibration position range.
- Store80 **exact-K** still fails at 32K. "Exact-K" means that attention weights use
  exact K, but the Value/output path still uses the position-sensitive Store80
  joint latent.

## 5. Recommended next steps

### 5.1 Immediate correction

1. Collect or synthesize post-RoPE statistics across absolute offsets spanning the
   native 32K context, for example 0, 2K, 4K, 8K, 16K, 24K, and 30K.
2. Refit the Store80 joint bank and its post-RoPE routing initialization with this
   position coverage. Position-offset augmentation of 2K windows is a cheaper
   first test than full 32K covariance collection.
3. Hold out both documents and absolute-position bands during validation.
4. Gate every new checkpoint with a same-token position sweep and a one-sample
   RULER smoke test before launching the 88-sample evaluation.

### 5.2 Structural alternatives

- Constrain the injected K subspace to lie in, or near, the payload decoder's
  nullspace so K coordinates cannot contaminate reconstructed Value output.
- Separate routing and payload storage when the nullspace constraint costs too
  much Value quality.
- Investigate a pre-RoPE routing representation with position-aware query/key
  rotation, avoiding a joint payload that is tied to the calibration's absolute
  post-RoPE phases.

## 6. Artifacts

- [V96 V-to-K and 64-to-32 rerank result](../results/evaluation/qwen3_8b_c1_v96_c64_r32_page64_32k_l0_17_35/summary.md)
- [V64 V-to-K result](../results/evaluation/qwen3_8b_c1_v64_to_k_top2048_page64_32k_l0_17_35/summary.md)
- [Failed Store80 RULER-32K result](../results/evaluation/qwen3_8b_s80_route32_ruler_32k_shadowkv11_s8/summary.md)
- [Same-token Store80 position diagnostic](../results/evaluation/qwen3_8b_s80_chunk512_prefill_diag_2k_same_tokens_s1/summary.md)

The failed RULER result should be retained only as a debugging artifact and should
not be reported as a Route32 accuracy result.

## 7. Reproduction and limitations

All reported runs used the `basis` conda environment, BF16, and an NVIDIA L40S on
the Lovelace host. The V-to-K runs used physical CUDA device 0; the RULER run and
same-token diagnostic used physical CUDA device 2. The commands recorded in the
result files were:

```bash
python evaluation/eval_qwen3_8b_dense_v_k_proxy.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --windows results/calibration/qwen3_8b_c4_32f4h_s32768/windows.safetensors \
  --c1-export results/checkpoints/qwen3_8b_c1_v64_als5 \
  --layers 0,17,35 --fit-start 0 --fit-windows 4 \
  --heldout-start 32 --heldout-windows 4 --sequence-length 32768 \
  --query-start 32767 --query-stride 32768 --budgets 512,1024,2048 \
  --page-size 64 --page-token-budget 2048 --relative-ridge 1e-6 \
  --batch-size 1 --torch-num-threads 4 --model-dtype bfloat16 \
  --output-dir results/evaluation/qwen3_8b_c1_v64_to_k_top2048_page64_32k_l0_17_35

python evaluation/eval_qwen3_8b_dense_v_k_proxy.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --windows results/calibration/qwen3_8b_c4_32f4h_s32768/windows.safetensors \
  --c1-export results/checkpoints/qwen3_8b_c1_v96_als5 \
  --layers 0,17,35 --fit-start 0 --fit-windows 4 \
  --heldout-start 32 --heldout-windows 4 --sequence-length 32768 \
  --query-start 32767 --query-stride 32768 --budgets 512,1024,2048 \
  --page-size 64 --page-token-budget 2048 --page-candidate-token-budget 4096 \
  --relative-ridge 1e-6 --batch-size 1 --torch-num-threads 4 \
  --model-dtype bfloat16 \
  --output-dir results/evaluation/qwen3_8b_c1_v96_c64_r32_page64_32k_l0_17_35

python evaluation/eval_qwen3_8b_s80_routing_ruler.py \
  --model-path /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --s80-export results/checkpoints/qwen3_8b_s80_bf_fast_r256_s35_adapteru_all36 \
  --dense-baseline results/evaluation/qwen3_8b_dense_ruler_32k_shadowkv11_s8/result.json \
  --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 \
  --sequence-length 32768 \
  --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,vt,fwe,qa_1,qa_2 \
  --samples-per-task 8 --routing-rank 32 --page-size 64 \
  --exact-token-budget 1024 --prefill-chunk-size 512 --dtype bfloat16 \
  --output-dir results/evaluation/qwen3_8b_s80_route32_ruler_32k_shadowkv11_s8

python evaluation/diagnose_qwen3_8b_s80_chunked_prefill.py \
  --model-path /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --s80-export results/checkpoints/qwen3_8b_s80_bf_fast_r256_s35_adapteru_all36 \
  --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 \
  --task niah_single_1 --sample-ordinal 0 --tokens 2048 --chunk-size 512 \
  --sequence-length 32768 --dtype bfloat16 \
  --output-dir results/evaluation/qwen3_8b_s80_chunk512_prefill_diag_2k_same_tokens_s1
```

The V-to-K study covers four fit windows, four held-out windows, three layers, and
one final-token query per held-out window. It is a routing probe rather than an
end-to-end quality benchmark. The same-token position diagnostic uses one RULER
sample, so it strongly localizes the failure dimension but does not by itself
measure its prevalence across tasks. The full 88-sample RULER run is invalid for
Route32 accuracy because the shared Store80 prefill is already broken before
routing begins.
