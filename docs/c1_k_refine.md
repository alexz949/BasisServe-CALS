# C1-KRefine oracle

C1-KRefine asks whether a small resident Key proxy plus selected exact-Key
pages can approach full exact-Key attention while C1-compressed Values remain
resident. This patch is a correctness and quality oracle. It does not move
Keys to CPU and makes no serving-throughput claim.

## Cache and factor semantics

C1 permanently stores the complete Value history in its learned latent width
`rV`. Every valid C1 Value participates in attention; the implementation never
reconstructs a dense Value cache. KRefine changes only how attention scores are
obtained.

Factors act after RoPE. Each physical KV head `g` owns one Key encoder
`E[g]` with shape `[head_dim, rK]`. Every query head `h` owns its own Query
factor `F[h]` with the same shape. All query heads in a GQA group consume the
same physical proxy `K[g] @ E[g]`; there is no per-query-head duplicate Key
cache.

The factors are initialized from per-group post-RoPE Key PCA and may be fitted
without backpropagation by alternating matrix least squares on valid causal Q/K
pairs. The objective is raw score regression,

```text
sum_n w_n (q_n^T k_n - (q_n^T F[h_n]) (k_n^T E[g_n]))^2 + ridge.
```

Each linear block uses the repository's matrix-free conjugate-gradient solver.
After a shared Key update, QR gauge fixing makes the Key encoder columns
orthonormal and transforms all consuming Query factors without changing proxy
scores.

## Attention policies

- `full_exact`: exact scores for every valid token.
- `proxy_only`: proxy scores for every valid token.
- `proxy_exact_refine`: exact scores replace proxy scores on selected pages;
  unselected tokens retain proxy scores.
- `sparse_exact`: selected pages use exact scores and all other valid tokens
  receive `-inf`.

Pages are ranked from proxy logits. `max` and `logsumexp` page statistics are
supported. Per-query-head page scores are reduced by a max across all query
heads sharing one physical KV head, so a selected exact page is fetched once
and reused across its GQA group.

`exact_token_budget` is rounded up to a deterministic page budget. Pages that
intersect `recent_exact_window` consume that budget first. If the forced recent
set alone is larger than the budget, recent-page correctness wins and the
reported actual token count exceeds the requested budget; otherwise the
budget is respected up to page granularity. Padding-only pages are never
selected.

`proxy_exact_refine` is still approximate unless all valid pages are refined.
Exact replacement corrects selected logits, but errors on unselected logits
still affect the denominator of the full-context softmax.

## Materialized and streaming references

The materialized reference exposes full proxy and mixed score tensors for
diagnostics. The streaming reference uses two passes:

1. scan proxy K one page at a time and retain only page-selection statistics;
2. resolve selected exact pages through `ExactKeyPageStore`, scan the context
   again, and accumulate a mixed online softmax in FP32.

The second pass uses exact logits on selected pages, proxy logits elsewhere,
and every resident C1 Value. Its running max, normalization sum, and C1 latent
numerator are `[batch, query_heads]` or `[batch, query_heads, rV]`; it never
retains `[batch, query_heads, sequence]` scores.

`GPUExactKeyPageStore` is the only backend in this patch. Both references fetch
selected exact pages through its protocol instead of directly indexing hidden
exact-Key state. This is the prepared boundary for a future pinned-CPU backend.

## Future runtime architecture

```text
GPU
├── full C1 Value cache
├── low-rank K proxy
├── recent exact K
└── hot exact-K pages

CPU
└── cold exact post-RoPE K pages

Decode
Q
→ proxy page scoring
→ selected exact-K page fetch
→ exact score replacement
→ full-context mixed softmax
→ full resident C1-V accumulation
→ C1 AR/AG backend
```

CPU offload, asynchronous transfer, prefetching, and hot-page reuse are future
runtime work. They should not be implemented until layer replay shows that a
small refinement budget materially improves over proxy-only attention and
approaches the full-exact C1 output.

## Captured-tensor workflow

First capture real post-RoPE tensors and uniformly weighted causal pair
samples. The capture is sharded by layer so it does not retain the complete
model-wide tensor bank in RAM:

```bash
python evaluation/capture_qwen3_8b_c1_k_refine.py \
  --model-path /path/to/Qwen3-8B \
  --c1-export /path/to/c1/export \
  --calibration-data /path/to/windows.safetensors \
  --window-start 0 \
  --windows 2 \
  --fit-pair-windows 1 \
  --sequence-length 2048 \
  --query-position 2047 \
  --queries-per-sequence 16 \
  --keys-per-query 64 \
  --recent-pair-fraction 0.25 \
  --top-score-pair-fraction 0.25 \
  --output-dir results/calibration/qwen3_8b_c1_k_refine_capture
```

The fitting CLI consumes that capture directory, or a single safetensors pair
bank containing
`layers.L.query`, `layers.L.key`, and `layers.L.query_head`. Rows must already
be valid causal post-RoPE pairs; optional `layers.L.weight` supplies weights.
For example:

```bash
python evaluation/fit_qwen3_c1_k_proxy.py \
  --pair-bank results/calibration/qwen3_8b_c1_k_refine_capture \
  --model-path /path/to/Qwen3-8B \
  --c1-export /path/to/c1/export \
  --num-query-heads 32 \
  --num-kv-heads 8 \
  --proxy-ranks 8,16,32,64 \
  --layers all \
  --als-sweeps 3 \
  --cg-iterations 32 \
  --ridge 1e-5 \
  --device cuda \
  --output-dir results/checkpoints/qwen3_c1_k_proxy
```

The layer oracle consumes captured tensors named `layers.L.query`,
`layers.L.exact_key`, and `layers.L.c1_value`, plus optional
`layers.L.attention_mask` and `layers.L.decoder`:

```bash
python evaluation/eval_qwen3_c1_k_refine_oracle.py \
  --capture results/calibration/qwen3_8b_c1_k_refine_capture \
  --c1-export /path/to/c1/export \
  --k-proxy-export results/checkpoints/qwen3_c1_k_proxy \
  --layers all \
  --example-start 1 \
  --examples 1 \
  --proxy-ranks 8,16,32,64 \
  --page-sizes 32,64,128 \
  --exact-token-budgets 0,64,128,256,512,1024 \
  --recent-exact-windows 0,128,256 \
  --page-score-modes max,logsumexp \
  --policies proxy_only,proxy_exact_refine,sparse_exact \
  --output-json results/evaluation/c1_k_refine_oracle.json \
  --output-markdown results/evaluation/c1_k_refine_oracle.md
```

The JSON includes score, page/token recall, attention, C1 latent, optional
decoded-output, logical byte/FLOP, timing, and streaming-equivalence metrics.
The Markdown file is a Pareto-oriented table. Reference timings are useful for
debugging only and are not production kernel measurements.

## Current limitations

- query length is restricted to one decode token;
- exact Keys remain on the same device as the oracle;
- capture, fitting, and replay remain separate reproducible stages;
- no Qwen attention module is monkey-patched, so the existing default C1 path
  remains unchanged;
- logical avoided-byte counts omit transfer scheduling and page-cache effects.
