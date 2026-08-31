# C1 rigorous evaluation: implementation audit

This note records the repository semantics that the rigorous C1 experiments
must preserve.  It is descriptive, not a new checkpoint format or runtime.

## Phase-1 decision: isolate `W_o`

The formal Phase-1 representation comparison keeps attention and the KV cache
dense on both arms.  It compresses only the post-attention `W_o` map:

```text
Wo-C1-AllGather:  sum_p (Z_p E_p) D_p
Wo-LR-AllReduce:  (sum_p Z_p F_p) D
```

For Qwen3-8B TP4, each `Z_p` has width 1024.  Wo-C1 uses four private
rank-512 sources, so its gathered width is 2048.  The equal-ideal-ring-traffic
LR-AllReduce control has shared rank 1024.  The capacity-matched LR-AllReduce
control has rank 2048 and can embed every private C1 source in disjoint shared
coordinates.  Consequently its globally optimized fit must be no worse than
the Wo-C1 fit; this is the first implementation kill gate.

This comparison has no V compression, no compressed-V attention kernel, and
no KV-cache reduction.  It avoids attributing the independent folded-V
constraint to the collective representation.  The existing Value-rank-64
checkpoint described below remains a separate system/quality branch.

## Audited Qwen3-8B TP4 geometry

The current uniform C1 checkpoint uses Qwen3-8B-Base with 36 layers, hidden
width 4096, 32 query heads, 8 physical KV heads, and head width 128.  TP4 owns
contiguous query/KV-head ranges: rank `p` owns query heads `[8p, 8p+8)` and KV
heads `[2p, 2p+2)`.  Query head `h` maps to KV group `h // 4`.

For one layer, `N` activation rows, and Value rank `r`:

| Quantity | Logical shape | Current uniform-r64 shape |
|---|---:|---:|
| Flattened dense attention output `Z` / `o_proj` input | `[N, 32*128]` | `[N, 4096]` |
| TP-local source output `Z_p` | `[N, 8*128]` | `[N, 1024]` |
| Physical-KV Value encoder `A_g` | `[128, r]` | `[128, 64]` |
| TP-source block-diagonal encoder `E_p` | `[8*128, 8*r]` | `[1024, 512]` |
| Per-query-head decoder `D_h` | `[r, 4096]` | `[64, 4096]` |
| TP-source decoder block `D_p` | `[8*r, 4096]` | `[512, 4096]` |
| Folded TP-local V projection | `[2*r, 4096]` | `[128, 4096]` |
| Persistent TP-local V cache per token | `[2, r]` | `[2, 64]` |
| TP-local AllGather latent | `[N, 8*r]` | `[N, 512]` |
| Global gathered latent | `[N, 32*r]` | `[N, 2048]` |
| Global C1 decoder | `[32*r, 4096]` | `[2048, 4096]` |
| Local-decode output before AllReduce | `[N, 4096]` | `[N, 4096]` |

The source-level matrices above are logical views.  The stored checkpoint does
not materialize `E_p`: it stores `value_coordinate_encoders` as `[8,128,r]`
and `head_output_decoders` as `[32,r,4096]`.  Within a TP source, `E_p` is
block diagonal and repeats the matching physical-KV encoder for each of its
four query heads.  `D_p` is the vertical concatenation of its eight `D_h`
blocks.

The current runtime flattens latent coordinates query-head-major within each
TP rank, gathers TP ranks in distributed rank order, and reshapes the decoder
rows as query heads `0..31`, then latent coordinate `0..r-1`.  The packed
feature-major arena changes physical layout only; it must not change this
logical order.

## Historical folded-V checkpoint geometry

The geometry above describes the existing Value-rank-64 C1 checkpoint and is
retained for provenance.  It is not the C1 arm in the Wo-only Phase-1 test.

## Canonical LR-AllReduce shapes

For TP4 and shared latent rank `r_AR`:

| Quantity | Shape |
|---|---:|
| Source-specific encoder `F_p` | `[1024, r_AR]` |
| Local latent `Z_p F_p` | `[N, r_AR]` |
| Reduced shared latent | `[N, r_AR]` |
| Replicated shared decoder `D` | `[r_AR, 4096]` |

For the Wo-only C1 source rank 512, the aggregate C1 width is 2048.  Therefore:

- capacity-matched LR-AR uses `r_AR=2048`;
- ideal ring-wire-matched LR-AR uses `r_AR=1024` because AllReduce sends two
  ring phases while AllGather sends one;
- capacity-matched LR-AR has the same decoder row count as C1 and approximately
  twice its logical ring traffic.

A general `E_p` or `F_p` can mix the eight query-head outputs owned by a TP
rank.  Both are post-attention `W_o` controls and neither is folded into the
physical V projection.  Neither arm has a compressed V cache.

## Fitting and deployment semantics

The audited uniform-r64 checkpoint is
`results/checkpoints/qwen3_8b_c1_v64_als5`:

- initialization: activation-weighted SVD;
- objective: complete attention-output MSE with cross-query-head covariance;
- optimization: five decoder-closed ALS sweeps with fixed 16-step encoder CG;
- fitting statistics: 256 C4 documents x all 2048 positions;
- selection statistics: a disjoint 64 C4 documents x all 2048 positions;
- covariance and solver work dtype: FP32;
- stored factors and current deployment dtype: BF16;
- covariance damping: `1e-5`; encoder damping and decoder jitter: zero.

The sufficient statistics store per-layer dense `o_proj` weights `[4096,4096]`
and fit/held-out covariances `[4096,4096]`.  They are sufficient to fit the
globally optimal activation-aware LR-AR control without replaying activations.

For the Wo-only TP-source fitter, each encoder coordinate has exactly one
Kronecker Hessian term,
`H_p(Delta) = C_pp Delta (D_p D_p^T)`.  Encoder updates therefore use an exact
FP64 two-sided Cholesky solve.  No CG tolerance, CG iteration count, or encoder
damping is part of this method; the shared covariance damping remains `1e-5`.

## Existing reusable components and gaps

- `basisserve/core/gqa_routed_ov_joint.py` is the current C1 decoder/encoder ALS
  implementation, including full cross-head covariance and ragged ranks.
- `evaluation/fit_llama2_mha_c1_joint.py`, selected through the Qwen3-8B profile,
  is the current uniform C1 fitting/export path.
- `basisserve/checkpoint/gqa_vo_qwen3.py` and
  `basisserve/core/qwen3_8b_tp4_decode.py` implement quality and TP4 folded-V
  deployment respectively.
- `basisserve/core/tp_output.py` already contains a distributed low-rank
  AllReduce runtime abstraction, but its standard factorizer is weight-only.
- `basisserve/core/qwen35_global_aa_svd.py` contains the desired activation-aware
  shared-decoder LR-AR mathematics, currently tied to a Qwen3.5 entrypoint.
- `evaluation/analyze_qwen3_o_proj_collective_endpoints.py` is an earlier
  equal-wire output-POD oracle.  Its private endpoint is not the current
  foldable C1-ALS solution and it lacks the capacity-matched inclusion gate.
- Existing Qwen3-8B TP4 eager benchmarks provide timing summaries and packed
  AllGather paths, but are not CUDA Graph benchmarks.  They cannot be relabeled
  as production CUDA Graph evidence.

## Terminology that must remain explicit

`layer output` means the summed attention output after the logical `o_proj`
boundary.  `terminal logits` means final-model logits.  These are separate
metrics and must not both be called `final output` in result files.

The present foldable C1 factors are private per physical KV group/query-head
block and can be grouped into TP sources.  A future unconstrained TP-source
encoder is a different representation and must not silently replace them.
Likewise, TP4 and TP8 alter ownership and communication even when a physical
KV-group factor bank is algebraically reusable; TP-dependent claims require
separately labelled experiments.
