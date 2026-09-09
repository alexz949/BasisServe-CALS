# Prompt-local residual core diagnostic

## Frozen setup

Qwen3-8B-Base; uniform C1-V96 checkpoint `qwen3_8b_c1_v96_32f4h_s32768_als6`; frozen Base16/E8/U8 bank `c1_v96_b16r8_qgram`. No C4 factors are refitted. Inputs are the first two saved examples of each of the six LongBench tasks: indices0,1,32,33,64,65,96,97,128,129,160,161. This is a12-prompt diagnostic, not a representative full benchmark estimate.

The model runs full-K C1-V96 FP16 memory-efficient SDPA prefill on V100. Hooks observe each layer's actual post-RoPE Q/K and C1-V96 codes without changing its outputs. All three routing diagnostic arms use FP32 Base/residual features and score evaluation, with FP64 small normal-equation solves. Consequently identity is a controlled FP32 diagnostic baseline, not a reproduction of the native BF16 deployment's37.0550 generation score. The original L40S/BF16 queue was cancelled before execution at the user's request to move to V100.

## Query positions and objective

For a prompt of length T,48 distinct positions are uniformly spaced from min(T-49,max(2048,floor(T/4))) through T-1. Positions with ordinal i mod3=1 are16 diagnostic queries; the other32 are fit queries. Each query's teacher distribution is restricted to its causal prefix. The fit uses no reference answers or generated queries. All positions come from an already completed prompt: this is not a test of causal adaptation during prefill and not a test of future decode-query generalization.

The exact-K teacher supplies non-sink Page-Fisher Grams after excluding pinned page0. For scaled query q, a=qU, and fixed E, the residual error is a S E-transpose minus q. The loss is its Gram-weighted quadratic plus a Frobenius penalty toward identity. Unknowns are0 for identity,8 for a diagonal core and64 for a full8x8 core per query head.

Normal equations are accumulated over fit queries. Both fitted variants use the same penalty coefficient:0.001 times the mean diagonal of the full64x64 Hessian, floored at1e-12 before multiplying by0.001. Zero-information cases return identity. No Adam, backprop, clipping or BCD sweeps are used.

## Measurements

- Fit and disjoint-query unregularized Page-Fisher losses.
- Exact-teacher full and non-sink attention mass retained by each selected set.
- Page overlap with exact-QK selection under the same Page32/shared B2048/pinned-page0 rule.
- Core distance from identity and separate small-solve timing.

Selection retains the existing per-head non-sink normalization followed by GQA-head maximum and fixed page budget. These objective values do not claim to optimize the exact page-LSE, discrete Top-B loss or generated output.

Fit information includes later prompt positions than some diagnostic queries, but never future generated tokens. Diagnostic queries are disjoint from fit queries; this is within-prompt evaluation, not an independent-prompt final held-out benchmark. Short prefixes that fit within2048 tokens have trivially full selected support and remain identifiable through the saved positions.

## Implementation and queued runs

Core solver: `basisserve/core/c1_prompt_core.py`. Tests: `tests/test_c1_prompt_core.py`. Driver: `evaluation/diagnose_c1_prompt_core.py`. CPU tests verify direct-quadratic agreement, regularized objective ordering and identity for zero information. Syntax checks passed; GPU behavior is gated by smoke.

-8301225/8301226/8301227: original L40S pipeline, cancelled before execution.
-8301228: V100 sample0, layers0/15/33/35 smoke; completed successfully.
-8301229: four independent V100 shards, all36 layers on12 prompts; started after smoke success.
-8301230: dependent CPU summary.

Outputs live in `results/evaluation/c1_prompt_core_v100`; fitted cores are stored per prompt alongside hashes, fit/diagnostic positions, per-layer metrics and protocol provenance. Existing benchmark artifacts are not overwritten. No end-to-end LongBench generation was submitted in this diagnostic.

V100 smoke prefill relative L2 error versus explicit FP32 reference was0.00023546595. All four smoke layers completed without numerical failures. Full cores reduced fit loss but increased diagnostic-query loss in all four sampled layers; this single-prompt smoke is not the final12-prompt result.
