# LRQK-equation routing with resident C1 Value payload

## Status

Implemented a Qwen3 C1 integration and validated it with five CPU tests and two real-model L40S smoke runs. No full LongBench/RULER accuracy evaluation or CPU-offload performance run has been performed for this integration.

The adapter replaces the Base16/R8 selector, not the C1 payload. Its inference sequence is:

1. Full causal C1-V96 prefill, while fitting LRQK factors from the same actual post-RoPE Q/K.
2. Incremental LRQK query/key-code and decoder updates on each decode step.
3. Per-query-head historical token Top-k plus the exact recent-token suffix.
4. Exact attention over selected uncompressed K and the corresponding resident C1-V96 latent values.
5. The unchanged C1 output decoder.

All36 layers participate. No C4 calibration or Base/Residual refit is needed for LRQK; its factor fit is prompt-local and online. This is not the offline KQ-SVD checkpoint or the Query-Gram residual-fitting sampler.

## Upstream provenance and retained equations

Official repository: [tenghuilee/LRQK](https://github.com/tenghuilee/LRQK).

Pinned commit: `caf16293db2e4423a84ab2e895bacf64479f1eb7`.

Inspected source: [lrqk_attention.py at the pinned commit](https://github.com/tenghuilee/LRQK/blob/caf16293db2e4423a84ab2e895bacf64479f1eb7/lrqk_attention.py), SHA256 `2bfbc6df2bf97316ce5b086249ce8af48481b0a78f89f6650e19463fe783fab5`.

The repository is cloned under `external/LRQK` without modifying its source or installing its CUDA extensions. Its MIT license notice is retained in the adapted numerical module. Model execution does not import the upstream monolithic Transformers/CUDA wrapper.

The numerical implementation follows upstream `_lrqk_prefill_inv_w1`, `_lrqk_decode_inv_w1`, and `_lrqk_decode_gd_B_lr` with all lambda weights equal to1. For each query head, the prefill objective has the form

\[
\|QK^\top-A_QA_K^\top\|_F^2
+\|Q-A_QB_Q\|_F^2
+\|K-A_KB_K\|_F^2.
\]

This is an unnormalized raw-QK/reconstruction objective, not softmax KL or Page-Fisher. Gram products avoid materializing the full length-squared score matrix. Factor fitting uses the complete prompt Q/K; the fitting objective itself does not impose a causal-pair mask, while actual prefill attention remains causal.

Decode uses the previous active exact keys and their matching stored A_K rows, plus the current q/k, to update new query/key coordinates. The B_Q/B_K update includes the upstream analytic-gradient step. There is no Adam or autograd backward, but this is **not exclusively closed-form/BCD**, and must not be labeled as satisfying that narrower optimizer restriction.

The upstream solvers' equations match exactly in the CPU parity test with identical FP32 inputs and initial factors. The adapter checks solve status and finite outputs rather than silently replacing failed solves. Solve completion does not prove convergence after a fixed iteration budget.

## Cache adaptation and non-equivalences

This is an **LRQK-equation resident-cache reference**, not a bitwise reproduction of the official end-to-end offload implementation:

- Original exact K remains in the physical GQA GPU cache; only selected K/C1-V is gathered for attention.
- The official CPU hit/miss pool, recent ring-buffer layout, and custom CUDA transfer kernels are not used.
- The decode fitting support is explicitly the same previous selected token IDs for exact K and A_K, before appending the current key code. It does not copy upstream ring-buffer update/indexing behavior.
- Recent support is explicitly the latest suffix, without a one-token gap at the historical/recent boundary.
- Historical IDs are sorted; recent IDs are chronological. No page aggregation, GQA-max page score, pinned page0, or physical-union cap is added.
- Initial factors use deterministic per-layer seeds in a local generator, rather than depending on the upstream global RNG stream.
- PyTorch operations run eagerly; upstream torch.compile decorators and the invalid FP32 CUDA-autocast wrapper are not required.

Therefore, equation parity is established, but official token-cache trajectory/accuracy parity is not claimed. The adapter gathers exact K and C1-V on the chosen support; this is not exact full-attention output.

## Default configuration and budget accounting

| Parameter | Default |
|---|---:|
| LRQK rank per query head |32|
| Historical Top-k per query head |2048|
| Always-kept recent tokens |64|
| Maximum per-query-head selected support |2112|
| Prefill alternating iterations |2|
| Decode alternating iterations |2|
| Tolerance |1e-8|
| Lambda weights |1,1,1,1|
| Initialization |FP32 Gaussian; seed0 + layer index|
| Stored factors/codes in BF16 inference |BF16|
| Solves and analytic-gradient updates |FP32, TF32 disabled in evaluation|

This is a **token-level, per-query-head budget**, not our Page32/B2048 shared-GQA budget. Four heads can select different historical tokens; the physical GQA union can exceed2112. The evaluator records that union instead of describing all runs as a2048-token physical retrieval.

A_K is also per query head. For Qwen3-8B,32 query heads × rank32 =1024 scalars/token, equal in element count to8 physical K heads ×128 dimensions. This adapter also retains exact K. It is a quality reference, not evidence of reduced total KV memory.

## Implementation files

- [Numerical factors, selection, and resident-cache state](../basisserve/core/c1_lrqk.py).
- [Qwen3 C1 adapter and request-owned cache](../basisserve/checkpoint/c1_lrqk_qwen3.py).
- [Correctness tests](../tests/test_c1_lrqk.py).
- [LongBench smoke/evaluation/summary entry point](../evaluation/eval_longbench_c1_lrqk.py).

Install the existing uniform C1 export first, then call `install_c1_lrqk(model, LRQKConfig(...))` and supply a new `C1LRQKCache(config=model.config)` per prompt. Qwen3 Q/K RMSNorm, RoPE, C1 V projection, and C1 output projection are retained. LRQK state belongs to the cache, not to a global layer counter or a previous request.

Supported reference scope: unpadded batch1, a complete initial prompt of at least rank tokens, then single-token decode. Beam reordering, chunked prefill, speculative cache transactions, sliding-window attention, and continuous batching are not implemented or validated. This adapter does not modify the original C1 attention implementation or previous benchmark artifacts.

## Validation results

Five CPU tests passed in `basis`:

1. Prefill and decode factors exactly match the pinned upstream FP32 equation functions for fixed inputs/initialization, including the analytic-gradient updates.
2. Historical Top-k/recent selection, disjointness and short-support behavior.
3. Selected exact attention matches an explicit full-logit mask reference under GQA with V96.
4. Incremental A_K growth, unchanged historical codes, recent eviction, and independent copied state.
5. Two-layer Qwen3 C1 full-budget equivalence through prefill and three decode steps, repeated with fresh request caches and compressed V6.

The `basis` environment lacks pytest. The five independent test functions were executed directly with their assertions; no environment dependencies were installed.

Real model: frozen Qwen3-8B-Base and `qwen3_8b_c1_v96_32f4h_s32768_als6`, same saved LongBench inputs, C1 Triton prefill, greedy generation capped at8 tokens for smoke. Jobs8301137_0–1 used one L40S each,2 CPUs and48GiB host RAM.

| Smoke input | Prompt tokens | Generated tokens | First/prefill logits | Peak allocated GPU memory | Job elapsed |
|---|---:|---:|---|---:|---:|
| Sample119,2wikimqa |1192|3(EOS)|Bitwise equal to full-K C1-V96|15.317GiB|27s|
| Sample175,qmsum |30431|8(cap)|Bitwise equal to full-K C1-V96|23.913GiB|40s|

All36 layers passed cache-width, state-length, finite-output and decode-step checks. Long input:7 sparse decode steps; final support2112 tokens/query head. At the **last smoke decode step**, physical union across four GQA heads averaged4324.6354 tokens over36 layers ×8 groups, range2620–6471. This is not an average across all generated steps or benchmark prompts. Short input:all1194 available tokens were selected at the last step.

All smoke jobs exited0 with no retries, non-finite values, or linear-solve failures. The only logged runtime warning was optional FuzzyWuzzy acceleration, unrelated to these unscored smoke outputs. These are not benchmark accuracy scores; no full192-prompt run was submitted.

## Commands and artifacts

Environment:`/home/zhangal/.conda/envs/basis/bin/python`; working directory:`/deac/csc/yangGrp/zhangal/BasisServe-CALS`.

CPU tests:

```bash
OMP_NUM_THREADS=2 /home/zhangal/.conda/envs/basis/bin/python -c 'import runpy; s=runpy.run_path("tests/test_c1_lrqk.py"); [(f(), print(n, "PASS")) for n,f in s.items() if n.startswith("test_")]'
```

GPU smoke(command differs only by smoke-index0 or1):

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_lrqk.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke --smoke-index 0
```

The entry point also exposes `evaluate` and `summarize` stages for the same frozen192-prompt set; they have not been run. Defaults are recorded in each sample JSON. Smoke records:`results/evaluation/longbench_c1_lrqk/smoke`. Logs:`logs/c1-lrqk-smoke-8301137_{0,1}.log`. Temporary submission file deleted after submission. No GitHub commit or push performed.
