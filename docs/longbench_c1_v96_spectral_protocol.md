# LongBench: prompt-specific closed-form residual R8

## Configuration

- Qwen3-8B-Base, basis environment, V100 FP16 model and KV cache; FP32 spectral fitting.
- Same frozen 192 LongBench prompts: six tasks, 32 each, original greedy decoding, EOS, task generation caps and scoring.
- Full causal C1-V96 prefill using the previously verified V100 memory-efficient SDPA implementation. C1-V96 payload retained during decode. No dense-V prefill.
- All 36 layers: fixed C4 Base16, prompt-specific residual rank8, Page32/B2048, pinned page0, GQA normalized page-mass maximum, no adaptive budget or forced current page.
- Per layer/group, use all post-RoPE residual Key rows and 256 prefill Q positions from the same query_positions rule as the three-prompt spectral diagnostic. Pool only the four Q heads associated with that KV group. No answer labels or generated queries enter fitting.
- Fit E/U using FP32 uncentered moments and shared-query-metric eigensolves with relative query ridge 1e-5. No PCG, BCD or Adam. Convert to model precision in the existing routing path; encode the prefix once with the new basis, then append new token codes with frozen E/U throughout decode.
- GPU-resident exact K and materialized Base128+R8 metadata. This is generation-quality evaluation, not actual CPU offload or throughput benchmarking.

## Comparisons and verification

Reuse completed FP16 full-K/C1-V96 records under results/evaluation/longbench_lrqk_fp16/full/evaluate. Validate sample identity, scores, and retain all reference file hashes. First generated tokens must match this FP16 reference. Old BF16 dense results are contextual only. A precision-matched offline-R8 comparison is not included in this run.

Smoke uses the shortest and longest inputs, checks deterministic repeated generation/logits, full-prefill logits, cache shapes/dtypes and sparse dispatch counts. Formal evaluation uses four independent model replicas, each assigned 48 prompts. Summary verifies all 192 predictions, scores, EOS/caps, shard coverage and reference first tokens.

The pending L40S smoke 8301247 was cancelled on user request. V100 smoke 8301248 completed successfully in 80 seconds: samples119/175, shortest/longest inputs, deterministic logits and full-prefill agreement verified. No NaN reported. Formal array: 8301249_0–3, four V100 model replicas. Outputs: results/evaluation/longbench_c1_v96_spectral_fp16. Logs: lbs_smoke_*, lbs_eval_* and lbs_sum_* at repository root.

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_longbench_c1_v96_spectral.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
```

Formal commands replace the stage with evaluate and add --shard-index 0, 1, 2 or 3. Final stage: summarize. Jobs use two CPUs per GPU and the basis environment.
