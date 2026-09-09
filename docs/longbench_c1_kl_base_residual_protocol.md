# Allocated C1 plus matched Base16 and residual R8

## Scope

Add page-sparse Base16/R8 decoding to the frozen alpha1 two-sided-KL average-rank80 C1 checkpoint. Preserve original dense prefill and compare on the same192 LongBench-v1 prompts with the completed KL full-exact-K result41.4687, uniform80 full-exact-K41.0180 and original dense43.0822. The new sparse arm has not produced benchmark scores at submission time. No old results are overwritten.

Checkpoint: `results/checkpoints/qwen3_8b_c1_twosided_r80_c4_32x32k`. Payload rank varies by layer between32 and128, average exactly80. The C1 checkpoint and rank schedule are frozen.

## Why the router is refitted

The old `mse_base_qgram32_r8` router consumes the original uniform C1-V80 coordinates. Its Base left factor has shape `(8,80,16)`. The allocated C1 has different widths and coordinates; even its rank80 layers were gauge-canonicalized during allocation export. The old Base cannot be directly attached. Residual is defined relative to that Base, so it also needs a matched fit.

This run fits a new Base and residual using existing immutable captures. It does not add a second V80 cache, truncate/pad old Base factors, train model weights, or change C1 payload factors.

## Router fitting

Use existing dense-model C4 captures:64 x32K fit windows(indices0–63),16 x32K diagnostic windows(indices64–79), all36 layers,8 KV groups. Base uses every token in the fit windows. Project raw V through the selected per-layer C1 encoder and solve affine rank16 pre-RoPE Key regression by the existing closed-form moment-whitening/truncated-SVD RRR routine. No optimizer or query sampling is used for Base.

For residual, retain the previously selected fit-only Query-Gram positions:32 Q per layer/window,8 from each8K stratum. Reuse the existing selected-Q captures and manifest unchanged. Compute exact post-RoPE residual relative to the new Base, then construct causal non-sink Page-Fisher statistics. Fit uniform R8 with40 BCD sweeps and PCG(relative damping1e-5,tolerance1e-5,max100 iterations). No Adam. Fit observations/head:64 x32=2048; diagnostic observations/head:16 x32=512. Diagnostic windows report final loss; they do not select the final factor endpoint or ranks.

Page size32; page0 excluded from residual Fisher and pinned at deployment. Target physical token budget2048. New bank: `results/checkpoints/c1_kl_b16r8_qgram`. Factors are FP32 on disk and cast to BF16 for deployment, matching the old router convention.

## Evaluation

Original Qwen3Attention dense K128/V128 prefill with a fresh RoutingDynamicCache for every prompt. Preserve original attention modules for subsequent prompts. Project cached V through each selected C1 encoder; K tensor identity remains unchanged. Dense-prefill argmax supplies the first generated token.

Decode on all36 layers with matched Base16/R8 routing, Page32/B2048, pinned page0, no adaptive budget and no forced current page. Per-head normalized non-sink page scores are combined by the existing GQA maximum rule. Attention uses exact K and resident C1 latent values within selected pages. Explicit full-support decode masks select the existing native BF16 sparse implementation. No routing during prefill.

This is a GPU-resident accuracy oracle with materialized `[post-RoPE Base K128, residual R8]` sidecars, not CPU offload or a performance benchmark. Each payload cache retains its actual layer rank. Uniform versus KL retains the previously documented checkpoint-export caveat; the KL sparse/full comparison uses exactly the same C1 payload checkpoint.

Same192 saved LongBench-v1 prompts, six tasks with32 prompts each; official QA F1 and summary ROUGE-L,0–100 scale and six-task arithmetic mean. Same references, input token IDs, greedy decoding, EOS and task caps128/64/32/32/512/512.32K is the input-plus-reserved-generation cap; actual prompts1,192–30,431 tokens. Not full LongBench.

Evaluation smoke covers shortest and longest inputs with four generated tokens, repeats the whole execution, compares logits exactly, independently checks dense-prefill logits, verifies actual-rank payload caches and Base128+R8 sidecar dimensions, and counts native sparse attention dispatches at all36 layers. Formal run checks every first token against the saved dense baseline. CPU summary re-decodes/re-scores all outputs, checks task means with the official scorer and verifies EOS/caps, source identity and shard coverage.

Output: `results/evaluation/longbench_c1_kl_b16r8`. Logs: `logs/kl-router-*`. Environment: `/home/zhangal/.conda/envs/basis/bin/python`; L40S on lovelace,2 CPUs and48GiB host memory per GPU worker.

## Submitted pipeline

| Stage | Job | Status at submission |
|---|---|---|
| Full-protocol fit checks: layers0(V128),35(V32) |8301005_0–1|Completed,exit0|
| Fit all layers, reusing layers0/35 |8301007_0–3|Four-GPU array submitted|
| Dense-prefill / sparse-decode smoke |8301011|After successful full fit|
| LongBench evaluation |8301012_0–3|After successful decode smoke|
| CPU score summary/audit |8301013|After successful evaluation|

All downstream stages require successful predecessors; invalid dependencies cancel downstream jobs. Temporary sbatch files were deleted after submission. No unrelated jobs were changed.

Layer0 full fit took84.6 seconds: fit/diagnostic Page-Fisher NMSE0.321425/0.632827. Layer35 took99.4 seconds:0.215729/0.317727. Both used the full64 fit/16 diagnostic windows and Q32, not a reduced-sample toy run. Factors were finite and had the expected dimensions. These are fitting diagnostics, not LongBench accuracy. The RoPE constructor emits a deprecation warning; it did not prevent either fit from completing.

## Completed pipeline and results

All 36 router layers completed. Full-fit worker durations: 14:27, 16:01, 16:20 and 14:27; the earlier full-protocol layers 0/35 were reused. All layer-file hashes, shared provenance, actual payload-rank Base dimensions, finite factors and finite fit/diagnostic losses were independently checked. Decode smoke job 8301011 completed in 58 seconds with exit 0. Both shortest and longest prompts passed repeated-logit, independently executed dense-prefill-logit, actual-rank cache and native sparse-dispatch checks.

All 192 LongBench predictions completed. Evaluation worker durations: 6:12, 5:15, 7:12 and 9:12; CPU score summary/audit took 18 seconds. Every job exited 0 without retries. First-token agreement with original dense: 192/192. Every decode step dispatched native conditional sparse attention at all 36 layers. Peak allocated memory: 26.060 GiB. Cap exits without EOS: 37/192. No non-finite logits or assertion failures; RoPE constructor deprecation and optional FuzzyWuzzy acceleration warnings did not prevent completion.

| Task | Dense | Uniform80 full K | KL avg80 full K | KL avg80 + Base16/R8 sparse |
|---|---:|---:|---:|---:|
| qasper |39.3018|34.9041|35.2802|34.9063|
| multifieldqa_en |52.8498|47.9694|48.6936|49.2924|
| hotpotqa |60.8872|59.3363|62.1941|62.6062|
| 2wikimqa |50.1190|47.2545|46.9940|46.9940|
| gov_report |29.1896|29.3136|28.6770|28.8272|
| qmsum |26.1460|27.3302|26.9735|26.2185|
| Mean |43.0822|41.0180|41.4687|41.4741|

Sparse minus KL full-K: +0.005358 score points. Paired scores: 34 improvements, 36 regressions, 122 ties. Sparse minus dense: -1.608126 points. Equal aggregate scores do not imply identical predictions or exact attention. This is the previously used 192-prompt pilot, not a proof of general accuracy equivalence.

Final result SHA256: `96ee7f31315e7159eecb440ccf9fa9941f70e5ac665752fa908e955499cdac42`. Result hash, recorded runtime source hashes, full sample coverage and sparse-dispatch counts were independently verified after scoring. Detailed result, audit and English summary are in `results/evaluation/longbench_c1_kl_b16r8`.

For clarity, the earlier `mse_base_q32_ruler32k`, `qgram32_ruler32k`, `terminal8_qgram_ruler32k` and `terminal8_uniform_ruler32k` RULER runs used shared full-attention **C1-V80 Triton prefill**, including the first generated token. They were not original dense-prefill runs. The current LongBench experiment uses original dense prefill as specified above. No new RULER evaluation was launched in this pipeline.

## Commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_c1_kl_base_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --layers 0
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_c1_kl_base_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --layers 35
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_c1_kl_base_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --shard-index "$SLURM_ARRAY_TASK_ID"
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_kl_router.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_kl_router.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --shard-index "$SLURM_ARRAY_TASK_ID"
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_kl_router.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```
