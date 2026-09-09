# LongBench C1-V96 prefill and decode

One new arm only: Qwen3-8B-Base, uniform C1-V96 on all36 layers, full exact K, no routing. Full-sequence causal C1 prefill uses the existing Triton kernel; decode uses full-K SDPA, matching the old V80 full-K arm. First generated token comes from C1-V96 prefill. No original-dense-prefill V96 experiment, offload, allocation, or refitting is included.

Checkpoint: `results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6`. Same C4 capture manifest and all non-rank fitting settings as the V80 checkpoint:32x32768 fit,4x32768 diagnostic, activation-weighted SVD initialization, six ALS sweeps and final decoder refit. Held-out local relative MSE: V80 0.0944713082; V96 0.0564016098. These are local fitting diagnostics, not LongBench scores.

Frozen dataset: `results/datasets/longbench_c1_32k`. Same192 prompts, six tasks x32: qasper, multifieldqa_en, hotpotqa,2wikimqa,gov_report,qmsum. Exact saved token IDs, official completion prompts, no chat template, greedy sampling, unchanged EOS and task-specific output caps. Input+reserved-output cap32768; actual inputs1192–30431 tokens. QA uses official F1; summaries official ROUGE-L, best alternative reference. Headline mean is arithmetic over six tasks. Not full LongBench.

Every sample gets a fresh RoutingDynamicCache without routing factors. A forwarding observer verifies36 real C1-V96 Triton prefill dispatches. All36 final cache shapes are checked: K128 and V96, BF16. Decode uses the existing explicit full-support-mask greedy loop with finite-logit checks. Exact K means uncompressed keys on this model trajectory, not the separate dense baseline's keys.

Smoke uses shortest and longest saved prompts with four-token caps, repeats each full execution, and requires bitwise-identical logits and generated IDs. Smoke is not scored. Formal summary independently re-decodes IDs, checks EOS/caps, re-scores with official metrics, verifies source/input/checkpoint provenance and four-shard coverage. Existing dense and C1-V80 results are reused without modification. Comparing V80 and V96 changes capacity in both prefill and decode, not prefill alone.

Environment: `/home/zhangal/.conda/envs/basis/bin/python`, BF16, TF32 disabled. Four independent L40S workers,2 CPUs and48GiB host memory each. Other A100/H200 GPUs were allocated; older idle V100 GPUs do not support the matched BF16 setting. Output: `results/evaluation/longbench_c1_v96`; logs:`logs/lb-v96-*`. No production kernel edits.

## Commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_v96.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_v96.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --shard-index "$SLURM_ARRAY_TASK_ID"
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_v96.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```

## Execution record

Smoke job8301026 completed in32 seconds with exit0. Both shortest and longest prompts passed all36-layer cache/dispatch checks and bitwise repeated-logit checks; no non-finite logits. The shortest stopped on EOS after3 tokens and the longest used the4-token smoke cap; these are not benchmark scores.

Formal array8301027_0–3 assigns48 prompts per L40S worker. CPU summary8301031 depends on all four workers succeeding. Temporary submission scripts were deleted after submission. Logs use the `lb-v96` prefix. Existing result folders and production code are unchanged.

## Completed results

All192 predictions passed the independent CPU summary audit. Worker durations:8:36,7:18,9:12,11:23. CPU summary:18 seconds. All jobs exited0, without retries. No non-finite logits or failed assertions. The optional FuzzyWuzzy acceleration warning does not affect the selected F1/ROUGE-L metrics. Maximum allocated GPU memory:21.613GiB.

| Task | Dense | C1-V80 prefill/decode | C1-V96 prefill/decode |
|---|---:|---:|---:|
| qasper |39.3018|19.6094|32.4074|
| multifieldqa_en |52.8498|30.7387|42.8418|
| hotpotqa |60.8872|29.7403|57.9676|
| 2wikimqa |50.1190|31.2642|40.2530|
| gov_report |29.1896|27.2987|30.5532|
| qmsum |26.1460|26.3270|27.6797|
| Six-task mean |43.0822|27.4964|38.6171|

V96 minus V80:+11.1207 score points; V96 minus dense:-4.4651. Paired versus V80:91 improvements,44 regressions,57 ties; versus dense:59 improvements,65 regressions,68 ties. V96 first-token agreement with dense:156/192. Generation-cap exits without EOS:39/192(V80:100/192; dense:16/192).

The observed score recovery supports sensitivity to C1 capacity in this pilot. The experiment changes V rank in both prefill and decode, so it does not separately identify the prefill contribution, prove an error-accumulation mechanism, or test a V96 dense-prefill arm. No confidence interval or independent benchmark generalization claim is made.

Detailed result:`results/evaluation/longbench_c1_v96/result.json`; SHA256:`1efcd87d4d887e4f47080a301089af59c14a328c6b827d52dd981c2adab6b01e`. `audit.json` records official score, generated text/token, stopping-rule, cache-shape, prefill-dispatch and shard-coverage checks. Runtime source hashes and all36 checkpoint artifact hashes were checked again after completion. English result summary:`results/evaluation/longbench_c1_v96/summary.md`.
