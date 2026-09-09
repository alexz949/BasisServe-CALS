# LongBench: two-sided KL average80 with original dense prefill

One new arm on the same frozen 192 LongBench-v1 prompts: original dense prefill, cache conversion, then full-exact-K compact C1 decode. Compare against completed original dense inference and uniform C1-V80 dense-prefill results. No routing, sparse pages, sidecar, offload, new calibration or factor fitting.

Checkpoint: `results/checkpoints/qwen3_8b_c1_twosided_r80_c4_32x32k`. Alpha=1; profiling anchor64; probes32/96; rank bank32/48/64/80/96/112/128. Rank counts are 3/3/11/6/5/3/5 layers, average exactly80. All eight KV groups in a layer have the same rank. Total latent rank across all layers/groups is23,040, matching uniform V80. Runtime retains actual per-layer ranks without padding to128.

Layer ranks, indices0–35:

`128,128,32,48,48,48,80,64,128,128,112,96,96,112,64,96,80,64,80,80,64,64,112,128,96,64,64,64,32,64,80,64,64,96,80,32`

Factor bank: the same C4 32 x 32K fit and4 x 32K held-out covariance set, six ALS encoder sweeps, full-layer output objective. Independent KL profile: 32 x 32K C4 windows, indices36–67; confirmation:12 x 32K, indices68–79. Alpha1 uses held-out local relative MSE; sampled terminal KL uses1,024 positions/window. Rank128 is the identity encoder/dense-O endpoint. LongBench is not used for fitting or allocation.

Important qualification: this is a comparison of existing exported checkpoints, not strictly a rank-index-only ablation. Allocation export canonicalized encoder coordinates and performed a closed-form decoder refit for non-anchor, non-full-rank layers. All six selected rank80 layers (6,16,18,19,30,34) have encoder and decoder tensors that are not bitwise equal to the original uniform80 tensors. They originate from that same factor bank; the differing coordinates and decoder closure are preserved, not silently replaced. This evaluation itself performs no refit.

Every prefill restores the original Qwen3Attention modules and a fresh DynamicCache. First-token argmax is checked against the saved original dense baseline. Cached BF16 V128 is multiplied by each selected C1 encoder; exact K tensor identity is unchanged. Subsequent tokens use compressed V and the C1 output decoder with the SDPA backend, as in the completed uniform dense-prefill experiment. Original V/O modules are retained for the next prompt; this is an accuracy run, not a memory/speed benchmark.

Same192 saved inputs, official prompts without chat template, references, greedy loop, EOS and task generation caps128/64/32/32/512/512. Tasks: qasper, multifieldqa_en, hotpotqa,2wikimqa,gov_report,qmsum;32 prompts each. Official QA F1 and summary ROUGE-L, on0–100 scale, six-task arithmetic mean.32K is the input-plus-generation cap; actual inputs1,192–30,431. This is a six-task pilot, not full LongBench.

Environment: basis. Four independent L40S workers on lovelace,48 prompts each,2 CPUs and48GiB host memory per worker. Outputs: `results/evaluation/longbench_c1_kl_denseprefill_32k`; logs:`logs/lb-kl-*`. Previous artifacts are unchanged.

Smoke job8300998 completed in32 seconds, exit0. Shortest and longest prompts passed exact repeated-logit and original-dense-prefill-logit checks, cache projection spot checks at all36 layers, factor installation checks, K tensor identity checks and actual-rank cache shape checks. Smoke is not a scored accuracy run.

Formal array:8300999_0–3. Summary is submitted with an after-success dependency. Temporary submission scripts are deleted after submission. CPU summary re-decodes and officially re-scores all192 predictions and checks EOS/caps, input identity, shard coverage and first-token agreement.

## Completed results

All192 prompts completed and passed the score audit. Summary job8301003 completed in18 seconds. Formal worker durations were9:25,7:44,9:30 and10:24; all jobs exited0 without retries. Peak allocated memory23.370GiB. First-token agreement with dense192/192. Generation-cap exits without EOS42/192, versus56/192 for uniform80 and16/192 for dense. No non-finite logits or assertion failures.

| Task | Dense | Uniform80 | Two-sided KL avg80 |
|---|---:|---:|---:|
| qasper |39.3018|34.9041|35.2802|
| multifieldqa_en |52.8498|47.9694|48.6936|
| hotpotqa |60.8872|59.3363|62.1941|
| 2wikimqa |50.1190|47.2545|46.9940|
| gov_report |29.1896|29.3136|28.6770|
| qmsum |26.1460|27.3302|26.9735|
| Mean |43.0822|41.0180|41.4687|

Two-sided minus uniform: +0.4507 score points; two-sided minus dense: -1.6135 points. Paired with uniform:46 improvements,37 regressions,109 ties. Paired with dense:42 improvements,56 regressions,94 ties. These are sample-level official score comparisons, not binary accuracy counts or a significance claim.

Result SHA256: `58492193ff089862c625a4ad128f7aebb67a5aef4f5f00550b8c3529fdcdc58a`. Final result and recorded runtime source hashes were independently checked after scoring. Detailed outputs and the English summary are in `results/evaluation/longbench_c1_kl_denseprefill_32k`.

## Commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_twosided_denseprefill.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_twosided_denseprefill.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --shard-index "$SLURM_ARRAY_TASK_ID"
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_twosided_denseprefill.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```
