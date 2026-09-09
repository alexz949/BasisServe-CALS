# Dense prefill followed by full-exact-K C1-V80 decode

This experiment adds one arm to the frozen 192-prompt LongBench-v1 pilot. It does not run routing or PaLU and does not refit factors. The previous results remain unchanged.

Every prompt starts with a fresh DynamicCache and the original Qwen3Attention modules, including the original V128 and output projections. After dense SDPA prefill, the first generated token is the dense argmax. Then each cached BF16 V128 is projected through the checkpoint's C1 encoder into BF16 V80. The exact K tensors are retained unchanged. Subsequent generation uses the installed C1 V projection and output decoder with full exact-K attention and the SDPA backend, matching the previous full-K arm's decode backend.

Original attention modules are retained outside the model and restored before EVERY prefill. This permits multiple prompts per worker even though the existing cache-transition helper releases its own dense V/O references. Q/K projections and Q/K norms are shared by identity between original and compressed attention modules. There is no sparse selection, sidecar, adaptive budget, or CPU offload.

Checkpoint: `results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6`, unchanged uniform V80, C4 32 x 32K fit and 4 x 32K held-out. Model: Qwen3-8B-Base, BF16. Same saved token IDs, references, greedy loop, EOS and task generation caps as the completed dense baseline and C1 four-arm run. Tasks: qasper, multifieldqa_en, hotpotqa, 2wikimqa, gov_report, qmsum; 32 prompts each. QA uses official F1 and summaries official ROUGE-L. The headline mean is the arithmetic mean of six task scores.

The 32K setting is an input-plus-reserved-generation cap. Actual prompt lengths are 1,192–30,431 tokens; none were truncated. This is a six-task pilot, not full LongBench or a fixed-32K-length test.

Smoke uses the shortest and longest saved prompts with a four-token generation cap, repeats complete executions, checks identical logits, compares prefill logits exactly to independently executed original dense inference, and checks first/last cached V-row projections at every layer. Every formal prompt checks first-token agreement with the saved dense baseline, K identity across conversion, and all 36 layers' K128/V80 cache shapes. CPU summary re-decodes and officially re-scores all predictions, checks EOS and caps, shard coverage, and input identity.

Environment: `/home/zhangal/.conda/envs/basis/bin/python`. Formal run: four independent L40S workers on lovelace, 48 prompts each, 2 CPUs and 48 GiB host memory per worker. Output: `results/evaluation/longbench_c1_denseprefill_32k`. Logs: `logs/lb-c1dp-*`. This is a quality experiment, not an optimized memory or timing benchmark; the original dense projections are deliberately retained for subsequent prompts.

## Commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_denseprefill.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_denseprefill.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --shard-index "$SLURM_ARRAY_TASK_ID"
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_denseprefill.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```

## Execution record

Smoke job `8300991` completed successfully in 33 seconds. Shortest prompt: sample 119, 1,192 tokens; longest: sample 175, 30,431 tokens. Both passed exact repeated-logit and independently executed dense-prefill-logit checks, as well as first/last-row C1 cache projection checks across all layers. These smoke predictions are not scored as benchmark results.

CPU checks passed: full-rank dense-prefill/cache-transition decode agrees with dense; reduced-dimensional cache projection agrees with an independent einsum; three repeated attention-module switch/restore cycles preserve original module identity.

Formal evaluation array: `8300992_0` through `8300992_3`, with 48 prompts per worker. CPU summary/audit: `8300996`, dependent on successful completion of all four workers. Temporary submission files were deleted after submission. No previous results or unrelated jobs were changed.

## Completed results

All 192 predictions completed and passed the CPU score audit. Worker durations were 10:08, 6:56, 11:23 and 13:44; CPU summary/audit took 18 seconds. All jobs exited with code 0, without retries. Maximum allocated GPU memory was 23.366 GiB. No non-finite logits or assertion failures were observed. The optional FuzzyWuzzy acceleration warning does not affect the selected F1/ROUGE-L metrics.

| Task | Original dense prefill/decode | C1 prefill/decode | Dense prefill, C1 decode |
|---|---:|---:|---:|
| qasper | 39.3018 | 19.6094 | 34.9041 |
| multifieldqa_en | 52.8498 | 30.7387 | 47.9694 |
| hotpotqa | 60.8872 | 29.7403 | 59.3363 |
| 2wikimqa | 50.1190 | 31.2642 | 47.2545 |
| gov_report | 29.1896 | 27.2987 | 29.3136 |
| qmsum | 26.1460 | 26.3270 | 27.3302 |
| Six-task mean | 43.0822 | 27.4964 | 41.0180 |

The new arm is 13.5216 score points above the prior C1-prefill arm and 2.0642 points below original dense inference. First tokens agree with original dense on 192/192 prompts. Generation-cap exits without EOS: 56/192, compared with 100/192 for C1-prefill and 16/192 for original dense. Paired scores versus dense: 42 improvements, 60 regressions, 90 ties; versus C1-prefill: 96 improvements, 45 regressions, 51 ties.

This tests the full prefill-path change, including the switch from compressed-V Triton prefill to original dense SDPA prefill. It does not independently separate prefill compression error from prefill backend numerical effects. The C1 decode backend, checkpoint and full-K support remain matched to the old full-K arm.

Detailed records: `results/evaluation/longbench_c1_denseprefill_32k/result.json`; score audit: `audit.json`; English summary: `summary.md` in the same directory. Final result and recorded source hashes were checked again after summary completion.
