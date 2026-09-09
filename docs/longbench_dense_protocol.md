# Dense K/V baseline for the frozen LongBench C1-V80 pilot

## Scope and controls

Add one unmodified Qwen3-8B-Base BF16 dense K128/V128 arm to the completed four-arm C1-V80 experiment. No C1 installation, routing sidecar, factor refit, page selection or CPU offload in this new arm. Both prefill and decode use dense attention. The first generated token comes from the dense model, not the C1 model.

Reuse all 192 saved prompts exactly: qasper, multifieldqa_en, hotpotqa, 2wikimqa, gov_report and qmsum, 32 prompts per task. Same saved input token IDs, references and official completion prompts without a chat template. Greedy decoding; model/tokenizer EOS; generation caps 128/64/32/32/512/512. Input plus reserved generation cap 32,768 tokens; actual input lengths 1,192–30,431, mean 9,244.59, no truncation. This is not a full LongBench run or a fixed-32K-length test.

Metrics: official LongBench-v1 F1 for QA, ROUGE-L for summaries, maximum across reference alternatives, followed by per-task means and the six-task arithmetic mean. Scores use a 0–100 scale and are not all accuracies.

Full-sequence prefill and single-token decode use unmodified Transformers SDPA. Decode reuses the same explicit full-support-mask greedy loop as the C1 experiment. This compares original dense inference with the C1 pipeline; it is not a matched-kernel V-compression-only ablation because the previous C1 prefill uses Triton.

## Validation and resources

Environment: `/home/zhangal/.conda/envs/basis/bin/python`. Four L40S on `lovelace` for the formal run, one independent 48-prompt shard per GPU. Each worker requests 2 CPUs and 48 GiB host memory. No unrelated jobs are modified.

Input validation checks the previously audited C1 result hash, identical model config, dataset manifest, source rows and input token hashes. Smoke covers the shortest and longest prompts with four generated tokens; it independently repeats prefill/decode and compares logits exactly, then compares token IDs against native `model.generate()`. Every prompt checks all 36 cache layers contain BF16 K and V of shape `(1, 8, cached_length, 128)`.

CPU summary verifies all shard indices, input/reference identity, generated token decoding, EOS/caps and all 192 official scores, including agreement with the official rounded per-task scorer. It combines the new dense baseline with the unchanged four-arm result, preserving paired per-sample comparisons. Original C1 artifacts are not overwritten.

Output: `results/evaluation/longbench_dense_32k/`, separate smoke/evaluate directories, per-prompt JSON predictions and provenance, `result.json`, `summary.md` and `audit.json`. Logs: `logs/lb-dense-{smoke,evaluate,summary}-{job}[_task].out/.err`. Temporary sbatch files are deleted after submission.

## Job record

Job IDs: smoke `8300958` (completed, exit 0, 29 seconds); formal four-worker array `8300959`; CPU summary/audit `8300963`, dependent on all four workers succeeding. The shortest-input smoke generated EOS after two tokens; the longest generated four tokens. Both passed exact repeated-logit and native-generation token checks. Peak allocated GPU memory in these smoke runs: 15.55 and 22.49 GiB respectively. Smoke scores are deliberately unset and are not included in formal evaluation.

## Completed results

All 192 dense predictions completed and passed summary validation. Formal worker elapsed times: 4:23, 3:45, 4:56 and 5:07. CPU summary and score audit: 18 seconds. All five jobs exited with code 0; no retries. Maximum formal allocated GPU memory: 22.48 GiB. The optional FuzzyWuzzy acceleration warning does not affect the F1/ROUGE-L metrics used here.

| Configuration | Six-task mean | Difference from dense K/V |
|---|---:|---:|
| Dense K128 + dense V128 | 43.0822 | 0.0000 |
| Full exact K + C1-V80 | 27.4964 | -15.5859 |
| Exact sparse K + C1-V80 | 29.1406 | -13.9416 |
| Base16/R8 Query-Gram Q32 + C1-V80 | 28.8070 | -14.2752 |
| Base16/R8 terminal Q32 + C1-V80 | 27.7305 | -15.3517 |

The full per-task table and paired changes are in `results/evaluation/longbench_dense_32k/summary.md`. Dense results differ from full-K/C1-V80 most on the four QA tasks. The latter uses no routing, so the gap is already present without page selection. This experiment does not identify whether compression, the C1 attention implementation, or their combination is responsible; prefill backends are not matched.

Additional saved-output checks: the first generated token differs between dense and full-K/C1-V80 on 47 of 192 prompts. Dense reaches the generation cap without EOS on 16 prompts; full-K/C1-V80 does so on 100 prompts. These are generation diagnostics, not explanations of the accuracy difference.

Audit: `results/evaluation/longbench_dense_32k/audit.json`, 192 predictions verified. Result SHA256: `9ae834f2b90cefef0cf07061d211f263fd45448be645f51eb2ba5385420cdac8`. Original four-arm results remain unchanged.

## Reproduction commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`.

### Smoke

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_dense.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
```

### Formal evaluation (four workers)

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_dense.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --shard-index "$SLURM_ARRAY_TASK_ID"
```

### CPU summary and score audit

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_dense.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```
