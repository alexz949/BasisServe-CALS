# LongBench-v1 C1-V80 four-arm32K-cap pilot

## User-requested comparison

1. full exact-K + C1-V80;
2. sparse exact-K selected with exact FP32 QK + C1-V80;
3. Base16/R8 routing, Query-Gram Q32 (eight calibration Q per8K bin across32K);
4. Base16/R8 routing, terminal Q32 (32 calibration Q in final8K).

All use the same Qwen3-8B-Base model and frozen C1-V80 checkpoint. The proxy banks are mse_base_qgram32_r8 and mse_base_q32_r8, NOT the newer terminal-Q8 banks. No factors are fit on LongBench. No model/weights/decoding parameters are tuned after observing this benchmark.

Six-task pilot, as stated before launch: qasper, multifieldqa_en, hotpotqa,2wikimqa,gov_report,qmsum;32 prompts per task,192 total. Selection is deterministic by SHA256(seed73:task:source_id), independent of labels, lengths and predictions. This is not full LongBench or LongBench-E.

## Length, prompts, metrics

Total sequence ceiling32768 tokens includes official task generation budget. Official base-completion prompts, no chat template. Keep original token IDs for inputs that fit. If oversized, preserve equally sized token prefix/suffix without decode/re-tokenize; this deliberately guarantees the exact total token cap and is documented as a token-level variant of middle truncation.

This fixed sample has NO truncated prompts. Actual lengths1192–30431 tokens; no padding to32K. Mean prompt lengths: qasper5193, multifieldqa_en6663, hotpotqa13001,2wikimqa7264,gov_report9869,qmsum13477 (rounded).

Use official LongBench-v1 metrics: QA F1 for four QA tasks, ROUGE-L for gov_report/qmsum; maximum over alternate references, then task arithmetic means. Official generation caps128/64/32/32/512/512. Greedy decoding with tokenizer/model EOS. Scores are not all exact-match accuracies. Aggregate six-task mean is a pilot mean, not the official whole-benchmark leaderboard metric.

Official source cloned under external/LongBench, commit2e00731f8d0bff23dc4325161044d0ed8af94c1e. Dataset HF revision, archive/task JSONL hashes, prompt/scorer hashes, sample IDs and exact token-input hashes are stored in results/datasets/longbench_c1_32k/manifest.json and samples.json.

## Attention controls

All four arms share one full C1-V80 Triton prefill and its first generated token; attention differences apply during subsequent decode. Independent immutable-prefix cache forks per arm. Full exact-K uses SDPA decode. Sparse arms use native BF16 selected-exact-K/C1-V attention, explicit full-support masks and all36 layers, Page32/B2048, pinned page0, no adaptive budget or forced current page. Short inputs select all available pages.

Exact sparse computes exact FP32 query-key logits from BF16 Q/K and applies the same per-head non-sink page-mass normalization, GQA-head max and physical64-page selector. It overrides proxy-based selection, but evaluates selected exact QK and C1-V through the unchanged native payload path. A terminal-bank sidecar is computed as plumbing and is not used for exact selection. This is an accuracy oracle with GPU-resident exact K, not an offload/latency benchmark.

## Implementation and verification

New scripts: evaluation/prepare_longbench_c1.py, evaluation/longbench_exact_pages.py, evaluation/eval_longbench_c1_fourarm.py. Tests in tests/test_longbench_c1_fourarm.py.

Four CPU tests passed: token cap/prefix-suffix, sample-ID selection independence, exact selector independence from proxy with manual sparse/full-support equivalence including partial pages, official F1/ROUGE and alternative references. Syntax/whitespace checks passed.

Environment basis. Official scorer dependencies installed in isolated results/tools/longbench_deps with --no-deps: rouge1.0.1, jieba0.42.1, fuzzywuzzy0.18.0; no global basis package versions changed. FuzzyWuzzy warns about optional python-Levenshtein absence; no code tasks use its similarity metric in this pilot.

Prepare job8300875 completed in18seconds. Smoke8300876 completed in32seconds, min-length sample119 and max-length sample175, four arms with repeated logits exactly equal, cache prefix unchanged, exact selector call counts checked. Smoke cap4 is not a task score.

Formal evaluation8300877 array0–3: four L40S on lovelace,2 CPUs/48GiB host memory per task,48 prompts per worker with all four arms per prompt. CPU summary8300878 depends on all four succeeding,2 CPUs/12GiB. Formal time limit6hours is a limit, not an estimate. Invalid dependencies cancel summary. Temporary sbatch scripts removed. No unrelated jobs modified or GitHub upload authorized.

Summary independently rescores every saved prediction using official metrics and verifies rounded per-task scores against official scorer. Per-sample predictions, generated tokens, lengths, caps, times and source hashes are saved. Resume accepts only completed samples with exactly matching protocol and input.

Logs: logs/lb-c1-{prepare,smoke,evaluate,summary}-{job}[_task].out/.err. Output root results/evaluation/longbench_c1_32k.

## Completed results

All 192 prompts and 768 arm predictions completed. Every formal worker exited with code 0; no failed or retried jobs. Worker elapsed times were 37:16, 36:27, 43:17 and 46:04. The official-score summary (8300878) completed in 34 seconds; independent audit (8300882) completed in 23 seconds. Environment: `basis`, four L40S GPUs on `lovelace`.

| Task | Full exact K + V80 | Sparse exact K + V80 | Base16/R8 Query-Gram Q32 | Base16/R8 terminal Q32 |
|---|---:|---:|---:|---:|
| qasper | 19.6094 | 18.8967 | 19.5324 | 19.7245 |
| multifieldqa_en | 30.7387 | 29.1173 | 29.8343 | 29.3452 |
| hotpotqa | 29.7403 | 32.8628 | 36.3228 | 30.6941 |
| 2wikimqa | 31.2642 | 37.0228 | 31.5359 | 33.3671 |
| gov_report | 27.2987 | 30.0715 | 29.4190 | 27.8504 |
| qmsum | 26.3270 | 26.8727 | 26.1977 | 25.4020 |
| Six-task arithmetic mean | 27.4964 | 29.1406 | 28.8070 | 27.7305 |

Query-Gram minus terminal: +1.0765 score points; Query-Gram minus exact sparse: -0.3336; exact sparse minus full exact K: +1.6443. These are descriptive differences on this fixed six-task pilot, not significance claims. Exact sparse is a selection control, not a guaranteed upper bound on downstream task scores. There is no dense-V baseline in this experiment. All arms use the same full-C1 prefill; only decode attention differs.

Independent audit checks passed for all source IDs and reference answers, reconstructed official prompts, retokenized inputs, four shard manifests, generated-token decoding, generation caps, official scores, and exact-selector invocation counts. All 36 C1 encoder tensors have shape `(8, 128, 80)`. Audit file: `results/evaluation/longbench_c1_32k/audit.json`. Audited result SHA256: `8aad51f62e5bd33e120ee2ef02efd18c290e63a544e6edf70d87a9ec68fa566b`.

Complete machine-readable results: `results/evaluation/longbench_c1_32k/result.json`; compact English report: `results/evaluation/longbench_c1_32k/summary.md`. Predictions and exact per-worker commands are preserved under `evaluate/sample_*.json`. The only scorer warning concerns the optional FuzzyWuzzy acceleration package; neither QA F1 nor summary ROUGE-L in this experiment uses that metric.

## Reproduction commands

Working directory /deac/csc/yangGrp/zhangal/BasisServe-CALS.

### Preparation

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/prepare_longbench_c1.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --output-dir results/datasets/longbench_c1_32k --samples-per-task 32 --sequence-length 32768 --seed 73
```

### smoke

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_fourarm.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --terminal-bank results/checkpoints/mse_base_q32_r8 --qgram-bank results/checkpoints/mse_base_qgram32_r8 --data-dir results/datasets/longbench_c1_32k --output-dir results/evaluation/longbench_c1_32k --num-shards 4 --stage smoke --shard-index 0
```

### evaluate

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_fourarm.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --terminal-bank results/checkpoints/mse_base_q32_r8 --qgram-bank results/checkpoints/mse_base_qgram32_r8 --data-dir results/datasets/longbench_c1_32k --output-dir results/evaluation/longbench_c1_32k --num-shards 4 --stage evaluate --shard-index "$SLURM_ARRAY_TASK_ID"
```

### summary

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_fourarm.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --terminal-bank results/checkpoints/mse_base_q32_r8 --qgram-bank results/checkpoints/mse_base_qgram32_r8 --data-dir results/datasets/longbench_c1_32k --output-dir results/evaluation/longbench_c1_32k --num-shards 4 --stage summarize --shard-index 0
```

### Independent audit

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/audit_longbench_c1.py --data-dir results/datasets/longbench_c1_32k --result-dir results/evaluation/longbench_c1_32k
```
