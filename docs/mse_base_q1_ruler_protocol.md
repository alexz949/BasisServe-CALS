# Closed-form Base16 + Q1 Fisher R8: RULER 32K

## Scope and status

The user selected a direct evaluation of the existing closed-form Base bank with its existing Q1 residual. No Q16 residual refitting, Base fitting, Adam optimization, or KL rank allocation is included.

The experiment is complete. Five small CPU regression tests passed in `basis`, and a read-only preflight verified all 36 layers, bank file hashes, finite FP32 tensors, expected Base/R8 shapes, original Base tensor equality, model configuration, C1 manifest, and RULER dataset manifest. Following the user's confirmation to use `yangGrp`, smoke, all four formal workers, and CPU summary completed successfully. All 88 paired prompts were scored and independently checked.

## Fixed settings

| Item | Setting |
| --- | --- |
| Model | Qwen3-8B-Base, BF16, 36 layers |
| Payload | Frozen C1-V80, `qwen3_8b_c1_v80_32f4h_s32768_als6` |
| Factor bank | `results/checkpoints/q8_residual_kl_bank` |
| Base | Per-group rank16 affine pre-RoPE K MSE-RRR, closed-form whitening and truncated SVD |
| Residual | Existing uniform R8, non-sink Page-Fisher BCD, terminal Q1/window |
| Residual fitting data | Existing 64 × 32768 C4 windows; 16 diagnostic windows |
| Context | RULER 32768-token configuration |
| Tasks/samples | Existing 11-task subset, eight samples/task, 88 prompts |
| Sparse layers | All 36 layers during decode |
| Selection | Page32, B2048, includes one pinned 32-token prefix page |
| Prefill | Full-support C1-V80 Triton prefill; shared first generated token |
| Decode | Native BF16 selected exact-K/C1-V attention; independent cache forks |
| Reference | Same-run full exact-K + C1-V80 |
| Generation | Greedy; official base prompt, task-specific caps and EOS |
| Storage | GPU-resident exact K and materialized Base128+R8 sidecar |
| Intended resources | Four L40S workers, two CPUs/worker, `basis` environment |

The evaluation dataset was used in prior experiments. This is a reused pilot, not a new held-out benchmark, not the complete 13-task RULER suite, and not an offload-speed measurement.

## Historical comparisons

| Configuration | Task-balanced RULER score |
| --- | ---: |
| Closed-form MSE-RRR Base16 + Q1 Fisher R8 | 75.35984848% |
| Adam Q-aware Base16 + Q1 Fisher R8 | 69.50757576% |
| Adam Q-aware Base16 + Q16 Fisher R8 | 80.43560606% |
| C1-V80 + full exact K | 85.20833333% |

The Q1 comparison matches residual query coverage. The Q16 score differs in both Base and residual fitting coverage relative to this run. The matched exact-K reference is rerun rather than substituted with the historical number. Cross-run comparison should check prompt coverage, exact-reference generations, and the executed attention code before attributing a score change to Base fitting.

## Program commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`. All GPU stages are to run through Slurm; these are the underlying program commands, not submission commands.

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_residual_rank_ruler.py \
  --stage smoke \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --bank results/checkpoints/q8_residual_kl_bank \
  --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 \
  --output-dir results/evaluation/mse_base_q1_ruler32k \
  --samples-per-task 8 --sequence-length 32768 \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

After smoke passes, use the same command with `--stage evaluate` and shard indices 0, 1, 2, and 3. After all four shards pass, use `--stage summarize --shard-index 0` on CPU. Smoke samples are excluded from formal accuracy. No result is claimed until all formal samples are complete and rescored.

## Implementation changes

- The existing evaluator now validates the original closed-form bank directly without rewriting its format or pretending it is a Q-aware bank.
- Its summary identifies the actual Base and residual query count.
- A CPU-only `preflight` stage is available.
- No attention, page-selection, fitted factors, or model-weight changes were made for this run.
- No Q16 fit was launched; the temporary Q16-only preparation edits were removed after the user's scope correction.

Evaluation source: [eval_qwen3_8b_residual_rank_ruler.py](../evaluation/eval_qwen3_8b_residual_rank_ruler.py).
Regression tests: [test_residual_rank_ruler.py](../tests/test_residual_rank_ruler.py).

## Submitted jobs

| Stage | Job ID | Resources | Dependency |
| --- | --- | --- | --- |
| Smoke | 8300649 | One L40S, two CPUs, 64 GiB RAM, `yangGrp` | None |
| Formal evaluation | 8300650, array 0–3 | Four L40S workers; two CPUs and 64 GiB RAM each, `yangGrp` | Successful smoke |
| CPU summary | 8300651 | Two CPUs, 8 GiB RAM, `small` | All four evaluation shards successful |

Smoke and CPU summary reserve 20 minutes each; formal workers reserve one hour. These are limits, not runtime estimates. Invalid dependencies cancel downstream jobs. Temporary submission files were removed after submission; logs and result artifacts are retained.

Logs: `logs/mse-q1-smoke-8300649.{out,err}`, `logs/mse-q1-evaluate-8300650_{0,1,2,3}.{out,err}`, and `logs/mse-q1-summarize-8300651.{out,err}`.

Smoke passed on `lovelace` in 25 seconds. The smoke prompt contained 32628 tokens; prefill took 6.10 seconds and maximum allocated GPU memory was 25.57 GiB. Its four-token generations are excluded from formal accuracy.

## Completed results and audit

Formal worker elapsed times were 6:43, 6:23, 7:08, and 6:25. CPU summary took 14 seconds. All jobs exited 0:0. Formal maximum allocated GPU memory was 25.19373894 GiB. No NaN, out-of-memory, or runtime failure was observed in the logs.

| Task | Closed-form Base + Q1 R8 | Adam Q-aware Base + Q1 R8, historical | Same-run exact K + C1-V80 |
| --- | ---: | ---: | ---: |
| niah_single_1 | 100.0000% | 100.0000% | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 100.0000% |
| niah_single_3 | 100.0000% | 75.0000% | 100.0000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 87.5000% |
| niah_multikey_2 | 25.0000% | 25.0000% | 87.5000% |
| niah_multiquery | 87.5000% | 84.3750% | 96.8750% |
| niah_multivalue | 65.6250% | 71.8750% | 93.7500% |
| vt | 92.5000% | 75.0000% | 92.5000% |
| fwe | 83.3333% | 58.3333% | 91.6667% |
| qa_1 | 50.0000% | 50.0000% | 50.0000% |
| qa_2 | 37.5000% | 37.5000% | 37.5000% |
| Task-balanced mean | 75.3598% | 69.5076% | 85.2083% |

The measured difference from historical Adam-Base/Q1 is +5.85227273 percentage points: 12 samples improved, four regressed, and 72 tied. Relative to same-run exact K, the difference is -9.84848485 points: one sample improved, 15 regressed, and 72 tied. These are paired scores on the reused 88-prompt pilot.

Independent CPU checking in `basis` reproduced the task-balanced means by directly computing case-insensitive reference substring hits, with fractional hits for `all` and any hit for `part`. All first tokens matched their shared prefill records. All 88 exact-K generated token sequences were identical to the historical Q1 run. Dataset identity, page/budget/prefix settings, prompting, generation, and attention backend settings also matched. Among the evaluator's tracked source hashes, only the evaluation entry changed; the recorded attention/cache implementation hashes matched the historical Q1 run.

Formal [result JSON](../results/evaluation/mse_base_q1_ruler32k/result.json) and [generated summary](../results/evaluation/mse_base_q1_ruler32k/summary.md) contain the per-sample records, current protocol, and executed commands. No Q16 residual was fitted or evaluated in this run, and no GitHub upload was performed.
