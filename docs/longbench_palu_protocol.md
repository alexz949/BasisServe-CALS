# LongBench-v1: matched-calibration PaLU M and G-LRD4

## Fixed comparison

Evaluate both newly built V-only PaLU Fisher checkpoints on the same 192 LongBench inputs as the completed dense and C1 experiments. Qwen3-8B-Base, BF16, full exact K, no routing or sparse attention for either new arm.

- M: `results/checkpoints/palu_m_fisher_r80_c1matched32k`, actual average per-KV-head rank 81.7778, nominal R80, eight independently factorized KV heads.
- G4: `results/checkpoints/palu_g4_fisher_r80_c1matched32k`, actual average equivalent rank 80, two groups of four adjacent KV heads, nominal group rank 320. Layer group ranks are Fisher allocated.

Both reuse exactly the C1 32 x 32,768 fit token IDs, matching whitening and the same newly measured dense-model V Fisher statistics. No LongBench refitting or configuration selection. The two realized latent budgets are reported separately rather than called exactly equal.

## Inputs and scoring

Reuse `results/datasets/longbench_c1_32k` without re-sampling or retokenizing. Six tasks, 32 prompts each: qasper, multifieldqa_en, hotpotqa, 2wikimqa, gov_report, qmsum. Prompt tokens, reference answers, official base-completion templates, greedy/EOS policy and generation caps 128/64/32/32/512/512 match the previous runs.

32K is the input plus reserved-output cap. Actual prompts are 1,192–30,431 tokens (mean 9,244.59), none truncated. Scores are official QA F1 or summary ROUGE-L, maximum across reference alternatives, on a 0–100 scale. The reported aggregate is the arithmetic mean of these six tasks, not full LongBench accuracy.

## Runtime semantics

Install saved BF16 PaLU writer/decoder factors using the existing `install_palu_m_factors` helper, which also supports grouped factors. Execute the latent V writer and per-group V reconstruction explicitly in both prefill and decode. Q, K and O projection module identities are verified unchanged, and installed writer/decoder weights are checked against checkpoint tensors.

Use native Transformers dense SDPA and the same greedy loop as the dense baseline. Cache stores exact K128 and reconstructed approximate V128, so this is a quality control, not a compact-cache memory or speed benchmark. No folding into a single approximate V projection weight is used in this run. Each arm computes its own prefill and first generated token; neither reuses the C1 prefix.

Previously evaluated C1 arms share full-C1 Triton prefill and differ only in decode attention. Thus PaLU versus C1 is an end-to-end pipeline comparison, not a matched-prefill-kernel ablation of factor objectives alone.

## Validation and resources

New evaluator: `evaluation/eval_longbench_palu.py`; existing evaluators and factors remain unchanged. Syntax and whitespace checks passed. Input checks authenticate the previously audited dense/C1 results, dataset hashes, checkpoint manifests/factors, matched calibration and Fisher hashes, all 36 layer ranks and 72 finite BF16 tensors per arm.

Smoke runs shortest and longest prompts for each arm, capped at four generated tokens. Repeated runs must have identical logits/tokens, and tokens must match native `model.generate()`. Scores are unset for smoke. Every run validates all 36 K/V cache shapes and finite logits.

Formal layout: four sample shards per arm, 48 prompts each, eight one-GPU tasks total with at most four concurrent workers on lovelace L40S. Each task requests two CPUs and 48 GiB host memory. Environment basis. Summary is CPU-only, checks all 384 new predictions, eight shard manifests, token decoding, EOS/caps and official scores, and combines unchanged prior dense/C1 scores.

Output: `results/evaluation/longbench_palu_32k/{m,g4}/{smoke,evaluate}`; combined `result.json`, `audit.json`, `summary.md`. Logs: `logs/lb-palu-{smoke,evaluate,summary}-{job}[_task].out/.err`. Smoke array: `8300979`. Temporary sbatch files are removed after submission. No unrelated jobs or GitHub state are changed.

## Job record

Both smoke tasks in array 8300979 completed with exit code 0 in 34 seconds each. Maximum allocated GPU memory on the longest prompt: M 22.392 GiB, G4 22.411 GiB. All repeated-logit and native-generation checks passed. The shortest G4 prompt generated EOS immediately; smoke uses the same EOS policy and does not score this capped test.

Formal array: `8300981_0..7`, concurrency capped at four; indices 0–3 run M shards 0–3, indices 4–7 run G4 shards 0–3. CPU summary/audit: `8300986`, dependent on all eight formal tasks succeeding. Existing result directories are not overwritten.

## Completed results

All 384 new predictions completed. All eight GPU tasks and the CPU summary exited with code 0, with no failed/retried jobs. M shard elapsed times: 6:54, 5:30, 8:24, 8:37. G4 shard elapsed times: 4:33, 3:26, 6:01, 4:51. The CPU scoring summary/audit took 24 seconds. Four-GPU concurrency means these arm times overlap; they are not latency benchmarks.

| Task | Dense K/V | PaLU M | PaLU G4 | C1 full exact K |
|---|---:|---:|---:|---:|
| qasper | 39.3018 | 27.2131 | 16.9565 | 19.6094 |
| multifieldqa_en | 52.8498 | 32.3183 | 36.6639 | 30.7387 |
| hotpotqa | 60.8872 | 16.1260 | 29.4210 | 29.7403 |
| 2wikimqa | 50.1190 | 28.8711 | 22.7083 | 31.2642 |
| gov_report | 29.1896 | 21.5618 | 25.3024 | 27.2987 |
| qmsum | 26.1460 | 25.2504 | 16.0173 | 26.3270 |
| Six-task mean | 43.0822 | 25.2234 | 24.5116 | 27.4964 |

M minus C1 full: -2.2729 points; G4 minus C1 full: -2.9848. M minus dense: -17.8588; G4 minus dense: -18.5707. These are descriptive differences on the fixed pilot, not significance claims or a diagnosis of the underlying cause.

Saved-generation diagnostics: M produced immediate EOS/empty text on 5/192 prompts and reached the output cap without EOS on 74/192; G4 produced immediate EOS/empty text on 52/192 and reached the cap without EOS on 33/192. These are completed generations under the fixed EOS policy, not missing records. Maximum allocated GPU memory: M 22.386 GiB, G4 22.405 GiB.

All 384 predictions were re-decoded and rescored; input/reference identities, factor metadata, eight shard manifests, EOS/caps and official rounded per-task scores passed validation. Result SHA256: `640a0ebc468e60be12cef07c4a42a1e954e33ecdf5dedc3a9a207d9f171b100a`. The only runtime warning is optional FuzzyWuzzy acceleration, unused by these F1/ROUGE-L task metrics.

Combined seven-arm results including C1 sparse/router controls: `results/evaluation/longbench_palu_32k/summary.md` and `result.json`. Audit: `results/evaluation/longbench_palu_32k/audit.json`. No changes were made to the original dense/C1 outputs or the two PaLU checkpoints.

## Execution commands

Working directory `/deac/csc/yangGrp/zhangal/BasisServe-CALS`.

Smoke, separately for `--arm m` and `--arm g4`:

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_palu.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke --arm m
```

Formal, for each arm and shard index 0–3:

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_palu.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --arm m --shard-index 0
```

Combined CPU summary and scoring audit:

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_palu.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```
