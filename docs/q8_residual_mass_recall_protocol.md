# Qwen3-8B residual attention-mass recall protocol

## Status

The diagnostic implementation and GPU pipeline are complete. Five new fixture-free CPU tests and all six existing residual KL/replay tests passed in the `basis` environment. The CLI help check passed. The first test run exposed an SDPA/native backend setting omission in the tiny test fixture; the fixture was corrected and the entire suite passed. GPU smoke, all 16 formal windows and aggregation completed successfully after explicit user confirmation. Existing KL/profile source files, factors, schedule and results are unchanged.

New files:

- `basisserve/core/residual_mass_recall.py`: shared teacher-query capture, exact attention-mass reference, current native page routing.
- `evaluation/eval_qwen3_8b_residual_mass_recall.py`: smoke, sharded evaluation and aggregation.
- `tests/test_residual_mass_recall.py`: causal/partial-page/GQA checks, exact native-selection comparison, sink normalization, teacher invariance/cache append equivalence and summary accounting.

## Fixed comparison

- Model: Qwen3-8B-Base, BF16, 36 layers, 32 Q heads / 8 KV groups.
- C1-V80 and per-group pre-RoPE Base16 remain frozen.
- Residual bank: `results/checkpoints/q8_residual_kl_bank`.
- Frozen schedule: `results/evaluation/q8_residual_kl_64x32k/schedule.json`.
- Uniform residual R8 versus the existing adaptive average-R8 schedule. The latter uses R4 in 16 layers, R8 in 12 layers and R16 in 8 layers. Both have total layer rank 288.
- No refitting, new allocation, budget sweep or configuration selection.
- C4 confirmation indices 64–79 from `results/calibration/qwen3_8b_c4_64f16h_s32768/windows.safetensors`.
- Each 32768-token window packs eight 4096-token C4 windows without added separators. These are not native 32K documents.
- These 16 windows were already used for factor diagnostics and terminal confirmation. This is post-allocation analysis, not a new untouched final test.
- Same 32640-token full-attention C1 prefix as the completed KL measurement. Only the last 128 queries are inspected, at all 36 layers and all 32 query heads.

## Shared teacher and actual selector

The teacher uses full exact-K attention with the same C1-V80 payload. Hooks observe the existing normalized queries and apply the model's RoPE; they do not replace or rerun Q projections. Diagnostic work is performed after the teacher suffix forward. Both routing arms receive identical teacher Q, exact K and C1 latent values at every layer. Sparse outputs never feed back into this teacher trajectory.

The proxy uses the true post-RoPE residual: exact K minus RoPE(Base prediction). Prefix and suffix sidecars are constructed separately to match the existing cached routing path. BF16 proxy contractions and `_selected_pages` are reused from the current native sparse attention implementation. Page32 LSE is normalized over non-sink pages separately per query head, then reduced by max across four Q heads per KV group. Each physical group selects 64 pages (B2048), including pinned page0. Causal masks and partially valid final pages are respected.

Unchanged-R8 layers reuse the same computed diagnostic result in both arms. Their identical recall is structural, not a separate replicate.

## Metrics

For exact teacher probabilities `p(h,i) = softmax_i(q_h K_i / sqrt(d))` over all causally valid tokens and selected-token union S:

- Total mass recall: sum of `p(h,i)` for `i in S`.
- Non-sink mass recall: selected mass outside page0 divided by all available mass outside page0.
- Sink mass: teacher probability mass in page0, reported separately.

Reference QK products and softmax use FP32 on the cached BF16 Q/K. The non-sink conditional distribution is normalized separately for numerical stability; it is mathematically the conditional version of the full teacher distribution. A query with no causally valid non-sink support is flagged and excluded from non-sink statistics, rather than assigned zero recall. No sparse-renormalized probabilities are used as the reference.

Retain per-layer/head/query observations in safetensors. Report pooled mean, P01, P10, median and minimum, per-layer means/quantiles, per-window paired differences and rank-group summaries. Pooled query/head quantiles are descriptive, not confidence intervals from independent documents. The paired standard error uses the 16 window means. Positive adaptive-minus-uniform differences indicate higher retained mass. Window64 remains in the main analysis and is listed separately like every other window.

This compares selectors on a shared teacher trajectory. It is not a remeasurement of the two students' different end-to-end trajectories, full-sequence sparse PPL, RULER accuracy, deployment storage, PCIe transfer or latency. Unweighted average recall need not improve merely because terminal KL improves.

## GPU smoke and provenance

Before formal evaluation, run window64 on one L40S:

1. Verify hooked and unhooked teacher suffix hidden states match bitwise.
2. Verify teacher NLL reproduces the prior L40S confirmation window within absolute tolerance 1e-7. Repeat this check in every formal window.
3. For layer0 R8, layer1 R8/R4 and layer33 R8/R16, run the actual native attention module on the captured teacher inputs, including real prefix-cache forks and suffix appends. Require bitwise agreement for the materialized sidecar, selected page IDs and validity masks.
4. Verify the original prefix cache signature remains unchanged.

Each result records the model/C1/bank/data/source hashes, frozen schedule hash, GPU, environment, command, runtime and raw observation hash. Resume accepts only matching complete artifacts. Formal results require L40S; cross-device data are not pooled.

At preparation time, four L40S were idle on `lovelace` in partition `yangGrp`, while another partition had an idle H200 node. The user explicitly approved using the four L40S despite that alternative to keep hardware consistent with the completed KL experiment.

The submitted pipeline is smoke job `8300221`, evaluation array `8300222` (shards 0–3, after successful smoke), and CPU aggregation job `8300223` (after all evaluation shards succeed). Dependencies are success-only, with cancellation when a dependency cannot succeed. Each GPU worker requests one L40S, two CPUs and 64 GiB host memory; aggregation requests two CPUs and 8 GiB host memory. Logs are `logs/mass-smoke-8300221.{out,err}`, `logs/mass-eval-8300222_{0,1,2,3}.{out,err}` and `logs/mass-summary-8300223.{out,err}`. Temporary sbatch files were deleted immediately after submission.

All jobs completed with exit code `0:0`. Slurm elapsed times were 26 seconds for smoke, 48 seconds for each of the four evaluation shards, and 17 seconds for aggregation. Smoke computation took 9.73 seconds with peak allocated memory 26.86 GiB. Formal per-window computation took 8.44–8.77 seconds; maximum allocated memory was 23.20 GiB. These are diagnostic runtimes, not deployment decode speed measurements. GPU stderr contained model-loading progress; no NaN, OOM, failed check or failed GPU task was observed.

All five native smoke comparisons passed: layer0 R8, layer1 R8/R4 and layer33 R8/R16 had bitwise-identical sidecars, page IDs and validity masks. Hooked and unhooked teacher hidden states matched exactly. Every formal window reproduced the prior teacher NLL exactly. Smoke and formal window64 raw arrays matched bitwise.

## Completed results

| Metric | Uniform R8 | Adaptive average R8 | Difference, percentage points |
|---|---:|---:|---:|
| Mean total mass recall | 86.151771% | 86.234851% | +0.083080 |
| Mean non-sink mass recall | 83.567803% | 83.684502% | +0.116699 |
| Total mass P01 | 28.393322% | 27.299566% | -1.093756 |
| Non-sink mass P01 | 18.610248% | 18.795598% | +0.185350 |

Total mass improved in 13/16 window means; non-sink recall improved in 14/16. Across the eight layers upgraded to R16, mean total mass increased by 1.572747 percentage points. Across the sixteen layers reduced to R4, it decreased by 0.599444 points. The twelve unchanged-R8 layers reuse identical observations. Mean recall improved slightly overall, but lower-tail recall did not improve uniformly.

Window64 contributed 17.59% of net total-mass improvement and 14.94% of net non-sink improvement. Excluding it descriptively leaves gains of 0.073031 and 0.105879 percentage points across the other fifteen windows; all main results retain window64. These shared-teacher local metrics do not, by themselves, explain the earlier student-trajectory terminal KL improvement.

Independent CPU checks verified all 16 protocol/schedule/raw-data hashes, unchanged-R8 equality, finite observations, 2,359,296 observations per arm, pooled means/quantiles, per-layer means and paired-window standard errors. The probability identity `total_mass = sink_mass + (1 - sink_mass) * non_sink_recall` held within maximum absolute error 7.1526e-7. No query lacked non-sink support.

Complete tables are in [the result summary](../results/evaluation/q8_residual_mass_16x32k/summary.md); all per-layer quantiles and rank-group statistics are in [result.json](../results/evaluation/q8_residual_mass_16x32k/result.json).

## Program commands

The completed CPU tests were run with:

```bash
/home/zhangal/.conda/envs/basis/bin/python - <<'PY'
import runpy
import torch
torch.set_num_threads(2)
for path in ('tests/test_residual_mass_recall.py', 'tests/test_residual_kl_replay.py'):
    namespace = runpy.run_path(path)
    for name, test in sorted(namespace.items()):
        if name.startswith('test_') and callable(test):
            test()
            print('PASS', path, name, flush=True)
PY
```

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`. Environment: `basis`. GPU work uses Slurm. The smoke is one GPU; formal evaluation has four independent window shards, one GPU and two CPUs each. Aggregation is CPU-only. A formal job must depend on successful smoke completion, and aggregation on all four successful evaluation shards. Logs will be retained; temporary sbatch files will be deleted after submission.

Smoke:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_residual_mass_recall.py \
  --stage smoke \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6
```

Formal evaluation, with `--shard-index` taking 0, 1, 2 and 3:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_residual_mass_recall.py \
  --stage evaluate --shard-index 0 --num-shards 4 \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6
```

Aggregation:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_residual_mass_recall.py \
  --stage summarize \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6
```

All three commands use the fixed default bank/windows/KL-root paths listed above, suffix length 128, query block size 8 and output directory `results/evaluation/q8_residual_mass_16x32k`.
