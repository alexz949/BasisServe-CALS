# Qwen3-8B-Base C1 V96 AllGather-to-FO-IO pipeline

## Scope

This document records the reproducible path from fresh calibration windows to
the current Qwen3-8B-Base C1 V96 private-AllGather and forward-only IO results.
It intentionally excludes K compression: the Key path is dense in every result
below.

The pipeline contains four distinct operations:

1. collect document-disjoint C4 windows and activation-aware routed V/O
   statistics;
2. fit the C1 V96 writer and folded per-head decoder;
3. fit and evaluate source-private AllGather wires at ranks 192, 256, 320, and
   384;
4. capture dense-teacher folded rows and test identity-geometry FRRR versus the
   finite-difference FO-IO pilot.

Large commands below are the Python payloads used inside Slurm jobs. They are
not intended to be launched as long-running jobs on a login node. Experiments
used the `lowrank` or `lowrankarena` conda environment as shown by each command.

## Fixed inputs

| Item | Value |
|:---|:---|
| Model | Qwen3-8B-Base, 36 layers, hidden size 4096, 32 Q heads, 8 KV heads |
| Model revision | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| Calibration corpus | `allenai/c4`, `en`, streaming train split |
| Window split | 192 fit + 64 held-out windows, document-disjoint |
| Window length | 2048 tokens; downstream statistics use the first 512 tokens |
| Statistics split | 192 fit + 32 validation + 16 test windows |
| Key path | Dense; no K factor directory or K-compression artifact |
| Value cache | C1: eight independent physical rank-96 caches |
| Evaluation | WikiText-2 test, sequence length 2048, batch size 1, 146 chunks |

Define these paths when reproducing the commands:

```bash
MODEL=/home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4
WINDOWS=results/cache/qwen3_8b_base_c4_s2048_fit192_select64_seed20260905
STATS=results/cache/q3base_pairgld_stats/globalcv_all36_s512_f192_v32_t16
C1=results/q3base_pairgld/c1v96_c2v192_w96_globalcv_all36_f192_v32_t16
ROWS=results/qwen3_c1_v96_ca_frrr/all36_rows_f192_s32_r32_p32_20260813
```

## Stage 1: document-disjoint C4 windows

Entrypoint: [`scripts/collect_ff2_teacher_rows.py`](../scripts/collect_ff2_teacher_rows.py).
Only its general `prepare-windows` stage is needed here; the FF2 row collector
is independent of the Value/AllGather pipeline.

```bash
/home/lz299/miniconda3/envs/lowrank/bin/python \
  scripts/collect_ff2_teacher_rows.py prepare-windows \
  --model "$MODEL" \
  --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --dataset allenai/c4 \
  --dataset-config en \
  --dataset-split train \
  --seed 20260905 \
  --fit-windows 192 \
  --heldout-windows 64 \
  --sequence-length 2048 \
  --shuffle-buffer-size 10000 \
  --output-dir "$WINDOWS" \
  --local-files-only \
  --resume
```

Output consumed by later stages: `$WINDOWS/windows.safetensors`. The tensor
artifact is deliberately not committed.

## Stage 2: activation-aware routed V/O statistics

Entrypoint:
[`scripts/collect_qwen3_grouped_routed_ov_stats.py`](../scripts/collect_qwen3_grouped_routed_ov_stats.py).
The collector records full transformed cross-head and cross-source covariance,
not a weight-only SVD objective.

```bash
/home/lz299/miniconda3/envs/lowrank/bin/python \
  scripts/collect_qwen3_grouped_routed_ov_stats.py \
  --model "$MODEL" \
  --input-windows "$WINDOWS/windows.safetensors" \
  --output-dir "$STATS" \
  --layers all \
  --supergroups 0,6:1,7:2,5:3,4 \
  --sequence-length 512 \
  --fit-windows 192 \
  --validation-windows 32 \
  --test-windows 16 \
  --batch-size 1 \
  --layers-per-calibration-pass 2 \
  --sample-chunk-size 128 \
  --head-pair-chunk-size 4 \
  --work-dtype float32 \
  --storage-dtype float32 \
  --model-dtype bfloat16 \
  --statistics-device layer \
  --device-map cuda \
  --require-full-gpu-residency \
  --local-files-only
```

## Stage 3: C1 V96 writer and folded decoder

Entrypoint:
[`scripts/analyze_qwen3_base_pair_gld.py`](../scripts/analyze_qwen3_base_pair_gld.py).

```bash
/home/lz299/miniconda3/envs/lowrank/bin/python \
  scripts/analyze_qwen3_base_pair_gld.py \
  --model "$MODEL" \
  --statistics-dir "$STATS" \
  --output-dir "$C1" \
  --layers 0-35 \
  --physical-cache-rank 96 \
  --pair-cache-rank 192 \
  --wire-rank-per-head 96 \
  --report-split test \
  --encoder-sweeps 5 \
  --minimum-encoder-sweeps 2 \
  --encoder-relative-tolerance 1e-6 \
  --encoder-patience 2 \
  --head-decoder-sweeps 20 \
  --head-decoder-minimum-sweeps 2 \
  --head-decoder-relative-tolerance 1e-6 \
  --head-decoder-patience 2 \
  --covariance-damping 1e-7 \
  --decoder-relative-jitter 0 \
  --encoder-relative-damping 1e-6 \
  --maximum-backtracks 10 \
  --encoder-cg-fixed-iterations 16 \
  --work-dtype float32 \
  --factor-dtype bfloat16 \
  --device cuda \
  --resume
```

All-layer activation-aware reconstruction results:

| Arm | Mean test relative MSE |
|:---|---:|
| C1, eight independent rank-96 physical caches | 0.0632895637 |
| C2, four pair-shared rank-192 logical caches | 0.0597627160 |

C2 improves the local metric by 5.573%, but the deployed private-AllGather
experiments below freeze the C1 writer. See
[`C1/C2 all-layer summary`](q3base_pairgld/c1v96_c2v192_w96_globalcv_all36_f192_v32_t16/summary.md).

## Stage 4: activation-aware source-private AllGather ranks

Entrypoint:
[`evaluation/fit_qwen3_vcache_private_ag.py`](../evaluation/fit_qwen3_vcache_private_ag.py).
For each rank in `192 256 320 384`, run:

```bash
/home/lz299/miniconda3/envs/lowrank/bin/python \
  evaluation/fit_qwen3_vcache_private_ag.py \
  --model "$MODEL" \
  --pair-factor-dir "$C1" \
  --statistics-dir "$STATS" \
  --output-dir "results/q3base_pairgld/c1_v96_private_ag/all36_activation_r${RANK}" \
  --arm c1 \
  --private-rank "$RANK" \
  --fit-method activation_rrr \
  --layers 0-35 \
  --report-split test \
  --relative-damping 1e-7 \
  --work-dtype float64 \
  --factor-dtype bfloat16 \
  --device cuda:0 \
  --torch-num-threads 2
```

The rank-384 endpoint is exact relative to the frozen C1 folded decoder. It
introduces no additional output bottleneck; it does not remove the upstream
C1 V96 error.

| Private rank/source | Gathered width | Ideal ring bytes/rank/token | Communication reduction vs dense AR | Mean report relative MSE |
|---:|---:|---:|---:|---:|
| 192 | 1536 | 2688 | 81.25% | 0.0730945153 |
| 256 | 2048 | 3584 | 75.00% | 0.0360479236 |
| 320 | 2560 | 4480 | 68.75% | 0.0132939341 |
| 384 | 3072 | 5376 | 62.50% | 0 |

Per-layer summaries:

- [`r192`](q3base_pairgld/c1_v96_private_ag/all36_activation_r192/summary.md)
- [`r256`](q3base_pairgld/c1_v96_private_ag/all36_activation_r256/summary.md)
- [`r320`](q3base_pairgld/c1_v96_private_ag/all36_activation_r320/summary.md)
- [`r384`](q3base_pairgld/c1_v96_private_ag/all36_activation_r384/summary.md)

The weight-only SVD controls use the same command with
`--fit-method weight_svd`. They show why the activation-aware covariance is a
material part of the pipeline:

| Private rank/source | Weight-only SVD MSE | Activation-aware RRR MSE | Relative MSE reduction |
|---:|---:|---:|---:|
| 192 | 0.2371007820 | 0.0730945153 | 69.17% |
| 256 | 0.1546449680 | 0.0360479236 | 76.69% |
| 320 | 0.0755159590 | 0.0132939341 | 82.40% |

Weight-only control summaries:

- [`r192 weight SVD`](q3base_pairgld/c1_v96_private_ag/all36_weight_svd_r192/summary.md)
- [`r256 weight SVD`](q3base_pairgld/c1_v96_private_ag/all36_weight_svd_r256/summary.md)
- [`r320 weight SVD`](q3base_pairgld/c1_v96_private_ag/all36_weight_svd_r320/summary.md)

## Stage 5: WikiText-2 PPL

Entrypoint:
[`evaluation/eval_qwen3_base_pair_gld_ppl.py`](../evaluation/eval_qwen3_base_pair_gld_ppl.py).
The dense and fixed-C1 controls use:

```bash
/home/lz299/miniconda3/envs/lowrank/bin/python \
  evaluation/eval_qwen3_base_pair_gld_ppl.py \
  --model "$MODEL" \
  --arm dense \
  --output-json results/q3base_pairgld/c1_v96_private_ag/ppl/dense.json \
  --dataset wikitext2 --split test --seqlen 2048 --batch-size 1 \
  --model-dtype bfloat16 --device cuda:0 --torch-num-threads 2

/home/lz299/miniconda3/envs/lowrank/bin/python \
  evaluation/eval_qwen3_base_pair_gld_ppl.py \
  --model "$MODEL" \
  --factor-dir "$C1" \
  --arm c1 \
  --output-json results/q3base_pairgld/ppl_all36_f192/c1.json \
  --dataset wikitext2 --split test --seqlen 2048 --batch-size 1 \
  --model-dtype bfloat16 --device cuda:0 --torch-num-threads 4
```

For a private rank, add:

```bash
  --output-collective rank_private_all_gather \
  --private-ag-factor-dir "results/q3base_pairgld/c1_v96_private_ag/all36_activation_r${RANK}"
```

The JSON result files are not committed; their key metrics are preserved here:

| Configuration | WikiText-2 test PPL |
|:---|---:|
| Dense model | 7.0033845834 |
| Fixed C1 V96, exact r384 private AllGather / logical C1 AllGather | 7.3727476446 |
| C1 V96 + private r320 | 7.4402561245 |
| C1 V96 + private r256 | 7.7634736527 |
| C1 V96 + private r192 | 10.7352937788 |

## Stage 6: dense-teacher folded rows

Entrypoint:
[`evaluation/capture_qwen3_c1_v96_dense_teacher_rows.py`](../evaluation/capture_qwen3_c1_v96_dense_teacher_rows.py).
This captures, for every layer, C1 source wires and the corresponding dense
teacher attention output on document-disjoint fit/select/report splits.

```bash
/home/lz299/miniconda3/envs/lowrank/bin/python \
  evaluation/capture_qwen3_c1_v96_dense_teacher_rows.py \
  --model-path "$MODEL" \
  --c1-factor-dir "$C1" \
  --windows "$WINDOWS" \
  --output-dir "$ROWS" \
  --layers 0-35 \
  --positions-per-window 32 \
  --position-seed 20260812 \
  --layers-per-pass 6 \
  --batch-size 1 \
  --model-dtype bfloat16 \
  --attn-implementation eager \
  --device cuda:0 \
  --torch-num-threads 4
```

The row tensors are calibration artifacts and are not committed.

## Stage 7: identity-geometry FRRR and FO-IO

The reusable collective-aware solver is in
[`basisserve/analysis/collective_aware_frrr.py`](../basisserve/analysis/collective_aware_frrr.py).
The general identity-geometry CLI is
[`evaluation/fit_qwen3_c1_v96_collective_aware_frrr.py`](../evaluation/fit_qwen3_c1_v96_collective_aware_frrr.py).
For example, the exact private-r384 control is represented by the rank split
`3072:0:384`.

The downstream-sensitive pilot adds:

- [`basisserve/forward_only_folded_io.py`](../basisserve/forward_only_folded_io.py):
  subspace construction, softmax-Fisher contraction, finite-difference and JVP
  geometry utilities;
- [`evaluation/run_qwen3_c1_v96_forward_only_io_probe.py`](../evaluation/run_qwen3_c1_v96_forward_only_io_probe.py):
  r256 identity versus FO-IO, with r384 as the exact control;
- [`evaluation/audit_qwen3_c1_v96_foio_epsilon.py`](../evaluation/audit_qwen3_c1_v96_foio_epsilon.py):
  batch-shape, epsilon, forward-dtype, and statistics-dtype audit;
- [`evaluation/audit_qwen3_c1_v96_foio_jvp.py`](../evaluation/audit_qwen3_c1_v96_foio_jvp.py):
  exact forward-mode JVP control without epsilon or reverse-mode/backprop.

The exact pilot and audit commands are in
[`the FO-IO attempt summary`](q3base_c1v96_foio_attempt_summary_20260815.md).

Current finite-difference conclusion:

- r384 reproduced the frozen C1 decoder exactly and had zero additional
  terminal KL at layers 0, 18, and 35;
- at r256, identity geometry beat finite-difference FO-IO at all three probe
  layers and FO-IO won only 3/12 paired windows;
- the original audit had a batch-size-one versus batch-size-two kernel confound;
- after correcting all paths to batch size one, FP16 forward improved the
  numerical floor, but no epsilon produced a stable, transferable local
  quadratic metric;
- this rejects the finite-difference estimator used here, not downstream-aware
  output geometry in general.

As of 2026-08-16, the exact forward-mode JVP controls remain queued:

| Job | Forward dtype | State | Reason |
|---:|:---|:---|:---|
| 160557 | FP16 | PENDING | Priority |
| 160558 | BF16 | PENDING | Priority |

Their code is included for reproducibility; their results should be added in a
later result-only commit after completion.

## Code provenance and commit boundary

The following core pipeline files are already present in commit `78dcc64`
(`Add Qwen and DeepSeek AllGather compression experiments`) and therefore do
not need to be re-added in the follow-up commit:

- `basisserve/qwen3_grouped_value_attention.py`
- `basisserve/qwen3_value_interface.py`
- `basisserve/qwen3_vcache_private_ag.py`
- `basisserve/qwen3_c1_v96_collective_aware.py`
- `basisserve/qwen3_pair_gld_attention.py`
- `basisserve/calibration/grouped_routed_ov_stats.py`
- `basisserve/core/grouped_dual_bottleneck_routed_output.py`
- `scripts/analyze_qwen3_base_pair_gld.py`
- `evaluation/fit_qwen3_vcache_private_ag.py`
- `evaluation/eval_qwen3_base_pair_gld_ppl.py`
- `evaluation/capture_qwen3_c1_v96_dense_teacher_rows.py`
- `evaluation/fit_qwen3_c1_v96_collective_aware_frrr.py`
- `evaluation/fit_qwen3_c1_v96_private_fit_only.py`

The follow-up commit should add only the missing from-scratch collection
entrypoints and dependencies, the FO-IO implementation/audits, their tests, and
the Markdown summaries. It should not include model weights, safetensors,
factor files, JSON outputs, logs, Slurm files, or unrelated K/Monarch changes.
