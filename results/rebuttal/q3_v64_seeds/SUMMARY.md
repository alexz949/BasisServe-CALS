# Qwen3-8B-Base nominal V64: calibration-seed sensitivity and MCQ confidence intervals

Calibration seeds: 42, 43, 44 (C4 train windows resampled per seed; WikiText-2, the C4-validation PPL bank and the MCQ test sets are fixed). Dense is evaluated once as a fixed reference.

## 1. Mean ± sample SD over calibration seeds

SD uses ddof=1 over the fitted seeds and describes the observed calibration variation only.

| Method | Retained V | WT2 PPL | C4 PPL | MCQ macro (%) | ARC-E | ARC-C | HellaS. | PIQA | WinoG. | BoolQ | OBQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense (fixed reference) | 1.0000 | 7.003 | 9.169 | 70.32 | 80.01 | 57.00 | 78.57 | 79.22 | 72.77 | 83.06 | 41.60 |
| BasisKV | 0.5000 | 8.220 ± 0.223 | 10.089 ± 0.023 | 67.34 ± 0.59 | 76.25 ± 1.75 | 51.79 ± 1.09 | 73.63 ± 0.24 | 78.60 ± 0.35 | 71.98 ± 0.16 | 79.54 ± 0.64 | 39.60 ± 0.53 |
| PaLU M-LRD | 0.4954 | 14.235 ± 0.892 | 16.257 ± 0.805 | 60.06 ± 0.64 | 59.16 ± 1.20 | 41.58 ± 0.42 | 69.90 ± 0.92 | 74.10 ± 0.86 | 67.56 ± 1.27 | 72.83 ± 0.65 | 35.27 ± 1.45 |
| PaLU G-LRD4 | 0.5017 | 10.645 ± 0.153 | 12.956 ± 0.128 | 64.94 ± 0.21 | 70.15 ± 1.14 | 49.97 ± 0.36 | 75.40 ± 0.29 | 76.84 ± 0.51 | 70.85 ± 0.43 | 71.87 ± 0.98 | 39.53 ± 0.83 |

## 2. Per-seed results

| Method | Seed | Retained V | WT2 PPL | C4 PPL | MCQ macro (%) | 95% CI (test samples) | ARC-E | ARC-C | HellaS. | PIQA | WinoG. | BoolQ | OBQA |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | - | 1.0000 | 7.0025 | 9.1686 | 70.32 | [69.39, 71.23] | 80.01 | 57.00 | 78.57 | 79.22 | 72.77 | 83.06 | 41.60 |
| BasisKV | 42 | 0.5000 | 8.1026 | 10.1073 | 66.92 | [66.01, 67.85] | 74.96 | 51.54 | 73.37 | 78.35 | 71.98 | 79.24 | 39.00 |
| BasisKV | 43 | 0.5000 | 8.0804 | 10.0972 | 68.02 | [67.09, 68.94] | 78.24 | 52.99 | 73.69 | 79.00 | 72.14 | 80.28 | 39.80 |
| BasisKV | 44 | 0.5000 | 8.4768 | 10.0634 | 67.09 | [66.16, 68.02] | 75.55 | 50.85 | 73.84 | 78.45 | 71.82 | 79.11 | 40.00 |
| PaLU M-LRD | 42 | 0.4861 | 15.2583 | 17.1808 | 59.34 | [58.39, 60.29] | 58.71 | 41.30 | 68.84 | 73.12 | 67.40 | 72.42 | 33.60 |
| PaLU M-LRD | 43 | 0.4931 | 13.6215 | 15.7053 | 60.58 | [59.65, 61.52] | 58.25 | 42.06 | 70.35 | 74.70 | 68.90 | 73.58 | 36.20 |
| PaLU M-LRD | 44 | 0.5069 | 13.8264 | 15.8844 | 60.25 | [59.31, 61.18] | 60.52 | 41.38 | 70.49 | 74.48 | 66.38 | 72.51 | 36.00 |
| PaLU G-LRD4 | 42 | 0.5017 | 10.5776 | 12.9320 | 64.99 | [64.06, 65.94] | 70.20 | 49.57 | 75.38 | 76.66 | 70.88 | 72.45 | 39.80 |
| PaLU G-LRD4 | 43 | 0.5017 | 10.8199 | 13.0950 | 64.72 | [63.78, 65.64] | 68.98 | 50.09 | 75.11 | 77.42 | 70.40 | 72.42 | 38.60 |
| PaLU G-LRD4 | 44 | 0.5017 | 10.5377 | 12.8423 | 65.12 | [64.18, 66.05] | 71.25 | 50.26 | 75.70 | 76.44 | 71.27 | 70.73 | 40.20 |

Layer ranks (one per layer; per physical KV head for BasisKV and M-LRD, per 4-head group for G-LRD4):

- BasisKV s42: `[112, 64, 32, 48, 48, 32, 64, 64, 80, 64, 80, 64, 48, 48, 64, 64, 64, 64, 64, 64, 64, 64, 112, 128, 96, 48, 64, 48, 32, 64, 64, 48, 64, 80, 64, 32]`
- BasisKV s43: `[112, 64, 32, 32, 48, 32, 64, 64, 80, 64, 80, 80, 64, 48, 64, 64, 64, 80, 80, 80, 64, 64, 96, 128, 96, 48, 64, 48, 32, 48, 64, 48, 48, 64, 64, 32]`
- BasisKV s44: `[96, 48, 32, 48, 48, 32, 64, 64, 80, 80, 80, 64, 48, 48, 64, 64, 64, 64, 64, 80, 64, 64, 96, 128, 96, 64, 64, 64, 32, 64, 64, 48, 48, 80, 64, 32]`
- PaLU M-LRD s42: `[64, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 32, 64, 64, 64, 64, 64, 96, 64, 96, 96, 64, 128, 64, 64, 64, 32, 64, 64, 32, 64, 32, 64, 64, 96, 96]`
- PaLU M-LRD s43: `[64, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 96, 64, 64, 128, 64, 64, 64, 32, 64, 32, 32, 64, 32, 64, 32, 96, 96]`
- PaLU M-LRD s44: `[64, 32, 32, 64, 32, 64, 64, 64, 64, 64, 64, 32, 64, 64, 64, 64, 64, 96, 64, 96, 64, 96, 128, 64, 64, 64, 64, 64, 64, 32, 64, 32, 64, 64, 96, 96]`
- PaLU G-LRD4 s42: `[256, 160, 160, 192, 192, 192, 224, 224, 224, 224, 256, 192, 256, 224, 288, 288, 320, 352, 256, 384, 320, 320, 512, 256, 256, 256, 192, 256, 192, 192, 256, 160, 256, 192, 384, 384]`
- PaLU G-LRD4 s43: `[320, 192, 192, 224, 256, 224, 256, 288, 256, 288, 288, 192, 288, 224, 256, 256, 288, 320, 224, 352, 288, 288, 480, 224, 256, 224, 192, 256, 192, 160, 224, 128, 256, 192, 320, 384]`
- PaLU G-LRD4 s44: `[256, 160, 160, 192, 192, 192, 224, 224, 224, 224, 256, 192, 256, 192, 288, 288, 288, 352, 256, 384, 320, 320, 512, 288, 288, 256, 192, 256, 192, 192, 256, 160, 288, 192, 352, 384]`

## 3. MCQ test-sample confidence intervals

Task-stratified percentile bootstrap, 10000 replicates, seed 20261113. Each replicate redraws the resampling units of every task with replacement (unit count unchanged), computes the seven task accuracies and averages them with equal weight. The same draw is applied to every method and calibration seed.

| Task | Resampling unit | Units | Examples |
| --- | --- | ---: | ---: |
| ARC-E | example | 2376 | 2376 |
| ARC-C | example | 1172 | 1172 |
| HellaS. | source_id | 8407 | 10042 |
| PIQA | example | 1838 | 1838 |
| WinoG. | example | 1267 | 1267 |
| BoolQ | passage | 2939 | 3270 |
| OBQA | example | 500 | 500 |

Paired differences in MCQ macro accuracy (percentage points):

| Comparison | Seed | Difference (pp) | 95% CI (pp) |
| --- | ---: | ---: | :---: |
| BasisKV − PaLU M-LRD | 42 | +7.58 | [+6.74, +8.40] |
| BasisKV − PaLU M-LRD | 43 | +7.44 | [+6.64, +8.27] |
| BasisKV − PaLU M-LRD | 44 | +6.84 | [+6.02, +7.67] |
| BasisKV − PaLU G-LRD4 | 42 | +1.93 | [+1.18, +2.67] |
| BasisKV − PaLU G-LRD4 | 43 | +3.30 | [+2.56, +4.06] |
| BasisKV − PaLU G-LRD4 | 44 | +1.97 | [+1.21, +2.74] |

Three-seed averages. Each replicate first scores every seed, then averages the seeds. **These intervals are test-sample uncertainty conditional on the three fitted models of each method; they are not confidence intervals over the population of calibration seeds.**

| Quantity | Value | 95% CI |
| --- | ---: | :---: |
| BasisKV MCQ macro (%) | 67.34 | [66.48, 68.23] |
| PaLU M-LRD MCQ macro (%) | 60.06 | [59.19, 60.92] |
| PaLU G-LRD4 MCQ macro (%) | 64.94 | [64.07, 65.82] |
| BasisKV − PaLU M-LRD (pp) | +7.28 | [+6.63, +7.96] |
| BasisKV − PaLU G-LRD4 (pp) | +2.40 | [+1.79, +3.02] |

Dense MCQ macro: 70.32 [69.39, 71.23].

## 4. Commands, environment, data protocol and run record

### Environment

- Machine `shadecloud`: 8 × NVIDIA A100-SXM4-80GB (driver 550.144.03), no Slurm; jobs ran under `nohup` with one full model per GPU.
- Conda env `basis`: Python 3.11.18, torch 2.6.0+cu124, transformers 5.15.1, datasets 5.0.0, lm-eval 0.4.11 (the versions recorded in the paper's quality results).
- Conda env `basis-win` (a clone of `basis` with datasets 4.4.2) was used only to sample C4 windows. datasets 5.0.0 orders the same streaming-shuffle seed differently; with 4.4.2 the sampler regenerates the paper's banks byte for byte (checked on the paper's seed, see below).
- Repository commit `b0a4cd2` plus the uncommitted changes of this experiment.

### Commands

```bash
# whole run (10 fits/evaluations, 41 minutes wall, launched 2026-11-13 03:03 UTC)
nohup scripts/run_v64_seeds.sh qwen3_8b launch "42 43 44" > results/rebuttal/q3_v64_seeds/logs/launch.log 2>&1 &
# summary, bootstrap and this file
python evaluation/summarize_v64_seeds.py --root results/rebuttal/q3_v64_seeds
```

The run itself was started as `scripts/launch_q3_v64_seeds.sh '42 43 44'`; that launcher and `run_q3_v64_seeds.sh` were merged afterwards into `scripts/run_v64_seeds.sh MODEL STAGE` with the same stage commands (the six rank banks of a seed are now fitted in parallel). Per calibration seed `S` (`C=calib/sS`, `B=basiskv/sS`, `M` = the pinned model snapshot):

```bash
# calibration windows (basis-win)
python evaluation/prepare_qwen3_8b_c4_windows.py --model M --output-dir C/windows_fit --samples 256 --sequence-length 2048 --seed S
python evaluation/prepare_qwen3_8b_c1_windows.py --model M --output-dir C/windows_fit_heldout --prefix-windows C/windows_fit/windows.safetensors \
  --fit-windows 256 --heldout-windows 64 --sequence-length 2048 --seed S
python evaluation/prepare_qwen3_8b_c4_windows.py --model M --output-dir C/windows_kl --samples 640 --sequence-length 4096 --seed $((S+7))

# BasisKV
python evaluation/capture_attention_o_proj_covariances.py --model-path M --windows C/windows_fit_heldout/windows.safetensors \
  --output-dir C/cov --expected-attention gqa --fit-windows 256 --heldout-windows 64 --device cuda:0
python scripts/fit_value.py fit-shard --snapshot-dir C/cov --output-dir B/banks/R<r> --rank <r> --covariance-damping 1e-5 --device cuda:0 --resume
python scripts/fit_value.py merge --output-dir B/banks/R<r>                       # r = 32 48 64 80 96 112
python evaluation/run_qwen3_8b_c1_two_sided_factorized_kl_sharded.py profile --model M --windows C/windows_kl/windows.safetensors \
  --snapshot-dir C/cov --profile-dir B/kl_profile --factor-dir <r>=B/banks/R<r> ... --anchor-rank 64 \
  --candidate-ranks 32,48,64,80,96,112,128 --window-start 320 --profile-windows 128 --confirmation-windows 16 \
  --sequence-length 2048 --batch-size 16 --covariance-damping 1e-5 --vocab-chunk-size 8192 --torch-num-threads 4 \
  --device-map balanced --max-memory-per-gpu-gib 44 --model-dtype bfloat16 --attn-implementation sdpa --local-files-only \
  --profile-shard-count 4 --factorized-probe-rank 96 --factorized-compression-probe-rank 32 --local-error-split heldout \
  --factorized-exponent 1 --profile-backend sampled_suffix --terminal-position-counts 64,128,256,512,1024 \
  --terminal-selection-position-count 1024 --terminal-position-seed 20260903 --profile-shard-index <0..3>
python evaluation/run_qwen3_8b_c1_two_sided_factorized_kl_sharded.py finalize <same arguments> --output-dir B/checkpoint \
  --target-average-rank 64 --force-selected-candidate two_sided_factorized_kl
python evaluation/build_c1_iclr_checkpoint.py --profile qwen3_8b --model M --allocation-dir B/checkpoint

# PaLU
python -m torch.distributed.run --standalone --nproc-per-node=1 evaluation/build_qwen3_8b_section3_palu.py prepare-whitening \
  --model M --output-dir C/whitening --windows C/windows_fit/windows.safetensors --samples 256
python evaluation/collect_gqa_palu_fisher.py --profile qwen3_8b --target v --model M --windows C/windows_fit/windows.safetensors \
  --output-dir C/fisher --samples 256 --sequence-length 2048 --retained-ratio 0.5 --device cuda:0
python evaluation/build_qwen3_8b_iclr_v_checkpoint.py --method palu-fisher --equivalent-rank 64 --head-group-size <1|4> --model M \
  --fisher-result C/fisher/fisher.json --whitening-dir C/whitening --output-dir palu_<m|g4>/sS/checkpoint --torch-num-threads 8

# quality, identical for Dense and all nine fits
python evaluation/eval_qwen3_8b_iclr_quality.py --run-id <id> --model M --checkpoint-dir <arm>/checkpoint \
  --c4-windows eval_inputs/c4_validation_128x2048/windows.safetensors --output-dir <arm>/quality \
  --batch-size 2 --lm-eval-batch-size 8 --max-memory-per-gpu-gib 44 --torch-num-threads 8 --expected-gpu-count 1
```

The fully expanded command of every stage is stored in its manifest or result JSON.

### Data protocol

- Model: Qwen/Qwen3-8B-Base, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, BF16, dense Keys, full attention.
- **Calibration seeds 42, 43, 44** were chosen for this run. They replace the paper's seed 20260821; the paper's own fit was not repeated. The paper's banks were regenerated only as a check of the sampler: `calib/s20260821` reproduces the PaLU windows (`c970fa6e…`), the 640×4096 KL bank (`784dfe8b…`, all 144 profile/confirmation documents in order) and the C4-validation bank (`31df55d6…`) byte for byte.
- A calibration seed changes the C4 train documents and window offsets (allenai/c4 `en`, revision `1588ec45…`, streaming shuffle buffer 10,000). Everything else is fixed: ALS initialization (activation-aware, seed 0), 6 sweeps, 16 CG iterations, damping 1e-5, FP32 fit, BF16 factors, KL position seed 20260903, lm-eval seeds, bootstrap seed.
- Within one seed all methods use the same documents: the 256 PaLU windows (whitening and Fisher) are the verified prefix of BasisKV's 256 fit + 64 held-out windows. No statistics, factors or rank schedules are shared between seeds.
- BasisKV's Two-Sided-KL profile uses, as in the paper, rows 320–447 (profile) and 448–463 (confirmation) of a second bank of 640 documents with at least 4,096 tokens, truncated to 2,048 tokens. The paper drew that bank with seed 20260828 = calibration seed + 7; this run keeps the rule (seeds 49, 50, 51).
- Document overlap is zero between the three seeds, between each seed's fit/held-out and KL windows, and between all calibration banks and the C4 evaluation documents.
- Evaluation inputs are identical for all ten runs: WikiText-2 test (298,862 scored tokens), 128 C4-validation documents × 2,048 tokens (seed 20260828, sha256 `31df55d6…`, the paper's bank), and seven zero-shot lm-eval tasks (ARC-E/ARC-C/HellaSwag/PIQA/OBQA version 1 with `acc_norm`; WinoGrande version 1 and BoolQ version 2 with `acc`).
- Covariance damping follows the paper's factor banks (1e-5); the paper appendix states 1e-7.

### Per-example MCQ records

Each `quality/commonsense-examples.jsonl` has one line per example (20,465 per run): task, `example_id` (`task:doc_id`), lm-eval document and prompt hashes, dataset-native ID and label, gold target, per-choice log-likelihoods, and `correct` (the per-example value of the reported metric). The evaluator refuses to finish unless the task means of `correct` equal lm-eval's aggregates, and the summary refuses to run unless all runs carry the same example IDs and document hashes.

WinoGrande twin sentences share no ID in the dataset, so WinoGrande is resampled by example; BoolQ is resampled by passage and HellaSwag by `source_id`.

### Run record

- All ten runs finished on the first attempt; no failure and no out-of-memory error. Peak allocation was 21 GB per evaluation and 30 GB per KL shard, on 80 GB GPUs.
- Wall time 41 minutes: covariance capture 1.9 min, six rank banks about 1 min each, four KL shards about 7.3 min each, allocation 0.6 min, whitening 1.5 min, Fisher 2.3 min, PaLU factorization about 0.6 min, quality about 10 min per run.
- Smoke test before the run: `smoke/SMOKE.md`. It found one defect, fixed before launch: the evaluator's GPU check failed on a single-GPU load.
- Differences from the paper's setup: A100 instead of L40S, one GPU per evaluation instead of four, whitening on one process. Dense reproduces the paper's WikiText-2 and C4 perplexities to all printed digits (7.002509, 9.168594); its MCQ macro is 70.32% against 70.43% in the paper (ARC-E, BoolQ, PIQA, HellaSwag, OBQA and ARC-C differ by at most 0.4 points, WinoGrande is identical).
- Uniform-rank held-out relative MSE of BasisKV matches the paper's level on all seeds (rank 64: 0.1360, 0.1353, 0.1371; paper 0.1368).

## 5. Conclusions

1. On each of the three calibration seeds BasisKV has lower WikiText-2 and C4 perplexity and higher MCQ macro accuracy than both PaLU variants. The MCQ gap is +7.28 pp over M-LRD (test-sample 95% CI [+6.63, +7.96]) and +2.40 pp over G-LRD4 ([+1.79, +3.02]) on the three-seed average; every per-seed interval excludes zero (smallest lower bound +1.18 pp, G-LRD4 on seed 42).
2. The gaps are several times the observed calibration variation: the seed SD of the MCQ macro is 0.59 pp (BasisKV), 0.64 pp (M-LRD) and 0.21 pp (G-LRD4). BasisKV is not the most stable method across seeds on this metric; G-LRD4 varies less.
3. The advantage over G-LRD4 is not uniform across tasks. G-LRD4 is better on HellaSwag on all three seeds (−2.01, −1.42, −1.86 pp) and the two are level on OpenBookQA (−0.8, +1.2, −0.2 pp). BoolQ alone (+6.8 to +8.4 pp) contributes about 1.1 pp of the 2.4 pp macro gap; ARC-E contributes most of the rest.
4. Perplexity ordering is the same on every seed. BasisKV's C4 perplexity is nearly seed-independent (10.09 ± 0.02); its WikiText-2 perplexity varies more (8.08–8.48), still far below G-LRD4 (10.54–10.82) and M-LRD (13.62–15.26).
5. Realized Value retention differs slightly by method and seed because PaLU's Fisher allocation rounds to 32-rank blocks: BasisKV 50.00% on all seeds, G-LRD4 50.17%, M-LRD 48.61%/49.31%/50.69%. M-LRD's deficit on seed 42 is part of its result there, not a like-for-like 50% comparison.
6. The paper's Table 1 values (seed 20260821, L40S) lie close to these seeds: BasisKV 8.26 / 10.06 / 67.6%, G-LRD 10.65 / 12.87 / 65.1%, M-LRD 14.84 / 16.87 / 61.1% against the means in Section 1.

Limits: three seeds describe the observed variation and cannot bound the seed population; the intervals in Section 3 are conditional on the fitted models; the paper's own seed was not refitted in this run.
