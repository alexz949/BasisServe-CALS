# Llama-3.1-8B-Instruct nominal V64: calibration-seed sensitivity and MCQ confidence intervals

Calibration seeds: 42, 43, 44 (C4 train windows resampled per seed; WikiText-2, the C4-validation PPL bank and the MCQ test sets are fixed). Dense is evaluated once as a fixed reference.

## 1. Mean ± sample SD over calibration seeds

SD uses ddof=1 over the fitted seeds and describes the observed calibration variation only.

| Method | Retained V | WT2 PPL | C4 PPL | MCQ macro (%) | ARC-E | ARC-C | HellaS. | PIQA | WinoG. | BoolQ | OBQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense (fixed reference) | 1.0000 | 7.216 | 9.773 | 70.85 | 79.63 | 55.20 | 79.15 | 81.07 | 73.95 | 83.98 | 43.00 |
| BasisKV | 0.5000 | 11.096 ± 0.075 | 12.529 ± 0.074 | 67.52 ± 0.58 | 73.79 ± 1.12 | 49.52 ± 1.05 | 73.68 ± 0.26 | 77.86 ± 0.42 | 72.51 ± 0.05 | 82.85 ± 0.64 | 42.47 ± 0.70 |
| PaLU M-LRD | 0.4740 | 15.592 ± 0.546 | 17.495 ± 0.504 | 59.59 ± 0.67 | 59.61 ± 1.14 | 38.20 ± 1.33 | 67.33 ± 0.65 | 73.65 ± 0.59 | 68.82 ± 1.03 | 72.05 ± 1.01 | 37.47 ± 0.23 |
| PaLU G-LRD4 | 0.4993 | 10.506 ± 0.357 | 13.040 ± 0.174 | 65.41 ± 0.38 | 69.47 ± 0.60 | 46.05 ± 0.69 | 73.43 ± 0.44 | 77.11 ± 0.13 | 71.77 ± 0.37 | 80.95 ± 0.57 | 39.07 ± 1.15 |

## 2. Per-seed results

| Method | Seed | Retained V | WT2 PPL | C4 PPL | MCQ macro (%) | 95% CI (test samples) | ARC-E | ARC-C | HellaS. | PIQA | WinoG. | BoolQ | OBQA |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | - | 1.0000 | 7.2165 | 9.7733 | 70.85 | [69.91, 71.77] | 79.63 | 55.20 | 79.15 | 81.07 | 73.95 | 83.98 | 43.00 |
| BasisKV | 42 | 0.5000 | 11.0796 | 12.5567 | 67.28 | [66.33, 68.21] | 72.90 | 49.23 | 73.65 | 77.64 | 72.45 | 82.72 | 42.40 |
| BasisKV | 43 | 0.5000 | 11.0295 | 12.4460 | 68.19 | [67.23, 69.12] | 75.04 | 50.68 | 73.95 | 78.35 | 72.53 | 83.55 | 43.20 |
| BasisKV | 44 | 0.5000 | 11.1778 | 12.5854 | 67.10 | [66.16, 68.02] | 73.44 | 48.63 | 73.43 | 77.58 | 72.53 | 82.29 | 41.80 |
| PaLU M-LRD | 42 | 0.4609 | 15.6468 | 18.0696 | 59.70 | [58.73, 60.62] | 60.27 | 38.05 | 67.36 | 74.32 | 68.11 | 72.17 | 37.60 |
| PaLU M-LRD | 43 | 0.4844 | 15.0209 | 17.1293 | 58.87 | [57.90, 59.80] | 58.29 | 36.95 | 66.67 | 73.23 | 68.35 | 70.98 | 37.60 |
| PaLU M-LRD | 44 | 0.4766 | 16.1087 | 17.2852 | 60.20 | [59.24, 61.15] | 60.27 | 39.59 | 67.96 | 73.39 | 70.01 | 73.00 | 37.20 |
| PaLU G-LRD4 | 42 | 0.4980 | 10.2679 | 12.8745 | 65.85 | [64.90, 66.77] | 70.16 | 46.84 | 73.92 | 77.26 | 72.06 | 80.31 | 40.40 |
| PaLU G-LRD4 | 43 | 0.5020 | 10.3330 | 13.0249 | 65.18 | [64.20, 66.10] | 69.02 | 45.73 | 73.31 | 77.04 | 71.35 | 81.38 | 38.40 |
| PaLU G-LRD4 | 44 | 0.4980 | 10.9169 | 13.2212 | 65.19 | [64.25, 66.12] | 69.23 | 45.56 | 73.05 | 77.04 | 71.90 | 81.16 | 38.40 |

Layer ranks (one per layer; per physical KV head for BasisKV and M-LRD, per 4-head group for G-LRD4):

- BasisKV s42: `[128, 112, 64, 112, 128, 80, 112, 80, 64, 64, 64, 64, 64, 80, 80, 112, 80, 96, 48, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]`
- BasisKV s43: `[96, 96, 64, 128, 112, 64, 112, 96, 80, 80, 64, 80, 64, 96, 64, 112, 64, 112, 48, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]`
- BasisKV s44: `[112, 112, 64, 128, 128, 80, 112, 96, 64, 64, 64, 64, 64, 96, 64, 96, 80, 96, 48, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]`
- PaLU M-LRD s42: `[96, 96, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 96, 64, 64, 64, 64, 64, 32, 32, 32, 32, 32, 64, 64, 64, 32, 32, 32]`
- PaLU M-LRD s43: `[128, 96, 96, 96, 96, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 32, 32, 32, 32, 32, 64, 64, 64, 32, 32, 32]`
- PaLU M-LRD s44: `[128, 96, 96, 96, 96, 64, 64, 96, 64, 64, 64, 96, 64, 64, 64, 64, 64, 64, 64, 64, 32, 32, 32, 32, 32, 32, 32, 32, 64, 32, 32, 32]`
- PaLU G-LRD4 s42: `[416, 352, 320, 320, 320, 288, 288, 288, 288, 256, 256, 288, 288, 288, 288, 352, 288, 288, 256, 224, 224, 192, 128, 160, 160, 160, 192, 224, 256, 160, 192, 160]`
- PaLU G-LRD4 s43: `[448, 416, 352, 320, 320, 288, 288, 320, 288, 288, 288, 320, 288, 256, 256, 320, 288, 288, 224, 224, 192, 192, 128, 160, 160, 160, 192, 192, 224, 192, 192, 160]`
- PaLU G-LRD4 s44: `[480, 416, 352, 352, 320, 320, 320, 320, 320, 288, 288, 320, 320, 288, 256, 320, 256, 256, 192, 192, 192, 160, 128, 128, 128, 128, 192, 192, 224, 160, 192, 160]`

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
| BasisKV − PaLU M-LRD | 42 | +7.59 | [+6.87, +8.34] |
| BasisKV − PaLU M-LRD | 43 | +9.32 | [+8.57, +10.09] |
| BasisKV − PaLU M-LRD | 44 | +6.90 | [+6.18, +7.63] |
| BasisKV − PaLU G-LRD4 | 42 | +1.44 | [+0.82, +2.05] |
| BasisKV − PaLU G-LRD4 | 43 | +3.01 | [+2.38, +3.64] |
| BasisKV − PaLU G-LRD4 | 44 | +1.91 | [+1.28, +2.54] |

Three-seed averages. Each replicate first scores every seed, then averages the seeds. **These intervals are test-sample uncertainty conditional on the three fitted models of each method; they are not confidence intervals over the population of calibration seeds.**

| Quantity | Value | 95% CI |
| --- | ---: | :---: |
| BasisKV MCQ macro (%) | 67.52 | [66.61, 68.41] |
| PaLU M-LRD MCQ macro (%) | 59.59 | [58.67, 60.48] |
| PaLU G-LRD4 MCQ macro (%) | 65.41 | [64.48, 66.29] |
| BasisKV − PaLU M-LRD (pp) | +7.94 | [+7.31, +8.57] |
| BasisKV − PaLU G-LRD4 (pp) | +2.12 | [+1.60, +2.64] |

Dense MCQ macro: 70.85 [69.91, 71.77].

## 4. Commands, environment, data protocol and run record

### Environment

- Machine `shadecloud`: 8 × NVIDIA A100-SXM4-80GB (driver 550.144.03), no Slurm; jobs ran under `nohup` with one full model per GPU. The large artifacts of this directory are stored under `/data/basisserve/l31i_v64_seeds` and linked into `results/rebuttal/l31i_v64_seeds`.
- Conda env `basis`: Python 3.11.18, torch 2.6.0+cu124, transformers 5.15.1, datasets 5.0.0, lm-eval 0.4.11, the same environment as the Qwen3-8B-Base run. The paper's Llama-3.1-8B-Instruct row was produced with torch 2.13.0+cu130 and transformers 5.17.0 on L40S.
- C4 windows were sampled with datasets 5.0.0. Under that version the sampler regenerates the paper's Llama banks: the C4-validation bank byte for byte (`3fe79cbd…`) and, with the paper's seed 20260821, all 144 KL-profile documents in order.
- Repository commit `b0a4cd2` plus the uncommitted changes of this experiment.

### Commands

```bash
# whole run (10 fits/evaluations, 36 minutes wall, launched 2026-11-13 04:11 UTC)
nohup scripts/run_v64_seeds.sh llama31_8b_instruct launch "42 43 44" > results/rebuttal/l31i_v64_seeds/logs/launch.log 2>&1 &
# summary, bootstrap and this file
python evaluation/summarize_v64_seeds.py --root results/rebuttal/l31i_v64_seeds
```

Per calibration seed `S` (`C=calib/sS`, `B=basiskv/sS`, `M` = the pinned model snapshot):

```bash
# calibration windows: one C4 stream, each bank a verified prefix of the next
python evaluation/prepare_llama31_8b_instruct_c4_windows.py --model M --output-dir C/windows_fit --samples 256 --sequence-length 2048 --seed S
python evaluation/prepare_llama31_8b_instruct_c1_windows.py --model M --output-dir C/windows_fit_heldout --prefix-windows C/windows_fit/windows.safetensors \
  --fit-windows 256 --heldout-windows 64 --sequence-length 2048 --seed S
python evaluation/prepare_llama31_8b_instruct_c1_windows.py --model M --output-dir C/windows_kl --prefix-windows C/windows_fit_heldout/windows.safetensors \
  --fit-windows 320 --heldout-windows 192 --sequence-length 2048 --seed S        # 512 windows; rows 320..463 are the KL windows

# BasisKV
python evaluation/capture_attention_o_proj_covariances.py --model-path M --windows C/windows_fit_heldout/windows.safetensors \
  --output-dir C/cov --expected-attention gqa --fit-windows 256 --heldout-windows 64 --device cuda:0
python scripts/fit_value.py fit-shard --snapshot-dir C/cov --output-dir B/banks/R<r> --rank <r> --covariance-damping 0 --device cuda:0 --resume
python scripts/fit_value.py merge --output-dir B/banks/R<r>                       # r = 32 48 64 80 96 112
python evaluation/run_llama31_8b_instruct_c1_two_sided_factorized_kl_sharded.py profile --model M --windows C/windows_kl/windows.safetensors \
  --snapshot-dir C/cov --profile-dir B/kl_profile --factor-dir <r>=B/banks/R<r> ... --anchor-rank 64 \
  --candidate-ranks 32,48,64,80,96,112,128 --window-start 320 --profile-windows 128 --confirmation-windows 16 \
  --sequence-length 2048 --batch-size 16 --covariance-damping 0 --vocab-chunk-size 8192 --torch-num-threads 4 \
  --device-map balanced --max-memory-per-gpu-gib 44 --model-dtype bfloat16 --attn-implementation sdpa --local-files-only \
  --profile-shard-count 8 --factorized-probe-rank 96 --factorized-compression-probe-rank 32 --local-error-split heldout \
  --factorized-exponent 1 --profile-backend sampled_suffix --terminal-position-counts 64,128,256,512,1024 \
  --terminal-selection-position-count 1024 --terminal-position-seed 20260903 --profile-shard-index <0..7>
python evaluation/run_llama31_8b_instruct_c1_two_sided_factorized_kl_sharded.py finalize <same arguments> --output-dir B/checkpoint \
  --target-average-rank 64 --force-selected-candidate two_sided_factorized_kl
python evaluation/build_c1_iclr_checkpoint.py --profile llama31_8b_instruct --model M --allocation-dir B/checkpoint

# PaLU
python -m torch.distributed.run --standalone --nproc-per-node=1 evaluation/build_llama31_8b_instruct_iclr_v_checkpoint.py prepare-whitening \
  --model M --output-dir C/whitening --windows C/windows_fit/windows.safetensors --samples 256
python evaluation/collect_gqa_palu_fisher.py --profile llama31_8b_instruct --target v --model M --windows C/windows_fit/windows.safetensors \
  --output-dir C/fisher --samples 256 --sequence-length 2048 --retained-ratio 0.5 --device cuda:0
python evaluation/build_llama31_8b_instruct_iclr_v_checkpoint.py --method palu-fisher --equivalent-rank 64 --head-group-size <1|4> --model M \
  --fisher-result C/fisher/fisher.json --whitening-dir C/whitening --output-dir palu_<m|g4>/sS/checkpoint --torch-num-threads 8

# quality, identical for Dense and all nine fits
python evaluation/eval_llama31_8b_instruct_iclr_quality.py --run-id <id> --model M --checkpoint-dir <arm>/checkpoint \
  --c4-windows eval_inputs/c4_validation_128x2048/windows.safetensors --output-dir <arm>/quality \
  --batch-size 2 --lm-eval-batch-size 8 --max-memory-per-gpu-gib 44 --torch-num-threads 8 --expected-gpu-count 1
```

The fully expanded command of every stage is stored in its manifest or result JSON. The `*_llama31_8b_instruct_*` entry points named in the paper's manifests were not in this repository; they were restored as thin wrappers over the shared modules with a new `llama31_8b_instruct` profile.

### Data protocol

- Model: meta-llama/Llama-3.1-8B-Instruct, revision `0e9e39f249a16976918f6564b8830bc894c89659`, BF16, dense Keys, full attention.
- The protocol is the one behind the paper's Llama-3.1-8B-Instruct row, read from its manifests. It differs from the Qwen3-8B-Base protocol in three places: covariance damping 0 (Qwen 1e-5), KL windows taken from the continuation of the calibration stream (Qwen: a separate 4,096-token bank), and 8 KL profile shards (Qwen 4).
- **Calibration seeds 42, 43, 44**, the same seeds as the Qwen run; the paper's seed 20260821 was not refitted. The Qwen windows could not be reused as token tensors because the tokenizers differ; the same seeds were resampled with the Llama tokenizer.
- Per seed one C4 train stream (allenai/c4 `en`, revision `1588ec45…`, shuffle buffer 10,000) gives 512 document-disjoint 2,048-token windows: rows 0–255 fit BasisKV and are the PaLU whitening/Fisher windows, rows 256–319 are BasisKV's held-out windows, rows 320–447 the KL-profile windows and rows 448–463 the KL-confirmation windows.
- Document overlap is zero between the three seeds and between all calibration windows and the C4 evaluation documents.
- Fixed across seeds: ALS initialization (activation-aware, seed 0), 6 sweeps, 16 CG iterations, FP32 fit, BF16 factors, candidate ranks 32–128 with anchor 64 and probes 32/96, KL position seed 20260903, lm-eval seeds, bootstrap seed.
- Evaluation inputs are identical for all ten runs: WikiText-2 test (288,627 scored tokens), 128 C4-validation documents × 2,048 tokens (seed 20260828, sha256 `3fe79cbd…`, the paper's bank), and the seven zero-shot lm-eval tasks with the same metrics as the Qwen run.
- Per-example MCQ records and bootstrap units are as in the Qwen run (BoolQ by passage, HellaSwag by `source_id`, the other tasks by example).

### Run record

- All ten runs finished on the first attempt of the formal run; no failure and no out-of-memory error. Peak allocation was 20 GB per evaluation and 29 GB per KL shard.
- Wall time 36 minutes: covariance capture 1.7 min, rank banks about 1 min each (fitted in parallel), 8 KL shards about 3.6 min each, whitening 1.3 min, Fisher 2.0 min, quality about 9 min per run.
- Smoke test before the run: `smoke/SMOKE.md` (perplexity end to end and ten questions per MCQ task).
- Dense agrees with the paper despite the different library versions and GPU: WikiText-2 7.2165 (paper 7.2164), C4 9.7733 (9.7740), MCQ macro 70.85% (70.89%), every task within 0.16 points.

## 5. Conclusions

1. On each of the three calibration seeds BasisKV has higher MCQ macro accuracy than both PaLU variants: +7.94 pp over M-LRD (test-sample 95% CI [+7.31, +8.57]) and +2.12 pp over G-LRD4 ([+1.60, +2.64]) on the three-seed average. Every per-seed interval excludes zero; the smallest lower bound is +0.82 pp (G-LRD4, seed 42).
2. **WikiText-2 perplexity favours G-LRD4 on every seed** (BasisKV 11.08 / 11.03 / 11.18 against 10.27 / 10.33 / 10.92; mean 11.10 ± 0.08 against 10.51 ± 0.36). The paper's row shows the same ordering (11.50 against 10.61), so this is a stable property of the method on this model, not a calibration accident.
3. C4 perplexity favours BasisKV on every seed (12.53 ± 0.07 against 13.04 ± 0.17), again as in the paper (12.45 against 12.90).
4. Unlike on Qwen3-8B-Base, the MCQ advantage over G-LRD4 is spread across tasks: BasisKV is ahead on six tasks on all three seeds (ARC-E +2.7 to +6.0 pp, ARC-C +2.4 to +5.0, OBQA +2.0 to +4.8, BoolQ +1.1 to +2.4, PIQA and WinoGrande below +1.4) and level on HellaSwag (−0.27, +0.64, +0.38 pp).
5. The observed calibration variation of the MCQ macro is 0.58 pp (BasisKV), 0.38 pp (G-LRD4) and 0.67 pp (M-LRD). The gap to G-LRD4 is about four times BasisKV's seed SD; BasisKV is again not the least seed-sensitive method on this metric.
6. Realized Value retention: BasisKV 50.00% on all seeds, G-LRD4 49.80% / 50.20% / 49.80%, M-LRD 46.09% / 48.44% / 47.66%. M-LRD retains less than the nominal budget on every seed because of PaLU's 32-rank rounding, as it does in the paper (47.66%).
7. The paper's Table 1 values (seed 20260821) lie close to these seeds: BasisKV 11.50 / 12.45 / 67.3%, G-LRD 10.61 / 12.90 / 65.6%, M-LRD 16.09 / 17.16 / 59.7%.

Limits: three seeds describe the observed variation and cannot bound the seed population; the intervals in Section 3 are conditional on the fitted models; the paper's own seed was not refitted; the library versions differ from the paper's Llama runs (Dense agrees to the figures above).
