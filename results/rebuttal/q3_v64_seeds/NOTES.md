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
