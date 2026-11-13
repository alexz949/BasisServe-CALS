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
