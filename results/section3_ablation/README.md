# Section 3 Appendix Ablations — Qwen3-8B-Base

Status: complete. The full formal R64/R96 short- and long-context matrix,
paired-document diagnostics, manifests, and appendix plots have been generated
and independently audited.

## Locked scope

- Model: `Qwen/Qwen3-8B-Base`, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Config SHA-256: `3bd01d7ad7a2e203ecbbe84e24087a51c6d2a108ee4bcc42d0016bf49564983a`.
- Uniform Value ranks only: R64 and R96 per physical KV head.
- PaLU methods: M-LRD and G4-LRD only. G4 uses group rank `4r`, hence 256 at R64 and 384 at R96.
- C1 uses joint complete-output fitting, activation-weighted initialization, exact decoder closure, and no rank allocator.
- Dense K and full dense attention are used everywhere. Routing, K offload, sparse selection, Page-Fisher, sink/recent rules, and Section 4 residuals are disabled.
- Environment: `basis`; BF16 model/factors; FP32 covariance accumulation; TF32 disabled.
- Preparation commit: `7db091e855d76dc5dfddcffe9e4016ffb10d0a07`.
  Because the experiment implementation is not yet committed, SHA-256 values
  for all eight executable preparation/fitting/evaluation/summary scripts are
  locked and verified in `qwen3_8b_base/experiment_manifest.json`.

## Calibration and evaluation data

WikiText-2 uses 128 train windows of 2048 tokens (262,144 calibration tokens) plus 32 disjoint validation windows. Exact character ranges, token hashes, dataset revision, and split boundaries are stored in `qwen3_8b_base/data/wikitext2/manifest.json`.

The standard C1 ALS experiment uses the existing C4 sufficient-statistics snapshot with 256 fit and 64 held-out windows at 2048 tokens.

The Dense short-context reference is reused from `ICLR-results/qwen3-8b/quality/Q3-8B-Dense/result.json`; it already contains complete WikiText-2 PPL and all seven MCQ tasks under the locked model and environment. Dense is not recalibrated or redundantly rerun.

The long-context experiment reuses the repository's document-disjoint C4 4K source bank. Every calibration condition contains exactly 1,048,576 fit tokens:

| Context | Fit windows | Held-out windows |
|---:|---:|---:|
| 2K | 512 | 128 |
| 8K | 128 | 32 |
| 32K | 32 | 8 |
| 128K | 8 | 2 |

Evaluation uses eight disjoint 128K sequences. Qwen3-8B-Base is native 32K, so all long-context covariance capture and all Dense/compressed evaluation use the repository's static YaRN configuration with factor 4.

Long-context likelihood evaluation uses a 128-token forward chunk and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for both Dense and C1. This execution-only chunking keeps the likelihood definition unchanged while fitting the full 128K Dense cache and temporary attention workspace on one L40S.

## Fitting and evaluation hyperparameters

- Standard C4 ALS/init sweep: 256 fit plus 64 held-out windows at length 2048;
  16 encoder sweeps; decoder-closed exports at S0/S1/S2/S4/S8/S12/S16.
- The locked production S6 point is a separate deterministic six-sweep replay
  with otherwise identical settings. Across all 36 layers and four
  rank/initialization conditions, its 6,048 recorded S0--S6 trajectory scalars
  match the original S16 run exactly (maximum absolute difference `0.0`).
- WikiText-2 and long-context C1 fits: six encoder sweeps and one final exact
  decoder refit. No early stopping is used.
- C1 numerics: activation-weighted-SVD initialization (or random-orthogonal
  with seed 73 for the initialization ablation), complete-layer output
  objective, FP32 work/covariance, BF16 factors, covariance ridge `1e-5`, fixed
  16-iteration CG, recorded relative tolerance `1e-8`, and TF32 disabled.
  There is no separate encoder damping, decoder jitter, or backtracking
  hyperparameter.
- The fixed-iteration CG policy deliberately executes all 16 iterations. At
  S16 each rank/initialization configuration therefore records 4,608 solves,
  mean/max iterations `16/16`, zero tolerance-triggered early convergence, and
  diagnostic cap-hit fraction `1.0`; these are fixed-budget solves, not runtime
  failures.
- Activation-aware Group-SVD uses pooled diagonal attention-output covariance
  independently within each physical KV group, ridge `1e-5`, and no ALS or
  cross-group covariance.
- Short evaluation uses WikiText-2 batch size 2 and lm-eval batch size 8. Long
  evaluation uses eight 128K sequences, chunk size 128, float32 loss
  accumulation, full dense attention, and static YaRN factor 4.
- GPU execution is on NVIDIA L40S except the memory-heavy 128K covariance
  capture, which used one H200. Dense and all compressed 128K likelihood runs
  themselves use L40S.

## Completed short-context findings

### Activation-aware Group-SVD versus Joint C1

| Method | Rank | Held-out rel-MSE | WT2 PPL | MCQ average |
|---|---:|---:|---:|---:|
| Dense | 128 | 0 | 7.0025 | 0.70430 |
| Activation-aware Group-SVD | 64 | 0.16806 | 9.2547 | 0.65507 |
| Joint C1 | 64 | 0.13602 | 8.7812 | 0.66886 |
| Activation-aware Group-SVD | 96 | 0.06932 | 7.3783 | 0.69038 |
| Joint C1 | 96 | 0.05469 | 7.2632 | 0.69477 |

Joint complete-output fitting improves held-out reconstruction, PPL, and MCQ
at both matched ranks. This supports the narrow objective-level comparison; it
does not isolate cross-group covariance as the only cause because the two
methods also differ in fitting objective.

### WikiText-2 calibration

| Method | R64 PPL | R64 MCQ | R96 PPL | R96 MCQ |
|---|---:|---:|---:|---:|
| Joint C1 | 7.6590 | 0.64672 | 7.1181 | 0.69535 |
| PaLU M-LRD | 10.9782 | 0.54263 | 9.0649 | 0.64734 |
| PaLU G4-LRD | 9.9710 | 0.62343 | 9.0087 | 0.68137 |

The common Dense reference is PPL `7.0025`, MCQ `0.70430`. Under identical
WikiText-2 windows and matched retained Value rank, Joint C1 is the strongest
compressed method at both ranks. At R96 it remains within `0.1156` PPL and
`0.00895` absolute MCQ of Dense.

### ALS sweeps and initialization

- The explicitly evaluated production S6 endpoints are: R64 activation-aware
  `0.13681 / 8.6747 / 0.66682`, R64 random
  `0.14209 / 8.5458 / 0.66939`, R96 activation-aware
  `0.05523 / 7.2679 / 0.69615`, and R96 random
  `0.05847 / 7.2450 / 0.68970` for held-out rel-MSE / PPL / MCQ.
- Six sweeps are the fixed production compute point, not a checkpoint selected
  using appendix test PPL. The S6 results reinforce that distinction: fitted
  reconstruction improves smoothly, while downstream metrics are not
  monotone in sweep count.
- R64 activation-aware initialization reaches its best PPL at S2 (`8.4330`),
  while reconstruction continues improving through S16 (`0.13602`). Random
  initialization nearly catches reconstruction by S16 (`0.13693`) and attains
  the best R64 MCQ (`0.67364`), but has a much weaker S0 endpoint.
- R96 activation-aware PPL improves gradually from `7.2791` at S0 to `7.2632`
  at S16; MCQ peaks at S4 (`0.69745`). Random initialization catches PPL by S8
  (`7.2422`) but remains below activation-aware initialization in MCQ across
  the recorded checkpoints.
- Consequently, more reconstruction sweeps do not monotonically improve
  downstream quality, especially at R64. The raw sweep and per-layer values
  remain the source of truth in the two ALS CSVs.

### Long-context calibration

All rows below evaluate the same eight document-disjoint 128K sequences with
static YaRN factor 4. `64–128K ΔNLL` is the paired mean NLL increase relative
to Dense in the final position bucket; the uncertainty is the standard error
over the eight paired documents.

| Calibration context | R64 PPL | R64 64–128K ΔNLL | R96 PPL | R96 64–128K ΔNLL |
|---:|---:|---:|---:|---:|
| 2K | 14.4213 | 0.36033 ± 0.01137 | 11.7532 | 0.10844 ± 0.00280 |
| 8K | 13.5152 | 0.27041 ± 0.00634 | 11.4446 | 0.07090 ± 0.00226 |
| 32K | 12.9243 | 0.20393 ± 0.00556 | 11.4234 | 0.06256 ± 0.00261 |
| 128K | 12.7827 | 0.18342 ± 0.00636 | 11.3802 | 0.05296 ± 0.00241 |

The Dense 128K reference is PPL `10.8622`. Longer calibration consistently
reduces the long-position loss gap, with a larger absolute effect at R64. At
R64, every adjacent increase in calibration length improves both overall NLL
and final-bucket NLL under paired-document analysis. At R96, 8K→32K is nearly
flat in overall NLL (`0.00185 ± 0.00139` improvement), although its 64–128K
bucket improves (`0.00834 ± 0.00143`); 32K→128K further improves overall NLL
by `0.00379 ± 0.00083` and final-bucket NLL by `0.00960 ± 0.00094`. Thus the
formal result supports length-matched calibration most clearly at low rank and
at long evaluation positions, without claiming that every small overall-PPL
difference is statistically resolved.

The eight long-calibration checkpoints were additionally evaluated on the
standard WikiText-2 test set using ordinary 2048-token windows and the native
pretrained RoPE configuration, without YaRN:

| Calibration context | R64 WT2 PPL | R96 WT2 PPL |
|---:|---:|---:|
| 2K | 8.3075 | 7.2646 |
| 8K | 8.3802 | **7.2295** |
| 32K | 8.3411 | 7.2515 |
| 128K | **8.1949** | 7.2815 |

The Dense reference is `7.0025`. Longer calibration does not cause monotonic
short-context degradation: 128K is best at R64, while the full R96 range is
only `0.0520` PPL. These data support short-context robustness, not a claim
that longer calibration necessarily improves WikiText-2.

## Phase 1 audit results

- Rank-128 C1 dense identity: analytic exact endpoint with zero fit and held-out relative MSE; sweep-0 export is decoder closed.
- ALS checkpoint export: sweep 0 is initialization plus decoder refit; later checkpoints are exported only after the decoder refit. R60 activation-aware held-out rel-MSE decreased `0.0240841 → 0.0232657 → 0.0228310` over sweeps 0/1/2. R96 random-orthogonal decreased `0.0934328 → 0.0800920 → 0.0730222`.
- The original weight-only Group-SVD smoke was superseded before formal reporting. The locked baseline is activation-aware per-group weighted SVD under pooled diagonal attention-output covariance, without cross-head fitting or ALS.
- WT2 manifest partitions fit and held-out data explicitly. PaLU whitening reads only the `fit` partition.
- M-R60 and G4-R96 checkpoint shapes and inference installation were exercised successfully.
- The 128K loader, YaRN factor 4, Dense cache, and actual compressed C1 cache were exercised on one 4096-token prefix. This was an interface smoke test, not a reported benchmark result.

Smoke jobs: `8338965`, `8338970`, `8338974`, `8338975`, `8338977`, `8338978`, `8338980`, and `8338983`. Logs are under `logs/s3-*`.

## Formal outputs

The formal run populates:

- `SECTION3_APPENDIX_SUMMARY.md`, the complete paper-oriented four-part report
- `wt2_calibration.csv` and `wt2_calibration_manifest.json`
- `als_init_sweep.csv` and `als_init_sweep_layers.csv`
- `long_context_calibration.csv`, paired document diagnostics, and
  `long_context_manifest.json`
- `long_calibration_short_wt2.csv`, its manifest, and the two-panel
  short-context robustness plot
- `group_svd_vs_joint.csv` and `group_svd_vs_joint_layers.csv`
- appendix plots derived from those CSVs

Formal Slurm chain:

- Earlier R60 exploratory factors/evaluations (`8338986`, `8338987`, `8339009`, `8339028`) are retained as provenance but excluded from the revised R64/R96 formal summaries.
- Corrected activation-aware Group-SVD smoke/build/evaluation: `8339051`–`8339053`; weight-only artifacts are excluded from formal summaries.
- Extended R64/R96 activation-aware/random ALS-to-S16 fit/merge/evaluation: fit/merge `8339074`–`8339075`; revised L40S evaluation `8339113`; phase-2 summary `8339115`. Superseded R60 evaluation `8339076` and pending R64 jobs `8339077`/`8339111` were replaced before formal summarization. Decoder-closed checkpoints are exported at S0/S1/S2/S4/S8/S12/S16.
- Supplemental production-S6 deterministic replay/evaluation: smoke `8339850`,
  fit `8339852`, merge `8339853`, and four-way L40S evaluation `8339854`.
  Every task completed with exit code `0:0`; evaluation elapsed time was
  `09:46--09:49`.
- Corrected R64 activation-aware Group-SVD build/evaluation: `8339135`, `8339136`; the completed R96 artifacts from `8339052`–`8339053` are reused. Initial build `8339112` exposed and fixed the obsolete `{60,96}` CLI rank restriction before producing artifacts.
- WikiText-2 R64/R96 covariance/whitening/C1/PaLU/evaluation/summary: `8339121`–`8339127`.
- Long-context R64/R96 capture/fitting/merge/summary: `8339128`–`8339134`. Dense 128K was moved from pending H200 job `8339132` to L40S; the 256-token chunk trial `8339137` reached 113K before a temporary-workspace OOM, so the locked chunk-128 Dense/C1 evaluations are `8339145`/`8339146`. Dense completed in `34:18`; the eight C1 L40S tasks completed successfully in `3:16:53`–`3:23:18` each, and the dependency-gated summary completed in `00:02`.
- Standard short-context WikiText-2 replay of the eight long-calibration
  checkpoints: L40S array `8340320`; all tasks completed with exit code `0:0`
  in `00:45`–`00:47`.

The machine-readable locked configuration and command templates are in `qwen3_8b_base/experiment_manifest.json`. Each checkpoint/evaluation result also records its fully expanded command, environment, hashes, runtime, and Slurm job ID.
