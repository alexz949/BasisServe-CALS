# ICLR V-side checkpoint and quality results

## Scope and outcome

This report covers the requested V-only matrix for Qwen3-8B-Base and
Llama-3.1-8B: Dense, Weight-SVD, and Fisher-allocated PaLU M-LRD,
G-LRD2, and G-LRD4 at equivalent ranks R96, R80, and R64. C1 is
intentionally excluded. All 26 requested checkpoints and all 26 quality
results are complete.

The broad expected behavior is reproduced on both models:

- Dense is best on full WikiText-2 PPL, fixed-128 C4 PPL, and the
  seven-task average accuracy.
- Weight-SVD is worst at every rank on all three aggregate metrics.
- PaLU generally improves as the grouping increases from M to G2 to G4.
  The strict nine-condition audit passes 8/9 checks on each model. The only
  failure on each model is R80; details are in the trend section below.

## Method definitions

- **Dense:** dense K and dense V.
- **Weight-SVD:** dense K; each physical V head is independently compressed
  with ordinary truncated SVD of the weight matrix. It does not use
  calibration activations or Fisher allocation.
- **PaLU:** dense K and compressed V. The factorization applies SVD to the
  activation-weighted matrix `W @ L`, where `L` is the C4 activation
  Cholesky factor, and maps the right factor back with a triangular solve.
  Layer budgets come from the official PaLU Fisher allocation with rank block
  size 32. Factors are exported as BF16; factorization uses FP64.
- **Rank convention:** R96/R80/R64 is the equivalent rank per physical head.
  M uses one physical head per group; G2 uses nominal group ranks 192/160/128;
  G4 uses nominal group ranks 384/320/256.

Only V is compressed. K remains dense in every compressed checkpoint.

## Evaluation protocol

| Item | Setting |
| --- | --- |
| Python environment | `basis` executable; Llama records `CONDA_DEFAULT_ENV=basis`, while Qwen was invoked by absolute environment path and records that optional field as null |
| `datasets` | 5.0.0 |
| `lm-eval` | 0.4.11 |
| GPUs per checkpoint/evaluation job | 4 x NVIDIA L40S on `lovelace`; Qwen canonical checkpoint artifacts were subsequently reproduced byte-for-byte in a conforming 4-GPU job |
| WikiText-2 | Complete `test` corpus, sequence length 2048, FP32 loss |
| C4 evaluation | Fixed 128 document-disjoint samples from `validation`, 2048 tokens each, FP32 loss |
| Commonsense | ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande, BoolQ, OpenBookQA; zero-shot |
| Metric selection | `acc_norm` when present, otherwise `acc` |
| Average | Unweighted arithmetic mean of the seven selected accuracies |

For PaLU, each model uses one fixed, document-disjoint C4 `train` calibration
set with 256 samples of length 2048 (524,288 tokens), seed 20260821. The C4
evaluation windows use the `validation` split and seed 20260828, so they are
not calibration samples. Fisher statistics target only `v_proj` and use
`sqrt(mean(per-sample gradient squared))`, followed by the matrix mean.

## Qwen3-8B-Base

Model revision:
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.

| Run ID | Method | Retained V | WT2 PPL | C4 PPL | ARC-E | ARC-C | HS | PIQA | WG | BoolQ | OBQA | Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q3-8B-Dense | Dense | 1.000000 | 7.002509 | 9.168594 | 0.801768 | 0.569113 | 0.786497 | 0.793798 | 0.727703 | 0.831193 | 0.420000 | 0.704296 |
| Q3-8B-SVD-R96 | Weight-SVD | 0.750000 | 4530.954327 | 847.913367 | 0.337963 | 0.261092 | 0.381299 | 0.563656 | 0.513812 | 0.576453 | 0.302000 | 0.419468 |
| Q3-8B-SVD-R80 | Weight-SVD | 0.625000 | 90181.147396 | 19263.675824 | 0.311869 | 0.282423 | 0.319259 | 0.537541 | 0.494081 | 0.544343 | 0.294000 | 0.397645 |
| Q3-8B-SVD-R64 | Weight-SVD | 0.500000 | 739229.370152 | 307515.739391 | 0.289141 | 0.269625 | 0.288588 | 0.521219 | 0.483820 | 0.557798 | 0.274000 | 0.383456 |
| Q3-8B-PALUM-R96 | PaLU M-LRD | 0.715278 | 8.663951 | 10.751474 | 0.740320 | 0.522184 | 0.768273 | 0.779108 | 0.719021 | 0.793578 | 0.394000 | 0.673783 |
| Q3-8B-PALUM-R80 | PaLU M-LRD | 0.645833 | 9.224107 | 11.294961 | 0.728956 | 0.500853 | 0.747162 | 0.780740 | 0.711918 | 0.782263 | 0.400000 | 0.664556 |
| Q3-8B-PALUM-R64 | PaLU M-LRD | 0.493056 | 14.839074 | 16.866396 | 0.626263 | 0.420648 | 0.699562 | 0.738847 | 0.670087 | 0.762997 | 0.356000 | 0.610629 |
| Q3-8B-PALUG2-R96 | PaLU G-LRD2 | 0.750000 | 8.298955 | 10.480836 | 0.768519 | 0.536689 | 0.778630 | 0.793798 | 0.707182 | 0.805199 | 0.416000 | 0.686574 |
| Q3-8B-PALUG2-R80 | PaLU G-LRD2 | 0.625000 | 9.713508 | 12.007475 | 0.679293 | 0.490614 | 0.763692 | 0.778020 | 0.709550 | 0.775229 | 0.396000 | 0.656057 |
| Q3-8B-PALUG2-R64 | PaLU G-LRD2 | 0.493056 | 12.386457 | 14.150045 | 0.611532 | 0.456485 | 0.729735 | 0.757889 | 0.696922 | 0.760245 | 0.376000 | 0.626972 |
| Q3-8B-PALUG4-R96 | PaLU G-LRD4 | 0.748264 | 8.244049 | 10.403133 | 0.771044 | 0.556314 | 0.786795 | 0.795974 | 0.718232 | 0.793272 | 0.408000 | 0.689947 |
| Q3-8B-PALUG4-R80 | PaLU G-LRD4 | 0.625000 | 9.567456 | 11.782581 | 0.755471 | 0.537543 | 0.773750 | 0.782372 | 0.709550 | 0.738226 | 0.404000 | 0.671559 |
| Q3-8B-PALUG4-R64 | PaLU G-LRD4 | 0.496528 | 10.650523 | 12.870647 | 0.697811 | 0.501706 | 0.754431 | 0.767138 | 0.709550 | 0.720183 | 0.406000 | 0.650974 |

Qwen trend detail:

- R96 and R64 strictly improve from M to G2 to G4 on both PPL metrics and
  average accuracy.
- At R80, G4 is best among the three PaLU geometries, but G2 has worse PPL and
  accuracy than M. The realized retained ratios differ: M 64.58%, G2 62.50%,
  G4 62.50%. Thus the M/G2 inversion is not an equal-realized-budget
  comparison.

## Llama-3.1-8B

Model revision:
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.

| Run ID | Method | Retained V | WT2 PPL | C4 PPL | ARC-E | ARC-C | HS | PIQA | WG | BoolQ | OBQA | Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L31-8B-Dense | Dense | 1.000000 | 6.239787 | 8.335565 | 0.811027 | 0.536689 | 0.789285 | 0.809576 | 0.735596 | 0.821101 | 0.452000 | 0.707896 |
| L31-8B-SVD-R96 | Weight-SVD | 0.750000 | 56.172585 | 87.709555 | 0.413300 | 0.275597 | 0.396335 | 0.634385 | 0.520916 | 0.489602 | 0.292000 | 0.431734 |
| L31-8B-SVD-R80 | Weight-SVD | 0.625000 | 173.884904 | 223.442079 | 0.372475 | 0.228669 | 0.331010 | 0.589771 | 0.532755 | 0.451070 | 0.272000 | 0.396821 |
| L31-8B-SVD-R64 | Weight-SVD | 0.500000 | 470.468302 | 397.687336 | 0.337542 | 0.233788 | 0.303724 | 0.554407 | 0.514601 | 0.407645 | 0.266000 | 0.373958 |
| L31-8B-PALUM-R96 | PaLU M-LRD | 0.742188 | 7.345675 | 9.537923 | 0.781145 | 0.516212 | 0.770862 | 0.800871 | 0.737174 | 0.774312 | 0.440000 | 0.688654 |
| L31-8B-PALUM-R80 | PaLU M-LRD | 0.656250 | 8.354250 | 10.598922 | 0.739899 | 0.482082 | 0.749751 | 0.793798 | 0.737964 | 0.772171 | 0.402000 | 0.668238 |
| L31-8B-PALUM-R64 | PaLU M-LRD | 0.476562 | 14.175872 | 14.663795 | 0.632576 | 0.395904 | 0.675463 | 0.754625 | 0.707972 | 0.648318 | 0.374000 | 0.598408 |
| L31-8B-PALUG2-R96 | PaLU G-LRD2 | 0.746094 | 7.184943 | 9.306987 | 0.775673 | 0.511945 | 0.778032 | 0.797062 | 0.727703 | 0.793272 | 0.440000 | 0.689098 |
| L31-8B-PALUG2-R80 | PaLU G-LRD2 | 0.628906 | 8.134802 | 10.210429 | 0.719697 | 0.472696 | 0.753734 | 0.784004 | 0.730860 | 0.784404 | 0.420000 | 0.666485 |
| L31-8B-PALUG2-R64 | PaLU G-LRD2 | 0.503906 | 10.842140 | 12.207576 | 0.669613 | 0.406143 | 0.700159 | 0.757889 | 0.709550 | 0.722018 | 0.378000 | 0.620482 |
| L31-8B-PALUG4-R96 | PaLU G-LRD4 | 0.751953 | 6.983324 | 9.076497 | 0.789141 | 0.522184 | 0.783011 | 0.804135 | 0.729282 | 0.800917 | 0.440000 | 0.695524 |
| L31-8B-PALUG4-R80 | PaLU G-LRD4 | 0.626953 | 7.635286 | 9.745016 | 0.750421 | 0.494027 | 0.767975 | 0.789445 | 0.726914 | 0.784404 | 0.424000 | 0.676741 |
| L31-8B-PALUG4-R64 | PaLU G-LRD4 | 0.498047 | 9.364653 | 11.124858 | 0.687290 | 0.445392 | 0.730532 | 0.772579 | 0.721389 | 0.758104 | 0.392000 | 0.643898 |

Llama trend detail:

- R96 and R64 strictly improve from M to G2 to G4 on both PPL metrics and
  average accuracy.
- At R80, both PPL metrics strictly improve M to G2 to G4. Average accuracy
  changes 0.668238 -> 0.666485 -> 0.676741, so the M-to-G2 change is a
  0.001753 decrease. The realized ratios are M 65.63%, G2 62.89%, and G4
  62.70%; G4 is best despite retaining the least of the three.

## Commands and Slurm records

Each manifest/result records its fully expanded direct command and SHA256
provenance. The checkpoint command interfaces used were:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/build_qwen3_8b_iclr_v_checkpoint.py \
  --method {dense|weight-svd|palu-fisher} \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --output-dir ICLR-results/qwen3-8b/checkpoints/<run-id> \
  [--equivalent-rank {96|80|64}] [--head-group-size {1|2|4}] \
  [--fisher-result results/fisher/qwen3_8b_palu_m_r64_fisher_c4_256x2048/fisher.json] \
  [--whitening-dir results/calibration/qwen3_8b_c4_256x2048_palu_whitening] \
  --torch-num-threads 8

/home/zhangal/.conda/envs/basis/bin/python evaluation/build_llama31_8b_iclr_v_checkpoint.py \
  --method {dense|weight-svd|palu-fisher} \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
  --output-dir ICLR-results/llama31-8b/checkpoints/<run-id> \
  [--equivalent-rank {96|80|64}] [--head-group-size {1|2|4}] \
  [--fisher-result ICLR-results/llama31-8b/fisher/v_fisher_c4_256x2048/fisher.json] \
  [--whitening-dir ICLR-results/llama31-8b/calibration/c4_train_256x2048_whitening] \
  --torch-num-threads 8
```

The quality command interfaces used were:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_<model>_iclr_quality.py \
  --run-id <run-id> --model <pinned-model-snapshot> \
  --checkpoint-dir ICLR-results/<model>/checkpoints/<run-id> \
  --c4-windows <fixed-validation-windows.safetensors> \
  --output-dir ICLR-results/<model>/quality/<run-id> \
  --batch-size 2 --lm-eval-batch-size 8 \
  --max-memory-per-gpu-gib 44 --torch-num-threads 8
```

Final successful Slurm jobs:

| Model | Stage | Job ID | Final status |
| --- | --- | --- | --- |
| Qwen3-8B | Initial Dense/Weight-SVD checkpoints (1-GPU allocation) | 8297781, tasks 0-3 | Completed |
| Qwen3-8B | Initial PaLU checkpoints after input-validation correction (1-GPU allocation) | 8297795, tasks 0-8 | Completed |
| Qwen3-8B | Full 4-GPU checkpoint reproduction and canonical hash audit | 8298296, tasks 0-12 | Completed; all matched |
| Qwen3-8B | Dense quality | 8297811 | Completed |
| Qwen3-8B | SVD/M quality | 8297823, successful tasks 0-5 | Completed |
| Qwen3-8B | Grouped PaLU quality after grouped-decoder loader correction | 8297861, tasks 0-5 | Completed |
| Llama-3.1-8B | Calibration, whitening, and Fisher | 8297897 | Completed |
| Llama-3.1-8B | All 13 checkpoints | 8297899, tasks 0-12 | Completed |
| Llama-3.1-8B | All 13 quality runs | 8297918, tasks 0-12 | Completed |

The initial Qwen PaLU checkpoint attempts in job 8297781 failed input
validation before producing final artifacts; job 8297795 regenerated all nine
PaLU checkpoints successfully. Those two early checkpoint jobs requested one
L40S because factorization itself is CPU-side. To satisfy the four-L40S job
requirement exactly, job 8298296 reran all 13 production commands on lovelace
with four allocated L40S GPUs: all 12 factor files matched the canonical
SHA256 values byte-for-byte, and the Dense semantic manifest matched. Initial
grouped-quality tasks in job 8297823
exposed a decoder-shape assumption in the loader; the loader was corrected and
all six grouped runs completed in job 8297861 using the same output paths.
There are no incomplete final checkpoints or results. The only recurring
runtime warning was lm-eval's HFLM notice that `pretrained` was an in-memory
model rather than a string, which is expected for evaluating the factorized
model object.

## Verification

- Audited 26 complete result files and 26 matching checkpoint manifests.
- Recomputed and matched all 78 stage hashes and all 24 non-dense factor
  artifact hashes.
- Full WikiText-2 evaluation consumed 298,862 scored tokens for every Qwen arm
  and 288,627 scored tokens for every Llama arm.
- Every C4 result contains 128 independent document losses and 262,016 scored
  tokens (`128 * (2048 - 1)`). Every arm of a model uses the same saved C4
  evaluation-window hash.
- Calibration/evaluation document overlap is zero for both models.
- Every result records `datasets==5.0.0`, `lm-eval==0.4.11`, four NVIDIA L40S
  devices, and nonzero peak allocation on all four GPUs.
- Slurm job 8298296 independently reproduces all Qwen checkpoints under a
  four-L40S allocation and verifies them against the artifacts used for the
  reported quality results.
- The two summary audits each verified 13 result formats/statuses, checkpoint
  manifest hashes, stage hashes, task sets, package versions, and GPU records.
- Relevant unit tests: 19 passed. All six production, evaluation, and summary
  Python entry points compile successfully.

## Artifact locations

- Qwen checkpoints: `ICLR-results/qwen3-8b/checkpoints/`
- Qwen per-run results: `ICLR-results/qwen3-8b/quality/`
- Qwen machine-readable summary: `ICLR-results/qwen3-8b/quality-summary.json`
- Llama checkpoints: `ICLR-results/llama31-8b/checkpoints/`
- Llama per-run results: `ICLR-results/llama31-8b/quality/`
- Llama machine-readable summary: `ICLR-results/llama31-8b/quality-summary.json`
