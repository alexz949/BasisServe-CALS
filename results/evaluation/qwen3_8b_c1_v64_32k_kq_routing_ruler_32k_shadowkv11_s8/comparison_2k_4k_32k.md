# Qwen3-8B uniform C1-V64 calibration-length comparison

## Controlled protocol

All three checkpoints use uniform rank 64 for every physical KV head in all
36 layers, the same ALS5 solver, 524,288 fit token rows, and BF16 factors. The
4K and 32K checkpoints also use exactly the same 128 fit documents and 16
validation documents:

- 4K: 128 fit windows and 16 validation windows;
- 32K: each consecutive group of eight 4K windows is packed without changing
  tokens, producing 16 fit windows and 2 validation windows.

The 32K packing introduces attention across unrelated C4 documents. It is a
controlled context-length experiment, not a natural-long-document calibration
corpus.

Evaluation uses the same 88 held-out RULER-v1 32K examples, BF16 dense
baseline, R32 routing factors, 64-token pages, B1024 exact-token budget, and
greedy decoding for every checkpoint.

## Aggregate accuracy

| Arm | 2K calibration | 4K calibration | 32K calibration | 4K to 32K |
|:---|---:|---:|---:|---:|
| Exact-K + C1-V64 | 66.12% | 76.23% | 81.84% | +5.61 pp |
| R32/B1024 routing + C1-V64 | 66.17% | 73.71% | 77.69% | +3.98 pp |
| BF16 dense reference | 86.48% | 86.48% | 86.48% | +0.00 pp |

For exact-K C1, the paired 4K-to-32K comparison contains 15 improvements, 5
regressions, and 68 ties. For the routing arm it contains 12 improvements, 8
regressions, and 68 ties.

The remaining exact-K C1 gap to BF16 is 4.64 pp. This gap cannot be assigned to
rank-64 capacity alone: Value rank, C4-to-RULER domain shift, finite calibration
data, packing boundaries, and the output-MSE objective remain confounded.

## Task-level 4K-to-32K comparison

| Task | C1 4K | C1 32K | Delta | Routing 4K | Routing 32K | Delta |
|:---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | +0.00 pp | 100.00% | 100.00% | +0.00 pp |
| niah_single_2 | 100.00% | 100.00% | +0.00 pp | 100.00% | 100.00% | +0.00 pp |
| niah_single_3 | 100.00% | 100.00% | +0.00 pp | 100.00% | 100.00% | +0.00 pp |
| niah_multikey_1 | 87.50% | 87.50% | +0.00 pp | 87.50% | 87.50% | +0.00 pp |
| niah_multikey_2 | 50.00% | 87.50% | +37.50 pp | 25.00% | 62.50% | +37.50 pp |
| niah_multiquery | 84.38% | 93.75% | +9.38 pp | 90.62% | 90.62% | +0.00 pp |
| niah_multivalue | 87.50% | 90.62% | +3.12 pp | 84.38% | 90.62% | +6.25 pp |
| vt | 87.50% | 82.50% | -5.00 pp | 90.00% | 77.50% | -12.50 pp |
| fwe | 79.17% | 83.33% | +4.17 pp | 58.33% | 58.33% | +0.00 pp |
| qa_1 | 50.00% | 50.00% | +0.00 pp | 50.00% | 62.50% | +12.50 pp |
| qa_2 | 12.50% | 25.00% | +12.50 pp | 25.00% | 25.00% | +0.00 pp |

Each task has only eight examples, so individual task deltas are noisy. The
monotonic aggregate exact-K trend is the stronger observation.

## Selector ceiling

As the C1 payload improves, the routing shortfall relative to exact-K C1
becomes visible:

| Calibration | Routing minus exact-K C1 |
|:---|---:|
| 2K | +0.06 pp |
| 4K | -2.52 pp |
| 32K | -4.15 pp |

The routing arm itself improves with the payload checkpoint; its ceiling simply
improves more slowly. With the 32K C1 factors frozen, selector budget/rank is
now a cleaner next variable than immediately changing the payload latent.

## Calibration diagnostics

| Calibration | Fit relative MSE | Validation relative MSE |
|:---|---:|---:|
| 2K | 0.11262 | 0.13705 |
| 4K | 0.10729 | 0.14031 |
| 32K | 0.10505 | 0.13726 |

The validation objectives correspond to different attention contexts and are
not substitutes for held-out RULER accuracy.

## Jobs and artifacts

- 32K covariance job `8287525`: completed in 2:13; capture took 121.576
  seconds and peaked at 21,069,542,400 allocated CUDA bytes.
- 4-GPU fit and RULER job `8287526`: completed in 24:30.
- Checkpoint: `results/checkpoints/qwen3_8b_c1_v64_32k_als5`.
- Full evaluation: `result.json` in this directory.
- Per-arm summary: `summary.md` in this directory.

