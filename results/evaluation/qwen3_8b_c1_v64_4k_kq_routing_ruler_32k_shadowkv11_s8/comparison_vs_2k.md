# Qwen3-8B C1-V64: 2K vs 4K calibration on held-out 32K RULER

## Controlled change

- Model: Qwen3-8B-Base.
- C1 layout: uniform Value rank 64 for every physical KV head in every layer.
- Old fit bank: 256 C4 windows x 2048 tokens = 524,288 fit rows.
- New fit bank: 128 C4 windows x 4096 tokens = 524,288 fit rows.
- New validation bank: 16 document-disjoint C4 windows x 4096 tokens.
- The K-routing factors, R32 routing rank, 64-token pages, B1024 exact-token
  budget, dense baseline, RULER prompts, and greedy decoding are unchanged.
- Evaluation: 11 tasks x 8 samples on the same held-out RULER-v1 32K bank.

The experiment therefore changes C1 calibration context geometry while holding
the number of C1 fit tokens and the runtime cache shape fixed.

## Aggregate accuracy

| Arm | 2K calibration | 4K calibration | Delta | Paired improvements | Paired regressions | Ties |
|:---|---:|---:|---:|---:|---:|---:|
| Exact-K + C1-V64 | 66.12% | 76.23% | +10.11 pp | 17 | 5 | 66 |
| R32/B1024 exact-K routing + C1-V64 | 66.17% | 73.71% | +7.54 pp | 15 | 6 | 67 |
| BF16 dense reference | 86.48% | 86.48% | +0.00 pp | - | - | 88 |

With the 4K checkpoint, routing is 2.52 pp below the exact-K C1 ceiling. The
same gap was +0.06 pp with the old checkpoint, where the much larger payload
error hid the selector loss.

## Task-level comparison

| Task | C1 2K | C1 4K | Delta | Routing 2K | Routing 4K | Delta |
|:---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | +0.00 pp | 100.00% | 100.00% | +0.00 pp |
| niah_single_2 | 100.00% | 100.00% | +0.00 pp | 100.00% | 100.00% | +0.00 pp |
| niah_single_3 | 87.50% | 100.00% | +12.50 pp | 75.00% | 100.00% | +25.00 pp |
| niah_multikey_1 | 75.00% | 87.50% | +12.50 pp | 75.00% | 87.50% | +12.50 pp |
| niah_multikey_2 | 25.00% | 50.00% | +25.00 pp | 12.50% | 25.00% | +12.50 pp |
| niah_multiquery | 87.50% | 84.38% | -3.12 pp | 90.62% | 90.62% | +0.00 pp |
| niah_multivalue | 78.12% | 87.50% | +9.38 pp | 78.12% | 84.38% | +6.25 pp |
| vt | 45.00% | 87.50% | +42.50 pp | 55.00% | 90.00% | +35.00 pp |
| fwe | 66.67% | 79.17% | +12.50 pp | 66.67% | 58.33% | -8.33 pp |
| qa_1 | 37.50% | 50.00% | +12.50 pp | 50.00% | 50.00% | +0.00 pp |
| qa_2 | 25.00% | 12.50% | -12.50 pp | 25.00% | 25.00% | +0.00 pp |

## Interpretation

The 10.11 pp exact-K improvement establishes that the earlier 2K C1 fit was a
material source of the 32K quality loss. It does not establish that 4K is the
optimal calibration length: exact-K C1 still trails BF16 by 10.25 pp, and the
evaluation has only eight samples per task.

The routing arm now trails its exact-K C1 ceiling, especially on `fwe` and
`niah_multikey_2`. Because the routing checkpoint and sparse budget were held
fixed, this new 2.52 pp gap is attributable to sparse selection/decode support,
not to the C1 calibration change itself. The clean next control is to improve
or enlarge the selector while keeping this 4K C1-V64 checkpoint frozen before
introducing the S16/P48/R16 joint latent.

## Jobs and artifacts

- Covariance job: `8286981`, completed in 2:12; covariance capture itself took
  86.278 seconds.
- Fit and RULER job: `8286996`, completed in 26:54.
- 4K C1 checkpoint: `results/checkpoints/qwen3_8b_c1_v64_4k_als5`.
- Full result: `result.json` in this directory.
- Per-arm summary: `summary.md` in this directory.

