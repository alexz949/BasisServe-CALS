# Qwen3-8B C1/K-routing calibration-scale comparison

## Protocol

The final evaluation is the same fixed 88-example RULER-v1 32K set used by the
previous run: 11 tasks, 8 examples per task, greedy decoding, page size 64,
K-routing rank 32, and an exact-K token budget of 1024.

| Configuration | C1 fit | C1 validation | K-routing fit | C1-V rank | Routing |
|:---|---:|---:|---:|---:|:---|
| Previous | 16 x 32K | 2 x 32K | 8 x 32K | 64/head | R32, B1024 |
| Expanded | 32 x 32K | 4 x 32K | 32 x 32K | 64/head | R32, B1024 |

The expanded C1 and routing fits share exactly the same first 32 packed 32K
contexts. Each context is a lossless concatenation of eight independently
sampled 4K C4 document windows. The expanded fit therefore covers 1,048,576
tokens from 256 source documents. Its four C1 validation contexts contain a
disjoint additional 131,072 tokens from 32 source documents. K routing has no
held-out selection step in this fixed-hyperparameter run.

## Main result

| Arm | Previous | Expanded | Delta vs previous | Gap to BF16 |
|:---|---:|---:|---:|---:|
| BF16 K + BF16 V | 86.48% | 86.48% | +0.00 pp | +0.00 pp |
| Exact K + C1-V64 | 81.84% | 79.00% | -2.84 pp | -7.48 pp |
| R32/B1024 exact-K routing + C1-V64 | 77.69% | 79.11% | +1.42 pp | -7.37 pp |

At the example level, expanded versus previous C1 has 5 improvements, 10
regressions, and 73 ties. Expanded end-to-end routing has 8 improvements, 5
regressions, and 75 ties.

Within the previous configuration, routing was 4.15 pp below its exact-K C1
ceiling. Within the expanded configuration, routing is 0.11 pp above exact-K
C1 in the task-balanced mean (5 improvements, 7 regressions, and 76 ties).
Sparse decoding can occasionally change an already incorrect C1 answer into a
correct one, so this near-zero aggregate difference is not an assertion that
routing exactly reproduces dense attention.

## Per-task comparison

| Task | Previous C1 | Expanded C1 | C1 delta | Previous routing | Expanded routing | Routing delta |
|:---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | +0.00 pp | 100.00% | 100.00% | +0.00 pp |
| niah_single_2 | 100.00% | 100.00% | +0.00 pp | 100.00% | 100.00% | +0.00 pp |
| niah_single_3 | 100.00% | 100.00% | +0.00 pp | 100.00% | 100.00% | +0.00 pp |
| niah_multikey_1 | 87.50% | 75.00% | -12.50 pp | 87.50% | 87.50% | +0.00 pp |
| niah_multikey_2 | 87.50% | 62.50% | -25.00 pp | 62.50% | 50.00% | -12.50 pp |
| niah_multiquery | 93.75% | 90.62% | -3.12 pp | 90.62% | 93.75% | +3.12 pp |
| niah_multivalue | 90.62% | 87.50% | -3.12 pp | 90.62% | 90.62% | +0.00 pp |
| vt | 82.50% | 95.00% | +12.50 pp | 77.50% | 90.00% | +12.50 pp |
| fwe | 83.33% | 70.83% | -12.50 pp | 58.33% | 58.33% | +0.00 pp |
| qa_1 | 50.00% | 50.00% | +0.00 pp | 62.50% | 62.50% | +0.00 pp |
| qa_2 | 25.00% | 37.50% | +12.50 pp | 25.00% | 37.50% | +12.50 pp |

## Interpretation

Increasing the number of independent 32K calibration contexts is not a
monotonic cure for this system. It improves the end-to-end routing arm, most
visibly on `vt` and `qa_2`, but the newly fitted C1-V64 subspace loses more on
`niah_multikey_2`, `fwe`, and `niah_multikey_1` than it gains elsewhere. The
best exact-K C1 checkpoint under this RULER metric therefore remains the
16-window checkpoint, while the best tested routing endpoint is the expanded
32/32 configuration.

This suggests that the earlier 16-window C1 result was not simply limited by
too few token rows. At fixed V rank 64, the attention-output objective must
trade off heterogeneous behaviors, and adding broader C4 contexts changes
that tradeoff. Calibration-domain coverage and objective alignment matter at
least as much as aggregate token count.

The C1 covariance diagnostics are 13.73% held-out relative MSE for the previous
checkpoint and 14.01% for the expanded checkpoint. The routing fit weighted
relative squared errors are 0.003196 and 0.003525, respectively. Neither pair
is a strict cross-run comparison because the expanded run uses a larger and
more diverse fit/validation distribution. The fixed RULER evaluation above is
the relevant common comparison.

Because both C1 and routing factors changed together, the observed +1.42 pp
end-to-end routing gain does not by itself isolate the routing-factor effect.
A strict factorial attribution would additionally evaluate previous-C1 with
expanded-routing and expanded-C1 with previous-routing on the same 88 examples.

## Artifacts and execution

- Expanded C1 checkpoint: `results/checkpoints/qwen3_8b_c1_v64_32k_n32_als5`
- Expanded routing checkpoint: `results/checkpoints/qwen3_8b_pairwise_kqsvd_r128_c4_n32_s32768`
- Expanded result: `result.json` in this directory (`status=complete`, 88 records)
- Calibration job: `8287624`, 1 x L40S, 8m37s, exit code 0
- Fit/evaluation job: `8287625`, 4 x L40S, 24m03s, exit code 0

The expanded 4K source bank preserves the previous 144 source windows as an
exact record-hash prefix. The 32K packing manifest records
`token_content_preserved=true` and shape `36 x 32768`.
