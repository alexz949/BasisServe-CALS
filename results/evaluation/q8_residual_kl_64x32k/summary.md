# Qwen3-8B residual two-sided KL

C1-V80 and Base16 frozen. C4 64×32768 profile, 16×32768 confirmation.
Windows are packed from eight 4096-token C4 samples, not native 32K documents.

Each window shares a full-attention C1 prefix across all arms; only the final 128 positions use the evaluated routing schedule.
Teacher: same C1-V80 with full exact-K. Page32, B2048, page0 pinned, per-layer R4/R8/R16, average R8.

| Arm | Confirmation teacher KL | Suffix NLL | Suffix PPL |
|---|---:|---:|---:|
| teacher | 0.00000000 | 1.90025533 | 6.68760178 |
| anchor | 0.01884865 | 1.91338754 | 6.77600395 |
| adaptive | 0.01242355 | 1.90617098 | 6.72728054 |

Layer ranks: `[8, 4, 8, 4, 4, 8, 4, 16, 4, 16, 4, 8, 4, 8, 4, 4, 4, 8, 8, 4, 16, 4, 8, 8, 16, 4, 8, 4, 4, 16, 8, 8, 16, 16, 16, 4]`

Predicted additive profile ΔKL: -0.00332971.
Measured confirmation ΔKL: -0.00642510; paired-window standard error: 0.00542688.

The three ranks use signed measured costs directly; no local-error interpolation or negative-slope clipping.
Profile windows overlap residual fitting. Confirmation windows were used for prior factor diagnostics, but not terminal allocation.
KL uses all 128 suffix positions; NLL uses 127 positions with known next-token targets. This is not full-corpus PPL or RULER accuracy.
GPU-resident exact K and materialized Base128+R sidecars are accuracy-oracle storage, not deployable memory or latency measurements.

## Allocation and paired-window diagnostics

Rank counts: R4 in 16 layers, R8 in 12 layers, R16 in 8 layers. Total layer rank is 288; average rank is 8. R16 layers, using zero-based indices, are 7, 9, 20, 24, 29, 32, 33 and 34.

The predicted additive profile delta is not a measured joint-schedule profile result. The adaptive schedule was frozen before confirmation and was not changed using confirmation results.

Mean confirmation KL decreased by 34.0878%; suffix PPL decreased by 0.7191%. KL improved in 11/16 windows and NLL in 10/16. Median paired delta KL is -0.00056838 and median paired delta NLL is -0.00354701. Deltas below are adaptive minus uniform R8; negative means improvement.

| Window index | Delta KL | Delta NLL |
|---|---:|---:|
| 64 | -0.08742837 | -0.06727018 |
| 65 | -0.00863078 | +0.03129380 |
| 66 | -0.00034061 | -0.00820724 |
| 67 | -0.00094505 | -0.03303993 |
| 68 | +0.00033169 | -0.00388028 |
| 69 | -0.00112946 | -0.00801112 |
| 70 | -0.00020534 | -0.00246006 |
| 71 | -0.00079614 | -0.01519596 |
| 72 | -0.00024428 | -0.00321375 |
| 73 | -0.00141712 | -0.01056463 |
| 74 | -0.00210423 | +0.00618959 |
| 75 | +0.00030473 | +0.00017476 |
| 76 | -0.00085430 | +0.00805022 |
| 77 | +0.00024818 | +0.00472526 |
| 78 | +0.00023107 | -0.01668450 |
| 79 | +0.00017847 | +0.00262907 |

Window 64 contributes 85.0457% of the total net KL gain: uniform KL 0.10125574 versus adaptive KL 0.01382737. Descriptively excluding that window leaves a mean delta KL of -0.00102488 across the other 15 windows. This exclusion is only an outlier diagnostic; the reported main result includes all 16 windows. The aggregate result does not establish a stable or statistically significant gain.

## Execution and verification

Environment: `basis`, PyTorch 2.6.0+cu124. All formal profile and confirmation windows ran on L40S. A100 smoke and initial partial profile data are preserved separately and excluded from allocation because the same inputs produced hardware-dependent terminal KL probe deltas. Within-device cached replay matched full suffix recomputation exactly. The three L40S smoke runs and formal window-0 rerun matched exactly for the checked anchor and probe KL arrays.

| Stage | Slurm job | Elapsed |
|---|---|---|
| Profile shard 0 | 8300193 | 14:38 |
| Profile shards 1/2/3, including smoke | 8300185 | 15:36 / 15:34 / 15:32 |
| Allocation | 8300194 | 00:26 |
| Confirmation shards 0/1/2/3 | 8300198 | 00:53 / 00:52 / 00:53 / 00:53 |
| Summary | 8300202 | 00:13 |

All listed jobs completed with exit code 0:0. Six small CPU tests passed. Independent post-run checks reproduced all profile costs, verified the exact-budget DP optimum, checked the frozen schedule hash in every confirmation window, and reproduced the reported aggregate metrics. Shared prefix caches remained unchanged. No NaN, CUDA OOM or failed formal task was observed.

The exact program commands, checkpoint paths, hardware discrepancy and complete measurement protocol are recorded in [the protocol](../../../docs/q8_residual_two_sided_kl_protocol.md). The frozen allocation is [schedule.json](schedule.json), and the machine-readable aggregate is [result.json](result.json).
