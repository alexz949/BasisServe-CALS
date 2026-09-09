# Qwen3-8B closed-form MSE-RRR Base16 + Q16 Page-Fisher R8: RULER 32K pilot

Frozen C1-V80 + closed-form MSE-RRR Base16 + Q16 Page-Fisher R8; C1 full exact-K reference rerun. No KL allocation or adaptive schedule.
11 tasks × 8 samples = 88 paired prompts. This is the existing 11-task subset, not the complete 13-task RULER suite.
Official base completion prompts, greedy generation, task-specific caps and EOS. No old checkpoint accuracy is reused.

| Task | Samples | C1 exact-K | Uniform R8 | R8-exact, pp | Improvements | Regressions |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 8 | 100.0000% | 100.0000% | +0.0000 | 0 | 0 |
| niah_single_2 | 8 | 100.0000% | 100.0000% | +0.0000 | 0 | 0 |
| niah_single_3 | 8 | 100.0000% | 100.0000% | +0.0000 | 0 | 0 |
| niah_multikey_1 | 8 | 87.5000% | 87.5000% | +0.0000 | 0 | 0 |
| niah_multikey_2 | 8 | 87.5000% | 50.0000% | -37.5000 | 0 | 3 |
| niah_multiquery | 8 | 96.8750% | 100.0000% | +3.1250 | 1 | 0 |
| niah_multivalue | 8 | 93.7500% | 93.7500% | +0.0000 | 0 | 0 |
| vt | 8 | 92.5000% | 92.5000% | +0.0000 | 1 | 1 |
| fwe | 8 | 91.6667% | 75.0000% | -16.6667 | 0 | 4 |
| qa_1 | 8 | 50.0000% | 50.0000% | +0.0000 | 0 | 0 |
| qa_2 | 8 | 37.5000% | 37.5000% | +0.0000 | 0 | 0 |
| Task-balanced mean | 88 | 85.2083% | 80.5682% | -4.6402 | 2 | 8 |

## Measurement scope

Both arms share one full-attention C1 Triton prefill. The first generated token is common; routing applies to subsequent decode forwards.
Every arm has an independent immutable-prefix cache fork. Page32, B2048 including page0, no forced current page and no adaptive budget.
An explicit full-support 4D decode mask locks routing to the native BF16 selector/attention used by the KL oracle, not the alternate fused decode path.
Ranks/factors were not refitted or selected using RULER. The dataset has been used in previous experiments, so this is not an untouched final benchmark.
Scores use the existing RULER case-insensitive substring metric: fraction of reference answers recovered for match_type=all; any-reference hit for match_type=part.
Eight examples per task constitute a pilot, not strong evidence about small accuracy differences. Aggregate scores are task-balanced means.
C1 exact-K is a reference for the added routing effect; this run does not include a new dense-V128 baseline.
Exact K remains GPU-resident and Base128+R coordinates are materialized. This is not an offload-memory or PCIe-latency benchmark.

Environment: basis, NVIDIA L40S, BF16. The result JSON records the factor provenance and exact evaluation commands.
