# Qwen3-8B Q-aware Base16 + uniform Fisher R8: RULER 32K pilot

Frozen C1-V80 + Q-aware Base16 + uniform Page-Fisher R8; C1 full exact-K reference rerun. No KL allocation or adaptive schedule.
11 tasks × 8 samples = 88 paired prompts. This is the existing 11-task subset, not the complete 13-task RULER suite.
Official base completion prompts, greedy generation, task-specific caps and EOS. No old checkpoint accuracy is reused.

| Task | Samples | C1 exact-K | Uniform R8 | R8-exact, pp | Improvements | Regressions |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 8 | 100.0000% | 100.0000% | +0.0000 | 0 | 0 |
| niah_single_2 | 8 | 100.0000% | 100.0000% | +0.0000 | 0 | 0 |
| niah_single_3 | 8 | 100.0000% | 75.0000% | -25.0000 | 0 | 2 |
| niah_multikey_1 | 8 | 87.5000% | 87.5000% | +0.0000 | 0 | 0 |
| niah_multikey_2 | 8 | 87.5000% | 25.0000% | -62.5000 | 0 | 5 |
| niah_multiquery | 8 | 96.8750% | 84.3750% | -12.5000 | 0 | 4 |
| niah_multivalue | 8 | 93.7500% | 71.8750% | -21.8750 | 0 | 3 |
| vt | 8 | 92.5000% | 75.0000% | -17.5000 | 0 | 3 |
| fwe | 8 | 91.6667% | 58.3333% | -33.3333 | 0 | 5 |
| qa_1 | 8 | 50.0000% | 50.0000% | +0.0000 | 0 | 0 |
| qa_2 | 8 | 37.5000% | 37.5000% | +0.0000 | 0 | 0 |
| Task-balanced mean | 88 | 85.2083% | 69.5076% | -15.7008 | 0 | 22 |

## Measurement scope

Both arms share one full-attention C1 Triton prefill. The first generated token is common; routing applies to subsequent decode forwards.
Every arm has an independent immutable-prefix cache fork. Page32, B2048 including page0, no forced current page and no adaptive budget.
An explicit full-support 4D decode mask locks routing to the native BF16 selector/attention used by the KL oracle, not the alternate fused decode path.
Ranks/factors were not refitted or selected using RULER. The dataset has been used in previous experiments, so this is not an untouched final benchmark.
Scores use the existing RULER case-insensitive substring metric: fraction of reference answers recovered for match_type=all; any-reference hit for match_type=part.
Eight examples per task constitute a pilot, not strong evidence about small accuracy differences. Aggregate scores are task-balanced means.
C1 exact-K is a reference for the added routing effect; this run does not include a new dense-V128 baseline.
Exact K remains GPU-resident and Base128+R coordinates are materialized. This is not an offload-memory or PCIe-latency benchmark.

Environment: basis, NVIDIA L40S, BF16. See [protocol and commands](../../../docs/q8_qbase_uniform_ruler_protocol.md).

## Execution and independent audit

Formal Slurm array `8300360` tasks 0–3 completed with exit code `0:0` in
06:42, 06:23, 07:07 and 06:42. CPU summary `8300361` completed in 00:17.
All 88 paired predictions were independently decoded from saved token IDs
and rescored; per-task means, the aggregate result and paired counts match.
The audit also verified sample/shard coverage, references, current source and
bank hashes, same-prefix first tokens, and EOS/cap behavior. Finite-logit
checks passed; no CUDA error or OOM occurred.

Paired counts: 22 regressions, 66 ties, zero improvements. Twenty exact-K and
25 uniform-R8 generations reached their official task cap without EOS.
Peak allocated GPU memory was 25.1932 GiB. The four-token smoke is excluded.

The current same-payload exact-K comparison measures the added sparse-routing
effect. It does not separate selector estimation from token-budget effects.
No matched old MSE-Base arm was rerun, so these results do not determine
whether replacing MSE Base with Q-aware Base improved RULER accuracy.
