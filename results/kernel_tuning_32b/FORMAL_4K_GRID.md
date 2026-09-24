# Qwen3-32B TP8 optimized 4K grid

Approved by the user on 2026-09-24. This supersedes the proposed 8K-input
grid in FORMAL_GRID.md; the 8K-input runs completed so far were smokes only.

Status: completed successfully on 2026-09-24. Both arms and the automatic
summary generator exited zero. See [the full results](../vllm_32b_tp8_sm89_4k_budget8k/summary.md).

- Environment: basis, eight L40S GPUs, TP8, BF16.
- Arms: Dense and C1 R64-S6 with the validated SM89 prefill/decode kernels.
- Prompt: 4096 tokens. Output: 128 tokens, ignoring EOS.
- Scheduler token budget: 8192, matching the historical 4K experiment.
- Cohorts: 1, 2, 4, 8, 16, 32, 64, 128, 256.
- Per arm/cohort: one warmup, three measurements (54 measured runs total).
- Separate rank-zero profiles: cohorts 1, 32, 256 for each arm.
- Memory utilization 0.8, synchronous chunked prefill, no prefix caching,
  compilation NONE, FULL_DECODE_ONLY CUDA Graphs.
- Structural factor validation only. No SHA256 checks.
- No MLP, routing, rank or decoder-design changes; no uploads.

Exact commands and model/factor paths are in the runner:

```bash
bash evaluation/run_qwen3_32b_tp8_sm89.sh
```

Output: `results/vllm_32b_tp8_sm89_4k_budget8k/`. Dense and C1 run serially,
with incremental JSON and separate logs. The existing summarizer runs after
both arms complete. Existing result JSON is never overwritten by the runner.
Historical `results/vllm_32b_tp8/` remains untouched.

Report medians, ranges, E2E throughput, TTFT, TPOT and preemptions. Cohort
size is not a constant execution batch. TPOT includes scheduling effects.
Large cohorts can be capacity affected even without preemptions. Profile
kernel sums are not wall time. These synthetic timing runs do not measure
model quality. The prior smoke and kernel tests are documented in SUMMARY.md.

## Completion Audit

- All 54 measured runs completed, containing 3066 requests and 392448
  generated tokens. Every request returned exactly 128 tokens.
- All eight C1 workers reported 64 loaded V64 layers, positive graph-capture
  counts and the expected SM89 prefill/H8 decode specializations.
- Six profiles were exported and parsed with valid decode graph attribution.
- No model OOM occurred. Dense batch 256 had one preemption in each measured
  run (three total); all other Dense runs and all C1 runs had zero.
- The two summary/metric regression tests passed. All eight GPUs were released.
- Runtime/driver source matched the initial `source.tar` by direct byte
  comparison. Only report wording changed during the run; its final source
  is preserved in `summary_source.py`. No SHA256 checks were performed.
- Detailed verification is in `audit.log` and `summary_tests.log` in the
  result directory. Startup, profile-export waits and nonfatal process-cleanup
  warnings remain in the arm logs. Nothing was uploaded or committed.

Current E2E speedups range from 0.975x at batch 1 to 1.229x at batch 128
and 1.307x at batch 256. The last point is capacity-sensitive. Relative to
the historical C1 CSV in `results/vllm_32b_tp8/summary.csv`, current C1
request times are 1.28%-2.97% lower across the nine cohorts. This historical
comparison covers the optimized prefill/decode/output path together; it
is not a controlled prefill-only ablation. Kernel-level speedups must not
be presented as full-model speedups.
