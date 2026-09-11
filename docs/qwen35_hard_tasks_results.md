# Qwen3.5-9B: MATH500, MBPP+, IFEval results

All nine full evaluations completed on 2026-09-10, with final engine shutdown at 13:15 EDT. The complete smoke/full pipeline exited with code 0. The final audit status is `complete_and_audited`: model and factor provenance match the prior audited GSM8K runs, prompts and target documents match across arms, generation protocols match, no input prompts were truncated, and every aggregate metric was recomputed from per-question scores.

## Main results

All values below are percentages. Thinking was disabled for all arms. MATH500 uses mathematical-equivalence verification; MBPP+ uses greedy pass@1 on official augmented tests; IFEval columns use prompt-level metrics.

| Configuration | MATH500 math_verify | MBPP+ plus pass@1 | IFEval prompt strict | IFEval prompt loose |
|---|---:|---:|---:|---:|
| Dense | 91.00 | 72.75 | 83.92 | 88.72 |
| Dense V + GDN Wo768 / Full Wo512 | 85.00 | 59.79 | 75.42 | 80.59 |
| Two-sided V128 + GDN Wo768 / Full Wo512 | 75.60 | 51.59 | 69.32 | 73.38 |

Adding Wo compression with Dense V reduces MATH500 by 6.00 percentage points, MBPP+ by 12.96 points, and IFEval prompt-strict by 8.50 points. Moving to the V128 arm reduces these by a further 9.40, 8.20, and 6.10 points. The V128 arm includes Wo refitted on its frozen compressed-V trajectory, so this comparison is not an isolated V-only ablation. The relatively small GSM8K regression does not generalize to these tasks.

## Other scoring metrics

| Configuration | MATH500 exact_match | MBPP base pass@1 | IFEval instruction strict | IFEval instruction loose |
|---|---:|---:|---:|---:|
| Dense | 15.80 | 87.57 | 88.97 | 92.45 |
| Dense V + GDN Wo768 / Full Wo512 | 9.00 | 73.28 | 83.09 | 86.57 |
| Two-sided V128 + GDN Wo768 / Full Wo512 | 59.80 | 61.90 | 78.30 | 81.53 |

MATH500 exact_match is strongly affected by extraction/normalization: Dense scores 15.8% with this metric but 91.0% with math_verify. Inspection confirmed an example with mathematically correct polar coordinates that failed exact_match. Both metrics are retained; no manual rescoring was substituted.

## Output behavior

| Configuration | MATH500 length capped /500 | MBPP+ length capped /378 | IFEval length capped /541 | Closing-think responses (math / code / instructions) |
|---|---:|---:|---:|---|
| Dense | 31 | 4 | 9 | 0 / 0 / 0 |
| Dense V + GDN Wo768 / Full Wo512 | 64 | 45 | 25 | 1 / 0 / 0 |
| Two-sided V128 + GDN Wo768 / Full Wo512 | 147 | 165 | 68 | 2 / 6 / 0 |

Length caps were 4096 tokens for MATH500 and 2048 for MBPP+/IFEval. More capped responses accompany compression, especially V128 MBPP+ (165/378), but this alone does not establish how many answers would improve with longer generation. Rare closing-think strings are model outputs despite disabled thinking, not evidence that thinking mode was enabled. Raw responses and finish reasons are retained.

## Execution and artifacts

- Environment: `lowrankarena` for vLLM and EvalPlus, `lowrank` for the final audit. EvalPlus 0.3.1, math-verify 0.9.0, antlr4 runtime 4.11.0. Full package versions are recorded in every result.
- Direct execution on this machine (no Slurm); GPUs 2/5/6 for Dense/Dense-Wo/V128-Wo respectively, TP1, OMP/MKL 2 threads per worker.
- MATH500: all 500 questions, fixed 4-shot examples; MBPP+: all 378 questions, 0-shot, greedy; IFEval: all 541 prompts, 0-shot. Same non-thinking chat template and seed 20260909.
- Batch limit 32, batched tokens 4096, model length 8192, fixed KV cache 6 GiB.
- All 378 MBPP+ canonical solutions passed both base and augmented reference checks before model evaluation. All nine two-question smoke jobs passed execution and protocol audit.

Executed commands:

```bash
bash scripts/run_qwen35_hard_tasks.sh smoke && bash scripts/run_qwen35_hard_tasks.sh full
# lowrank, OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u -m evaluation.summarize_qwen35_hard_tasks --smoke
python -u -m evaluation.summarize_qwen35_hard_tasks
```

The exact per-task Python commands and bank paths are in `scripts/run_qwen35_hard_tasks.sh` and `docs/qwen35_hard_tasks_protocol.md`.

- Full raw results: `results/q35_hybrid/hard_tasks/full/<arm>_<task>.json` (nine files).
- Audited summary: `results/q35_hybrid/hard_tasks/summary.json`.
- Logs: `results/q35_hybrid/hard_tasks/logs/full_<arm>_<task>.log`.
- Reference and audit logs: `mbpp_reference.log`, `smoke_audit.log`, `full_audit.log` in that log directory.
- Local checks: three existing tests passed; shell syntax and `git diff --check` passed.

Warnings: vLLM printed process-group cleanup warnings after engine shutdown and FLA short-sequence shape warnings. All nine outputs were produced, pipeline exit was 0, and the final audit passed. Existing V factors still represent finite-iteration ALS; these experiments do not establish full ALS convergence. The vLLM adapter uses padded V cache at TP1, so these quality results do not measure physical cache-memory or communication savings.
