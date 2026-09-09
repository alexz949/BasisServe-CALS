# Qwen3-32B R80: MATH500, MBPP+, BBH

## Resumed 2026-09-08

User authorized resuming the 12 compressed-arm full evaluations, preferring
available A100/H200 GPUs and otherwise four L40S GPUs. At submission, all
A100 GPUs and the operational H200 node were allocated; lovelace was idle.
Submitted L40S TP=2 smoke array `8302327` and full array `8302329`.
Each full group depends on its corresponding smoke group (`aftercorr`).
Group 0: C1 and PaLU-M; group 1: PaLU-G2 and PaLU-G4, three benchmarks each.
Dense's existing H200 results are retained without rerunning.

`max_num_batched_tokens` is now **8192**, down from 16384 to reduce prefill
peak memory after the prior PaLU-M BBH OOM. `max_num_seqs=64`, memory
utilization 0.85, context 8192, generation caps, prompts, few-shot settings,
and scoring remain unchanged. Runtime scheduling differs from the earlier
smoke and H200 runs; cross-backend outputs are not guaranteed bitwise equal.
Smoke runs BBH first, starting with PaLU-M/G4, then checks all remaining arm/task
combinations using `--limit 2` in `smoke-l40s-b8k/`. Full outputs reuse `full/`.
Logs are `logs/l40-smoke-8302327_*.log` and `logs/l40-full-8302329_*.log`.

Prior smoke `8300975` completed 11/12 stages; PaLU-M/BBH failed from GPU OOM.
Its dependent full array `8300977` was canceled without running, consistent
with the user's subsequent pause request. All prior artifacts are retained.

## L40S continuation (2026-09-06)

Dense's three complete results below remain from H200 TP=1. The remaining
12 evaluations were moved, with user approval, to L40S TP=2. GPU memory
utilization is 0.85 on all four compressed arms, because one GPU consistently
had 38.99 GiB free, below the 40.07 GiB requested at 0.90. Generation and
scoring parameters are unchanged; GPU architecture and TP reduction order
differ, so this is not a bitwise-identical backend comparison.

| Completed Dense task | Metric | Score |
|---|---|---:|
| MATH500 | math_verify | 0.796 |
| MATH500 | exact_match | 0.694 |
| MBPP+ | base_pass_at_1 | 0.4920634921 |
| MBPP+ | plus_pass_at_1 | 0.4576719577 |
| BBH | sample-weighted exact_match | 0.4688987867 |

Dense MATH500 wrote complete results but hung on process exit and hit its
eight-hour Slurm limit. Explicit vLLM engine shutdown with a 30-second timeout
has now been added; normal exits must be verified in the new smoke jobs.

Two arrays, each with two concurrent tasks requesting two L40S GPUs and eight
CPUs per task: smoke `8300975`, dependent full evaluation `8300977`.
Group 0 runs C1 then PaLU-M (three benchmarks each); group 1 runs PaLU-G2 then
PaLU-G4. Both smoke groups must complete before the full array starts.
Pending H200 array tasks `8300822_[3-14]` were canceled. Other experiments were
not canceled. Existing Dense results and prior logs are retained.

Run `bash evaluation/run_qwen3_32b_hard_l40s.sh smoke` or `full` inside the
corresponding Slurm allocation with `SLURM_ARRAY_TASK_ID=0` or `1`. Expanded
Python commands are logged. The Python entrypoint is
`evaluation/eval_qwen3_32b_hard_l40s_vllm.py`; all CLI options match the command
below, except `--gpu-memory-utilization 0.85`. Smoke uses `smoke-l40s-m85/`
with `--limit 2`; full results use `full/`.
C1 TP slices contiguous KV/query heads and their corresponding V/O factors;
PaLU TP slices reconstructed V rows. Synthetic nonuniform-rank C1 and all
PaLU group-size numerical checks passed before submission.

Initial 0.90 smoke `8300970` had two C1 startup failures due to the same free
memory check. Its PaLU-G2 MATH500 stage completed and explicitly shut down
successfully. That completed stage is retained in `smoke-l40s/`. The old smoke
and dependent full array `8300974` were canceled before resubmitting both
groups at 0.85; no full L40S results existed at that point.

## Scope and protocol

Five arms: `Q3-32B-Dense`, `Q3-32B-C1-R80`, `Q3-32B-PALUM-R80`,
`Q3-32B-PALUG2-R80`, `Q3-32B-PALUG4-R80`. Existing checkpoints only;
no fitting or allocation changes. PaLU uses Fisher allocation. C1 uses
Two-Sided-KL allocation. Compressed factors are folded into dense projection
slots; this is a quality comparison, not a compressed-kernel speed benchmark.

| Task | lm-eval task | Few-shot | Generation cap |
|---|---|---:|---:|
| MATH500 | minerva_math500 | 4 | 4096 |
| MBPP+ | mbpp_plus_full (local task) | 0 | 2048 |
| BBH | bbh_cot_fewshot, all 27 subtasks | 3 | 1024 |

All arms: greedy decoding, no chat template, context 8192, max_num_seqs 64,
max_num_batched_tokens 16384, GPU memory utilization 0.90, TP=1, H200 NVL,
TRITON_ATTN. BBH retains the harness's per-task stop strings and answer filter;
its aggregate is sample-weighted. Raw outputs and subtask metrics are retained.

Environment: basis-derived `/deac/csc/yangGrp/zhangal/.cache/vllm/cu128-v0.18.0/venv`,
vLLM `0.18.1.dev0+gbcf2be961.d20260828.cu128`, torch `2.10.0+cu128`,
lm-eval `0.4.11`, datasets `5.0.0`. Added math-verify `0.9.0`,
latex2sympy2_extended `1.11.0`, antlr4-python3-runtime `4.11.0`.

## MBPP+ scoring

The installed harness `mbpp_plus` inherits a target containing only three base
assertions. The local task instead uses EvalPlus `0.3.1`'s `untrusted_check`,
official MBPP+ `v0.2.0` inputs, tolerances, special oracles, and default adaptive
timeouts. Each problem is scored in a clean CPU interpreter, outside vLLM's
CUDA process. The address-space ceiling is 16 GiB (including the large
reference-output mapping for extreme MBPP/255 inputs), identical for all arms.
Expected outputs are computed with EvalPlus `trusted_exec` from
official reference solutions on the Slurm node and cached by dataset hash.
The 378 task IDs must match the Hugging Face dataset used for prompts.
Both `base_pass_at_1` and `plus_pass_at_1` are reported; plus requires both base
and augmented tests to pass. Code fences are stripped consistently for every
arm. This is a custom zero-shot lm-eval prompt with official EvalPlus scoring,
not a claim that prompts match every EvalPlus leaderboard protocol.

Zero-shot avoids reusing the harness's default MBPP examples (IDs 2, 3, 4), which
also occur in this test set. Initial scoring preflights exposed missing
`test_imports`, then frozen floating/complex expected-output comparisons and
an unsuitable whole-program timeout in the exported tests (IDs 590, 599).
The exported-test scorer was replaced with official EvalPlus before any model
evaluation completed. A full canonical-solution preflight is required before
generation smoke.

## Commands

`evaluation/run_qwen3_32b_hard_generation.sh` records each expanded Python
command in the Slurm log. Equivalent evaluation command:

```bash
/deac/csc/yangGrp/zhangal/.cache/vllm/cu128-v0.18.0/venv/bin/python \
  evaluation/eval_qwen3_32b_hard_generation_vllm.py \
  --run-id "$RUN_ID" \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
  --checkpoint-dir "ICLR-results/qwen3-32b/checkpoints/$RUN_ID" \
  --output-dir "ICLR-results/qwen3-32b/hard-r80/full/$RUN_ID/$TASK" \
  --task "$TASK" --max-length 8192 --max-num-seqs 64 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.90 \
  --torch-num-threads 4 --confirm-run-unsafe-code
```

Smoke uses `--limit 2` and `smoke/` instead of `full/`. For BBH the limit is per
subtask, not two examples for the entire group.

## Submission status

Submitted 2026-09-06: smoke `8300821`; dependent full array `8300822` (15 tasks,
one concurrent H200 allocation). Original preflight `8300811` and dependent
array `8300812` were canceled to include missing test imports; `8300816` failed
the exported-test reference check and dependency `8300817` did not run.
`8300819`/`8300820` were canceled to isolate scoring in clean CPU processes.
Logs are kept
under `logs/`; retries reuse the same result directories. No benchmark scores
are claimed until the corresponding full result is complete.
