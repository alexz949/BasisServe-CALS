# Qwen3-32B: corrected BBH and MBPP+ pass@3

## Scope

Five arms: Dense, C1 Two-Sided-KL R80 (alpha=1), PaLU-M/G2/G4 Fisher R80.
Existing checkpoints are unchanged. Earlier `hard-r80/` results are preserved.
This comparison reruns Dense on the same hardware and TP configuration as the
compressed arms; do not mix the previous H200 Dense row into this table.

All arms: two L40S GPUs, TP=2, bf16, TRITON_ATTN, context 8192,
max_num_seqs=64, max_num_batched_tokens=8192, GPU memory utilization 0.85.
No chat template. Environment: basis-derived
`/deac/csc/yangGrp/zhangal/.cache/vllm/cu128-v0.18.0/venv`,
vLLM `0.18.1.dev0+gbcf2be961.d20260828.cu128`, torch `2.10.0+cu128`,
lm-eval `0.4.11`, datasets `5.0.0`, EvalPlus `0.3.1`.

## BBH

All 27 subtasks, 6511 examples, standard 3-shot CoT prompts and get-answer
regex. Greedy, max generation 1024 tokens, unchanged from the first run.
Removed double-newline and bare `Q` stop strings. Stop only at tokenizer EOS
or the explicit next-question boundary `\nQ:`. Keep the token cap fixed to
isolate the stop-condition change; inspect length-limit rates before treating
this as a clean reasoning comparison. Report answer extraction rate alongside
sample-weighted exact match, with all subtask scores retained.

## MBPP+

378 problems, zero-shot prompts unchanged. Three independently sampled
completions per problem (`repeats=3`, `take_first_k=3`), temperature=0.2,
top_p=0.95, max generation 2048 tokens, stop `[DONE]` or EOS.
The seed for sample i is `(first 32 bits of SHA256(UTF8(prompt)) + i) mod 2^31`,
identical across model arms and distinct for the three samples of each prompt.
Samples may still produce identical code; no deduplication or resampling.

Each completion uses the existing isolated CPU EvalPlus scorer (official
MBPP+ v0.2.0 base and augmented inputs, reference-scaled timeouts, 16 GiB
address-space ceiling). Per problem, with c successful completions:

- pass@1 = c/3 (unbiased single-sample estimate at this sampling temperature).
- pass@3 = 1 if c > 0, otherwise 0.

Aggregate by averaging over problems, separately for base and plus tests.
This sampled pass@1 is not the earlier greedy pass@1 protocol.

## Artifacts and commands

Full lm-eval samples contain all raw completions and filtered responses.
`generation_records.json` adds task/doc/sample identifiers, prompt hash, seed,
original/retained prompt-token counts, generated-token counts, finish reason,
stop reason, stop strings and generation limit.

Run `bash evaluation/run_qwen3_32b_bbh_mbpp3.sh smoke` or `full` in a Slurm
allocation with `SLURM_ARRAY_TASK_ID` set. The script logs the expanded command:

```bash
/deac/csc/yangGrp/zhangal/.cache/vllm/cu128-v0.18.0/venv/bin/python \
  evaluation/eval_qwen3_32b_bbh_mbpp3_vllm.py \
  --run-id "$RUN_ID" \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
  --checkpoint-dir "ICLR-results/qwen3-32b/checkpoints/$RUN_ID" \
  --output-dir "ICLR-results/qwen3-32b/bbh-mbpp3/full/$RUN_ID/$TASK" \
  --task "$TASK" --max-length 8192 --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 \
  --torch-num-threads 4 --confirm-run-unsafe-code
```

Smoke adds `--limit 2`, using `smoke/`; BBH limits apply per subtask.
Submitted 2026-09-08: smoke array **8302588**, tasks 0–3 (Dense/C1, both tasks);
dependent full array **8302589**, tasks 0–9. At most two TP=2 jobs run at once.
All four smoke jobs must succeed before full evaluation starts. At submission
the L40S node was occupied by another user, so these jobs were pending.

Local checks passed for syntax, three-response scoring, pass@k calculations,
distinct reproducible sampling seeds, and generation-record capture. Actual
GPU execution and full harness integration remain subject to the smoke gate.
