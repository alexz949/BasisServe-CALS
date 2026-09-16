# Two-stage RULER 64K, 88 prompts

Model: Llama-3.1-8B base; four independent L40S workers, one GPU per prompt. BF16; full FlashAttention prefill, original Dense V and Wo, CPU K offload, vector missing-K fetch and Triton slot attention. Greedy generation, native EOS, original per-task caps. Both arms use identical saved token IDs from the prior 88-prompt run.

Full arm: current 8-warp B16R16 full scan. Two arm: exact post-RoPE Page32 min/max, 512 total candidates per KV head including sink/recent pages, B16R16 candidate scan and candidate normalization. Both use sink32 + recent64 within hard2048. The two-stage branch never scans all routing codes during decode.

| Task | Full B16R16 | Two-stage | Difference (pp) |
|---|---|---|---|
| niah_single_1 | 100.00 | 100.00 | +0.00 |
| niah_single_2 | 100.00 | 100.00 | +0.00 |
| niah_single_3 | 100.00 | 100.00 | +0.00 |
| niah_multikey_1 | 100.00 | 100.00 | +0.00 |
| niah_multikey_2 | 50.00 | 62.50 | +12.50 |
| niah_multiquery | 93.75 | 93.75 | +0.00 |
| niah_multivalue | 90.62 | 90.62 | +0.00 |
| vt | 92.50 | 97.50 | +5.00 |
| fwe | 75.00 | 75.00 | +0.00 |
| qa_1 | 37.50 | 37.50 | +0.00 |
| qa_2 | 50.00 | 50.00 | +0.00 |
| **All 88** | **80.85** | **82.44** | **+1.59** |

Changed scores: 7; identical generated sequences: 39/88. All paired prefill first tokens match; all evaluated logits finite. Per-request min/max and slot states are reset. Runtime numerical behavior differs from the older Transformers evaluations, so their scores are not used as the paired baseline.

Environment: basis. Commands:

```bash
python -m evaluation.eval_two_stage_ruler --smoke
python -m evaluation.eval_two_stage_ruler --shard 0 --shards 4
# Other workers use --shard 1, 2, 3.
python -m evaluation.eval_two_stage_ruler --summarize
```

All raw per-sample predictions and source hashes are retained in samples/. No refit, production default change, commit, or push.
