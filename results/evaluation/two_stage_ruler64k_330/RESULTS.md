# Two-stage RULER 64K, 330 prompts

Model: Llama-3.1-8B base; four independent L40S workers, one GPU per prompt. BF16; full FlashAttention prefill, original Dense V and Wo, CPU K offload, vector missing-K fetch and Triton slot attention. Greedy generation, native EOS, original per-task caps. Both arms use identical token IDs from the frozen official dataset.

Full arm: current 8-warp B16R16 full scan. Two arm: exact post-RoPE Page32 min/max, 512 total candidates per KV head including sink/recent pages, B16R16 candidate scan and candidate normalization. Both use sink32 + recent64 within hard2048. The two-stage branch never scans all routing codes during decode.

| Task | Full B16R16 | Two-stage | Difference (pp) |
|---|---|---|---|
| niah_single_1 | 100.00 | 100.00 | +0.00 |
| niah_single_2 | 100.00 | 100.00 | +0.00 |
| niah_single_3 | 100.00 | 100.00 | +0.00 |
| niah_multikey_1 | 100.00 | 100.00 | +0.00 |
| niah_multikey_2 | 63.33 | 73.33 | +10.00 |
| niah_multiquery | 96.67 | 96.67 | +0.00 |
| niah_multivalue | 97.50 | 96.67 | -0.83 |
| vt | 90.00 | 92.67 | +2.67 |
| fwe | 86.67 | 90.00 | +3.33 |
| qa_1 | 46.67 | 46.67 | +0.00 |
| qa_2 | 43.33 | 43.33 | +0.00 |
| **All 330** | **84.02** | **85.39** | **+1.38** |

Changed scores: 22; identical generated sequences: 139/330. All paired prefill first tokens match; all evaluated logits finite. Per-request min/max and slot states are reset. Runtime numerical behavior differs from the older Transformers evaluations, so their scores are not used as the paired baseline.

Environment: basis. Commands:

```bash
python -m evaluation.eval_two_stage_ruler --smoke
python -m evaluation.eval_two_stage_ruler --shard 0 --shards 4
# Other workers use --shard 1, 2, 3.
python -m evaluation.eval_two_stage_ruler --summarize
```

All raw per-sample predictions and source hashes are retained in samples/. No refit, production default change, commit, or push.
