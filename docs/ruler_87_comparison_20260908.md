# RULER32K: paired 87-sample comparison

Snapshot: 2026-09-08. Qwen3-8B-Base, C1-V96, reused 11-task pilot with 8 prompts/task. Exclude global sample index 86 from every arm. This is the mean of the 87 saved per-sample scores (partial credit included), not an equal-weight mean of 11 task means. QA2 has 7 samples; each other task has 8.

| Method | Samples | Mean score (%) |
|---|---:|---:|
| Full-K FP16 | 87 | 88.371648 |
| LRQK R32/k1152/recent64 | 87 | 88.429119 |
| Ours Base16/R16 Q32 Page32/B2048 | 87 | 87.145594 |
| Ours plus recent64 | 87 | 87.528736 |
| Full-K BF16 | 87 | 88.888889 |
| ShadowKV plus C1-V96 | 87 | 87.624521 |

## Interpretation and limitations

- FP16 arms use the V100 evaluation path; ShadowKV and its Full-K reference use the BF16 L40S path. Cross-group score differences do not isolate routing alone.
- Ours reserves page 0 (32 tokens) within B2048. The recent64 variant unions the sliding last 64 tokens, including the current token, without subtracting routing budget; duplicates are masked. Maximum support is 2112 tokens/group.
- ShadowKV routes 256 chunks of 8 tokens (2048), plus 48 outlier chunks (384 tokens), 32–39 prompt-local tokens and generated tokens. Only the routed-token budget matches ours; total attention support does not.
- LRQK uses per-query-head k1152 plus recent64. Its recorded last-decode-step GQA union averages approximately 2662 tokens; it is not a hard B2048 budget.
- At this snapshot, ShadowKV has 87/88 saved formal samples. Sample 86 produced first-token EOS and hit a statistics assertion. The repair is documented in docs/shadowkv_v96_protocol.md; the missing record is not silently scored as zero. No final 88-sample ShadowKV score is claimed here.

## Reproduction

Environment: basis (`/home/zhangal/.conda/envs/basis/bin/python`). This comparison reads existing sample JSON only; no model evaluation was launched.

Calculation: load `sample_*.json` under each directory below, align `result.index`, exclude index 86, and compute `100 * sum(result.score) / 87`. Verified matching task, references and prompt-token count at every retained index.

- `results/evaluation/ruler_lrqk_v96_fp16/full/evaluate`
- `results/evaluation/ruler_lrqk_v96_fp16/k1152/evaluate`
- `results/evaluation/ruler_v96_r16_fp16/evaluate`
- `results/evaluation/ruler_v96_r16_recent64_fp16/evaluate`
- `results/evaluation/ruler_shadowkv_v96_bf16/full/evaluate`
- `results/evaluation/ruler_shadowkv_v96_bf16/shadowkv/evaluate`

Original evaluation commands, model path, hardware, hashes and configuration are retained in each sample JSON. Protocols: `docs/lrqk_ruler_v96_protocol.md`, `docs/ruler_v96_r16_protocol.md`, `docs/ruler_v96_recent64_protocol.md`, `docs/shadowkv_v96_protocol.md`.
