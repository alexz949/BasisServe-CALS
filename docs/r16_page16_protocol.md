# Page16 vs Page32 at B2048

## Fixed experiment

Qwen3-8B-Base, C1-V96 payload, Base16, offline Page-Fisher residual R16. Same C4 captures:64x32768 fit windows and16x32768 diagnostic windows; original Query-Gram Q32 positions for both.40 BCD sweeps, PCG maximum100, relative damping/tolerance1e-5. Same spectral initialization rule, top16 group-summed fit Fisher Gram eigenvectors with U0=E0. Residual factors are refitted for Page16; do not reuse Page32 factors as the Page16 result. Base uses unchanged deterministic closed-form MSE-RRR fitting; payload checkpoint is unchanged.

Page size16, B2048 tokens per GQA group, at most128 selected pages. Pin2 pages so the first32 tokens stay fixed, matching the Page32 baseline's1 pinned page. Page-Fisher excludes those same32 prefix tokens before normalizing the non-sink distribution. Thus126 routed pages replace63 routed Page32 pages, with the same total token budget.

Evaluation: same192 LongBench prompts, six tasks x32, C1-V96 full-causal prefill and sparse decode at all36 layers, native group-max normalized page-mass selection. No adaptive budget or forced current page. V100 FP16 generation, FP32 fitting. Compare to completed Page32/Q32/R16/40/100 FP16 mean38.8146757987. Baseline factors were fitted on L40S; new factors on V100, so fitting is not a bitwise same-device control. This is an accuracy oracle with GPU-resident exact K and materialized Base128+R16 metadata, not a CPU-offload latency benchmark or new optimized kernel.

## Tests and pipeline

Two CPU tests passed: Page16 physical budget/pinned-prefix/uniqueness across short, boundary and ragged lengths; Page-Fisher Gram and energy against direct calculation, including invariance to changes in the excluded32-token prefix. GPU smoke additionally checks actual dispatched page16/B2048/pin2 arguments, selected page count, uniqueness and prefix IDs, deterministic generation and full-prefill logits.

- 8301578: full-protocol fitting smoke, layers0/35; results reused.
- 8301579_0–3: all36-layer fitting.
- 8301580: shortest/longest LongBench smoke.
- 8301581_0–3:192-prompt evaluation.
- 8301582: scoring and artifact audit.
- 8301583: matched Page32/Page16 comparison.

Jobs use basis, one V100 and two CPUs each; no fixed node restriction, allowing available V100 nodes to run shards. Later stages are gated by successful dependencies. Bank:results/checkpoints/c1_v96_b16r16_p16_q32. Evaluation:results/evaluation/longbench_r16_p16_q32_fp16. Comparison:results/evaluation/r16_page16. Logs:p16_* at repository root. Existing Page32 artifacts remain unchanged.

## Commands

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/fit_c1_v96_r16_page16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --layers 0,35
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_longbench_c1_v96_page16_fp16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
/home/zhangal/.conda/envs/basis/bin/python evaluation/summarize_r16_page16.py
```

Formal fitting replaces --layers with --shard-index0/1/2/3. Formal evaluation uses --stage evaluate --shard-index0/1/2/3; per-bank summary uses --stage summarize. No accuracy claim before completion.
