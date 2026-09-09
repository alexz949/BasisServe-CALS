# Offline R16: Query-Gram Q64

## Active protocol: user changed solver budget to40/100

The user explicitly requested40 BCD sweeps and PCG100 for the real Q64 experiment. This supersedes the50/150 settings in the historical preparation below. Jobs8301407–8301412 were cancelled before formal fitting started; pre-existing partial50/150 artifacts are preserved but not consumed. Completed selection8301405 and extraction8301406 are reused.

Active jobs:8301414 fitting smoke;8301415_0–3 full fit;8301416 LongBench smoke;8301417_0–3 evaluation;8301418 score verification;8301419 paired comparison. All GPU jobs use V100. Fit/diagnostic remain64/16 C4 windows; fit Q64, diagnostic Q32; initialization rule and tolerances unchanged.

New bank:results/checkpoints/c1_v96_b16r16_q64_s40p100. Evaluation:results/evaluation/longbench_r16_q64_s40p100_fp16. Reference:results/evaluation/longbench_r16_s40p100_fp16, mean38.8146757987. Generation is matched V100 FP16; Q32 factors were originally fitted on L40S whereas Q64 factors are fitted on V100, both in FP32. Paired summary remains results/evaluation/r16_query_gram64. Exact commands below retain the same script names but now execute40/100; logs use q64_40_*.

## Completed40/100 results

All36 layers, smoke,192 predictions and both summary jobs completed successfully. Evaluation shard3 was moved while pending from gpu-v100-02 to an available V100 on gpu-v100-03; its command and protocol were unchanged. Final result/audit SHA256:88a6d661fe990bc8a99f8e924affebdc66e7a95984e7bd1d9ffa40f15a9cbf92. Rechecked36 checkpoint hashes and actual Q64/diagnosticQ32/40/100 settings. Prediction audit verifies192 samples, official scores, first tokens, cache/dispatch and shard coverage.

Matched FP16 LongBench mean:Q32=38.8146757987;Q64=38.6002985640;delta=-0.2143772347 points. Per-sample score changes:23 improvements,25 regressions,144 ties. gov_report improves29.6477→30.9495; qmsum decreases28.3561→26.3186. Detailed table:results/evaluation/r16_query_gram64/summary.md.

On the unchanged diagnostic Q32 captures, all36 layers improve Page-Fisher NMSE. The layer-average decreases0.13163288755→0.12348013212. Despite that surrogate improvement, the192-prompt LongBench mean decreases. This does not by itself establish overfitting or statistical significance. Every layer still has a final PCG query subsolve reaching100 iterations; maximum recorded final relative residual is0.03972608224. Do not claim full solver convergence.

## Historical preparation (superseded solver budget)

## Matched comparison

Extend fit queries from32 to64 per window, with16 deterministic pivots per8K bin over32K contexts. Reuse the original fit-only whitening and position Grams from results/evaluation/qgram32/statistics.safetensors. Verify that the first8 pivots of each bin reproduce the old Q32 and that repeated16-pivot selection is deterministic. No new candidate or model capture is needed.

All64 C4 fit windows are extracted from the existing512-position BF16 candidate Q captures. Keep the16 diagnostic windows at the original Q32 positions/captures, making diagnostic losses comparable. No answers, diagnostic observations or LongBench queries enter selection. This replaces the initially considered plan to recapture diagnostic Q64; no A100 capture job was submitted.

Frozen C1-V96, closed-form Base16, offline Page-Fisher R16,50 BCD sweeps,PCG150, damping/tolerance1e-5 remain unchanged. The initialization rule remains top16 eigenvectors of the summed fit Page-Fisher Gram, U0=E0; numerical initial factors may change when the fit statistics change. Fit input count per head increases from2048 to4096 query-window observations; diagnostic remains512. All32768 K/V rows per window are retained.

GPU computation uses V100: FP32 fitting and FP16 LongBench generation. Inputs remain the existing BF16 captures. Same192 prompts, six tasks x32, C1-V96 prefill/decode, all36 layers, Page32/B2048, pinned page0. Compare against completed Q32/50/150 FP16 mean38.8145235175. This is an accuracy oracle, not a CPU-offload speed benchmark.

## Pipeline

- 8301405: CPU Gram extension; completed,36-layer old-subset and deterministic-pivot checks passed.
- 8301406: CPU fit-Q extraction; completed,64 windows, no model forward.
- 8301407: V100 full-protocol fit smoke on layers0/35.
- 8301408_0–3: V100 all-layer fit, smoke outputs reused.
- 8301409: V100 shortest/longest LongBench smoke.
- 8301410_0–3: V100192-prompt evaluation.
- 8301411: scoring/audit summary.
- Paired Q32/Q64 comparison depends on8301411.

All downstream stages require successful dependencies. Two CPUs per GPU,64 GiB host RAM for the larger fit statistics. New artifacts are separate from Q32: results/evaluation/qgram64, results/calibration/qgram64_fit, results/checkpoints/c1_v96_b16r16_q64_s50p150, results/evaluation/longbench_r16_q64_s50p150_fp16, results/evaluation/r16_query_gram64. Root logs use q64_* prefixes.

## Commands (basis environment)

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/extend_query_gram64.py
/home/zhangal/.conda/envs/basis/bin/python evaluation/extract_qgram64_fit.py
/home/zhangal/.conda/envs/basis/bin/python evaluation/fit_c1_v96_r16_qgram64.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --layers 0,35
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_longbench_c1_v96_q64_fp16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
/home/zhangal/.conda/envs/basis/bin/python evaluation/summarize_r16_query_gram64.py
```

Formal fitting replaces --layers with --shard-index0/1/2/3. Formal evaluation uses --stage evaluate --shard-index0/1/2/3; aggregation uses --stage summarize. No results are asserted before completion.
