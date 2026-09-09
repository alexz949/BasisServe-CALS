# C1-V96 prefill with matched Base16/R8 sparse decode

## Scope

Add exactly one sparse-decode arm to the completed uniform C1-V96 full-K LongBench pilot. The reference six-task mean is38.6171. Both arms use identical C1-V96 Triton full causal prefill, including the first generated token. The new arm changes only decoding to matched Base16/R8 routing and exact-K/C1-V96 attention on selected pages. No original-dense-prefill V96 arm is included.

Model: Qwen3-8B-Base. Frozen payload checkpoint:`results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6`, rank96 on all36 layers and8 KV groups. No C1 factor refitting, rank allocation, or model-weight training. The prior full-K baseline result and runtime inputs are hash checked.

## Matched router fitting

The old V80 Base consumes80 coordinates and cannot be directly applied to96-dimensional C1 latents. Both Base and its conditional residual are newly fitted in the frozen V96 coordinates. Numerical fitting routines are reused unchanged from the allocated-C1 experiment; the new wrapper supplies uniform-V96 artifact paths and records the correct checkpoint provenance.

Use the same dense-model C4 captures:64x32768 fit windows(indices0–63),16x32768 diagnostic windows(indices64–79). This is distinct from the C1 payload's32 fit/4 diagnostic windows. Base uses all fit token rows and closed-form affine reduced-rank regression to pre-RoPE K, rank16, reconstruction-MSE objective. No Adam or Q sampling for Base.

Residual R8 uses the same existing fit-only Query-Gram selection manifest and Q captures:32 Q positions per layer/window,8 in each8K stratum. The post-RoPE exact-K residual is recomputed against the new Base. Statistics use causal non-sink Page-Fisher, Page32, excluding page0. Fit observations/head:2048; diagnostic observations/head:512. The window-major builder reuses token features across Q. No benchmark prompts or labels enter fitting.

Residual fitting:40 BCD sweeps; PCG damping1e-5,tolerance1e-5,maximum100 iterations, matching the previous router. Fixed final endpoint and uniform R8; diagnostic loss does not select factors or ranks. FP32 factors are stored and cast to BF16 during deployment. Fit permits TF32, as in the previous fitting implementation; evaluation disables TF32.

Output bank:`results/checkpoints/c1_v96_b16r8_qgram`. Per-layer Base-left shape(8,96,16); Base-right(8,16,128); bias(8,128); residual encoder(8,128,8); query factor(32,128,8). All factors must be finite and match expected dimensions.

## Evaluation

Same frozen192 prompts:6 tasks x32(qasper,multifieldqa_en,hotpotqa,2wikimqa,gov_report,qmsum). Saved input token IDs, official completion prompts without chat template, greedy decoding, identical EOS and caps128/64/32/32/512/512. Actual inputs1192–30431 tokens, total input+reserved-output cap32768. This is a reused six-task pilot, not full LongBench or a new independent test set. QA uses official F1 and summaries official ROUGE-L, best reference; headline score is six-task arithmetic mean on0–100 scale.

Every prompt starts with a fresh full-attention C1-V96 prefill. Its first token must match the saved V96 full-K baseline. Then create an immutable-prefix fork and attach the matched router. Decode uses all36 layers, Page32/B2048(64 pages including pinned page0), no adaptive budget or forced current page. Per-head normalized non-sink page scores are combined by the existing GQA maximum rule. Selected pages supply uncompressed exact K and resident C1-V96 values.

This is a GPU-resident accuracy oracle, not CPU offload or a performance benchmark. The current reference path materializes a post-RoPE Base128+R8 sidecar, width136. Exact K means uncompressed keys from this trajectory, not keys copied from a separately executed dense model.

Smoke uses shortest and longest prompts with four-token caps, repeated complete executions and bitwise-logit comparisons. It independently runs full-K V96 and checks the initial logits exactly. All executions count36 real Triton-prefill calls; each subsequent token must invoke native sparse attention on36 layers. K128/V96/sidecar136 BF16 cache dimensions and immutable-prefix signatures are checked. Formal evaluation checks every first token against saved full-K V96. CPU summary verifies192 records, shard coverage, IDs/text, EOS/caps, cache/dispatch checks and official scores.

## Jobs and environment

Environment:`/home/zhangal/.conda/envs/basis/bin/python`. L40S on lovelace; other A100/H200 GPUs were allocated. Each GPU worker requests2 CPUs and48GiB host RAM. CPU summary uses2 CPUs and8GiB. Temporary submission scripts are deleted after submission. Logs:`logs/v96-router-*`. Existing checkpoints, results, and production code are unchanged.

| Stage | Job | Status at submission |
|---|---|---|
| Full-protocol fit smoke,layers0/35 |8301037_0–1|Completed,exit0|
| Fit all36 layers,reuse0/35 |8301039_0–3|Running on4 L40S|
| Sparse decode smoke |8301040|After successful fit|
| LongBench192 evaluation |8301041_0–3|After successful smoke|
| CPU summary/audit |8301042|After successful evaluation|

All downstream stages require successful predecessors; invalid dependencies cancel downstream jobs. No partial-result scoring or benchmark-driven tuning is scheduled.

Layer0 fitting took88.2 seconds:fit/diagnostic residual Page-Fisher NMSE0.292923/0.557809. Layer35 took101.6 seconds:0.220877/0.327183. Both used the full64/16-window Q32 protocol. These are residual fitting diagnostics, not LongBench scores. Some PCG query solves hit the100-iteration cap: final maximum relative residuals0.001165(layer0),0.005217(layer35), above the requested1e-5 tolerance. Factors and losses are finite; numerical convergence is not claimed. The existing solver budget is retained unchanged for the comparison. The RoPE constructor's deprecation warning did not prevent completion.

Output:`results/evaluation/longbench_c1_v96_b16r8`, including per-sample JSON, final result, audit and English summary. No sparse LongBench score exists at submission time.

## Commands

Working directory:`/deac/csc/yangGrp/zhangal/BasisServe-CALS`.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_c1_v96_base_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --layers 0
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_c1_v96_base_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --layers 35
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_c1_v96_base_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --shard-index "$SLURM_ARRAY_TASK_ID"
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_v96_router.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_v96_router.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --shard-index "$SLURM_ARRAY_TASK_ID"
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_longbench_c1_v96_router.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```
