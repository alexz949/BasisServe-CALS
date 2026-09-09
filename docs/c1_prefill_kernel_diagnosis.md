# C1 prefill kernel numerical diagnosis

Read-only diagnosis of the existing uniform C1-V80 Triton prefill path used in the earlier LongBench run. No production kernel, model weight or C1 factor is changed. This is not a new accuracy evaluation or a RULER rerun.

Use the first saved prompt from each of six LongBench tasks plus shortest/longest saved prompts, fixed before diagnosis:

| Sample | Task | Length | Length mod32 | Length mod64 |
|---|---|---:|---:|---:|
|0|qasper|5984|0|32|
|32|multifieldqa_en|4948|20|20|
|64|hotpotqa|9856|0|0|
|96|2wikimqa|7647|31|31|
|128|gov_report|13572|4|4|
|160|qmsum|12118|22|22|
|119|2wikimqa|1192|8|40|
|175|qmsum|30431|31|31|

At each of36 layers on the original Triton C1 trajectory, compare exactly the same post-RoPE Q/K and C1-V80 tensors:

1. Actual `compressed_v_prefill_attention` Triton kernel, not a mocked dispatch.
2. Forced PyTorch Flash-SDPA, causal GQA, V feature dimension zero-padded from80 to128 and output sliced to80. Q/K, attention scaling and token support are unchanged.
3. Independent explicit per-head FP32 QK, causal softmax and PV at fixed query positions: sequence starts,32/64/128/256 block boundaries, quarter/mid/three-quarter positions, terminal boundaries and final rows. TF32 disabled.

Report full-tensor Triton/Flash relative L2, sampled-query errors against FP32, per-query errors, and corresponding C1-decoded output errors using the same output weight in FP32. The local observer returns the real Triton output so later layers retain the old trajectory. Every layer's local record is written separately.

Then execute the entire C1 prefill with the Flash reference, using identical factors and input. Compare final logits and top1 with the Triton trajectory and independently executed original dense prefill. Verify original Triton top1 matches the prior saved C1 LongBench result. All passes use fresh caches; no generation. Terminal teacher KL uses the entire vocabulary in FP32.

Reference padding is solely a numerically equivalent attention reference; it does not change the compression factors or add capacity. Eight samples and sampled FP32 query rows cannot rule out every possible input or tiny backend differences; conclusions must be restricted to the measured cases.

Environment: basis, BF16, PyTorch2.6.0+cu124, TF32 disabled, four independent L40S workers,2 CPU cores/48GiB host memory each. Two prompts per worker. Output: `results/evaluation/c1_prefill_kernel`; logs: `logs/prefill-diag-*`. Original experimental artifacts are unchanged.

Smoke job8301020 completed in13 seconds, exit0. Random BF16 GQA inputs of lengths129 and4097, Q/K dim128 and V dim80: Triton/FP32 sampled relative L2 approximately0.001385 and0.001389; Flash/FP32 approximately0.001386 and0.001388. This validates the reference execution path, not real-input correctness.

Real-input probe array8301021_0–3; CPU summary8301025 after successful probes. Temporary submission scripts deleted after submission.

## Completed diagnosis

All four probe workers completed successfully in24,23,20 and34 seconds. CPU summary took13 seconds. All jobs exited0 without retries. The original Triton terminal top1 matched the previous C1 LongBench output on all eight samples. All288 layer/input-pair metrics and all recorded sampled-query comparisons were finite.

| Maximum relative L2 over layer/input pairs | Triton | Flash reference |
|---|---:|---:|
| Sampled latent versus explicit FP32 |0.00195404|0.00185143|
| Sampled C1-decoded output versus explicit FP32 |0.00253238|0.00250694|

Full-tensor Triton versus Flash maximum relative L2:0.00184106. Across6,624 sampled layer/query positions, worst single-row Triton/FP32 relative L2:0.00300417(sample96,layer34,query255); row norms aggregate all32 query heads.

Replacing all36 C1 prefill kernels with Flash changed none of the eight terminal top1 tokens. Three C1/dense top1 disagreements persisted. Mean full-vocabulary terminal KL(dense,Triton-C1):0.502051; KL(dense,Flash-C1):0.496512; KL(Triton-C1,Flash-C1):0.003239. These are final-prompt-position distribution measurements, not task accuracy.

No obvious Triton-specific numerical defect was identified on these real inputs, including the measured non-aligned lengths and block boundaries. Kernel substitution did not eliminate the observed C1/dense terminal distribution difference. This is evidence against a large kernel defect in the checked cases, not proof that all inputs or complete generation sequences behave identically. No production fix or full reference-backend LongBench generation run was performed.

Outputs: `results/evaluation/c1_prefill_kernel/{result.json,summary.md,audit.json}` plus per-sample and per-layer records. Result SHA256:`25cd6a21356b355d6379f0335886dddf6ed3b1001dffa546d638c05f013d348f`. Runtime source hashes and finite metrics were checked independently after summary completion.

## Commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/diagnose_c1_prefill_kernel.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/diagnose_c1_prefill_kernel.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage probe --shard-index "$SLURM_ARRAY_TASK_ID"
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/diagnose_c1_prefill_kernel.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```
