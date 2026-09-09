# Offline R16: 50 BCD sweeps / PCG150

The user requested increasing BCD40 to50 and the PCG iteration cap100 to150, permitting V100/A100 if L40S is unavailable. All L40S and A100 GPUs were allocated; V100 node gpu-v100-02 had four available GPUs.

## Fixed experiment settings

C1-V96, Base16, offline Page-Fisher R16, same C4 64x32768 fit and16x32768 diagnostic captures, identical Query-Gram32 positions. Relative damping and PCG tolerance remain1e-5. Start from the same deterministic top16 group Page-Fisher Gram eigenvectors with U0=E0; do not warm-start the old40-sweep endpoint. Export the final50-sweep endpoint without diagnostic selection. Fitting uses FP32 on V100; old fitting used FP32 on L40S (TF32 permitted there), so this is not a bitwise cross-device numerical control.

LongBench uses the same192 prompts, six tasks x32, fixed C1-V96 prefill/decode, all36 layers, Page32/B2048 and pinned page0. V100 uses FP16 and the existing explicit memory-efficient prefill implementation. Both old40/100 and new50/150 banks are evaluated on this same FP16 path; old BF16 scores are not treated as a precision-matched comparison. Full-K FP16 reference records are reused and first-token agreement is checked.

## Files and jobs

New bank: results/checkpoints/c1_v96_b16r16_s50p150. Old bank remains unchanged at results/checkpoints/c1_v96_b16r16_qgram.

- 8301364: new fit smoke, complete protocol on layers0/35, reused by formal fit.
- 8301365: old40/100 FP16 LongBench smoke on shortest/longest prompts.
- 8301366_0–3: new50/150 all-layer fitting.
- 8301367: new-bank LongBench smoke.
- 8301368_0–3: new-bank192-prompt evaluation.
- 8301369: new-bank score verification/summary.
- 8301370_0–3: old-bank192-prompt FP16 evaluation.
- 8301371: old-bank score verification/summary.
- Paired summary depends on both per-bank summaries.

GPU jobs request one V100 and two CPUs each; Slurm limits concurrency to available resources. Success dependencies gate later stages. Original files and results are not overwritten. Output directories: results/evaluation/longbench_r16_s40p100_fp16, results/evaluation/longbench_r16_s50p150_fp16, and results/evaluation/r16_solver_budget. Logs at repository root use s50_* and s40_* prefixes.

## Commands

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/fit_c1_v96_r16_s50p150.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --layers 0,35
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_longbench_c1_v96_r16_fp16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_longbench_c1_v96_r16_fp16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke --bank results/checkpoints/c1_v96_b16r16_qgram --output-dir results/evaluation/longbench_r16_s40p100_fp16
/home/zhangal/.conda/envs/basis/bin/python evaluation/summarize_r16_solver_budget.py
```

Formal fitting replaces --layers with --shard-index0/1/2/3. Formal evaluation uses --stage evaluate --shard-index0/1/2/3; per-bank aggregation uses --stage summarize. Environment: basis. No claim of improved accuracy or PCG convergence is made before results.

## Completed result

All36 layers and both192-prompt evaluations completed. Per-bank summary and paired comparison job8301375 completed successfully. Rechecked checkpoint hashes and result/audit hashes. New-run result SHA256:90c98bbf1c03ebf623230456a2ac18d2318c1aca939008522d98040cd33e1f12.

Matched V100 FP16 means:40/100=38.8146757987;50/150=38.8145235175; delta=-0.0001522812 points. Per-sample scores:21 improved,17 regressed,154 tied. Three task means improve, two remain unchanged and qmsum decreases. Detailed scores: results/evaluation/r16_solver_budget/summary.md. New-run first-token agreement192/192;44 generation-cap exits without EOS; peak allocated memory25.454 GiB.

Across36 layers, mean fit Page-Fisher NMSE changes0.075008686→0.074920394 (32 layers improve); mean diagnostic NMSE changes0.131632888→0.131666637 (23 layers improve). Final query-factor maximum relative residual improves in all36 layers; its layer mean changes0.002608481→0.000889817. All36 layers still have at least one final PCG solve reaching150 iterations; the largest final relative residual is0.006529858. This is not evidence of fully converged fits. Diagnostic changes are small and mixed, while generation mean is effectively unchanged.

Both generations use FP16 on V100, but old factors were fitted on L40S and new factors on V100. Do not interpret this as a bitwise same-hardware fitter comparison or compare these means directly with the previous BF16 score37.9563.
