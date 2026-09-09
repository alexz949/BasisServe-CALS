# V100 FP16 LRQK LongBench pilot

This is a separate precision/backend experiment. The existing L40S BF16 artifacts and queued jobs are unchanged.

## Protocol

- Qwen3-8B-Base; basis environment; FP16 model, exact K cache and frozen uniform C1-V96 cache.
- Same frozen 192 LongBench prompts: six tasks, 32 examples each, unchanged greedy generation, EOS and task caps.
- Three arms: full exact-K; LRQK R32 historical Top1152 + recent64 per query head; LRQK R32 historical Top1280 + recent64 per query head.
- Full causal C1-V96 prefill in every arm, not dense-V prefill. Explicit CUTLASS memory-efficient SDPA with repeated GQA K/V; no quadratic math-backend fallback for prefill.
- Full-K decode uses SDPA. LRQK uses the existing online FP32 factor equations, FP16 stored factors and selected exact-K/C1-V attention. Rank32, seed0+layer, two prefill/decode iterations, recent64. Adapted resident cache, not upstream CPU/ring-cache reproduction.
- Physical token union is recorded at each prompt's final decode step across all36 layers and8 KV groups. It is not an all-step average or a hard shared B2048 cap.
- Four independent single-GPU workers on gpu-v100-02; two CPUs and48GiB host memory per worker. No TP.

## Compatibility checks

The original Triton prefill aborted during compilation on V100 with `LLVM ERROR: Failed to compute parent layout for slice layout.` The V100 evaluator therefore dispatches prefill to explicit memory-efficient SDPA in its own process; production Triton source remains unchanged.

An initial smoke also exposed a missing `num_shards` input argument, corrected before GPU testing. Failed jobs8301156 and8301160 produced no accuracy results. No environment packages were changed.

Successful smoke jobs8301164 and8301168 tested all three arms on samples119/175 (1192/30431 input tokens), generating up to8 tokens. All six cases passed finite-logit and FP16 cache-shape checks. LRQK prefill logits matched the same-process FP16 full-K reference bitwise.

The small random causal GQA prefill check against explicit FP32 attention had relative L2 error0.00023546595. Maximum observed smoke GPU allocation was23.9104GiB. Long-prompt measured generation sections were16.65s full-K,18.61s k1152 and18.67s k1280; these are smoke measurements, not performance benchmark results.

## Submission

Formal array8301170 contains12 single-GPU shards, at most4 concurrent, gated by both successful smoke arrays. CPU summary8301175 depends on formal completion. At submission, four full-K shards started; no formal accuracy was available yet.

Evaluator: `evaluation/eval_longbench_lrqk_fp16.py`. Outputs: `results/evaluation/longbench_lrqk_fp16`. Commands, source fingerprints, input/checkpoint provenance and device information are recorded in individual sample JSON files. Formal logs: `logs/lrqk-v100-8301170_*.log`.
