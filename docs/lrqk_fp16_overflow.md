# LRQK FP16 overflow diagnosis and FP32 routing state

This document describes our historical FP16 overflow workaround, not a
requirement to keep BF16 experiments' routing state in FP32. Upstream LRQK
at revision `caf16293db2e4423a84ab2e895bacf64479f1eb7` performs prefill/decode
factor solves in FP32 and casts the returned factors back to the input K
dtype (`cast_lrqk_prefill` / `cast_lrqk_decode`,
[official source](https://github.com/tenghuilee/LRQK/blob/caf16293db2e4423a84ab2e895bacf64479f1eb7/lrqk_attention.py#L784-L840)).
Current Qwen3-32B and Nemotron K-routing evaluations use BF16 persistent
LRQK state with FP32 solves. Existing FP32-state results retain their original
precision provenance and must not be silently relabeled or mixed with BF16 runs.

The original FP16 V100 run completed the full-K control but encountered non-finite LRQK states on long generations. This diagnosis preserved the original arithmetic and observed the FP32 outputs immediately before the FP16 cast.

| k1152 sample | Layer (zero-based) | Decode update | Maximum absolute new K-code | Nonfinite FP32 | Nonfinite after FP16 |
|---|---:|---:|---:|---:|---:|
|137|32|498|77817.1015625|0|6|
|168|32|492|73926.4609375|0|3|
|170|32|444|89420.2421875|0|2|

In all three cases BQ, BK and Q-code remained finite even after conversion. New K-code exceeded FP16's largest finite value,65504. This establishes cast overflow for these reproductions; it does not establish why the online factor scaling grows or guarantee every future FP32 solve is stable.

Diagnostic array8301183 stopped at the original assertion after writing `failure.json` for each sample under `results/evaluation/lrqk_fp16_diagnosis`. These expected assertion exits are not failed diagnostic measurements.

## Precision change

`evaluation/eval_longbench_lrqk_fp32route.py` keeps BQ, BK, accumulated AK, new Q/K codes and routing score scans in FP32 from the initial prompt onward. Model weights, exact K and C1-V96 payload caches remain FP16. The online update equations, seed, rank32, Top-k/recent policy and exact selected attention are unchanged. No clipping, resets or skipped layers are introduced. Existing production LRQK files and L40S BF16 jobs are unchanged.

This changes numerical routing trajectories and doubles routing-state scalar storage versus FP16. Both192-prompt sparse arms are rerun independently; successful samples from the old FP16-state run are not mixed in. The completed FP16 full-K control is reused with per-record hashes and matching-protocol checks.

Two CPU tests passed: FP16 output with FP32 routing state, and bitwise agreement with the original all-FP32 update path on a small deterministic case.

## Regression checks

Array8301187 completed all five original-cap checks:

- k1152 sample137:512 generated tokens, past the original failure at update498.
- k1152 sample168:200 tokens, normal EOS. Its changed trajectory did not reach the old failure step.
- k1152 sample170:512 generated tokens.
- k1280 sample170:512 generated tokens.
- k1280 sample175:30431 input tokens and512 generated tokens.

All returned finite logits and valid FP16 K/C1-V96 caches. No all-dataset accuracy or unconditional stability claim follows from these checks.

Formal array8301192 reruns k1152 and k1280 across eight single-GPU shards, at most four concurrent V100 workers. Summary8301193 depends on successful completion. Outputs: `results/evaluation/longbench_lrqk_fp32route`. Logs: `logs/lrqk-fp32-route-8301192_*.log`. At this snapshot four k1152 shards were running; no formal sparse scores were available.
