# LRQK-equation routing + C1-V96: integration smoke

Qwen3-8B-Base, basis, BF16, two independent L40S workers. This is a correctness smoke, not an accuracy benchmark.

| Sample | Input tokens | Generated tokens | Full-K C1 prefill logits identical | Peak memory |
|---|---:|---:|---|---:|
|119|1192|3(EOS)|Yes, bitwise|15.317GiB|
|175|30431|8(cap)|Yes, bitwise|23.913GiB|

Config:rank32 per query head, historical Top2048 + recent64,2/2 prefill/decode iterations, tolerance1e-8, lambda weights1, Gaussian initialization seed0+layer. Factors are fitted online from post-RoPE prompt Q/K; no C4 calibration. Decode includes upstream analytic-gradient B updates, not just closed-form/BCD.

All36 layer states and cache shapes verified. On the last long-input decode step,2112 tokens selected per query head; physical GQA union over36x8 groups:mean4324.6354,min2620,max6471. This is not a shared2048-token budget.

Jobs8301137_0–1 completed in27/40 seconds, exit0, no retries or numerical failures. Five CPU tests passed, including pinned-upstream factor-equation parity. No192-prompt LongBench score exists for this arm.

This implementation adapts upstream equations to a resident C1 cache, with aligned previous-active K/AK and an exact recent suffix. It does not reproduce official CPU/ring-cache behavior or establish full upstream accuracy parity. Detailed provenance, scope, code and commands: [integration document](../../../../docs/c1_lrqk_integration.md).
