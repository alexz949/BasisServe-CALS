# Llama-3.1-8B Base ShadowKV with HF KL96 values

Paired extension of docs/llama31_base_shadow_protocol.md. Same model revision,
BF16, frozen88 prompts, seed42,32K total cap, greedy and official generation caps.
Primary sample mean excludes index86; full88 also reported. FWE64..71 run first.

HF alexz949/BasisServe-CALS revision35fb02cb6361bc1caa01aba7806fa14c3bf4148b,
ICLR-results/llama31-8b/checkpoints/L31-8B-C1-R96. Two-sided terminal KL allocation,
average V96 with per-layer ranks. Verified model config/index identity, manifest,
result and all32 factor hashes. Each layer fuses its value-coordinate encoder
into the original V projection and uses its corresponding head-output decoder.
Q/K projections and original RoPE remain unchanged. Both arms use the compressed
V payload during full causal Triton prefill and decode.

Full-K: exact K with C1 latent values. ShadowKV: rank160 reconstructed historical K,
exact local/outlier/generated K, chunk8,2048 routed tokens,48 outlier chunks,
original local policy; same resident C1 latent values. No new fitting.
This remains an accuracy adaptation, not an official Instruct reproduction or
CPU-offload throughput benchmark. Original exact-V results are preserved.

Environment lowrank; GPU3(full) and6(shadowkv), two CPU threads each, no Slurm.
Output results/evaluation/llama31_base_shadow_v96; logs results/logs/llama31_shadow_v96.

```bash
python -u evaluation/eval_llama31_shadow_v96.py smoke --arm full
python -u evaluation/eval_llama31_shadow_v96.py smoke --arm shadowkv
python -u evaluation/eval_llama31_shadow_v96.py evaluate --arm full
python -u evaluation/eval_llama31_shadow_v96.py evaluate --arm shadowkv
python -u evaluation/eval_llama31_shadow_v96.py summarize
```

Smoke repeats two prompts; cross-arm first tokens must agree. Final audit rechecks
all176 records, official scores, EOS/caps, protocols and first-token agreement.
