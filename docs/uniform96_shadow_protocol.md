# Uniform V96 ShadowKV on frozen local RULER prompts

HF revision: `f1a6253b5d5c747a2475cbf9e704a67d97930b31`, repository
`alexz949/BasisServe-CALS`. Payloads: `Q3-8B-C1U-R96` and `L31-8B-C1U-R96`,
under `results/hf/ICLR-results/{qwen3,llama31}-8b/checkpoints/`.
Their manifests reference the corresponding `c1/factor-banks/R96-S6` files.
All physical KV heads in every layer have rank96. Factors are BF16, six encoder
sweeps followed by decoder refitting. Manifest, factor and model config/index
hashes are checked before evaluation.

User requested ShadowKV only. There is no new uniform Full-K reference in this run.
Compare absolute ShadowKV scores with the completed local KL96 ShadowKV arms;
this comparison cannot measure the routing/reconstruction penalty relative to
uniform Full-K. Identity with the older remote L40S ALS checkpoint remains unverified.

Reuse the exact input token IDs, answers, task names, caps and sample indices from:

- Qwen3: `results/evaluation/ruler_kl96_seed42/shadowkv/evaluate`
- Llama3.1: `results/evaluation/llama31_base_shadow_v96/shadowkv/evaluate`

Both are Base models, seed42 RULER, 11 tasks x8 prompts, context limit32768.
The Llama dataset used generator margin8. Evaluate all88; report both the
sample-weighted87 mean excluding global index86 and the full88 mean.
Prompts match within each model; prompts across models are not paired.

BF16 inference, TF32 disabled, seed0, greedy decoding, official saved task caps
and model EOS. Common full causal C1 Triton prefill. ShadowKV SVD rank160,
chunk8, routed2048, 48 exact outlier chunks per KV head, last4 full prompt
chunks plus remainder and all generated tokens exact. Routed K is reconstructed;
C1 V remains resident. This is an accuracy evaluation, not an offload benchmark.
Qwen preserves Q/K RMSNorm and uses the existing ShadowKV adapter; Llama uses
the same forward and generation functions as its completed KL96 evaluation.

Environment `lowrank`; direct shell execution, no Slurm. GPU3 for Qwen3 and GPU6
for Llama3.1, two CPU threads per worker. Commands:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 python -u evaluation/eval_uniform96_shadow.py --model-family qwen3
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=6 python -u evaluation/eval_uniform96_shadow.py --model-family llama31
```

Logs: `results/logs/uniform96_shadow/{qwen3,llama31}.log` and `download.log`.
Outputs: `results/evaluation/{qwen3,llama31}_uniform96_shadow/`.
Each worker first repeats sample64 and sample0 with a four-token cap, requiring
identical logits and token IDs. Then it evaluates all88, starting with FWE.
The final audit verifies immutable protocol/sample identities, scoring, decoding,
EOS and caps. Source and reference hashes and exact commands are saved per sample.
