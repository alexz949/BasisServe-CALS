# Llama-3.1-8B Base / ShadowKV K / exact V

Model meta-llama/Llama-3.1-8B, revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b.
Original V projection and output projection are retained. No C1 checkpoint is used.
Both arms BF16, full causal Triton prefill, greedy decoding with official task caps
and original EOS. Full arm uses exact K/V; ShadowKV uses rank160 SVD-reconstructed
historical K and exact V, chunks8,2048 routed tokens,48 outlier chunks, last4 full
prompt chunks plus remainder, and exact generated K/V. This is the official
accuracy algorithm through an HF Llama adapter; it is not the paper's Instruct
model or original optimized offload runtime.

Data: new official RULER generation, revision c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a,
seed42, base template,11 tasks x8 samples. Llama tokenizer; generator length32760
leaves8 tokens for final prompt overhead, validated total prompt+generation <=32768.
Data directory results/datasets/llama31_base_ruler_seed42_margin8.
Earlier no-margin data failed the length gate and is not evaluated.
Primary mean excludes global index86 (87 samples), with full88 also reported.
FWE indices64..71 are evaluated first; task order/global indices remain unchanged.

Environment lowrank, direct execution without Slurm, GPU3(full) and6(shadowkv),
two CPU threads each. Logs results/logs/llama31_shadow, output
results/evaluation/llama31_base_shadow. Commands:

```bash
python -u evaluation/eval_llama31_shadow.py smoke --arm full
python -u evaluation/eval_llama31_shadow.py smoke --arm shadowkv
python -u evaluation/eval_llama31_shadow.py evaluate --arm full
python -u evaluation/eval_llama31_shadow.py evaluate --arm shadowkv
python -u evaluation/eval_llama31_shadow.py summarize
```

CPU tests: Full adapter matches native HF Llama prefill/decode logits and exact KV
cache; three ShadowKV equation/cache tests passed. Smoke repeats FWE64 and NIAH0
with four generated tokens. Final audit covers all176 records, scores, stopping
and shared first generated tokens. Results are accuracy measurements; V and exact
diagnostic K caches remain GPU resident, with no offload-throughput claim.
