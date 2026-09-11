# Local HF KL96 fixed recent-budget comparison

New dataset; this does not reproduce the previous remote-server 87 prompts.
NVIDIA RULER revision c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a,
seed42, base template, total length32768, eight samples for each task in order:
niah_single_1, niah_single_2, niah_single_3, niah_multikey_1,
niah_multikey_2, niah_multiquery, niah_multivalue, vt, fwe, qa_1, qa_2.
Generate and evaluate all88; primary sample-weighted mean excludes global index86
for the user-requested87 scope. Also report the full88 mean, including first-token EOS.

Model: Qwen3-8B-Base revision49e3418fbbbca6ecbdf9608b4d22e5a407081db4.
HF payload: alexz949/BasisServe-CALS, Q3-8B-C1-R96, two-sided terminal KL,
average V rank96 with variable layer ranks. All arms BF16 with the same causal
C1 Triton prefill, greedy generation, official task-specific caps and model EOS.

Reuse results/checkpoints/v96kl_b16r16, fitted for this exact HF payload:
Base16 affine pre-RoPE MSE, Page-Fisher R16, Q32,64x32768 fit and16x32768
diagnostic C4 windows,40 sweeps/PCG100. No new fitting or validation selection.

Arms:
- full: full exact K and C1 payload.
- recent_extra: Page32/B2048 including pinned sink32, then union exact recent64;
  at most2112 unique physical tokens per KV group.
- recent_fixed: sink32 plus exact sliding recent64, plus61 complete historical
  pages disjoint from both; at most2048 tokens. Short contexts use all tokens.
- lrqk: R32, per-query k1152 plus recent64, FP32 state,2/2 iterations, seed0.
  Physical GQA union is measured, not hard-capped2048.
- shadowkv: SVD160, chunks8,256 routed chunks plus48 outlier chunks,
 32-39 prompt-local tokens plus generated tokens; resident C1 values.
  Adaptation of official revision e51904cdeab7d4d34013370f09f2cf5fcd655e15.

Environment lowrank, direct execution without Slurm. Smoke GPU workers3/6;
formal mapping: full=GPU2, recent_extra=GPU4, recent_fixed=GPU6,
lrqk=GPU0, shadowkv=GPU3. One worker runs all88 samples per arm.
OMP_NUM_THREADS=2, MKL_NUM_THREADS=2, TOKENIZERS_PARALLELISM=false.
Data: results/datasets/ruler_kl96_seed42.
Results: results/evaluation/ruler_kl96_seed42.
Logs: results/logs/ruler_kl96.

Commands after activating lowrank:

```bash
python -u evaluation/eval_ruler_kl96.py smoke --arm ARM
python -u evaluation/eval_ruler_kl96.py evaluate --arm ARM --num-shards 1
python -u evaluation/eval_ruler_kl96.py summarize
```

ARM is full/recent_extra/recent_fixed/lrqk/shadowkv.
Smoke repeats two prompts at four generated tokens. Final audit checks every
sample, data/protocol hashes, official scores, EOS/caps and cross-arm first tokens.
These are resident accuracy measurements, not CPU-offload throughput benchmarks.
