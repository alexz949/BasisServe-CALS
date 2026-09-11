# Layer-0 Base-only attention mass recall

The fixed linear-only weights plus newly fitted bias recover substantial attention mass, but remain below the original affine Base. At the longest sampled prefix and a 4096-token budget, recall rises from 59.17% to 77.74%, versus 90.73% for affine and 91.13% for exact-score selection with the same routing rule.

## Protocol

Qwen3-8B-Base, Two-sided average V96 checkpoint; layer 0 has V rank 112, Base rank 16. Use the same 16 diagnostic windows (IDs 64–79), 32768-token windows, and 32 query positions previously selected from training data. No weights or biases are fitted in this evaluation. All three arms are Base only, without residual correction.

For each causal prefix, compute exact dense QK probabilities using BF16 captured Q/K and FP32 scoring. Proxy scores use the same Q and each reconstructed K. Page size is 32. Pin the first page within the total physical token budget per KV group. Rank remaining pages by per-head conditional non-sink page probability, taking the maximum across the four query heads sharing each KV group. Attention mass recall is the sum of exact dense attention probabilities over selected tokens. Non-sink recall separately renormalizes exact probabilities outside the first page.

The exact-score reference uses the same GQA group-max selection rule; it is not a per-head optimal upper bound. Dense attention retaining all tokens has mass 100% by definition.

## Longest sampled prefix

Query position 32639, 32640 visible tokens; equal average over 16 windows × 32 query heads (512 observations per arm and budget).

| Base / selection | 1024 tokens | 2048 tokens | 4096 tokens |
|---|---:|---:|---:|
| Original affine | 80.5065% | 85.7842% | 90.7255% |
| Linear-only | 50.9421% | 54.9466% | 59.1695% |
| Frozen linear-only + fitted bias | 60.6386% | 67.9004% | 77.7393% |
| Exact-score reference | 80.9390% | 86.2327% | 91.1272% |

At budget 4096, non-sink mass recall is respectively 90.7249%, 59.1689%, 77.7388%, and 91.1266%; the sink page contributes little to this layer's result. The fifth percentile across window/head observations is 56.3437% for affine, approximately 0.00000147% for linear-only, and 0.02060% for linear-only plus bias. The mean therefore hides severe misses on some query heads. These observations are correlated within windows and heads; they are not independent statistical trials.

## All 32 sampled queries

Equal average over 16 windows × 32 positions × 32 query heads. Short prefixes make these averages easier than the longest-prefix comparison, and prefixes shorter than a budget retain all tokens.

| Base / selection | 1024 tokens | 2048 tokens | 4096 tokens |
|---|---:|---:|---:|
| Original affine | 86.3537% | 90.4910% | 94.3933% |
| Linear-only | 62.4177% | 67.4327% | 74.0623% |
| Frozen linear-only + fitted bias | 76.1935% | 83.0003% | 89.7808% |
| Exact-score reference | 86.8248% | 90.9513% | 94.7510% |

These are layer-0 routing diagnostics, not full-model recall or downstream task accuracy. No R16 residual was added or refitted.

## Execution and verification

Environment `lowrank`; direct execution on GPU 6, OMP/MKL two threads, exit code 0. Script-reported runtime 20.62 seconds. The only startup warning was the `torch_dtype` deprecation. Source hashes were unchanged, bias-arm left/right factors exactly match linear-only factors, and the linear-only bias is zero. The audit checked all 6144 unique window/query/arm/budget records, 32 heads each: finite bounded recall, physical budget compliance, and nondecreasing recalled mass with increasing budget. Syntax compilation and `git diff --check` passed.

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
python -u -m evaluation.eval_linear_base_mass_recall \
  --linear-bank results/checkpoints/v96kl_linear_b16r16 \
  --bias-result results/evaluation/linear_base_bias \
  --token-budgets 1024 2048 4096 \
  --output results/evaluation/linear_base_bias/mass_recall.json \
  > results/logs/linear_base/mass_recall.log 2>&1
```

- Script: `evaluation/eval_linear_base_mass_recall.py`.
- Full per-head observations, summaries, and provenance: `results/evaluation/linear_base_bias/mass_recall.json`.
- Execution and audit log: `results/logs/linear_base/mass_recall.log`.
