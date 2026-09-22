# Qwen3.5-9B Mixed Calibration V128 + Wo Results

## Checkpoint

- Model: `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- V bank SHA-256: `3b63aa275786358ffafc1a6c02c61950cfda03b3518f44ce28ed23b410558fa8`
- V factor SHA-256: `8248969c51254cae1f6784ed0b2efb0d67b98083fda0075e56042622ebfda55d`
- Wo bank SHA-256: `e26843b9f5829d39fee673ea09f0929258daeba4bc62c1ebf9dbdac072e48945`
- V schedule: `80, 96, 96, 96, 192, 160, 192, 112` (average rank 128)
- Rank allocation: frozen from the C4 two-sided V128 run
- Factor fit: `128 C4 + 64 Math + 64 Code`, each 2048 tokens
- Heldout: `64 C4` windows
- Wo ranks: GDN 768 per source; full-attention 512 per source

## Protocol

- Non-thinking, greedy generation
- Seed: `20260909`
- Eight independent TP1 vLLM workers on eight L40S GPUs
- Each task split by even/odd original lm-eval `doc_id`
- vLLM continuous batching: max 32 sequences, 4096 batched tokens
- Model length: 8192
- Environment: `basis`
- Versions: torch 2.13.0, vLLM 0.29.0, Transformers 5.17.0, lm-eval 0.4.11,
  EvalPlus 0.3.1, math-verify 0.9.0, ANTLR runtime 4.11.0
- Command: `bash /workspace/runs/qwen35-mixed-eval/run_8gpu.sh`

## Results

| Task | Questions | Metrics | Length-capped |
|---|---:|---|---:|
| GSM8K | 1319 | strict exact match 89.23%; flexible exact match 90.60% | 58 |
| MATH500 | 500 | math-verify 83.00%; exact match 8.80% | 93 |
| MBPP+ | 378 | base pass@1 83.33%; plus pass@1 69.84% | 73 |
| IFEval | 541 | prompt strict 77.82%; instruction strict 84.53%; prompt loose 82.44%; instruction loose 87.77% | 49 |

Exact counts:

- GSM8K: strict `1177/1319`, flexible `1195/1319`
- MATH500: math-verify `415/500`, exact match `44/500`
- MBPP+: base `315/378`, plus `264/378`
- IFEval: prompt strict `421/541`, instruction strict `705/834`, prompt loose
  `446/541`, instruction loose `732/834`

The slowest shard took 558.3 seconds, including model initialization, CUDA Graph setup,
generation, and task scoring.

## Comparison with Reasoning-Only Calibration

Both checkpoints use the same frozen V rank schedule and Wo ranks, so this isolates the
factor calibration-domain change.

| Metric | Mixed | Reasoning-only | Mixed delta |
|---|---:|---:|---:|
| GSM8K strict | 89.23% | 88.93% | +0.30 pp |
| GSM8K flexible | 90.60% | 90.37% | +0.23 pp |
| MATH500 math-verify | 83.00% | 85.60% | -2.60 pp |
| MBPP+ base pass@1 | 83.33% | 85.98% | -2.65 pp |
| MBPP+ plus pass@1 | 69.84% | 72.49% | -2.65 pp |
| IFEval prompt strict | 77.82% | 72.64% | +5.18 pp |
| IFEval instruction strict | 84.53% | 81.29% | +3.24 pp |
| IFEval prompt loose | 82.44% | 77.45% | +4.99 pp |
| IFEval instruction loose | 87.77% | 84.65% | +3.12 pp |

Mixed calibration substantially improves instruction following, is essentially neutral to
slightly positive on GSM8K, and gives back about 2.6 points on MATH500 and MBPP+ relative
to reasoning-only calibration.

## Audit

- All 2738 task documents are present exactly once across the eight generation shards.
- GSM8K's two answer filters are both present for all 1319 documents.
- All model, V-bank, Wo-bank, tokenizer, and chat-template hashes match across shards.
- The eight raw output SHA-256 values are recorded in
  `mixed_v128_wo768_512_summary.json`.
- Summary status: `complete_and_audited`.
