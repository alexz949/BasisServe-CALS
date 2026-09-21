# Qwen3.5-9B Reasoning-Only Calibration V128 + Wo Results

- V schedule: `80, 96, 96, 96, 192, 160, 192, 112` (average rank 128).
- Rank allocation is frozen from the C4 Two-Sided run; only V/Wo fitting uses reasoning calibration.
- Fit calibration: `128 Math + 128 Code`, each 2048 tokens.
- Wo ranks: GDN 768 per source; full-attention 512 per source.
- Evaluation: non-thinking, greedy, matched Qwen3.5 vLLM protocol.

| Task | Questions | Metrics | Length-capped |
|---|---:|---|---:|
| gsm8k | 1319 | exact_match,strict-match=88.93%; exact_match,flexible-extract=90.37% | 58 |
| minerva_math500 | 500 | exact_match,none=0.00%; math_verify,none=85.60% | 92 |
| ifeval | 541 | prompt_level_strict_acc,none=72.64%; inst_level_strict_acc,none=81.29%; prompt_level_loose_acc,none=77.45%; inst_level_loose_acc,none=84.65% | 61 |
| mbpp_plus_full | 378 | base_pass_at_1,none=85.98%; plus_pass_at_1,none=72.49% | 75 |

The frozen rank schedule makes this a calibration-domain ablation, not a reasoning-domain allocator result.
