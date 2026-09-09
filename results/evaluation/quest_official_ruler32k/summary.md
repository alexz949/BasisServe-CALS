# Official QUEST accuracy forward: C1-V80 RULER 32K

Qwen3-8B-Base, same 88 reused prompts, shared full C1 prefill/first token, greedy generation, L40S BF16.
Both sparse arms retain full attention in layers 0 and 1. No factors are refitted.
QUEST directly calls unmodified mit-han-lab/Quest evaluation/quest_attention.py at commit 01c1623bf9395009520874e989e29f683203b357.
Page32, 2048 tokens per Q head, no pinned page. Our Q8 Base16/Q16 Fisher R8 selects 2048 physical tokens per KV group including page0.
The model bridge preserves Qwen3 norm/RoPE and C1-V80; it is not original-model paper accuracy or an optimized offload/kernel benchmark.

| Task | Full exact K | Base16+R8, full2 | Official QUEST, full2 |
|---|---:|---:|---:|
| niah_single_1 | 100.0000% | 100.0000% | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 75.0000% |
| niah_single_3 | 100.0000% | 100.0000% | 12.5000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 37.5000% |
| niah_multikey_2 | 87.5000% | 50.0000% | 25.0000% |
| niah_multiquery | 96.8750% | 100.0000% | 37.5000% |
| niah_multivalue | 93.7500% | 93.7500% | 28.1250% |
| vt | 92.5000% | 90.0000% | 92.5000% |
| fwe | 91.6667% | 79.1667% | 75.0000% |
| qa_1 | 50.0000% | 50.0000% | 37.5000% |
| qa_2 | 37.5000% | 37.5000% | 37.5000% |
| Task-balanced mean | 85.2083% | 80.7197% | 50.7386% |

Mean unique physical tokens / sparse-layer KV group (weighted by generated decode steps):
- uniform_r8_full2: 2033.744
- quest_official_full2: 3579.526

QUEST's per-head budget is not an equal physical GQA budget. Neither method's count includes the two full layers.
