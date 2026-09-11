# Llama-3.1-8B Base: exact V versus HF two-sided KL V96

Same88 frozen prompts and BF16; primary87 excludes index86. Both176-record audits passed; all352 sample identities match.

| Configuration | Mean87 | Mean88 | FWE (8) |
|---|---:|---:|---:|
| Exact V / Full K | 89.367816 | 89.488636 | 100.000000 |
| Exact V / ShadowKV | 87.969349 | 88.106061 | 91.666667 |
| V96 / Full K | 83.716475 | 83.901515 | 79.166667 |
| V96 / ShadowKV | 77.298851 | 77.556818 | 75.000000 |

ShadowKV minus Full-K: exact V -1.398467 points; V96 -6.417625 points. Gap increases by5.019157 points. This small pilot supports a compression interaction; it does not identify its mechanism or establish a universal effect.

V96 per-task means below also exclude86 (QA2 has7 samples):

| Task | Full-K + V96 | ShadowKV + V96 |
|---|---:|---:|
| niah_single_1 | 100.000000 | 100.000000 |
| niah_single_2 | 100.000000 | 100.000000 |
| niah_single_3 | 100.000000 | 87.500000 |
| niah_multikey_1 | 100.000000 | 100.000000 |
| niah_multikey_2 | 75.000000 | 87.500000 |
| niah_multiquery | 81.250000 | 59.375000 |
| niah_multivalue | 100.000000 | 68.750000 |
| vt | 100.000000 | 87.500000 |
| fwe | 79.166667 | 75.000000 |
| qa_1 | 50.000000 | 50.000000 |
| qa_2 | 28.571429 | 28.571429 |

Environment lowrank; direct GPUs3/6,2CPU threads each; no Slurm.
Commands: python -u evaluation/eval_llama31_shadow_v96.py evaluate --arm full/shadowkv; then summarize.
Protocol: docs/llama31_base_shadow_v96_protocol.md. Logs: results/logs/llama31_shadow_v96.
No evaluation failures after launch; 32 checkpoint factors and model identity verified. Results use greedy task-capped generation. Resident accuracy adaptation, not an official Instruct or offload-throughput reproduction.
