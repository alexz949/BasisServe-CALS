# Qwen3-8B Wo-only C1 vs LR-AllReduce: whole-model quality

Every arm keeps V and the KV cache dense. Factorized collective maps are folded into equivalent BF16 `o_proj` weights, so these results isolate approximation quality from runtime kernels.

| Arm | WikiText-2 PPL | Δ vs dense | C4 validation PPL | Δ vs dense |
|---|---:|---:|---:|---:|
| `dense` | 7.00250905 | +0.000% | 9.16859426 | +0.000% |
| `wo_c1_ag` | 7.27974718 | +3.959% | 9.32014486 | +1.653% |
| `wo_lr_ar_wire` | 7.54842926 | +7.796% | 9.73659137 | +6.195% |
| `wo_lr_ar_capacity` | 7.03978690 | +0.532% | 9.20900684 | +0.441% |

Protocol:

- WikiText-2 test, concatenated non-overlapping 2048-token chunks.
- C4 validation, 128 document-disjoint 2048-token windows.
- BF16 model execution, SDPA attention, FP32 cross-entropy accumulation.
- C4 audit documents are disjoint from Phase-1 C4 train documents.
