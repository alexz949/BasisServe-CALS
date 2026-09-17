# Llama-3.1-8B-Instruct Chunk8 Landmark Geometry

This is the no-fit Stage-0 diagnostic for direct Chunk8 Fisher landmarks. It uses Dense V128, the frozen existing Base16/B16R16 factors, exact selected K, Dense V, and original Wo.

## Pooled results

| Variant | Scan dims/token | Chunk rel-MSE | Recall vs Exact Chunk8 | Recall vs B16R16 Chunk8 | Attention mass | Non-sink mass | Output rel-MSE | Wo rel-MSE | Support tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TOKEN_B16R16_PAGE32 | n/a | n/a | 0.6436 | 0.6697 | 0.9218 | 0.8248 | 0.00650413 | 0.00542337 | 2036.6 |
| EXACT_QK_PAGE32 | 128 | n/a | 0.6677 | 0.6432 | 0.9257 | 0.8335 | 0.00551689 | 0.00456578 | 2036.5 |
| TOKEN_B16R16_CHUNK8_LSE | n/a | 0.00360868 | 0.8118 | 1.0000 | 0.9308 | 0.8458 | 0.00497517 | 0.00406824 | 2048.0 |
| EXACT_QK_CHUNK8_LSE | 128 | 0 | 1.0000 | 0.8118 | 0.9350 | 0.8555 | 0.00421435 | 0.00343885 | 2048.0 |
| EXACT_K_MEAN_CHUNK8 | 16 | 0.0170449 | 0.7909 | 0.7466 | 0.9077 | 0.7700 | 0.00737214 | 0.00606806 | 2048.0 |
| BASE16_MEAN_CHUNK8 | 16 | 0.0280329 | 0.6565 | 0.6669 | 0.8928 | 0.7333 | 0.013066 | 0.0115091 | 2048.0 |
| BASE16_OLD_R16_MEAN_CHUNK8 | 18 | 0.0193391 | 0.7408 | 0.7980 | 0.9048 | 0.7636 | 0.00816554 | 0.00674574 | 2048.0 |

## Geometry decomposition

- Chunk8 granularity ceiling: Exact-QK Wo rel-MSE 0.00456578 (Page32) -> 0.00343885 (Chunk8).
- Mean-pooling gap: Exact Chunk8-LSE 0.00343885 -> Exact-K mean 0.00606806.
- Frozen B16 representation gap before pooling: Exact Chunk8-LSE 0.00343885 -> token B16R16 Chunk8-LSE 0.00406824.
- Existing residual-mean contribution: Base16 mean 0.0115091 -> Base16+old-R16 mean 0.00674574.

Chunk8 arms use exactly 4 pinned sink chunks + 244 freely routed historical chunks + exact recent64, for 2048 unique logical attention tokens. No fit or learned landmark was used in this stage.
