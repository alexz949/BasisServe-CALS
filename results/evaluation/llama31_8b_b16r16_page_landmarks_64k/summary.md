# Llama-3.1-8B-Instruct B16R16 Parameter-Free Page Landmarks

## Conclusion

Not yet under the declared diagnostic rule: LM8x4 loses too much selection/output fidelity, so no RULER follow-up should be launched from this result.

This is a parameter-free diagnostic of the existing B16R16 fit. No Base/Residual refit, ALS, SVD/RRR landmark fit, or learned pooling was used.

## Pooled diagnostic

| Variant | Reps/Page32 | Scan dims/token | Page rel-MSE | Pearson | Recall vs teacher | Recall vs Exact-QK | Attention mass | Non-sink mass | Output rel-MSE | Wo rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TOKEN_TEACHER | 32 | n/a | 0 | 1.000000 | 1.0000 | 0.8344 | 0.9218 | 0.8248 | 0.00650413 | 0.00542337 |
| LM32x1 | 1 | 4.5 | 0.0409173 | 0.954005 | 0.7832 | 0.7453 | 0.8969 | 0.7417 | 0.0106421 | 0.0088727 |
| LM16x2 | 2 | 9 | 0.0343923 | 0.958170 | 0.7990 | 0.7553 | 0.8976 | 0.7433 | 0.010281 | 0.00858408 |
| LM8x4 | 4 | 18 | 0.0270395 | 0.963850 | 0.8139 | 0.7634 | 0.8984 | 0.7455 | 0.00992289 | 0.00830229 |
| LM4x8 | 8 | 36 | 0.0182181 | 0.972505 | 0.8317 | 0.7722 | 0.8996 | 0.7488 | 0.00936878 | 0.00776081 |
| EXACT_K_LM8x4 | 4 | 16 | 0.0290423 | 0.956874 | 0.7710 | 0.8062 | 0.9007 | 0.7503 | 0.00907401 | 0.007567 |
| EXACT_QK | 32 | n/a | 0.00397272 | 0.988360 | 0.8344 | 1.0000 | 0.9257 | 0.8335 | 0.00551689 | 0.00456578 |

The teacher's deployable persistent state is Base16+Residual16 (32 dimensions/token) plus transient Base reconstruction. The 144D materialized teacher state used by this diagnostic is not claimed as equivalent storage. Landmark scan costs are BF16 coordinates scanned per original token.

## LM8x4 decision audit

The diagnostic rule was declared in the result artifact: routed page recall against TOKEN_TEACHER >= 0.80, attention-mass loss <= 0.01, and Wo rel-MSE <= 1.25x TOKEN_TEACHER. It is only a gate for the specified RULER follow-up, not a general quality claim.

- Routed page recall vs teacher: 0.813943
- Attention-mass change vs teacher: -0.023354
- Wo rel-MSE ratio vs teacher: 1.530836x
- Follow-up gate: FAIL

## Protocol and audits

- 32 layers, 16 held-out C4 64K windows, 256 continuous online tail queries per window.
- Hard B=2048 support, Page32, pinned Page0/sink32, exact recent64, unchanged physical GQA aggregation.
- Exact selected post-RoPE K and Dense V are used after routing; Wo is the original dense projection.
- Full pages are pooled once; the moving ragged historical boundary is rebuilt using only currently historical cached tokens.
- TOKEN_TEACHER uses the existing selector directly; landmark scores re-enter that selector through an exact Page-LSE-preserving proxy.
- Synthetic identical-token, weighted ragged-LSE, and proxy round-trip audits passed.
