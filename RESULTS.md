# Promoted results

This file indexes results already produced by the preserved code. It does not
claim that the CALS extraction reran the large GPU experiments. Commands and
fuller provenance remain in the copied summaries under results/.

## Llama-2-7B, K dense, V retained at 75%

WikiText-2, matched V-cache width 3072:

| Method | V latent structure | PPL |
|---|---:|---:|
| Dense | 32 × 128 | 5.472056328 |
| PaLU M-LRD | 32 × 96 | 6.946421792 |
| PaLU G-LRD4 | 8 × 384 | 6.400272684 |
| **C1 joint** | **32 × 96** | **6.231372767** |

C1 improves over PaLU M-LRD by 0.7150 PPL and over G-LRD4 by 0.1689 PPL.
Unlike G-LRD4, each query head consumes only a 96-wide Value latent.

The full algorithm/protocol discussion is copied at
[results/llama2_mha_v25/summary.md](results/llama2_mha_v25/summary.md).

## Llama-2 V64 initialization/ALS ablation

The 32-layer weight-only initialization experiment used 128 × 2048 C4 fit
windows, decoder-closed selection, up to 20 encoder sweeps and CG32:

| Metric | Value |
|---|---:|
| Mean fit relative MSE | 0.11606255 |
| Mean held-out relative MSE | 0.17868945 |
| WikiText-2 PPL, 146 × 2048 windows | 6.82520229 |

See
[the V64 summary](results/llama2_mha_v25/c1_weight_only_v64_c4_128x2048_closed_s20_cg32/summary.md).

## Per-TP-source Global-KL at V64 budget

Candidate ranks were {32, 48, 64, 80, 96} under an exact ideal ragged-wire
budget equal to uniform V64:

| Ownership/allocation | Uniform PPL | Selected PPL | Confirmation KL |
|---|---:|---:|---:|
| Contiguous TP sources + mean-DP | 6.896116100 | **6.578383605** | 0.120138462 |
| CKA-balanced TP groups + mean-DP | 6.895931596 | **6.600288685** | 0.106762856 |

The selected schedules have 41–44% padding overhead under rectangular
AllGather, so their communication saving requires the included exact-size
ragged collective rather than padded native AllGather.

See the copied
[contiguous-source summary](results/llama2_mha_v25/c1_tp_source_global_kl_v64_r32_96_b4_c4p8c8_s512_wt146/summary.md)
and
[CKA-group summary](results/llama2_mha_v25/c1_cka_tp8_global_kl_v64_r32_96_c4p8c8_s512_wt146/summary.md).

## Qwen3-8B Base GQA

At V-cache width 768 (75% V retention):

| Method | PPL |
|---|---:|
| Dense PaLU protocol | 6.998721290 |
| PaLU M-LRD | 9.211224584 |
| PaLU G-LRD2 | 9.124840272 |
| PaLU G-LRD4 | 9.126794526 |
| Dense routed-C1 protocol | 7.003384583 |
| Routed C1 V96 | 7.372747645 |
| Pair C2 | **7.343676206** |

The two evaluator protocols have a 0.0047 dense-baseline offset, so raw
cross-protocol PPL should not be interpreted as bitwise matched.

The private-AllGather pipeline and its rank/communication-quality curve are
copied at
[the Qwen3 pipeline summary](results/q3base_c1v96_allgather_foio_pipeline_summary_20260816.md).

## Scope limits

- The Qwen3.5 hybrid builder/runtime/PPL code is included, but this index does
  not promote an unverified adaptive-rank Qwen3.5 result.
- The CKA and Global-KL allocations above are quality-reference runs.
- Padded folded-Hugging-Face evaluation preserves the compressed function but
  is not a throughput claim for the custom collective kernels.
