# Qwen3-8B Store80 chunked-prefill diagnostic

Each row compares final next-token logits on the identical token segment and position IDs.

## Interpretation

Chunked prefill is not the failure source: dense and Store80 each preserve their full-prefill Top1 under chunk512, and explicit versus implicit causal masks give the same Store80 result. On the same tail tokens, Store80 matches dense Top1 at positions 0--2047 but diverges at the prompt's actual 30K--32K positions. The Store80-versus-dense KL rises from `0.211071` to `6.741167`. This isolates absolute-position/RoPE extrapolation as the primary failure in this sample, before Route32 is enabled.

## Comparisons

| segment | comparison | top1 | top10 overlap | rel-RMSE | cosine | KL |
|:---|:---|---:|---:|---:|---:|---:|
| tail_zero_positions | dense_chunked_implicit_vs_full | yes | 1.0000 | 0.014891 | 0.999904 | 0.000163 |
| tail_zero_positions | dense_chunked_explicit_vs_full | yes | 1.0000 | 0.019856 | 0.999848 | 0.000093 |
| tail_zero_positions | store80_full_vs_dense_full | yes | 0.6000 | 0.207208 | 0.978611 | 0.211071 |
| tail_zero_positions | store80_chunked_implicit_vs_full | yes | 0.9000 | 0.015749 | 0.999879 | 0.007834 |
| tail_zero_positions | store80_chunked_explicit_vs_full | yes | 0.9000 | 0.015749 | 0.999879 | 0.007834 |
| tail_zero_positions | store80_chunked_implicit_vs_dense_full | yes | 0.7000 | 0.205325 | 0.978938 | 0.222430 |
| actual_tail_positions | dense_chunked_implicit_vs_full | yes | 1.0000 | 0.014414 | 0.999896 | 0.006921 |
| actual_tail_positions | dense_chunked_explicit_vs_full | yes | 1.0000 | 0.015902 | 0.999880 | 0.004787 |
| actual_tail_positions | store80_full_vs_dense_full | no | 0.1000 | 0.659599 | 0.763161 | 6.741167 |
| actual_tail_positions | store80_chunked_implicit_vs_full | yes | 0.9000 | 0.020226 | 0.999803 | 0.003706 |
| actual_tail_positions | store80_chunked_explicit_vs_full | yes | 0.9000 | 0.020226 | 0.999803 | 0.003706 |
| actual_tail_positions | store80_chunked_implicit_vs_dense_full | no | 0.1000 | 0.659650 | 0.762588 | 6.717779 |

## Next-token predictions

### tail_zero_positions

- `dense_full`: `220` / `' '`
- `dense_chunked_implicit_mask`: `220` / `' '`
- `dense_chunked_explicit_mask`: `220` / `' '`
- `store80_full`: `220` / `' '`
- `store80_chunked_implicit_mask`: `220` / `' '`
- `store80_chunked_explicit_mask`: `220` / `' '`

### actual_tail_positions

- `dense_full`: `220` / `' '`
- `dense_chunked_implicit_mask`: `220` / `' '`
- `dense_chunked_explicit_mask`: `220` / `' '`
- `store80_full`: `1182` / `' back'`
- `store80_chunked_implicit_mask`: `1182` / `' back'`
- `store80_chunked_explicit_mask`: `1182` / `' back'`
