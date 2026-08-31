# Qwen3-8B Store80 chunked-prefill diagnostic

Each row compares final next-token logits on the identical token segment and position IDs.

| segment | comparison | top1 | top10 overlap | rel-RMSE | cosine | KL |
|:---|:---|---:|---:|---:|---:|---:|
| prefix_positions | dense_chunked_implicit_vs_full | yes | 1.0000 | 0.013350 | 0.999913 | 0.000000 |
| prefix_positions | dense_chunked_explicit_vs_full | yes | 1.0000 | 0.016000 | 0.999878 | 0.000000 |
| prefix_positions | store80_full_vs_dense_full | yes | 0.8000 | 0.171221 | 0.985689 | 0.000036 |
| prefix_positions | store80_chunked_implicit_vs_full | yes | 1.0000 | 0.019030 | 0.999819 | 0.000000 |
| prefix_positions | store80_chunked_explicit_vs_full | yes | 1.0000 | 0.019030 | 0.999819 | 0.000000 |
| prefix_positions | store80_chunked_implicit_vs_dense_full | yes | 0.8000 | 0.168068 | 0.986191 | 0.000035 |
| actual_tail_positions | dense_chunked_implicit_vs_full | yes | 1.0000 | 0.014414 | 0.999896 | 0.006921 |
| actual_tail_positions | dense_chunked_explicit_vs_full | yes | 1.0000 | 0.015902 | 0.999880 | 0.004787 |
| actual_tail_positions | store80_full_vs_dense_full | no | 0.1000 | 0.659599 | 0.763161 | 6.741167 |
| actual_tail_positions | store80_chunked_implicit_vs_full | yes | 0.9000 | 0.020226 | 0.999803 | 0.003706 |
| actual_tail_positions | store80_chunked_explicit_vs_full | yes | 0.9000 | 0.020226 | 0.999803 | 0.003706 |
| actual_tail_positions | store80_chunked_implicit_vs_dense_full | no | 0.1000 | 0.659650 | 0.762588 | 6.717779 |

## Next-token predictions

### prefix_positions

- `dense_full`: `576` / `' The'`
- `dense_chunked_implicit_mask`: `576` / `' The'`
- `dense_chunked_explicit_mask`: `576` / `' The'`
- `store80_full`: `576` / `' The'`
- `store80_chunked_implicit_mask`: `576` / `' The'`
- `store80_chunked_explicit_mask`: `576` / `' The'`

### actual_tail_positions

- `dense_full`: `220` / `' '`
- `dense_chunked_implicit_mask`: `220` / `' '`
- `dense_chunked_explicit_mask`: `220` / `' '`
- `store80_full`: `1182` / `' back'`
- `store80_chunked_implicit_mask`: `1182` / `' back'`
- `store80_chunked_explicit_mask`: `1182` / `' back'`
